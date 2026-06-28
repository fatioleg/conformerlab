"""openconf backend (optional).

openconf is Rowan's open-source Monte Carlo conformer generator
(https://github.com/rowansci/openconf) -- the same one behind the web UI you
were looking at. Installing it (``pip install openconf``) makes this backend
available; if it is absent, ``is_available()`` returns False and the factory
falls back to RDKit rather than crashing.

The conformer *energies* here are still MMFF94s (openconf scores with a
forcefield) and are reported in kcal/mol straight from ``ConformerEnsemble``.
To get xTB/MLIP-quality energies, generate with this backend and then re-rank
with ``refine.mlip`` -- that is the local equivalent of Rowan's "generate then
refine" pipeline.

Geometry bridge: openconf hands back its own ``ConformerEnsemble``; downstream
analysis (RMSD, r_g, SASA) needs an RDKit ``Mol`` with one conformer per record
whose id matches ``ConformerRecord.conf_id``. We rebuild that mol from the
ensemble's SDF, assigning conformer ids 0..n-1 aligned with the energy order.
"""

from __future__ import annotations

import os
import tempfile

from rdkit import Chem

from conformerlab.backends.base import ConformerBackend
from conformerlab.core.errors import BackendNotAvailableError, EmptyEnsembleError
from conformerlab.core.molecule import mol_from_input
from conformerlab.core.types import (
    ConformerRecord,
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
)

_OPENCONF_PRESETS = {"rapid", "ensemble", "docking", "macrocycle"}


def _ensemble_to_rdkit(ensemble_mol: object, n_expected: int) -> Chem.Mol:
    """Rebuild a single RDKit Mol (Hs kept) with conformer ids 0..n-1.

    Reads the openconf ``ConformerEnsemble`` via its SDF export and packs every
    conformer onto one mol, with ids aligned to the ensemble's energy order so
    each ``ConformerRecord.conf_id`` indexes the matching geometry.
    """
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".sdf")
        os.close(fd)
        ensemble_mol.to_sdf(path)  # type: ignore[attr-defined]
        supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        mols = [m for m in supplier if m is not None]
    except AttributeError as exc:  # pragma: no cover - API drift guard
        raise BackendNotAvailableError(
            f"openconf API surface changed unexpectedly: {exc}"
        ) from exc
    finally:
        if path is not None and os.path.exists(path):
            os.remove(path)

    if len(mols) != n_expected:
        raise BackendNotAvailableError(
            f"openconf SDF had {len(mols)} conformers, expected {n_expected}."
        )

    base = Chem.Mol(mols[0])
    base.RemoveAllConformers()
    for i, m in enumerate(mols):
        conf = m.GetConformer()
        conf.SetId(i)
        base.AddConformer(conf, assignId=False)
    return base


class OpenConfBackend(ConformerBackend):
    name = "openconf"

    def is_available(self) -> bool:
        try:
            import openconf  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(
        self, molecule: MoleculeInput, settings: GenerationSettings
    ) -> EnsembleResult:
        if not self.is_available():
            raise BackendNotAvailableError(
                "openconf is not installed. Run `pip install openconf`, "
                "or use the rdkit backend."
            )
        from openconf import generate_conformers, preset_config

        mol = mol_from_input(molecule, add_hs=True)
        preset = settings.preset if settings.preset in _OPENCONF_PRESETS else "rapid"
        config = preset_config(preset)
        config.max_out = settings.max_conformers
        config.energy_window_kcal = settings.energy_window_kcal
        config.random_seed = settings.random_seed

        ensemble_mol = generate_conformers(mol, config=config)
        energies = [float(e) for e in ensemble_mol.energies]  # kcal/mol
        if not energies:
            raise EmptyEnsembleError("openconf returned no conformers.")

        rdkit_mol = _ensemble_to_rdkit(ensemble_mol, n_expected=len(energies))
        records = [
            ConformerRecord(conf_id=i, backend=self.name, energy_kcal=e)
            for i, e in enumerate(energies)
        ]
        result = EnsembleResult(
            molecule=molecule, settings=settings,
            backend=self.name, records=records,
        )
        return result.attach_mol(rdkit_mol)
