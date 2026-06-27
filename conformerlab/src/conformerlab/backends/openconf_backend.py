"""openconf backend (optional).

openconf is Rowan's open-source Monte Carlo conformer generator
(https://github.com/rowansci/openconf) -- the same one behind the web UI you
were looking at. Installing it (``pip install openconf``) makes this backend
available; if it is absent, ``is_available()`` returns False and the factory
falls back to RDKit rather than crashing.

The conformer *energies* here are still MMFF94s (openconf scores with a
forcefield). To get xTB/MLIP-quality energies, generate with this backend and
then re-rank with ``refine.mlip`` -- that is the local equivalent of Rowan's
"generate then refine" pipeline.
"""

from __future__ import annotations

from conformerlab.backends.base import ConformerBackend
from conformerlab.core.errors import BackendNotAvailableError, EmptyEnsembleError
from conformerlab.core.molecule import mol_from_smiles
from conformerlab.core.types import (
    ConformerRecord,
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
)


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

        mol = mol_from_smiles(molecule.smiles, add_hs=True)
        config = preset_config(settings.preset if settings.preset in
                               {"rapid", "ensemble", "docking", "macrocycle"}
                               else "rapid")
        config.max_out = settings.max_conformers
        config.energy_window_kcal = settings.energy_window_kcal
        config.random_seed = settings.random_seed

        ensemble_mol = generate_conformers(mol, config=config)
        # openconf returns an object exposing per-conformer MMFF energies;
        # we read them through its documented API. Kept defensive so a minor
        # API change degrades to a clear error, not a silent wrong number.
        try:
            energies = list(ensemble_mol.energies_kcal)  # type: ignore[attr-defined]
            rdkit_mol = ensemble_mol.to_rdkit()           # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover - API drift guard
            raise BackendNotAvailableError(
                f"openconf API surface changed unexpectedly: {exc}"
            ) from exc

        if not energies:
            raise EmptyEnsembleError("openconf returned no conformers.")

        records = [
            ConformerRecord(conf_id=i, backend=self.name, energy_kcal=float(e))
            for i, e in enumerate(energies)
        ]
        result = EnsembleResult(
            molecule=molecule, settings=settings,
            backend=self.name, records=records,
        )
        return result.attach_mol(rdkit_mol)
