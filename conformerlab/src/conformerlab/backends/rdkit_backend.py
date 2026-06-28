"""RDKit ETKDGv3 + MMFF94s backend.

This is the default, dependency-free generator: it runs anywhere RDKit runs,
with no credits and no external services. It is the reference implementation
of the ConformerBackend contract and the one the tests exercise end to end.

Energies are MMFF94s in kcal/mol straight out of RDKit.
"""

from __future__ import annotations

from rdkit.Chem import AllChem

from conformerlab.backends.base import ConformerBackend
from conformerlab.core.errors import EmptyEnsembleError
from conformerlab.core.molecule import mol_from_input
from conformerlab.core.types import (
    ConformerRecord,
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
)

_PRESET_TO_NCONF = {
    "rapid": 50,
    "ensemble": 200,
    "docking": 200,
    "macrocycle": 300,
}


class RDKitBackend(ConformerBackend):
    name = "rdkit-etkdg-mmff94s"

    def is_available(self) -> bool:
        return True  # RDKit is a hard dependency of the core

    def generate(
        self, molecule: MoleculeInput, settings: GenerationSettings
    ) -> EnsembleResult:
        mol = mol_from_input(molecule, add_hs=True)
        mol.RemoveAllConformers()  # SDF/XYZ inputs carry coords; ETKDG re-embeds

        n_request = min(
            settings.max_conformers,
            _PRESET_TO_NCONF.get(settings.preset, settings.max_conformers),
        )
        params = AllChem.ETKDGv3()
        params.randomSeed = settings.random_seed
        params.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_request, params=params)
        if len(conf_ids) == 0:
            raise EmptyEnsembleError(
                f"ETKDG embedded no conformers for {molecule.name!r}."
            )

        # MMFF94s optimisation; returns (converged_flag, energy_kcal) per conf.
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        results = []
        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            ff.Minimize(maxIts=2000)
            results.append((cid, ff.CalcEnergy()))

        records = [
            ConformerRecord(conf_id=cid, backend=self.name, energy_kcal=energy)
            for cid, energy in results
        ]
        ensemble = EnsembleResult(
            molecule=molecule,
            settings=settings,
            backend=self.name,
            records=records,
        )
        return ensemble.attach_mol(mol)
