"""Loading molecules from SDF and XYZ files, plus MoleculeInput validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from conformerlab.backends.rdkit_backend import RDKitBackend
from conformerlab.core.errors import InvalidMoleculeFileError
from conformerlab.core.molecule import mol_from_input, mol_from_sdf, mol_from_xyz
from conformerlab.core.types import GenerationSettings, MoleculeInput


def _write_3d(smiles: str, ext: str, path):
    """Write a SMILES as an embedded 3D molecule to an SDF or XYZ file."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    if ext == "sdf":
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
    else:
        Chem.MolToXYZFile(mol, str(path))
    return path


def test_mol_from_sdf(tmp_path):
    p = _write_3d("CCO", "sdf", tmp_path / "ethanol.sdf")
    mol = mol_from_sdf(p)
    assert rdMolDescriptors.CalcMolFormula(mol) == "C2H6O"


def test_mol_from_xyz(tmp_path):
    p = _write_3d("CCO", "xyz", tmp_path / "ethanol.xyz")
    mol = mol_from_xyz(p, charge=0)
    assert rdMolDescriptors.CalcMolFormula(mol) == "C2H6O"


def test_mol_from_input_dispatch(tmp_path):
    sdf = _write_3d("CCO", "sdf", tmp_path / "e.sdf")
    mol = mol_from_input(MoleculeInput(sdf_path=str(sdf), name="ethanol"))
    assert rdMolDescriptors.CalcMolFormula(mol) == "C2H6O"


def test_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValidationError):
        MoleculeInput(name="no-source")
    with pytest.raises(ValidationError):
        MoleculeInput(smiles="CCO", xyz_path="x.xyz")


def test_missing_file_raises():
    with pytest.raises(InvalidMoleculeFileError):
        mol_from_sdf("/no/such/file.sdf")
    with pytest.raises(InvalidMoleculeFileError):
        mol_from_xyz("/no/such/file.xyz")


def test_backend_runs_from_sdf(tmp_path):
    sdf = _write_3d("CCO", "sdf", tmp_path / "e.sdf")
    ensemble = RDKitBackend().generate(
        MoleculeInput(sdf_path=str(sdf), name="ethanol"),
        GenerationSettings(max_conformers=3, random_seed=1),
    )
    assert ensemble.records
    assert ensemble.molecule.sdf_path == str(sdf)
