"""Tests for analysis.align — planar-scaffold conformer alignment."""

from rdkit import Chem
from rdkit.Chem import AllChem

from conformerlab.analysis.align import (
    _largest_ring_system_ids,
    align_conformers,
    planar_atom_ids,
)


def _mol_with_conformers(smiles: str, n: int = 3) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    return mol


# ── planar_atom_ids ────────────────────────────────────────────────────────────

def test_planar_atoms_benzene():
    mol = Chem.MolFromSmiles("c1ccccc1")
    ids = planar_atom_ids(mol)
    # all 6 ring carbons are heavy and in a ring
    assert len(ids) == 6
    assert all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ids)


def test_planar_atoms_acetylene():
    mol = Chem.MolFromSmiles("C#C")
    ids = planar_atom_ids(mol)
    # triple bond ≥ 2 — both carbons qualify
    assert len(ids) == 2


def test_planar_atoms_ethanol_no_ring_no_double():
    mol = Chem.MolFromSmiles("CCO")
    ids = planar_atom_ids(mol)
    # no rings, no double bonds → empty → fallback handled by align_conformers
    assert ids == []


def test_planar_atoms_ibuprofen_returns_only_ring():
    # ibuprofen has one benzene ring — planar_atom_ids should return exactly those 6 carbons
    mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    ids = planar_atom_ids(mol)
    assert len(ids) == 6
    assert all(mol.GetAtomWithIdx(i).IsInRing() for i in ids)


def test_largest_ring_system_two_separate_rings():
    # biphenyl-ether: two phenyl rings connected by O — two separate ring systems of 6
    mol = Chem.MolFromSmiles("c1ccc(Oc2ccccc2)cc1")
    ids = _largest_ring_system_ids(mol)
    # each ring has 6 atoms; both are the same size, so one is returned
    assert len(ids) == 6


def test_largest_ring_system_fused_is_bigger():
    # naphthalene (fused bicyclic, 10 ring atoms) + a separate benzene (6) connected by a chain
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1CCc1ccccc1")
    ids = _largest_ring_system_ids(mol)
    # naphthalene system has 10 atoms, benzene has 6 → largest wins
    assert len(ids) == 10


# ── align_conformers ───────────────────────────────────────────────────────────

def test_align_returns_mol_with_same_conformer_count():
    mol = _mol_with_conformers("c1ccccc1", n=4)
    aligned = align_conformers(mol, ref_conf_id=0)
    assert aligned.GetNumConformers() == mol.GetNumConformers()


def test_align_ref_conformer_unchanged():
    """Coordinates of the reference conformer must not move."""
    mol = _mol_with_conformers("c1ccccc1CC", n=3)
    ref_id = 0
    ref_before = list(mol.GetConformer(ref_id).GetPositions().flatten())
    aligned = align_conformers(mol, ref_conf_id=ref_id)
    ref_after = list(aligned.GetConformer(ref_id).GetPositions().flatten())
    for a, b in zip(ref_before, ref_after):
        assert abs(a - b) < 1e-4


def test_align_fallback_no_planar_atoms():
    """Molecules with no planar atoms (alkane) must still align without error."""
    mol = _mol_with_conformers("CCCCCC", n=3)
    aligned = align_conformers(mol, ref_conf_id=0)
    assert aligned.GetNumConformers() == 3
