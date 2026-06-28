"""Conformer alignment using the largest rigid ring system as reference scaffold.

Priority for anchor selection:
  1. Largest *connected* ring system (fused rings counted together): single stable
     reference frame even when the molecule has multiple ring systems separated by
     a flexible linker (e.g. two phenyl groups on a PEG chain).
  2. Atoms on double/aromatic bonds when no rings exist (conjugated systems).
  3. All heavy atoms when no planar features are found (saturated chains).

Using ALL ring atoms of a multi-ring molecule (two separate aromatic groups, for
instance) splits the RMSD budget between both systems and gives a poor overlay —
identical to the failure of using all heavy atoms.  Anchoring on one ring system
keeps the scaffold fixed and lets the flexible linker vary freely.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def _largest_ring_system_ids(mol: Chem.Mol) -> list[int]:
    """Heavy-atom indices of the largest connected set of fused rings.

    Rings that share at least one atom are merged into one system.  The largest
    system (by atom count) is returned.  Returns [] when the molecule has no rings.
    """
    atom_rings = mol.GetRingInfo().AtomRings()
    if not atom_rings:
        return []
    systems: list[set[int]] = []
    for ring in atom_rings:
        ring_set = set(ring)
        merged: list[set[int]] = []
        remaining: list[set[int]] = []
        for sys in systems:
            (merged if sys & ring_set else remaining).append(sys)
        combined = ring_set.union(*merged) if merged else ring_set
        systems = remaining + [combined]
    largest = max(systems, key=len)
    return sorted(i for i in largest if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def planar_atom_ids(mol: Chem.Mol) -> list[int]:
    """Return heavy-atom indices that form the best rigid scaffold for alignment.

    Returns the largest connected ring system when rings exist; falls back to
    atoms on double/aromatic bonds (no rings); returns [] for saturated systems.
    """
    # Prefer: largest connected ring system
    ring_ids = _largest_ring_system_ids(mol)
    if ring_ids:
        return ring_ids
    # Fallback: atoms on double or aromatic bonds
    ids: set[int] = set()
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() >= 2 or bond.GetIsAromatic():
            for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1:
                    ids.add(idx)
    return sorted(ids)


def align_conformers(mol: Chem.Mol, ref_conf_id: int = 0) -> Chem.Mol:
    """Align all conformers to ref_conf_id using the largest rigid scaffold.

    Returns a new Mol with aligned conformers.  Falls back to all heavy atoms
    when no planar features are detected (e.g. saturated linear chain).
    """
    mol_aligned = Chem.Mol(mol)
    anchor_ids = planar_atom_ids(mol_aligned)
    if not anchor_ids:
        anchor_ids = [
            a.GetIdx() for a in mol_aligned.GetAtoms() if a.GetAtomicNum() > 1
        ]
    all_ids = [c.GetId() for c in mol_aligned.GetConformers()]
    ordered = [ref_conf_id] + [cid for cid in all_ids if cid != ref_conf_id]
    AllChem.AlignMolConformers(mol_aligned, atomIds=anchor_ids, confIds=ordered)
    return mol_aligned
