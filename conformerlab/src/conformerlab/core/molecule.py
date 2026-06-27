"""Molecule validation and lightweight descriptors.

Pure cheminformatics: no conformer generation, no energies. The single job
here is to turn a (possibly bad) SMILES into a trustworthy RDKit Mol and
report basic descriptors, raising a controlled error on garbage input.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from conformerlab.core.errors import InvalidSmilesError


@dataclass(frozen=True)
class MoleculeDescriptors:
    formula: str
    molecular_weight: float      # g/mol
    n_heavy_atoms: int
    n_rotatable_bonds: int


def mol_from_smiles(smiles: str, add_hs: bool = True) -> Chem.Mol:
    """Parse a SMILES into an RDKit Mol, adding explicit hydrogens by default.

    Raises:
        InvalidSmilesError: if the string is empty or unparseable.
    """
    if not smiles or not smiles.strip():
        raise InvalidSmilesError("Empty SMILES string.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSmilesError(f"RDKit could not parse SMILES: {smiles!r}")
    if add_hs:
        mol = Chem.AddHs(mol)
    return mol


def is_valid_smiles(smiles: str) -> bool:
    """Return True if the SMILES parses, without raising."""
    try:
        mol_from_smiles(smiles, add_hs=False)
        return True
    except InvalidSmilesError:
        return False


def describe(smiles: str) -> MoleculeDescriptors:
    """Compute formula, MW, heavy-atom count and rotatable-bond count."""
    mol = mol_from_smiles(smiles, add_hs=True)
    return MoleculeDescriptors(
        formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        n_heavy_atoms=mol.GetNumHeavyAtoms(),
        n_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
    )
