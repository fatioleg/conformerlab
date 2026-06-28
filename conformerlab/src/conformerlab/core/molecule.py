"""Molecule validation and lightweight descriptors.

Pure cheminformatics: no conformer generation, no energies. The job here is to
turn a (possibly bad) input -- a SMILES string, an SDF file, or an XYZ file --
into a trustworthy RDKit Mol and report basic descriptors, raising a controlled
error on garbage input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds, rdMolDescriptors

from conformerlab.core.errors import InvalidMoleculeFileError, InvalidSmilesError

if TYPE_CHECKING:
    from conformerlab.core.types import MoleculeInput


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


def mol_from_sdf(path: str | Path, add_hs: bool = True) -> Chem.Mol:
    """Load the first molecule from an SDF file as an RDKit Mol (topology).

    Existing 3D coordinates are kept but are irrelevant to generation, which
    re-embeds from the topology.

    Raises:
        InvalidMoleculeFileError: if the file is missing or has no readable mol.
    """
    p = Path(path)
    if not p.is_file():
        raise InvalidMoleculeFileError(f"SDF file not found: {str(path)!r}")
    supplier = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise InvalidMoleculeFileError(f"No readable molecule in SDF: {str(path)!r}")
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)
    return mol


def mol_from_xyz(path: str | Path, charge: int = 0, add_hs: bool = True) -> Chem.Mol:
    """Load an XYZ file and perceive bonds into an RDKit Mol.

    XYZ stores coordinates but no connectivity, so bond orders are inferred with
    RDKit's ``rdDetermineBonds`` using the total ``charge``.

    Raises:
        InvalidMoleculeFileError: if the file is missing, unparseable, or bonds
            cannot be perceived.
    """
    p = Path(path)
    if not p.is_file():
        raise InvalidMoleculeFileError(f"XYZ file not found: {str(path)!r}")
    raw = Chem.MolFromXYZFile(str(p))
    if raw is None:
        raise InvalidMoleculeFileError(f"RDKit could not parse XYZ: {str(path)!r}")
    mol = Chem.Mol(raw)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except (ValueError, RuntimeError) as exc:
        raise InvalidMoleculeFileError(
            f"Could not perceive bonds from XYZ {str(path)!r}: {exc}"
        ) from exc
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)
    return mol


def mol_from_input(molecule: MoleculeInput, add_hs: bool = True) -> Chem.Mol:
    """Build an RDKit Mol from a ``MoleculeInput``, dispatching on its source.

    Exactly one of ``smiles``/``sdf_path``/``xyz_path`` is set (enforced by the
    model), so backends call this once instead of knowing about input formats.
    """
    if molecule.smiles is not None:
        return mol_from_smiles(molecule.smiles, add_hs=add_hs)
    if molecule.sdf_path is not None:
        return mol_from_sdf(molecule.sdf_path, add_hs=add_hs)
    if molecule.xyz_path is not None:
        return mol_from_xyz(
            molecule.xyz_path, charge=molecule.charge, add_hs=add_hs
        )
    raise InvalidMoleculeFileError("MoleculeInput has no molecule source set.")


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
