import math

import pytest

from conformerlab.core.errors import InvalidSmilesError
from conformerlab.core.molecule import describe, is_valid_smiles, mol_from_smiles


def test_butane_descriptors():
    d = describe("CCCC")
    assert d.formula == "C4H10"
    assert math.isclose(d.molecular_weight, 58.12, abs_tol=0.05)
    assert d.n_heavy_atoms == 4
    assert d.n_rotatable_bonds == 1


def test_ibuprofen_parses():
    assert is_valid_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    d = describe("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert d.formula == "C13H18O2"
    assert d.n_heavy_atoms == 15


def test_resveratrol_parses():
    smi = "Oc1ccc(cc1)/C=C/c1cc(O)cc(O)c1"
    assert is_valid_smiles(smi)
    d = describe(smi)
    assert d.n_heavy_atoms == 17


def test_invalid_smiles_raises():
    assert not is_valid_smiles("this-is-not-smiles((")
    with pytest.raises(InvalidSmilesError):
        mol_from_smiles("")
    with pytest.raises(InvalidSmilesError):
        mol_from_smiles("C(C(C")
