"""Tests for refine.mlip — ASE is mocked so no ML dep is required at test time."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from conformerlab.core.types import (
    ConformerRecord,
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
    RefinementSettings,
)
from conformerlab.refine.mlip import (
    _dedup_records,
    list_methods,
    refine_with_mlip,
    register_calculator,
)

_EV_TO_KCAL = 23.060547830619026


@pytest.fixture()
def ethane_ensemble() -> EnsembleResult:
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
    records = [
        ConformerRecord(conf_id=0, backend="rdkit", energy_kcal=-1.0),
        ConformerRecord(conf_id=1, backend="rdkit", energy_kcal=-0.5),
    ]
    ens = EnsembleResult(
        molecule=MoleculeInput(smiles="CC"),
        settings=GenerationSettings(),
        backend="rdkit",
        records=records,
    )
    ens.attach_mol(mol)
    return ens


def _ase_ctx(num_atoms: int, energy_ev: float = -10.0):
    """Return (context_manager, atoms_mock) patching sys.modules with mock ASE."""
    atoms_inst = MagicMock()
    atoms_inst.get_potential_energy.return_value = energy_ev
    atoms_inst.get_positions.return_value = [[0.0, 0.0, 0.0]] * num_atoms

    mock_ase = MagicMock()
    mock_ase.Atoms.return_value = atoms_inst

    mock_optimize = MagicMock()
    mock_optimize.LBFGS.return_value = MagicMock()

    ctx = patch.dict("sys.modules", {"ase": mock_ase, "ase.optimize": mock_optimize})
    return ctx, atoms_inst


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_methods_includes_builtins():
    assert "aimnet2" in list_methods()
    assert "mace-off" in list_methods()


def test_register_adds_method():
    register_calculator("_reg_test", lambda device: None)
    assert "_reg_test" in list_methods()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_method_raises(ethane_ensemble):
    settings = RefinementSettings(method="no_such_mlip_xyz_999")
    ctx, _ = _ase_ctx(ethane_ensemble.mol.GetNumAtoms())
    with ctx, pytest.raises(ValueError, match="Unknown MLIP method"):
        refine_with_mlip(ethane_ensemble, settings)


# ---------------------------------------------------------------------------
# Non-destructive contract
# ---------------------------------------------------------------------------


def test_refine_does_not_mutate_original(ethane_ensemble):
    orig_energies = [r.energy_kcal for r in ethane_ensemble.records]
    n = ethane_ensemble.mol.GetNumAtoms()

    register_calculator("_nd_test", lambda device: MagicMock())
    ctx, _ = _ase_ctx(n, energy_ev=-5.0)
    with ctx:
        refine_with_mlip(ethane_ensemble, RefinementSettings(method="_nd_test"))

    assert [r.energy_kcal for r in ethane_ensemble.records] == orig_energies


def test_refine_returns_new_object(ethane_ensemble):
    n = ethane_ensemble.mol.GetNumAtoms()
    register_calculator("_new_test", lambda device: MagicMock())
    ctx, _ = _ase_ctx(n)
    with ctx:
        refined = refine_with_mlip(
            ethane_ensemble, RefinementSettings(method="_new_test")
        )

    assert refined is not ethane_ensemble
    assert refined.records is not ethane_ensemble.records


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_refine_sets_mlip_method_on_records(ethane_ensemble):
    n = ethane_ensemble.mol.GetNumAtoms()
    register_calculator("_mm_test", lambda device: MagicMock())
    ctx, _ = _ase_ctx(n)
    with ctx:
        refined = refine_with_mlip(
            ethane_ensemble, RefinementSettings(method="_mm_test")
        )

    for r in refined.records:
        assert r.mlip_method == "_mm_test"


def test_refine_converts_energy_to_kcal(ethane_ensemble):
    energy_ev = -7.5
    n = ethane_ensemble.mol.GetNumAtoms()
    register_calculator("_eu_test", lambda device: MagicMock())
    ctx, _ = _ase_ctx(n, energy_ev=energy_ev)
    with ctx:
        refined = refine_with_mlip(
            ethane_ensemble, RefinementSettings(method="_eu_test")
        )

    expected = energy_ev * _EV_TO_KCAL
    for r in refined.records:
        assert abs(r.energy_kcal - expected) < 1e-9


def test_refine_progress_callback_fires_correctly(ethane_ensemble):
    n = ethane_ensemble.mol.GetNumAtoms()
    register_calculator("_pb_test", lambda device: MagicMock())

    calls: list[tuple[int, int]] = []
    ctx, _ = _ase_ctx(n)
    with ctx:
        refine_with_mlip(
            ethane_ensemble,
            RefinementSettings(method="_pb_test"),
            progress_callback=lambda d, t: calls.append((d, t)),
        )

    assert calls == [(1, 2), (2, 2)]


def test_force_trajectory_none_when_lbfgs_mocked(ethane_ensemble):
    """force_trajectory is None when the mocked LBFGS never fires the observer."""
    n = ethane_ensemble.mol.GetNumAtoms()
    register_calculator("_ft_test", lambda device: MagicMock())
    ctx, _ = _ase_ctx(n)
    with ctx:
        refined = refine_with_mlip(
            ethane_ensemble, RefinementSettings(method="_ft_test")
        )
    for r in refined.records:
        assert r.force_trajectory is None


def test_energy_only_force_trajectory_is_none(ethane_ensemble):
    """Single-point mode never populates force_trajectory."""
    n = ethane_ensemble.mol.GetNumAtoms()
    register_calculator("_sp_ft_test", lambda device: MagicMock())
    ctx, _ = _ase_ctx(n)
    with ctx:
        refined = refine_with_mlip(
            ethane_ensemble, RefinementSettings(method="_sp_ft_test", energy_only=True)
        )
    for r in refined.records:
        assert r.force_trajectory is None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _two_conf_mol(identical: bool):
    """Ethanol with two conformers; second is a copy of the first when *identical*."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=7)
    if identical:
        c0 = mol.GetConformer(0)
        c1 = mol.GetConformer(1)
        for i in range(mol.GetNumAtoms()):
            c1.SetAtomPosition(i, c0.GetAtomPosition(i))
    return mol


def test_dedup_collapses_identical_geometries():
    mol = _two_conf_mol(identical=True)
    recs = [
        ConformerRecord(conf_id=0, backend="x", energy_kcal=-1.0),
        ConformerRecord(conf_id=1, backend="x", energy_kcal=-0.5),
    ]
    kept = _dedup_records(mol, recs, rmsd_thresh=0.125)
    assert len(kept) == 1
    assert kept[0].conf_id == 0  # lowest-energy representative is kept


def test_dedup_keeps_distinct_geometries():
    mol = _two_conf_mol(identical=False)
    recs = [
        ConformerRecord(conf_id=0, backend="x", energy_kcal=-1.0),
        ConformerRecord(conf_id=1, backend="x", energy_kcal=-0.5),
    ]
    kept = _dedup_records(mol, recs, rmsd_thresh=0.01)
    assert len(kept) == 2


def test_dedup_disabled_by_default_in_settings():
    assert RefinementSettings().dedup_rmsd is None
