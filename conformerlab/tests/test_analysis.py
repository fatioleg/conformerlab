import math

from conformerlab.analysis.boltzmann import add_boltzmann_weights
from conformerlab.analysis.energy import add_delta_e_kcal
from conformerlab.analysis.selection import (
    select_by_energy_window,
    select_top_n,
)
from conformerlab.core.types import (
    ConformerRecord,
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
)


def _synthetic():
    recs = [
        ConformerRecord(conf_id=0, backend="t", energy_kcal=10.0),
        ConformerRecord(conf_id=1, backend="t", energy_kcal=12.282),
        ConformerRecord(conf_id=2, backend="t", energy_kcal=13.666),
    ]
    return EnsembleResult(
        molecule=MoleculeInput(smiles="CCCC"),
        settings=GenerationSettings(temperature_k=298.15),
        backend="t",
        records=recs,
    )


def test_delta_e_and_sort():
    e = add_delta_e_kcal(_synthetic())
    assert e.records[0].delta_e_kcal == 0.0
    assert math.isclose(e.records[1].delta_e_kcal, 2.282, abs_tol=1e-6)


def test_boltzmann_weights_sum_to_one():
    e = add_boltzmann_weights(add_delta_e_kcal(_synthetic()))
    total = sum(r.boltzmann_weight for r in e.records)
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    # 2.282 kcal/mol gap at 298 K -> ~2% population for conformer 2
    assert 0.01 < e.records[1].boltzmann_weight < 0.04
    assert e.records[0].boltzmann_weight > 0.9


def test_selection():
    e = add_delta_e_kcal(_synthetic())
    select_top_n(e, 2)
    assert [r.selected for r in e.records] == [True, True, False]
    select_by_energy_window(e, 2.5)
    assert [r.selected for r in e.records] == [True, True, False]
