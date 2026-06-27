import math

from conformerlab.analysis.pipeline import analyze
from conformerlab.backends.factory import get_backend
from conformerlab.io.export import to_dataframe, write_csv
from conformerlab.core.types import GenerationSettings, MoleculeInput


def _run(smiles, name, n=10):
    backend = get_backend("rdkit")
    mol_in = MoleculeInput(smiles=smiles, name=name)
    settings = GenerationSettings(preset="rapid", max_conformers=n, random_seed=1)
    ensemble = backend.generate(mol_in, settings)
    return analyze(ensemble)


def test_ibuprofen_generates_and_analyses():
    e = _run("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ibuprofen", n=8)
    assert len(e.records) >= 1
    assert e.records[0].delta_e_kcal == 0.0
    assert e.records[0].rmsd_to_min_ang == 0.0
    total = sum(r.boltzmann_weight for r in e.records)
    assert math.isclose(total, 1.0, abs_tol=1e-6)
    assert all(r.radius_of_gyration_ang > 0 for r in e.records)


def test_resveratrol_generates():
    e = _run("Oc1ccc(cc1)/C=C/c1cc(O)cc(O)c1", "resveratrol", n=6)
    assert len(e.records) >= 1


def test_csv_has_expected_columns(tmp_path):
    e = _run("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ibuprofen", n=5)
    df = to_dataframe(e)
    for col in ["conf_id", "energy_kcal", "delta_e_kcal", "boltzmann_weight"]:
        assert col in df.columns
    out = write_csv(e, tmp_path / "ibuprofen.csv")
    assert out.exists() and out.stat().st_size > 0
