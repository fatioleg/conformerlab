"""End-to-end check of the openconf backend against the real package.

Skipped when openconf is not installed, so the core test suite stays
dependency-free. When present, it guards the API bridge (energies in kcal/mol +
RDKit geometry with conf ids aligned to records) that downstream analysis
relies on.
"""

from __future__ import annotations

import pytest

from conformerlab.analysis.pipeline import analyze
from conformerlab.backends.openconf_backend import OpenConfBackend
from conformerlab.core.types import GenerationSettings, MoleculeInput

pytest.importorskip("openconf")


def test_openconf_generates_and_analyses():
    backend = OpenConfBackend()
    assert backend.is_available()

    mol = MoleculeInput(smiles="CCO", name="ethanol")
    ensemble = backend.generate(
        mol, GenerationSettings(preset="rapid", max_conformers=20, random_seed=42)
    )

    assert ensemble.backend == "openconf"
    assert ensemble.records, "expected at least one conformer"

    rdkit_mol = ensemble.mol
    assert rdkit_mol is not None
    # Each record's conf_id must index a real conformer (the analysis contract).
    conf_ids = {c.GetId() for c in rdkit_mol.GetConformers()}
    assert {r.conf_id for r in ensemble.records} == conf_ids

    analyze(ensemble)
    weights = [r.boltzmann_weight for r in ensemble.records]
    assert all(w is not None for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert min(r.delta_e_kcal for r in ensemble.records) == 0.0
