"""Relative energies and ordering. All energies in kcal/mol."""

from __future__ import annotations

from conformerlab.core.types import EnsembleResult


def add_delta_e_kcal(ensemble: EnsembleResult) -> EnsembleResult:
    """Fill ``delta_e_kcal`` = energy - min(energy) and sort ascending.

    Returns the same EnsembleResult, records sorted by absolute energy so the
    lowest-energy conformer is first (and is the RMSD reference).
    """
    if not ensemble.records:
        return ensemble
    e_min = min(r.energy_kcal for r in ensemble.records)
    for r in ensemble.records:
        r.delta_e_kcal = r.energy_kcal - e_min
    ensemble.records.sort(key=lambda r: r.energy_kcal)
    return ensemble
