"""Boltzmann populations at a configurable temperature.

w_i = exp(-dE_i / RT) / sum_j exp(-dE_j / RT)

with dE in kcal/mol and RT in kcal/mol. Uses dE (not absolute E) and shifts by
the minimum for numerical safety -- exp of a large positive number overflows.
"""

from __future__ import annotations

import math

from conformerlab.core.types import EnsembleResult
from conformerlab.core.units import R_KCAL_PER_MOL_K


def add_boltzmann_weights(
    ensemble: EnsembleResult, temperature_k: float | None = None
) -> EnsembleResult:
    """Fill ``boltzmann_weight`` for every record (weights sum to 1)."""
    if not ensemble.records:
        return ensemble
    t = temperature_k if temperature_k is not None else ensemble.settings.temperature_k
    rt = R_KCAL_PER_MOL_K * t

    # Ensure delta_e is present; if not, derive from absolute energies.
    e_min = min(r.energy_kcal for r in ensemble.records)
    deltas = [
        (r.delta_e_kcal if r.delta_e_kcal is not None else r.energy_kcal - e_min)
        for r in ensemble.records
    ]
    boltz = [math.exp(-d / rt) for d in deltas]
    z = sum(boltz)
    for r, b in zip(ensemble.records, boltz):
        r.boltzmann_weight = b / z
    return ensemble
