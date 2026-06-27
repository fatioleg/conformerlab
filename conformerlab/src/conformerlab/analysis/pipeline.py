"""Convenience pipeline: run the full analysis in the correct order.

Order matters: energy (sets delta_e and sorts) -> rmsd (needs the sorted
reference) -> boltzmann -> geometry. Each step is independently testable; this
just wires them so callers don't have to remember the sequence.
"""

from __future__ import annotations

from conformerlab.analysis.boltzmann import add_boltzmann_weights
from conformerlab.analysis.energy import add_delta_e_kcal
from conformerlab.analysis.geometry import add_geometry
from conformerlab.analysis.rmsd import add_rmsd_to_min
from conformerlab.core.types import EnsembleResult


def analyze(
    ensemble: EnsembleResult, temperature_k: float | None = None
) -> EnsembleResult:
    add_delta_e_kcal(ensemble)
    add_rmsd_to_min(ensemble)
    add_boltzmann_weights(ensemble, temperature_k)
    add_geometry(ensemble)
    return ensemble
