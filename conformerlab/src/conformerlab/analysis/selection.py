"""Conformer selection: energy window, top-N, Boltzmann coverage, manual.

Selection only flips ``record.selected``; it never deletes data, so a later
choice can re-select without re-running anything.
"""

from __future__ import annotations

from conformerlab.core.types import EnsembleResult


def _reset(ensemble: EnsembleResult) -> None:
    for r in ensemble.records:
        r.selected = False


def select_by_energy_window(
    ensemble: EnsembleResult, window_kcal: float
) -> EnsembleResult:
    """Select all conformers with delta_e <= window_kcal."""
    _reset(ensemble)
    for r in ensemble.records:
        if r.delta_e_kcal is not None and r.delta_e_kcal <= window_kcal:
            r.selected = True
    return ensemble


def select_top_n(ensemble: EnsembleResult, n: int) -> EnsembleResult:
    """Select the n lowest-energy conformers (records assumed energy-sorted)."""
    _reset(ensemble)
    for r in ensemble.records[: max(0, n)]:
        r.selected = True
    return ensemble


def select_by_boltzmann_coverage(
    ensemble: EnsembleResult, coverage: float = 0.95
) -> EnsembleResult:
    """Select the smallest set of conformers whose weights sum to >= coverage."""
    _reset(ensemble)
    acc = 0.0
    for r in sorted(
        ensemble.records, key=lambda x: (x.boltzmann_weight or 0.0), reverse=True
    ):
        if acc >= coverage:
            break
        r.selected = True
        acc += r.boltzmann_weight or 0.0
    return ensemble


def select_manual(ensemble: EnsembleResult, conf_ids: list[int]) -> EnsembleResult:
    """Select exactly the conformers whose conf_id is in conf_ids."""
    _reset(ensemble)
    wanted = set(conf_ids)
    for r in ensemble.records:
        r.selected = r.conf_id in wanted
    return ensemble
