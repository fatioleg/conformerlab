"""MLIP re-ranking of a conformer ensemble (optional).

This is the piece that removes your dependence on Rowan credits: generate
conformers locally (RDKit or openconf), then re-optimise and re-score each one
with a machine-learned interatomic potential through ASE -- the local
equivalent of Rowan's "g-xTB // GFN2-xTB" final step.

Supported calculators (install one):
  * AIMNet2  -> pip install "aimnet[ase]"   (neutral, charged, organics)
  * MACE-OFF -> pip install mace-torch       (organics, ~0.25 kcal/mol)

Both are gas-phase / implicit-solvent single-molecule potentials. For a
flexible polymer like PEG, better single-molecule energies do NOT fix the
underlying physics question (the relevant observable is the r_g/SASA
distribution, not the global minimum). Use this to sharpen energies, not to
answer a question that conformer search cannot answer.
"""

from __future__ import annotations

from conformerlab.core.errors import BackendNotAvailableError
from conformerlab.core.types import EnsembleResult


def _build_calculator(method: str):
    method = method.lower()
    if method == "aimnet2":
        try:
            from aimnet.calculators import AIMNet2ASE  # type: ignore
        except ImportError as exc:
            raise BackendNotAvailableError(
                'AIMNet2 not installed: pip install "aimnet[ase]"'
            ) from exc
        return AIMNet2ASE("aimnet2")
    if method in {"mace", "mace-off", "maceoff"}:
        try:
            from mace.calculators import mace_off  # type: ignore
        except ImportError as exc:
            raise BackendNotAvailableError(
                "MACE not installed: pip install mace-torch"
            ) from exc
        return mace_off(model="medium", device="cpu")
    raise ValueError(f"Unknown MLIP method: {method!r}")


def refine_with_mlip(
    ensemble: EnsembleResult,
    method: str = "aimnet2",
    fmax: float = 0.05,
    max_steps: int = 200,
) -> EnsembleResult:
    """Re-optimise each conformer with an MLIP and overwrite energy_kcal.

    Leaves delta_e/Boltzmann stale on purpose -- re-run analysis.pipeline.analyze
    afterwards so the whole ensemble is consistent at the new level of theory.
    """
    try:
        from ase import Atoms  # type: ignore
        from ase.optimize import LBFGS  # type: ignore
    except ImportError as exc:
        raise BackendNotAvailableError(
            "ASE not installed: pip install ase"
        ) from exc

    mol = ensemble.mol
    if mol is None or not ensemble.records:
        return ensemble

    calc = _build_calculator(method)
    EV_TO_KCAL = 23.060547830619026

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    for r in ensemble.records:
        conf = mol.GetConformer(r.conf_id)
        positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=max_steps)
        # write optimised coordinates back and store MLIP energy (kcal/mol)
        new_pos = atoms.get_positions()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, tuple(new_pos[i]))
        r.energy_kcal = float(atoms.get_potential_energy()) * EV_TO_KCAL
        r.backend = f"{ensemble.backend}+{method}"
    return ensemble
