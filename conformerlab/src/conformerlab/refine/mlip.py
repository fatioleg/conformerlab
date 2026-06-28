"""MLIP re-ranking of a conformer ensemble (optional).

Re-optimise conformers locally using a machine-learned interatomic potential
through ASE, producing sharper energies without external compute credits.

Supported calculators (install one):
  * AIMNet2  -> pip install "aimnet[ase]"   (neutral/charged organics)
  * MACE-OFF -> pip install mace-torch       (organics, ~0.25 kcal/mol accuracy)

Both are gas-phase / implicit-solvent potentials. They sharpen energies; they
do not change the qualitative distribution of conformers.

## Extending with a new calculator

Register any ASE-compatible calculator:

    from conformerlab.refine.mlip import register_calculator
    register_calculator("xtb", lambda device: XTB())

For GPU cloud (e.g. Modal.com): implement a thin shim that satisfies the two
ASE methods used here (.get_potential_energy() -> float [eV] and
.get_positions() -> array[N,3]), then register it:

    register_calculator("mace-modal", ModalMaceCalculator)

The rest of the pipeline is unchanged — Modal is just another registry entry.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from typing import Any

# Disable torch.compile / TorchDynamo before any torch import. In environments
# without g++ (e.g. WSL without build-essential) torch._inductor raises
# InvalidCxxCompiler. Setting these env vars here catches the case where torch
# was not yet imported when this module loads; the _suppress_torch_compile()
# call below handles the already-imported case via the Python API.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_SUPPRESS_ERRORS", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # PyTorch ≥ 2.4


def _suppress_torch_compile() -> None:
    """Patch torch dynamo/inductor config if torch is already imported."""
    try:
        import torch  # noqa: PLC0415
        torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
        torch._dynamo.config.disable = True  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


_suppress_torch_compile()

# E402: estes imports vêm depois do setup de env vars de propósito — as vars de
# supressão do torch.compile precisam ser definidas antes de qualquer import que
# toque o torch.
from conformerlab.core.errors import BackendNotAvailableError  # noqa: E402
from conformerlab.core.types import (  # noqa: E402
    EnsembleResult,
    RefinementSettings,
)

_EV_TO_KCAL = 23.060547830619026

# ---------------------------------------------------------------------------
# Calculator registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[str], Any]] = {}


def register_calculator(name: str, factory: Callable[[str], Any]) -> None:
    """Register an ASE-compatible calculator factory.

    *factory* receives ``device`` ("cpu" | "cuda" | any string) and returns
    an ASE calculator.  Future remote-GPU adapters (e.g. Modal.com) plug in
    here without touching any other code.
    """
    _REGISTRY[name.lower()] = factory


def list_methods() -> list[str]:
    """Return the names of all registered MLIP calculators."""
    return sorted(_REGISTRY)


def mlip_available(method: str) -> bool:
    """Return True if *method* is registered and its package is importable."""
    key = method.lower()
    if key not in _REGISTRY:
        return False
    if key == "aimnet2":
        try:
            import aimnet  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False
    if key in {"mace", "mace-off"}:
        try:
            import mace  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False
    return True  # custom-registered method: assume caller manages deps


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def _aimnet2_factory(device: str) -> Any:
    _suppress_torch_compile()  # re-apply in case torch was imported after module load
    try:
        from aimnet.calculators import AIMNet2ASE  # type: ignore
    except ImportError as exc:
        raise BackendNotAvailableError(
            'AIMNet2 not installed: pip install "aimnet[ase]"'
        ) from exc
    return AIMNet2ASE("aimnet2")


def _mace_factory(device: str) -> Any:
    _suppress_torch_compile()
    try:
        from mace.calculators import mace_off  # type: ignore
    except ImportError as exc:
        raise BackendNotAvailableError(
            "MACE not installed: pip install mace-torch"
        ) from exc
    # compile=False avoids torch._inductor (requires g++) when not needed
    try:
        return mace_off(model="medium", device=device, compile=False)
    except TypeError:
        # older mace-torch without compile kwarg
        return mace_off(model="medium", device=device)


register_calculator("aimnet2", _aimnet2_factory)
register_calculator("mace-off", _mace_factory)
register_calculator("mace", _mace_factory)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedup_records(mol: Any, records: list, rmsd_thresh: float) -> list:
    """Drop near-identical geometries, keeping the lowest-energy representative.

    Uses RDKit's symmetry-aware best-RMSD so ring flips / equivalent atoms do not
    count as different structures. Operates on a throwaway copy of *mol* because
    ``GetBestRMS`` aligns (mutates) the probe conformer.
    """
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    probe = Chem.Mol(mol)  # isolate alignment side-effects from the real geometry
    kept: list = []
    for r in sorted(records, key=lambda x: x.energy_kcal):
        is_dup = False
        for k in kept:
            try:
                rms = AllChem.GetBestRMS(probe, probe, r.conf_id, k.conf_id)
            except Exception:  # noqa: BLE001
                rms = float("inf")
            if rms < rmsd_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------


def refine_with_mlip(
    ensemble: EnsembleResult,
    settings: RefinementSettings | None = None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    step_callback: Callable[[int, int, float, float], None] | None = None,
) -> EnsembleResult:
    """Re-optimise each conformer and return a **new** EnsembleResult (non-destructive).

    Energies are in kcal/mol.  ``delta_e_kcal`` and ``boltzmann_weight`` are
    left stale — re-run ``analysis.pipeline.analyze`` on the result to refresh
    them.  *progress_callback(done, total)* fires after each conformer.
    *step_callback(conf_id, step, energy_kcal, max_force_ev_ang)* fires after
    each optimizer step.
    """
    if settings is None:
        settings = RefinementSettings()

    method = settings.method.lower()
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown MLIP method {method!r}. Available: {list_methods()}"
        )

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

    from rdkit import Chem  # type: ignore

    mol_copy = Chem.RWMol(mol)
    refined = copy.deepcopy(ensemble)
    refined.attach_mol(mol_copy)

    if settings.max_conformers is not None:
        refined.records.sort(key=lambda r: r.energy_kcal)
        refined.records = refined.records[: settings.max_conformers]

    calc = _REGISTRY[method](settings.device)
    symbols = [a.GetSymbol() for a in mol_copy.GetAtoms()]
    total = len(refined.records)

    for done_count, r in enumerate(refined.records):
        conf = mol_copy.GetConformer(r.conf_id)
        positions = [
            list(conf.GetAtomPosition(i)) for i in range(mol_copy.GetNumAtoms())
        ]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc
        conf_id = r.conf_id  # snapshot — avoids loop-variable capture in closure

        if settings.energy_only:
            e = float(atoms.get_potential_energy()) * _EV_TO_KCAL
            r.energy_kcal = e
            r.energy_trajectory = [e]
            r.backend = f"{r.backend}+{method}(sp)"
            if step_callback is not None:
                step_callback(conf_id, 1, e, 0.0)  # no force in single-point
        else:
            import numpy as np  # type: ignore

            trajectory: list[float] = []
            force_traj: list[float] = []
            step_count = [0]

            def _record(
                _atoms: object = atoms,
                _traj: list = trajectory,
                _ftraj: list = force_traj,
                _sc: list = step_count,
                _cid: int = conf_id,
                _scb: object = step_callback,
            ) -> None:
                e = float(_atoms.get_potential_energy()) * _EV_TO_KCAL  # type: ignore[union-attr]
                f = float(np.max(np.abs(_atoms.get_forces())))  # type: ignore[union-attr]
                _traj.append(e)
                _ftraj.append(f)
                _sc[0] += 1
                if _scb is not None:
                    _scb(_cid, _sc[0], e, f)

            opt = LBFGS(atoms, logfile=None)
            opt.attach(_record)
            opt.run(fmax=settings.fmax, steps=settings.max_steps)

            new_pos = atoms.get_positions()
            for i in range(mol_copy.GetNumAtoms()):
                conf.SetAtomPosition(i, tuple(new_pos[i]))

            r.energy_kcal = float(atoms.get_potential_energy()) * _EV_TO_KCAL
            r.energy_trajectory = trajectory if trajectory else [r.energy_kcal]
            r.force_trajectory = force_traj if force_traj else None
            r.backend = f"{r.backend}+{method}"

        r.mlip_method = method

        if progress_callback is not None:
            progress_callback(done_count + 1, total)

    # Geometry optimisation can collapse several starting conformers onto the same
    # minimum; drop the duplicates (single-point keeps input geometries, so skip).
    if (
        settings.dedup_rmsd is not None
        and not settings.energy_only
        and len(refined.records) > 1
    ):
        refined.records = _dedup_records(
            mol_copy, refined.records, settings.dedup_rmsd
        )

    return refined
