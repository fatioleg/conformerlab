"""Exporters: CSV, XYZ, SDF, and a project JSON snapshot.

CSV columns are fixed and documented so downstream scripts (or a future UI)
can rely on them: conf_id, energy_kcal, delta_e_kcal, boltzmann_weight,
rmsd_to_min_ang, radius_of_gyration_ang, sasa_ang2, selected, backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from conformerlab.core.types import EnsembleResult

_CSV_COLUMNS = [
    "conf_id",
    "energy_kcal",
    "delta_e_kcal",
    "boltzmann_weight",
    "rmsd_to_min_ang",
    "radius_of_gyration_ang",
    "sasa_ang2",
    "selected",
    "backend",
    "mlip_method",
]


def to_dataframe(ensemble: EnsembleResult) -> pd.DataFrame:
    rows = [r.model_dump() for r in ensemble.records]
    df = pd.DataFrame(rows)
    return df.reindex(columns=_CSV_COLUMNS)


def write_csv(ensemble: EnsembleResult, path: str | Path) -> Path:
    path = Path(path)
    to_dataframe(ensemble).to_csv(path, index=False)
    return path


def write_xyz(ensemble: EnsembleResult, path: str | Path) -> Path:
    path = Path(path)
    mol = ensemble.mol
    if mol is None:
        raise ValueError("No geometry attached; cannot write XYZ.")
    n = mol.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    blocks = []
    for r in ensemble.records:
        conf = mol.GetConformer(r.conf_id)
        comment = (
            f"conf_id={r.conf_id} E={r.energy_kcal:.4f}kcal/mol "
            f"dE={r.delta_e_kcal} w={r.boltzmann_weight}"
        )
        lines = [str(n), comment]
        for i in range(n):
            p = conf.GetAtomPosition(i)
            lines.append(f"{symbols[i]:<2s} {p.x:14.6f} {p.y:14.6f} {p.z:14.6f}")
        blocks.append("\n".join(lines))
    path.write_text("\n".join(blocks) + "\n")
    return path


def write_sdf(ensemble: EnsembleResult, path: str | Path) -> Path:
    path = Path(path)
    mol = ensemble.mol
    if mol is None:
        raise ValueError("No geometry attached; cannot write SDF.")
    writer = Chem.SDWriter(str(path))
    for r in ensemble.records:
        mol.SetProp("_Name", f"{ensemble.molecule.name}_conf{r.conf_id}")
        mol.SetProp("energy_kcal", f"{r.energy_kcal:.6f}")
        if r.delta_e_kcal is not None:
            mol.SetProp("delta_e_kcal", f"{r.delta_e_kcal:.6f}")
        if r.boltzmann_weight is not None:
            mol.SetProp("boltzmann_weight", f"{r.boltzmann_weight:.6f}")
        mol.SetProp("backend", r.backend)
        if r.mlip_method is not None:
            mol.SetProp("mlip_method", r.mlip_method)
        writer.write(mol, confId=r.conf_id)
    writer.close()
    return path


def write_project_json(ensemble: EnsembleResult, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "molecule": ensemble.molecule.model_dump(),
        "settings": ensemble.settings.model_dump(),
        "backend": ensemble.backend,
        "records": [r.model_dump() for r in ensemble.records],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
