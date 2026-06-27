"""Per-conformer geometry descriptors: radius of gyration and SASA.

These are the columns that actually matter for a flexible system (PEG and
friends): the *distribution* of r_g/SASA across the ensemble describes the
molecule far better than the single lowest conformer. r_g is mass-weighted.
SASA uses RDKit's rdFreeSASA when classification succeeds, else None.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFreeSASA

from conformerlab.core.types import EnsembleResult


def _radius_of_gyration(mol: Chem.Mol, conf_id: int) -> float:
    conf = mol.GetConformer(conf_id)
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    com = np.average(coords, axis=0, weights=masses)
    d2 = np.sum((coords - com) ** 2, axis=1)
    return float(np.sqrt(np.average(d2, weights=masses)))


def _sasa(mol: Chem.Mol, conf_id: int) -> float | None:
    try:
        radii = rdFreeSASA.classifyAtoms(mol)
        return float(rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id))
    except Exception:
        return None


def add_geometry(ensemble: EnsembleResult) -> EnsembleResult:
    """Fill ``radius_of_gyration_ang`` and ``sasa_ang2`` for each record."""
    mol = ensemble.mol
    if mol is None or not ensemble.records:
        return ensemble
    for r in ensemble.records:
        r.radius_of_gyration_ang = round(_radius_of_gyration(mol, r.conf_id), 4)
        s = _sasa(mol, r.conf_id)
        r.sasa_ang2 = round(s, 4) if s is not None else None
    return ensemble
