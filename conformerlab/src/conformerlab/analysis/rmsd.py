"""Heavy-atom RMSD of every conformer to the lowest-energy one.

Uses RDKit's GetBestRMS (symmetry-aware, with alignment). Requires the RDKit
Mol attached to the ensemble; if it is missing (e.g. a backend that did not
keep geometry), RMSD is left as None rather than guessed.
"""

from __future__ import annotations

from rdkit.Chem import AllChem

from conformerlab.core.types import EnsembleResult


def add_rmsd_to_min(ensemble: EnsembleResult) -> EnsembleResult:
    """Fill ``rmsd_to_min_ang`` for each record (the minimum itself = 0.0)."""
    mol = ensemble.mol
    if mol is None or not ensemble.records:
        return ensemble

    # records are expected sorted by energy (analysis.energy ran first);
    # the reference is the first record's conformer.
    ref_conf_id = ensemble.records[0].conf_id
    for r in ensemble.records:
        if r.conf_id == ref_conf_id:
            r.rmsd_to_min_ang = 0.0
            continue
        try:
            r.rmsd_to_min_ang = AllChem.GetBestRMS(
                mol, mol, refId=ref_conf_id, prbId=r.conf_id
            )
        except Exception:  # geometry mismatch / missing conf -> leave None
            r.rmsd_to_min_ang = None
    return ensemble
