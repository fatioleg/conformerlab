"""Data models shared across the whole pipeline.

These are the *only* contract between backends, analysis, and IO. A new
backend (openconf, CREST, MLIP, DFT) must return an ``EnsembleResult`` and
nothing else needs to change downstream.

Energy convention: every energy field is in kcal/mol. There is no other unit
anywhere in the core. If a backend speaks Hartree, it converts at its edge
(see ``core.units``).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class MoleculeInput(BaseModel):
    """User-facing description of a molecule to analyse."""

    smiles: str
    name: str = "molecule"
    charge: int = 0
    multiplicity: int = 1


class GenerationSettings(BaseModel):
    """Knobs for conformer generation, backend-agnostic."""

    preset: str = "rapid"            # rapid | ensemble | docking | macrocycle
    max_conformers: int = 200
    energy_window_kcal: float = 10.0
    temperature_k: float = 298.15
    random_seed: int = 42


class ConformerRecord(BaseModel):
    """One conformer and its analysed properties."""

    conf_id: int
    backend: str
    energy_kcal: float                       # absolute, kcal/mol
    delta_e_kcal: Optional[float] = None      # relative to the minimum
    boltzmann_weight: Optional[float] = None  # 0..1, sums to 1 over ensemble
    rmsd_to_min_ang: Optional[float] = None   # heavy-atom RMSD vs lowest E
    radius_of_gyration_ang: Optional[float] = None
    sasa_ang2: Optional[float] = None
    selected: bool = False


class EnsembleResult(BaseModel):
    """Everything a backend hands back: the records plus provenance."""

    molecule: MoleculeInput
    settings: GenerationSettings
    backend: str
    records: list[ConformerRecord] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The RDKit Mol with embedded conformers is kept as a private attribute
    # (it is not serialisable). Backends attach it via .attach_mol(); analysis
    # reads it via .mol. This keeps the serialisable data model clean while
    # still allowing geometry-based analysis (RMSD, r_g, SASA).
    _mol: object = PrivateAttr(default=None)

    def attach_mol(self, mol: object) -> "EnsembleResult":
        self._mol = mol
        return self

    @property
    def mol(self) -> object:
        return self._mol
