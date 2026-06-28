"""Data models shared across the whole pipeline.

These are the *only* contract between backends, analysis, and IO. A new
backend (openconf, CREST, MLIP, DFT) must return an ``EnsembleResult`` and
nothing else needs to change downstream.

Energy convention: every energy field is in kcal/mol. There is no other unit
anywhere in the core. If a backend speaks Hartree, it converts at its edge
(see ``core.units``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class MoleculeInput(BaseModel):
    """User-facing description of a molecule to analyse.

    Provide exactly one source: ``smiles``, ``sdf_path``, or ``xyz_path``.
    ``charge`` is also used for bond perception when reading XYZ.
    """

    smiles: str | None = None
    sdf_path: str | None = None
    xyz_path: str | None = None
    name: str = "molecule"
    charge: int = 0
    multiplicity: int = 1

    @model_validator(mode="after")
    def _exactly_one_source(self) -> MoleculeInput:
        sources = [self.smiles, self.sdf_path, self.xyz_path]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Provide exactly one molecule source: smiles, sdf_path, or "
                "xyz_path."
            )
        return self


class GenerationSettings(BaseModel):
    """Knobs for conformer generation, backend-agnostic."""

    preset: str = "rapid"            # rapid | ensemble | docking | macrocycle
    max_conformers: int = 200
    energy_window_kcal: float = 10.0
    temperature_k: float = 298.15
    random_seed: int = 42


class RefinementSettings(BaseModel):
    """Knobs for MLIP re-optimisation, calculator-agnostic."""

    method: str = "aimnet2"          # registered calculator name; see refine.mlip
    fmax: float = 0.05               # convergence threshold, eV/Å
    max_steps: int = 200
    device: str = "cpu"              # "cpu" | "cuda"
    max_conformers: int | None = None  # None = refine all; N = top-N by energy
    energy_only: bool = False          # True = single-point (no geometry change)
    # heavy-atom best-RMSD (Å) below which two refined geometries are treated as
    # duplicates (the lowest-energy one is kept); None disables deduplication.
    # Off by default at the API level; the UI enables it (slider default 0.125).
    dedup_rmsd: float | None = None


class ConformerRecord(BaseModel):
    """One conformer and its analysed properties."""

    conf_id: int
    backend: str
    energy_kcal: float                       # absolute, kcal/mol
    delta_e_kcal: float | None = None      # relative to the minimum
    boltzmann_weight: float | None = None  # 0..1, sums to 1 over ensemble
    rmsd_to_min_ang: float | None = None   # heavy-atom RMSD vs lowest E
    radius_of_gyration_ang: float | None = None
    sasa_ang2: float | None = None
    selected: bool = False
    mlip_method: str | None = None         # set after MLIP refinement
    energy_trajectory: list[float] | None = None  # per-step kcal/mol during MLIP opt
    # per-step max-force eV/Å during MLIP geometry optimisation
    force_trajectory: list[float] | None = None


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

    def attach_mol(self, mol: object) -> EnsembleResult:
        self._mol = mol
        return self

    @property
    def mol(self) -> object:
        return self._mol
