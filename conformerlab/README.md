# conformerlab

A small, modular core for conformer generation and ensemble analysis. Free and
local: RDKit by default, openconf and MLIP refinement as optional drop-ins. No
web service, no credits.

## Install

```bash
pip install -e .            # core (RDKit, numpy, pandas, pydantic)
pip install -e ".[openconf]"  # + openconf generator (same one Rowan uses)
pip install -e ".[mlip]" && pip install "aimnet[ase]"  # + MLIP re-ranking
pip install -e ".[dev]"     # + pytest, ruff
```

## Minimal use

```python
from conformerlab.backends.factory import get_backend
from conformerlab.analysis.pipeline import analyze
from conformerlab.io.export import write_csv
from conformerlab.core.types import MoleculeInput, GenerationSettings

backend = get_backend("auto")          # openconf if installed, else rdkit
mol = MoleculeInput(smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O", name="ibuprofen")
ensemble = backend.generate(mol, GenerationSettings(preset="rapid", max_conformers=50))
analyze(ensemble)                      # delta_e, RMSD, Boltzmann, r_g, SASA
write_csv(ensemble, "ibuprofen.csv")
```

## Local "generate then refine" (the Rowan-without-credits path)

```python
from conformerlab.refine.mlip import refine_with_mlip
backend = get_backend("openconf")
ensemble = backend.generate(mol, GenerationSettings(max_conformers=200))
refine_with_mlip(ensemble, method="aimnet2")  # re-optimise + re-score
analyze(ensemble)                              # re-run at the new level
```

## Layout

```
core/      data models, units, molecule validation, errors
backends/  ConformerBackend interface + rdkit / openconf implementations
analysis/  energy, boltzmann, rmsd, geometry (r_g/SASA), selection, pipeline
refine/    optional MLIP re-ranking via ASE
io/        CSV / XYZ / SDF / project-JSON export
tests/     molecule, analysis math, end-to-end RDKit run, export
```

See `AGENTS.md` before extending.
