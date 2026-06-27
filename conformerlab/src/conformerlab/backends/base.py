"""The one interface every conformer generator must satisfy.

Add CREST, an MLIP-driven search, or DFT later by writing a new subclass.
Nothing downstream (analysis, IO, any future UI) imports a concrete backend;
they all speak ``EnsembleResult``.
"""

from __future__ import annotations

import abc

from conformerlab.core.types import (
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
)


class ConformerBackend(abc.ABC):
    """Abstract base for all conformer-generation backends."""

    #: short, stable identifier written into every ConformerRecord.backend
    name: str = "base"

    @abc.abstractmethod
    def generate(
        self, molecule: MoleculeInput, settings: GenerationSettings
    ) -> EnsembleResult:
        """Generate conformers and return an unanalysed EnsembleResult.

        The result must carry absolute energies in kcal/mol and have the RDKit
        Mol (with embedded conformers) attached via ``.attach_mol``. Energy
        ordering, Boltzmann weights, RMSD and selection are NOT this method's
        job; the analysis layer adds them.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend's dependencies are importable."""
        raise NotImplementedError
