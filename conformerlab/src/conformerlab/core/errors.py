"""Controlled exceptions for conformerlab.

Every failure that the user could plausibly cause (bad SMILES, missing
backend, empty ensemble) raises one of these, never a bare RDKit/segfault.
"""


class ConformerLabError(Exception):
    """Base class for all conformerlab errors."""


class InvalidSmilesError(ConformerLabError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


class InvalidMoleculeFileError(ConformerLabError):
    """Raised when an SDF/XYZ input file is missing or cannot be parsed."""


class BackendNotAvailableError(ConformerLabError):
    """Raised when an optional backend (openconf, MLIP) is not installed."""


class EmptyEnsembleError(ConformerLabError):
    """Raised when conformer generation produced zero usable conformers."""
