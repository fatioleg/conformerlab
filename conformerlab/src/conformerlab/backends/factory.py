"""Backend factory: pick a generator by name, fall back gracefully."""

from __future__ import annotations

from conformerlab.backends.base import ConformerBackend
from conformerlab.backends.openconf_backend import OpenConfBackend
from conformerlab.backends.rdkit_backend import RDKitBackend


def get_backend(name: str = "auto") -> ConformerBackend:
    """Return a usable backend.

    "auto" prefers openconf if installed, otherwise RDKit. Explicit names
    ("rdkit", "openconf") are honoured; an unavailable explicit choice raises
    via the backend itself when ``generate`` is called.
    """
    name = name.lower()
    if name == "rdkit":
        return RDKitBackend()
    if name == "openconf":
        return OpenConfBackend()
    if name == "auto":
        oc = OpenConfBackend()
        return oc if oc.is_available() else RDKitBackend()
    raise ValueError(f"Unknown backend: {name!r}")
