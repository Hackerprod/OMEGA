"""Native kernel loader for OMEGA."""

from __future__ import annotations

from importlib import import_module


def load_kernels():
    try:
        return import_module("omega.core.native._kernels")
    except ModuleNotFoundError as exc:
        raise ImportError("OMEGA native kernels are not built") from exc


def has_native() -> bool:
    try:
        load_kernels()
    except ImportError:
        return False
    return True
