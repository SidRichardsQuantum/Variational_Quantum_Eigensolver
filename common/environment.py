"""
Runtime environment metadata for reproducible benchmark records.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, Any]:
    """
    Return JSON-safe runtime metadata for research-grade result records.

    The package versions are collected through importlib.metadata rather than
    importing scientific packages solely for metadata. This keeps the helper
    cheap and avoids extra import side effects.
    """
    packages = {
        name: _package_version(name)
        for name in (
            "vqe-pennylane",
            "pennylane",
            "numpy",
            "scipy",
            "openfermion",
            "openfermionpyscf",
            "pyscf",
        )
    }
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def ensure_environment_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """
    Add an ``environment`` block to a result dictionary when missing.

    The input dictionary is returned after mutation so callers can use this
    helper inline while preserving existing record objects.
    """
    result.setdefault("environment", environment_metadata())
    return result
