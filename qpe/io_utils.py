"""
File I/O, caching, and path management utilities for QPE.
"""

import os
import json
import hashlib

# === Base directories (parallel to vqe) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "package results")
IMG_DIR = os.path.join(BASE_DIR, "qpe", "images")


# ---------------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------------
def ensure_dirs():
    """Ensure that results and image directories exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# File Utilities
# ---------------------------------------------------------------------
def signature_hash(molecule, n_ancilla, t, noise, shots):
    """Generate a reproducible hash key for caching results."""
    key = json.dumps(
        {"mol": molecule, "anc": n_ancilla, "t": t, "noise": noise, "shots": shots},
        sort_keys=True,
    )
    return hashlib.md5(key.encode()).hexdigest()


def cache_path(molecule, hash_key):
    """Build cache file path for results."""
    ensure_dirs()
    safe_mol = molecule.replace("+", "plus")
    return os.path.join(RESULTS_DIR, f"{safe_mol}_QPE_{hash_key}.json")


def save_qpe_result(result):
    """Save QPE result to JSON in the global package results directory."""
    ensure_dirs()
    key = signature_hash(
        result["molecule"],
        result["n_ancilla"],
        result["t"],
        result["noise"],
        result["shots"],
    )
    path = cache_path(result["molecule"], key)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Saved QPE result → {path}")


def load_qpe_result(molecule, hash_key):
    """Load a cached QPE result if available."""
    path = cache_path(molecule, hash_key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_qpe_plot(name: str):
    """Return the full image path in qpe/images/ and ensure directory exists."""
    ensure_dirs()
    return os.path.join(IMG_DIR, name)
