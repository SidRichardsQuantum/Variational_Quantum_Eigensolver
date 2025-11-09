# package_tests/conftest.py
import os
import sys
import pytest

# Ensure project root is on PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture(scope="session")
def test_seed():
    """Global random seed fixture for reproducibility."""
    import numpy as np
    np.random.seed(42)
    return 42


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for energy comparisons."""
    return 1e-2
