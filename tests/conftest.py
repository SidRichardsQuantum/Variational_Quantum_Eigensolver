from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VQE_TEST_MODE", "1")
    monkeypatch.setenv("QPE_TEST_MODE", "1")
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    yield
