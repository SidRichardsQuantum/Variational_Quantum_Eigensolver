import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    # Low-resource mode for continuous integration
    monkeypatch.setenv("VQE_TEST_MODE", "1")
    monkeypatch.setenv("QPE_TEST_MODE", "1")
    yield
