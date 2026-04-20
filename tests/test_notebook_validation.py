from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_NOTEBOOK_ROOT = REPO_ROOT / "notebooks" / "benchmarks"
BENCHMARK_NOTEBOOKS = tuple(sorted(BENCHMARK_NOTEBOOK_ROOT.rglob("*.ipynb")))
EXECUTION_SMOKE_NOTEBOOKS = tuple(
    sorted((REPO_ROOT / "tests" / "fixtures" / "notebooks").glob("*.ipynb"))
)


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def test_benchmark_notebooks_are_discovered() -> None:
    assert BENCHMARK_NOTEBOOKS


def test_execution_smoke_notebooks_are_discovered() -> None:
    assert EXECUTION_SMOKE_NOTEBOOKS


@pytest.mark.parametrize(
    "path", BENCHMARK_NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_benchmark_notebook_smoke(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook.get("nbformat") == 4
    assert notebook.get("cells")

    metadata = notebook.get("metadata", {})
    assert isinstance(metadata, dict)

    kernelspec = metadata.get("kernelspec", {})
    assert isinstance(kernelspec, dict)
    assert kernelspec.get("language") == "python"

    language_info = metadata.get("language_info", {})
    assert isinstance(language_info, dict)
    assert language_info.get("name") == "python"

    for index, cell in enumerate(notebook["cells"], start=1):
        assert isinstance(cell, dict)
        if cell.get("cell_type") != "code":
            continue

        source = _cell_source(cell)
        compile(source, f"{path.relative_to(REPO_ROOT)}:cell-{index}", "exec")

        error_outputs = [
            output
            for output in cell.get("outputs", [])
            if isinstance(output, dict) and output.get("output_type") == "error"
        ]
        assert not error_outputs, f"cell {index} contains saved error output"


@pytest.mark.parametrize(
    "path", EXECUTION_SMOKE_NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_notebook_execution_smoke(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook_smoke__"}

    for index, cell in enumerate(notebook["cells"], start=1):
        assert isinstance(cell, dict)
        if cell.get("cell_type") != "code":
            continue

        source = _cell_source(cell)
        code = compile(source, f"{path.relative_to(REPO_ROOT)}:cell-{index}", "exec")
        exec(code, namespace)
