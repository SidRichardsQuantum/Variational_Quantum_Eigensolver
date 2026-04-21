# Caching And Artifacts

This repository uses two different kinds of generated output:

- local run caches and plots under `results/` and `images/`
- curated benchmark artifacts under `notebooks/benchmarks/_artifacts/`

Keep those categories separate. The local cache is for fast development and
reruns. Curated artifacts are the reviewed result surfaces that can be linked
from docs and committed to the repository.

## Local Run Records

Solver runs write JSON records through each package's I/O helpers:

- VQE records under `results/vqe/`
- QPE records under `results/qpe/`
- QITE and QRTE records under `results/qite/`

The exact filename is derived from a run signature. The signature captures the
resolved problem and algorithm settings that affect the result.

Typical signature inputs include:

- molecule or explicit geometry
- charge, basis, unit, and mapping
- active-space settings
- seed
- ansatz and ansatz keyword arguments
- optimizer, step count, and stepsize
- QPE ancillas, shots, evolution time, and Trotter settings
- VarQITE or VarQRTE step size and linear-system controls
- noise settings
- expert-mode Hamiltonian fingerprint, qubit count, and reference state

Changing a meaningful setting should create a different cache record.

## Cache Hits

Most high-level APIs return cache metadata:

| Field | Meaning |
| ----- | ------- |
| `cache_hit` | `True` when the result was loaded from an existing record |
| `runtime_s` | Wall time for the current call, including cache lookup |
| `compute_runtime_s` | Original compute time, or compute time for a fresh run |

Use this distinction when comparing method runtime. A cache hit is useful for
notebook iteration, but it is not the same thing as solver compute time.

## Forcing Recompute

Use `force=True` from Python:

```python
from vqe import run_vqe

result = run_vqe(molecule="H2", force=True, plot=False)
```

Or `--force` from the CLI:

```bash
python -m vqe --molecule H2 --force
python -m qite run --molecule H2 --force
```

Use forced recomputation when:

- validating a changed implementation
- refreshing benchmark notebooks
- checking runtime without cache reuse
- repairing a stale or incomplete local cache record

## Stale Cache Handling

The package includes regression coverage for stale cache invalidation. Solver
entrypoints refresh older records when required runtime metadata is missing.

Some cache records are still intentionally rejected. For example, VarQITE and
VarQRTE records must include final parameters and parameter shape so downstream
workflows can reuse the prepared circuit. If a legacy cache is missing required
fields, rerun with `force=True`.

## Local Plots

Plotting helpers write figures under `images/`. These files are ignored by git
and should be treated as generated local output.

If a plot should appear in the published docs, promote a reviewed copy into a
tracked artifact path through the benchmark artifact exporter rather than
linking directly to `images/`.

## Curated Benchmark Artifacts

Tracked benchmark artifacts live under:

```text
notebooks/benchmarks/_artifacts/
```

This directory contains selected figures and tables that are safe to publish in
the Sphinx docs. These are the result surfaces linked by
`notebooks/benchmarks/RESULTS.md` and `docs/benchmarks/results.md`.

Current curated artifact categories include:

- H2 VQE ansatz comparison figures
- H2 mapping comparison figures
- low-qubit VQE benchmark figures and tables
- H2 and LiH cross-method tables
- QPE H2 decision-map tables
- H2 VQE noise-robustness reference tables

## Export Workflow

After rerunning important benchmark notebooks, export reviewed artifacts:

```bash
python scripts/export_benchmark_artifacts.py
```

The exporter:

1. copies selected generated figures into `_artifacts/figures/`
2. extracts selected notebook output tables into `_artifacts/tables/`
3. writes CSV and Markdown table copies
4. rewrites the generated section in `notebooks/benchmarks/RESULTS.md`

Do not manually edit the generated block between:

```text
<!-- benchmark-artifacts:start -->
<!-- benchmark-artifacts:end -->
```

Rerun the exporter instead.

## Validation

Use these checks before committing benchmark artifact changes:

```bash
python -m sphinx -W -b html docs docs/_build/html
pytest -q tests/test_benchmark_artifacts.py tests/test_notebook_validation.py
python -m ruff check .
```

The benchmark artifact tests verify that referenced artifacts exist, artifact
types are expected, sizes stay reasonable, and the exporter is idempotent.

## What To Commit

Commit:

- notebook source changes that intentionally update a benchmark
- curated `_artifacts/` files produced by the exporter
- matching `notebooks/benchmarks/RESULTS.md` updates
- docs pages that link to curated results

Do not commit:

- raw `results/` cache records
- raw `images/` plot outputs
- `.pytest_cache/`
- local build outputs under `docs/_build/`

## Reading Benchmark Evidence

For published result summaries, start with:

- `docs/benchmarks/summary.md`
- `docs/benchmarks/results.md`
- `notebooks/benchmarks/SUMMARY.md`
- `notebooks/benchmarks/RESULTS.md`

Use raw local cache files only for debugging or rerun acceleration. Treat the
curated benchmark artifacts as the stable evidence surface.

