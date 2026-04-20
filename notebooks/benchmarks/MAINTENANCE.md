# Benchmark Maintenance

This directory contains two kinds of benchmark outputs:

- raw local outputs in `results/` and `images/`, which are ignored by git
- curated published artifacts in `notebooks/benchmarks/_artifacts/`, which are tracked

Only curated artifacts should be referenced from `RESULTS.md` or the Sphinx
site.

## Updating Published Benchmark Results

Use this workflow when rerunning important benchmark notebooks:

1. Run the benchmark notebook or notebooks you want to refresh.
2. Confirm the notebook output tables and generated plots look correct.
3. Export curated artifacts:

   ```bash
   python scripts/export_benchmark_artifacts.py
   ```

4. Review the generated changes:

   ```bash
   git diff notebooks/benchmarks/RESULTS.md notebooks/benchmarks/_artifacts
   ```

5. Run the validation checks:

   ```bash
   python -m sphinx -W -b html docs docs/_build/html
   pytest -q tests/test_benchmark_artifacts.py tests/test_notebook_validation.py
   python -m ruff check .
   ```

6. Commit the notebook changes, `RESULTS.md`, and `_artifacts/` changes together.

## Artifact Rules

- Keep `results/` ignored. It is a local cache, not a public dataset.
- Keep `images/` ignored. It is a local plot-output directory.
- Track `notebooks/benchmarks/_artifacts/` because it contains curated result tables and figures.
- Do not manually edit the generated section between:

  ```text
  <!-- benchmark-artifacts:start -->
  <!-- benchmark-artifacts:end -->
  ```

  Rerun `python scripts/export_benchmark_artifacts.py` instead.

## Adding A New Curated Artifact

1. Add a figure or table entry to `scripts/export_benchmark_artifacts.py`.
2. Rerun the exporter.
3. Confirm `RESULTS.md` displays the artifact correctly in the local Sphinx site.
4. Add or adjust tests in `tests/test_benchmark_artifacts.py` if the artifact has special requirements.
