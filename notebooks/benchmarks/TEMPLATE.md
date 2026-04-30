# Benchmark Notebook Template

Use this outline when adding or substantially revising a benchmark notebook.
The goal is to make each notebook answer one research or method-selection
question without duplicating existing benchmark coverage.

## Question

State the decision or research question in one sentence.

Examples:

- Which method gives the best H2 ground-state estimate for a fixed small budget?
- How sensitive is QPE to ancilla count and evolution time on H2?
- Which low-qubit molecules are suitable for heavier VQE benchmarking?

## Scope

List the systems, methods, and settings included. Also state what is excluded.

Recommended fields:

- systems or Hamiltonians
- methods
- ansatzes, mappings, optimizers, or noise channels
- seeds, shots, and sweep ranges
- active-space settings

## Reference

Describe the baseline used for comparison:

- exact diagonalization
- Hartree-Fock
- analytical model result
- no reference, with a reason

## Metrics

Report only metrics that support the question.

Common metrics:

- energy
- exact/reference energy
- absolute error
- runtime and compute runtime
- cache-hit state
- convergence steps
- seed or shot spread
- method-specific failure diagnostics

## Execution

Run through the shared package APIs rather than duplicating Hamiltonian or cache
logic inside the notebook. Prefer `force=True` only when measuring fresh compute
time or intentionally refreshing evidence.

## Aggregation

For stochastic or optimizer-sensitive workflows, summarize across seeds or
shots with explicit aggregate fields such as mean error, standard deviation,
worst-case error, and failure rate.

## Limitations

State limitations that affect interpretation, such as:

- small-system-only evidence
- H2-calibrated defaults
- noiseless-only solver paths
- cache-hit timing versus fresh compute timing
- simulation-only results, not hardware execution

## Artifact Export

If a table or figure should be part of the published evidence set:

1. Add an entry to `scripts/export_benchmark_artifacts.py`.
2. Export with `python scripts/export_benchmark_artifacts.py`.
3. Confirm `RESULTS.md` and `_artifacts/benchmark_manifest.json` update.
4. Keep table columns aligned with `SCHEMA.md` where practical.
