# Notebook Benchmark Roadmap

This roadmap is intentionally narrow: add notebooks that answer method-selection or research-validation questions, not just more usage demos.

## Highest-Value Remaining Additions
### 1. QPE Calibration Benchmark Follow-on

Goal:
turn the existing QPE calibration work into a decision-grade configuration map rather
than repeating another broad sweep.

Recommended outputs:

- absolute energy error vs exact reference
- aliasing / branch-selection failures
- best-configuration table
- explicit summary of when defaults are good enough vs when retuning is warranted

Priority molecules:

- `H2`

## Scope Guardrail

New notebooks should close a decision gap, not create another near-duplicate example.

When a new notebook overlaps strongly with an existing one, prefer replacing,
merging, or repurposing the older notebook instead of expanding the inventory.

## Current Additions In 0.3.10

- `notebooks/benchmarks/qpe/H2/Calibration_Sweep.ipynb`
- `notebooks/benchmarks/comparisons/H2/Cross_Method_Comparison.ipynb`

These two notebooks were chosen first because they improve decision-grade benchmarking rather than adding more introductory coverage.

## Current Additions In 0.3.12

- `notebooks/benchmarks/comparisons/H2/Reproducibility_Benchmark.ipynb`
- `notebooks/benchmarks/comparisons/multi_molecule/Scaling_Benchmark.ipynb`
- `notebooks/benchmarks/defaults/VQE_Default_Calibration.ipynb`
- `notebooks/benchmarks/defaults/VarQITE_Default_Calibration.ipynb`
- `notebooks/benchmarks/defaults/QPE_Default_Calibration.ipynb`

These additions close the two biggest benchmark gaps left after the earlier calibration and cross-method notebooks:

- reproducibility now has an H2 benchmark with seed spread, noisy-vs-noiseless comparisons, cache timing, and artifact inspection
- scaling now has a multi-molecule benchmark covering H2, LiH, and BeH2 with runtime, qubit-count, error, and proxy-size reporting
- defaults now have dedicated calibration notebooks so package defaults can be justified from benchmark evidence instead of ad hoc intuition

## Current Additions After 0.3.12

- `notebooks/benchmarks/vqe/H2/Noise_Robustness_Benchmark.ipynb`
- `notebooks/benchmarks/comparisons/LiH/Cross_Method_Comparison.ipynb`
- `notebooks/benchmarks/comparisons/LiH/Reproducibility_Benchmark.ipynb`

This notebook closes the remaining `H2` cross-channel VQE noise gap by comparing:

- depolarizing
- amplitude damping
- phase damping
- bit flip
- phase flip

under one shared multi-seed protocol with energy-bias, exact-error, fidelity, and sensitivity-ranking outputs.

The `LiH` cross-method notebook extends the existing `H2` comparison pattern onto a larger chemistry problem with:

- one shared active-space Hamiltonian
- exact-ground comparison
- runtime and cache-hit reporting
- explicit fairness notes about baseline `QPE` settings

The `LiH` reproducibility notebook extends the same active-space benchmark setup into:

- per-seed energy spread across methods
- noisy-versus-noiseless variance where supported
- cache-hit versus forced-rerun timing
- artifact-inspection notes for cached JSON outputs
