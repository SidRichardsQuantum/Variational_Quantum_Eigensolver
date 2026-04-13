# Notebook Benchmark Roadmap

This roadmap is intentionally narrow: add notebooks that answer method-selection or research-validation questions, not just more usage demos.

## Highest-Value Remaining Additions
### 1. Noise Robustness Benchmark

Goal:
compare channel-specific degradation instead of only single-channel examples in isolation.

Recommended outputs:

- depolarizing / amplitude damping / phase damping / bit flip / phase flip comparison
- energy bias and variance
- channel ranking by sensitivity

Priority molecules:

- `H2`

### 2. LiH Cross-Method Extension

Goal:
extend the existing cross-method benchmarking pattern beyond `H2` so the repo has at least one comparison notebook on a larger chemistry problem.

Recommended outputs:

- final energy
- absolute error to exact ground state
- wall-clock runtime
- cache / reproducibility notes

Priority molecules:

- `LiH`

### 3. LiH Reproducibility Extension

Goal:
repeat the reproducibility benchmark on a system larger than `H2` to test whether cache and seed behavior remain decision-grade once active-space choices matter.

Recommended outputs:

- per-seed energy spread
- mean and standard deviation across runs
- cache-hit vs forced-rerun timing
- JSON artifact comparison notes

Priority molecules:

- `LiH`

### 4. QPE Calibration Benchmark Follow-on

Goal:
map the tradeoff surface between ancilla resolution, evolution time, Trotterization, and shots.

Recommended outputs:

- absolute energy error vs exact reference
- aliasing / branch-selection failures
- best-configuration table

Priority molecules:

- `H2`

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
