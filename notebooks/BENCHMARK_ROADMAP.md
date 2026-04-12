# Notebook Benchmark Roadmap

This roadmap is intentionally narrow: add notebooks that answer method-selection or research-validation questions, not just more usage demos.

## Highest-Value Next Additions

### 1. Reproducibility Benchmark

Goal:
measure seed sensitivity, cache reuse, and noisy-vs-noiseless variance on the same problem.

Recommended outputs:

- per-seed energy spread
- mean and standard deviation across runs
- cache-hit vs forced-rerun timing
- JSON artifact comparison notes

Priority molecules:

- `H2`
- `LiH`

### 2. Scaling Benchmark

Goal:
show what changes as the chemistry problem grows.

Recommended outputs:

- qubit count by molecule
- runtime by method
- convergence quality / final energy error
- rough memory and circuit-size proxies

Priority molecules:

- `H2`
- `LiH`
- `BeH2`

### 3. QPE Calibration Benchmark

Goal:
map the tradeoff surface between ancilla resolution, evolution time, Trotterization, and shots.

Recommended outputs:

- absolute energy error vs exact reference
- aliasing / branch-selection failures
- best-configuration table

Priority molecules:

- `H2`

### 4. Cross-Method Comparison

Goal:
compare VQE, VarQITE, and QPE on one shared Hamiltonian with a common reference and consistent reporting.

Recommended outputs:

- final energy
- absolute error to exact ground state
- wall-clock runtime
- cache / reproducibility notes

Priority molecules:

- `H2`
- `LiH`

### 5. Noise Robustness Benchmark

Goal:
compare channel-specific degradation instead of only single-channel examples in isolation.

Recommended outputs:

- depolarizing / amplitude damping / phase damping / bit flip / phase flip comparison
- energy bias and variance
- channel ranking by sensitivity

Priority molecules:

- `H2`

## Current Additions In 0.3.10

- `notebooks/benchmarks/qpe/H2/Calibration_Sweep.ipynb`
- `notebooks/benchmarks/comparisons/H2/Cross_Method_Comparison.ipynb`

These two notebooks were chosen first because they improve decision-grade benchmarking rather than adding more introductory coverage.
