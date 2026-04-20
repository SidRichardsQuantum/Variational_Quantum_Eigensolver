# Benchmark Summary

This page summarizes the benchmark coverage currently present under
`notebooks/benchmarks/` and the generated artifact state observed in the local
workspace.

The benchmark notebooks are the source of truth. The ignored `results/` and
`images/` directories are local/generated artifacts and are not packaged or
committed.

## Coverage At A Glance

| Area | Notebooks | Main question answered |
| --- | ---: | --- |
| Cross-method chemistry comparisons | 4 | How do VQE, VarQITE, and QPE compare on H2 and LiH? |
| Multi-molecule chemistry panels | 5 | Which registry systems are ready for broader benchmarks, scaling, and ionization studies? |
| Default calibration | 3 | Are the package defaults justified by calibration sweeps? |
| Non-molecule model Hamiltonians | 3 | Does expert-mode Hamiltonian input work beyond chemistry? |
| QPE calibration and noise | 3 | How sensitive is QPE to ancilla count, evolution time, shots, and noise? |
| VQE H2 benchmark studies | 10 | How do ansatzes, mappings, noisy channels, and excited-state solvers behave on H2? |
| VQE H3+ benchmark studies | 2 | How do ansatzes behave on a larger hydrogen ion panel? |
| QITE / QRTE dynamics | 1 | How does projected real-time evolution compare with exact evolution on H2? |

Total benchmark notebooks: **31**.

## Method Selection Notes

| Need | Start with | Why |
| --- | --- | --- |
| Fast ground-state baseline | VQE | Lowest overhead and broadest notebook coverage. |
| Adaptive ansatz construction | ADAPT-VQE | Useful when fixed ansatz choice is the main uncertainty. |
| Excited-state estimates after a reference VQE | QSE, EOM-QSE, LR-VQE, EOM-VQE | Reuses a converged VQE reference and exposes post-VQE excited-state diagnostics. |
| Direct variational excited states | SSVQE or VQD | Useful when excited states are the optimization target rather than a post-processing step. |
| Phase / spectral information | QPE | Best represented by the H2 calibration and decision-map notebooks. |
| Imaginary-time relaxation | VarQITE | Useful as an alternative state-preparation or relaxation path. |
| Real-time dynamics | VarQRTE | Covered by the exact QRTE benchmark on H2. |
| Non-chemistry qubit models | Expert-mode Hamiltonian inputs | Covered by TFIM, XXZ Heisenberg, and SSH benchmark notebooks. |

## Current Benchmark Notebook Inventory

### Cross-Method Comparisons

- `comparisons/H2/Cross_Method_Comparison.ipynb`
- `comparisons/H2/Reproducibility_Benchmark.ipynb`
- `comparisons/LiH/Cross_Method_Comparison.ipynb`
- `comparisons/LiH/Reproducibility_Benchmark.ipynb`

### Multi-Molecule Comparisons

- `comparisons/multi_molecule/Atomic_Ionization_Energy_Benchmark.ipynb`
- `comparisons/multi_molecule/Hydrogen_Family_VQE_Benchmark.ipynb`
- `comparisons/multi_molecule/Low_Qubit_VQE_Benchmark.ipynb`
- `comparisons/multi_molecule/Registry_Coverage.ipynb`
- `comparisons/multi_molecule/Scaling_Benchmark.ipynb`

### Default Calibration

- `defaults/QPE_Default_Calibration.ipynb`
- `defaults/VQE_Default_Calibration.ipynb`
- `defaults/VarQITE_Default_Calibration.ipynb`

### Non-Molecule Model Hamiltonians

- `non_molecule/Heisenberg_Chain_Benchmark.ipynb`
- `non_molecule/SSH_Chain_Benchmark.ipynb`
- `non_molecule/TFIM_Cross_Method_Benchmark.ipynb`

### QITE / QRTE

- `qite/H2/Exact_QRTE_Benchmark.ipynb`

### QPE

- `qpe/H2/Calibration_Decision_Map.ipynb`
- `qpe/H2/Calibration_Sweep.ipynb`
- `qpe/H2/Noisy.ipynb`

### VQE

- `vqe/H2/Ansatz_Comparison.ipynb`
- `vqe/H2/Mapping_Comparison.ipynb`
- `vqe/H2/Noise_Robustness_Benchmark.ipynb`
- `vqe/H2/Noise_Scan.ipynb`
- `vqe/H2/Noisy_Ansatz_Comparison.ipynb`
- `vqe/H2/Noisy_Ansatz_Convergence.ipynb`
- `vqe/H2/Noisy_Optimizer_Comparison.ipynb`
- `vqe/H2/Noisy_Optimizer_Convergence.ipynb`
- `vqe/H2/SSVQE_Comparisons.ipynb`
- `vqe/H2/VQD_Comparisons.ipynb`
- `vqe/H3plus/Ansatz_Comparison_Noiseless.ipynb`
- `vqe/H3plus/Ansatz_Comparison_Noisy.ipynb`

## Artifact Audit

Observed local generated artifacts:

| Directory | Files | Size | Git status |
| --- | ---: | ---: | --- |
| `results/` | 733 JSON files | 6.2 MiB | ignored |
| `images/` | 3 PNG files | 464 KiB | ignored |

The result cache is useful for local reruns, but it is not clean enough to act
as a committed benchmark dataset. It still mixes:

- benchmark-style H2, H3+, H4, LiH, and hydrogen-family runs
- QPE calibration sweeps with hundreds of per-seed/per-configuration JSON files
- multiple historical cache signatures for similar benchmark configurations

Those generated cache artifacts are expected for local development, but they
should not be interpreted as a curated benchmark dataset.

The stale smoke/cache result files and orphaned images identified during the
0.3.23 artifact audit have been removed from the local workspace.

Referenced generated images remaining locally:

- `images/vqe/H2/ansatz_conv_Adam_s0.png`
- `images/vqe/H2/mapping_comparison_UCCSD_Adam.png`
- `images/vqe/multi_molecule/low_qubit_benchmark_UCCSD_Adam_jordan_wigner_max10q.png`

## Recommended Cleanup Policy

Keep `results/` and `images/` ignored. Before publishing a benchmark result as
documentation, promote only curated tables or plots into a reviewed docs asset
path. Avoid using the raw local cache as a public benchmark source unless it is
regenerated from a pinned script and validated in CI.
