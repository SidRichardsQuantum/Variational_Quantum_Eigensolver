# Notebooks

This directory contains curated Jupyter notebooks demonstrating **VQE**, **ADAPT-VQE**, **LR-VQE**, **EOM-VQE**, **QSE**, **EOM-QSE**, **SSVQE**, **VQD**, **QPE**, **VarQITE**, and **VarQRTE** workflows using the packaged code in:

- `vqe/`
- `qpe/`
- `qite/`
- `common/`

Most notebooks are written as **pure package clients**: they use the repository’s public package APIs and do not redefine their own devices, QNodes, engines, caching, or plotting utilities.

A smaller number of notebooks also build **exact reference spectra** locally (for example via `pennylane.qchem`) for validation plots and energy-difference checks.

For the main project docs, see:

- **[README.md](../README.md)** — project overview and quickstart
- **[USAGE.md](../USAGE.md)** — CLI and Python workflows
- **[THEORY.md](../THEORY.md)** — algorithms and methodology
- **[BENCHMARK_ROADMAP.md](./BENCHMARK_ROADMAP.md)** — recommended next benchmark / research notebooks

---

## How to use these notebooks

These notebooks are intended to serve three roles:

- **conceptual introductions** for learning the algorithms
- **package-client examples** showing recommended API usage
- **benchmark / comparison workflows** for reproducible experiments

Assumptions:

- the repository has been installed into the active Python environment
- package imports such as `vqe`, `qpe`, `qite`, and `common` are available
- generated outputs are written through the package’s standard caching and plotting utilities

Default output locations:

- `results/vqe/`, `results/qpe/`, `results/qite/` — JSON run records
- `images/vqe/`, `images/qpe/`, `images/qite/` — saved plots

---

## Directory Structure

```
notebooks/
├── README_notebooks.md
│
├── benchmarks/
│   ├── comparisons/
│   ├── defaults/
│   ├── qite/
│   ├── qpe/
│   └── vqe/
│
├── getting_started/
│   ├── 01_vqe_vs_qpe_from_scratch_h2.ipynb
│   └── 11_getting_started_qrte_h2.ipynb
│
├── vqe/
│   ├── H2/
│   └── H2O/
│
└── qite/
    └── H2/
```

---

## Getting Started

If you are new to the repository, begin with:

`notebooks/getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb`

This notebook provides compact, conceptual implementations of **VQE** and **QPE** before moving to the packaged workflows used elsewhere in the repository.

Fast path:

- start with `getting_started/02_getting_started_vqe_h2.ipynb` for the basic VQE API
- use `getting_started/07_getting_started_qite_h2.ipynb` for VarQITE
- use `getting_started/11_getting_started_qrte_h2.ipynb` for prepared-state VarQRTE usage
- use `benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb` when you want to validate VarQRTE against exact evolution

---

## VQE Notebooks

### H2

Path: `notebooks/vqe/H2/`

H2 is the main educational benchmark in this repository: small enough to run quickly, but rich enough to demonstrate ansatz choice, optimizer behaviour, geometry dependence, noise modelling, and excited-state workflows.

| Notebook      | Purpose                                                              | Style          |
| ------------- | -------------------------------------------------------------------- | -------------- |
| `QSE.ipynb`         | QSE spectrum from a converged VQE reference vs exact eigenvalues | Package client |
| `EOM_QSE.ipynb`     | EOM-QSE roots from a converged VQE reference vs exact eigenvalues | Package client |
| `LR_VQE.ipynb`      | LR-VQE tangent-space excitations vs exact eigenvalues            | Package client |
| `EOM_VQE.ipynb`     | EOM-VQE full-response tangent-space excitations vs exact eigenvalues | Package client |
| `SSVQE.ipynb`       | Excited-state calculation with SSVQE, including validation against exact energies | Package client |
| `VQD.ipynb`         | Excited-state calculation with VQD, including validation against exact energies | Package client |

Notes:

* `QSE.ipynb`, `EOM_QSE.ipynb`, `LR_VQE.ipynb`, and `EOM_VQE.ipynb` are all **post-VQE** workflows built on converged noiseless VQE reference states.
* `EOM_QSE.ipynb` studies a generally non-Hermitian reduced problem and uses physical-root selection heuristics.
* `LR_VQE.ipynb` demonstrates the tangent-space **Tamm-Dancoff approximation (TDA)**.
* `EOM_VQE.ipynb` demonstrates the **full-response** tangent-space workflow.

---

### H2O

Path: `notebooks/vqe/H2O/`

H2O is included primarily to demonstrate a bond-angle scan workflow.

| Notebook           | Purpose                                              | Style          |
| ------------------ | ---------------------------------------------------- | -------------- |
| `Bond_Angle.ipynb` | Two-stage H–O–H angle scan with local refinement and geometry visualization using the package geometry-scan API | Package client |

---

## QPE Notebooks

QPE introductory material now lives in `notebooks/getting_started/`, while QPE benchmarking and calibration workflows live under `notebooks/benchmarks/qpe/`.

---

## QITE / Projected-Dynamics Notebooks

### H2

Path: `notebooks/qite/H2/`

VarQITE and VarQRTE are demonstrated on H2 as package-client workflows.

| Notebook                           | Purpose                    | Style          |
| ---------------------------------- | -------------------------- | -------------- |
| `Real_Time.ipynb`                  | Noiseless VarQRTE on H2    | Package client |
| `getting_started/11_getting_started_qrte_h2.ipynb` | Prepared-state VarQRTE intro on H2 | Package client |

Note:

* if a noisy-evaluation notebook is added in future, it should follow the repository convention used elsewhere: perform the parameter-update stage noiselessly, then evaluate the converged circuit under noise
* both VarQITE and VarQRTE are implemented as projected pure-state variational dynamics workflows; they do not optimize under mixed-state noise

---

## Benchmark Notebooks

These notebooks are dedicated comparison, scan, or exact-reference workflows and now live under `notebooks/benchmarks/`.

### VQE Benchmarks

Paths:

* `notebooks/benchmarks/vqe/H2/`
* `notebooks/benchmarks/vqe/H3plus/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/vqe/H2/Ansatz_Comparison.ipynb` | Compare H2 VQE ansätze, including exact-reference checks | Mixed |
| `benchmarks/vqe/H2/Mapping_Comparison.ipynb` | Compare fermion-to-qubit mappings for H2 | Package client |
| `benchmarks/vqe/H2/Noise_Scan.ipynb` | Multi-seed H2 noise statistics benchmark | Package client |
| `benchmarks/vqe/H2/Noisy_Ansatz_Comparison.ipynb` | Compare H2 ansatz families under noise | Package client |
| `benchmarks/vqe/H2/Noisy_Ansatz_Convergence.ipynb` | Compare noisy H2 ansatz convergence traces | Package client |
| `benchmarks/vqe/H2/Noisy_Optimizer_Comparison.ipynb` | Compare H2 optimizers under noise | Package client |
| `benchmarks/vqe/H2/Noisy_Optimizer_Convergence.ipynb` | Compare noisy H2 optimizer convergence traces | Package client |
| `benchmarks/vqe/H2/SSVQE_Comparisons.ipynb` | Noiseless SSVQE sweeps across ansatz / optimizer choices | Package client |
| `benchmarks/vqe/H2/VQD_Comparisons.ipynb` | Noiseless VQD sweeps across ansatz / optimizer choices | Package client |
| `benchmarks/vqe/H3plus/Ansatz_Comparison_Noiseless.ipynb` | Noiseless H3plus ansatz comparison for UCCS / UCCD / UCCSD | Package client |
| `benchmarks/vqe/H3plus/Ansatz_Comparison_Noisy.ipynb` | Noisy H3plus ansatz comparison for UCCS / UCCD / UCCSD | Package client |

### QPE Benchmarks

Path: `notebooks/benchmarks/qpe/H2/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/qpe/H2/Noisy.ipynb` | Noisy QPE distribution and multi-seed noise sweep | Package client |
| `benchmarks/qpe/H2/Calibration_Sweep.ipynb` | Diagnostic QPE calibration sweep with analytic baselines, phase-bin diagnostics, and aliasing / branch-selection checks against an exact H2 reference | Mixed |

### Cross-Method Benchmarks

Path: `notebooks/benchmarks/comparisons/H2/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/comparisons/H2/Cross_Method_Comparison.ipynb` | Compare VQE, QPE, and VarQITE on one shared H2 Hamiltonian and exact reference | Mixed |
| `benchmarks/comparisons/H2/Reproducibility_Benchmark.ipynb` | Measure seed spread, cache timing, and noisy-vs-noiseless variance on one shared H2 problem | Mixed |
| `benchmarks/comparisons/multi_molecule/Scaling_Benchmark.ipynb` | Compare runtime, qubit count, exact-energy error, and proxy-size metrics across H2, LiH, and BeH2 | Mixed |

### Default Calibration Benchmarks

Path: `notebooks/benchmarks/defaults/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/defaults/VQE_Default_Calibration.ipynb` | Calibrate robust VQE defaults across ansatz, optimizer, stepsize, step budget, and seeds | Mixed |
| `benchmarks/defaults/VarQITE_Default_Calibration.ipynb` | Calibrate robust VarQITE defaults across ansatz, `dtau`, step budget, and seeds | Mixed |
| `benchmarks/defaults/QPE_Default_Calibration.ipynb` | Calibrate baseline QPE defaults across ancillas, evolution time, Trotter depth, shots, and seeds | Mixed |

### QITE / VarQRTE Benchmarks

Path: `notebooks/benchmarks/qite/H2/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb` | Exact-vs-VarQRTE quench benchmark on H2 | Mixed |

Notes:

* `benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb` is the main small-system correctness notebook for VarQRTE: it compares the variational trajectory against exact real-time evolution of the same post-quench initial state
* benchmark notebooks are meant to complement, not replace, the smaller usage demos in `getting_started/` and the specialized algorithm notebooks that remain under `notebooks/vqe/` and `notebooks/qite/`
* the benchmark backlog is tracked in `notebooks/BENCHMARK_ROADMAP.md`

---

## Recommended Reading Order

1. **Conceptual starting point**

   * `getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb`

2. **Core VQE workflow**

   * `getting_started/02_getting_started_vqe_h2.ipynb`
   * `getting_started/09_bond_scan_h2.ipynb`

3. **Noise studies**

   * `benchmarks/comparisons/H2/Reproducibility_Benchmark.ipynb`
   * `benchmarks/vqe/H2/Noise_Scan.ipynb`
   * `benchmarks/qpe/H2/Noisy.ipynb`

4. **Default calibration**

   * `benchmarks/defaults/VQE_Default_Calibration.ipynb`
   * `benchmarks/defaults/VarQITE_Default_Calibration.ipynb`
   * `benchmarks/defaults/QPE_Default_Calibration.ipynb`

5. **Excited-state methods**

   * `vqe/H2/QSE.ipynb`
   * `vqe/H2/EOM_QSE.ipynb`
   * `vqe/H2/LR_VQE.ipynb`
   * `vqe/H2/EOM_VQE.ipynb`
   * `vqe/H2/SSVQE.ipynb`
   * `vqe/H2/VQD.ipynb`

6. **Larger systems and geometry**

   * `getting_started/10_adapt_vqe_h3plus.ipynb`
   * `benchmarks/comparisons/multi_molecule/Scaling_Benchmark.ipynb`
   * `benchmarks/vqe/H3plus/Ansatz_Comparison_Noiseless.ipynb`
   * `benchmarks/vqe/H3plus/Ansatz_Comparison_Noisy.ipynb`
   * `vqe/H2O/Bond_Angle.ipynb`

7. **Projected dynamics**

   * `getting_started/07_getting_started_qite_h2.ipynb`
   * `getting_started/11_getting_started_qrte_h2.ipynb`
   * `benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb`
   * `qite/H2/Real_Time.ipynb`

---

## Reproducibility

These notebooks use the same caching, naming, and output conventions as the package CLI workflows described in `USAGE.md`.

That means:

* repeated runs can reuse cached results when configurations match
* plot and JSON output naming follows the shared repository conventions
* notebook experiments are aligned with the packaged solver infrastructure rather than separate one-off code paths

---

**Author:** Sid Richards (SidRichardsQuantum)

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).
