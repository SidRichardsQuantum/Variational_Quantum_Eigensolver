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
│   ├── qite/
│   ├── qpe/
│   └── vqe/
│
├── getting_started/
│   ├── vqe_vs_qpe_from_scratch_h2.ipynb
│   └── 13_getting_started_qrte_h2.ipynb
│
├── vqe/
│   ├── H2/
│   ├── H2O/
│   └── H3plus/
│
├── qpe/
│   └── H2/
│
└── qite/
    └── H2/
```

---

## Getting Started

If you are new to the repository, begin with:

`notebooks/getting_started/vqe_vs_qpe_from_scratch_h2.ipynb`

This notebook provides compact, conceptual implementations of **VQE** and **QPE** before moving to the packaged workflows used elsewhere in the repository.

---

## VQE Notebooks

### H2

Path: `notebooks/vqe/H2/`

H2 is the main educational benchmark in this repository: small enough to run quickly, but rich enough to demonstrate ansatz choice, optimizer behaviour, geometry dependence, noise modelling, and excited-state workflows.

| Notebook      | Purpose                                                              | Style          |
| ------------- | -------------------------------------------------------------------- | -------------- |
| `Bond_Length.ipynb` | H2 bond-length scan using the package geometry-scan API         | Package client |
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

### H3plus

Path: `notebooks/vqe/H3plus/`

H3plus is used as a larger benchmark than H2 while still keeping notebook runtimes practical.

| Notebook          | Purpose                                                      | Style          |
| ----------------- | ------------------------------------------------------------ | -------------- |
| `Adapt.ipynb`     | ADAPT-VQE convergence and operator-growth workflow on H3plus | Package client |

---

### H2O

Path: `notebooks/vqe/H2O/`

H2O is included primarily to demonstrate a bond-angle scan workflow.

| Notebook           | Purpose                                              | Style          |
| ------------------ | ---------------------------------------------------- | -------------- |
| `Bond_Angle.ipynb` | H–O–H angle scan using the package geometry-scan API | Package client |

---

## QPE Notebooks

### H2

Path: `notebooks/qpe/H2/`

These notebooks demonstrate the QPE pipeline on H2, including:

* controlled time evolution
* inverse QFT on ancillas
* phase-to-energy recovery
* optional noise studies and sweeps

| Notebook          | Purpose                                           | Style          |
| ----------------- | ------------------------------------------------- | -------------- |
| `Noiseless.ipynb` | Noiseless QPE distribution and ancilla sweep      | Package client |

All QPE notebooks are package clients built on the `qpe` module and shared chemistry infrastructure.

---

## QITE / Projected-Dynamics Notebooks

### H2

Path: `notebooks/qite/H2/`

VarQITE and VarQRTE are demonstrated on H2 as package-client workflows.

| Notebook                           | Purpose                    | Style          |
| ---------------------------------- | -------------------------- | -------------- |
| `Noiseless.ipynb`                  | Noiseless VarQITE on H2    | Package client |
| `Real_Time.ipynb`                  | Noiseless VarQRTE on H2    | Package client |
| `getting_started/13_getting_started_qrte_h2.ipynb` | Prepared-state VarQRTE intro on H2 | Package client |

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
| `benchmarks/vqe/H3plus/Ansatz_Comparison_Noiseless.ipynb` | Noiseless H3plus ansatz comparison for UCC-S / UCC-D / UCCSD | Package client |
| `benchmarks/vqe/H3plus/Ansatz_Comparison_Noisy.ipynb` | Noisy H3plus ansatz comparison for UCC-S / UCC-D / UCCSD | Package client |

### QPE Benchmarks

Path: `notebooks/benchmarks/qpe/H2/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/qpe/H2/Noisy.ipynb` | Noisy QPE distribution and multi-seed noise sweep | Package client |

### QITE / VarQRTE Benchmarks

Path: `notebooks/benchmarks/qite/H2/`

| Notebook | Purpose | Style |
| -------- | ------- | ----- |
| `benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb` | Exact-vs-VarQRTE quench benchmark on H2 | Mixed |

Notes:

* `benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb` is the main small-system correctness notebook for VarQRTE: it compares the variational trajectory against exact real-time evolution of the same post-quench initial state
* benchmark notebooks are meant to complement, not replace, the smaller usage demos in `getting_started/` and the algorithm-specific package-client notebooks

---

## Recommended Reading Order

1. **Conceptual starting point**

   * `getting_started/vqe_vs_qpe_from_scratch_h2.ipynb`

2. **Core VQE workflow**

   * `getting_started/01_getting_started_vqe_h2.ipynb`
   * `vqe/H2/Bond_Length.ipynb`

3. **Noise studies**

   * `getting_started/12_noise_scan_h2.ipynb`
   * `benchmarks/vqe/H2/Noise_Scan.ipynb`
   * `benchmarks/qpe/H2/Noisy.ipynb`

4. **Excited-state methods**

   * `vqe/H2/QSE.ipynb`
   * `vqe/H2/EOM_QSE.ipynb`
   * `vqe/H2/LR_VQE.ipynb`
   * `vqe/H2/EOM_VQE.ipynb`
   * `vqe/H2/SSVQE.ipynb`
   * `vqe/H2/VQD.ipynb`

5. **Larger systems and geometry**

   * `vqe/H3plus/Adapt.ipynb`
   * `benchmarks/vqe/H3plus/Ansatz_Comparison_Noiseless.ipynb`
   * `benchmarks/vqe/H3plus/Ansatz_Comparison_Noisy.ipynb`
   * `vqe/H2O/Bond_Angle.ipynb`

6. **Projected dynamics**

   * `qite/H2/Noiseless.ipynb`
   * `getting_started/13_getting_started_qrte_h2.ipynb`
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
