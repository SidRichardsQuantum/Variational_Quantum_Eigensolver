# 📘 Notebooks

This directory contains curated Jupyter notebooks demonstrating **VQE**, **ADAPT-VQE**, **QSE/VQD/SSVQE** (excited states), **QPE**, and **VarQITE** workflows using the packaged code in:

- `vqe/`
- `qpe/`
- `qite/`
- `common/`

Most notebooks are written as **pure package clients**: they call public APIs (e.g., `vqe.core`, `qpe.core`, `qite.core`) and do not define their own devices, QNodes, engines, caching, or plotting infrastructure.

For background and CLI usage:

- **[THEORY.md](../THEORY.md)** — algorithms and methodology
- **[USAGE.md](../USAGE.md)** — command-line usage and flags
- **[README.md](../README.md)** — project overview

---

## Directory Structure

```
notebooks/
├── README_notebooks.md
│
├── getting_started/
│   └── H2_VQE_vs_QPE_From_Scratch.ipynb
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

## 🚀 Getting Started

If you are new to this repository, start here:

`notebooks/getting_started/H2_VQE_vs_QPE_From_Scratch.ipynb`

This notebook provides minimal, conceptual implementations of **VQE** and **QPE** to explain what the algorithms are doing (before using the package APIs).

---

## ⚛️ VQE Notebooks

### H₂ (educational + production workflows)

Path: `notebooks/vqe/H2/`

H₂ is the primary educational benchmark: it is small enough to run quickly while still demonstrating the full VQE pipeline (ansatz choice, optimizers, geometry dependence, noise modelling, and excited-state methods).

| Notebook                            | Purpose                                                                                                           | Style                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `Ansatz_Comparison.ipynb`           | Compare ansätze with an educational section plus a production package-client workflow                             | Mixed (educational + package client) |
| `Bond_Length.ipynb`                 | H₂ bond-length scan using the package geometry-scan API                                                           | Package client                       |
| `Mapping_Comparison.ipynb`          | Compare fermion-to-qubit mappings for H₂                                                                          | Package client                       |
| `Noise_Scan.ipynb`                  | **Multi-seed** noise statistics for H₂ (robustness under noise)                                                   | Package client                       |
| `Noisy_Ansatz_Comparison.ipynb`     | Compare ansätze under noise (summary metrics / curves)                                                            | Package client                       |
| `Noisy_Ansatz_Convergence.ipynb`    | Noisy convergence behaviour for ansatz choices                                                                    | Package client                       |
| `Noisy_Optimizer_Comparison.ipynb`  | Compare optimizers under noise (summary metrics / curves)                                                         | Package client                       |
| `Noisy_Optimizer_Convergence.ipynb` | Noisy convergence behaviour for optimizer choices                                                                 | Package client                       |
| `QSE.ipynb`                         | **Quantum Subspace Expansion (QSE)**: post-VQE subspace spectrum vs exact eigenvalues (noiseless)               | Package client                       |
| `SSVQE.ipynb`                       | k-state excited states via **SSVQE** (noiseless + noisy validation; prints ΔEᵢ vs exact)                        | Package client                       |
| `SSVQE_Comparisons.ipynb`           | **Noiseless** SSVQE sweeps (optimizer / ansatz / full grid), pick “best” config, multi-seed validation (mean ± std) | Package client                     |
| `VQD.ipynb`                         | k-state excited states via **VQD** (noiseless + noisy validation; prints ΔEᵢ vs exact)                          | Package client                       |
| `VQD_Comparisons.ipynb`             | **Noiseless** VQD sweeps (optimizer / ansatz / full grid), pick “best” config, multi-seed validation (mean ± std) | Package client                       |

Notes:

- `Noise_Scan.ipynb` is intentionally **multi-seed** (statistical behaviour), not a single-seed demonstration notebook.
- `Ansatz_Comparison.ipynb` contains an explicitly educational “toy ansatz” section; the remainder demonstrates the production workflow.
- `QSE.ipynb` demonstrates **Quantum Subspace Expansion** as a *post-processing* method built on a converged VQE reference state.

---

### H₃⁺ (larger system benchmarks)

Path: `notebooks/vqe/H3plus/`

H₃⁺ is used as the “next step up” from H₂ (more qubits, more structure), but notebooks here remain focused to keep runtimes reasonable.

| Notebook          | Purpose                                                   | Style          |
| ----------------- | --------------------------------------------------------- | -------------- |
| `Adapt.ipynb`     | ADAPT-VQE smoke test / convergence on H₃⁺ (operator growth + inner-loop optimization) | Package client |
| `Noiseless.ipynb` | Noiseless VQE comparison for UCC-S / UCC-D / UCCSD on H₃⁺ | Package client |
| `Noisy.ipynb`     | Noisy VQE comparison for UCC-S / UCC-D / UCCSD on H₃⁺     | Package client |

---

### H₂O (geometry example)

Path: `notebooks/vqe/H2O/`

H₂O is included primarily to demonstrate a **bond-angle scan** workflow.

| Notebook           | Purpose                                              | Style          |
| ------------------ | ---------------------------------------------------- | -------------- |
| `Bond_Angle.ipynb` | H–O–H angle scan using the package geometry-scan API | Package client |

---

## 🔷 QPE Notebooks

### H₂ (noiseless + noisy QPE)

Path: `notebooks/qpe/H2/`

These notebooks demonstrate the QPE pipeline on H₂, including:

- controlled time evolution via `ApproxTimeEvolution`
- inverse QFT on ancillas
- phase → energy unwrapping using a Hartree–Fock reference
- optional noise models and parameter sweeps

They are kept intentionally minimal for runtime and clarity.

| Notebook          | Purpose                                         | Style          |
| ----------------- | ----------------------------------------------- | -------------- |
| `Noiseless.ipynb` | Noiseless QPE distribution and ancilla sweep    | Package client |
| `Noisy.ipynb`     | Noisy QPE distribution + multi-seed noise sweep | Package client |

All QPE notebooks are pure package clients, importing exclusively from `qpe.core`, `qpe.hamiltonian`, `qpe.io_utils`, and `qpe.visualize`.

---

## 🟣 QITE / VarQITE Notebooks

### H₂ (VarQITE)

Path: `notebooks/qite/H2/`

VarQITE is demonstrated on H₂ as a pure package client.

| Notebook          | Purpose                     | Style          |
| ----------------- | --------------------------- | -------------- |
| `Noiseless.ipynb` | Noiseless VarQITE on H₂      | Package client |

Note:
- If you add a noisy-evaluation notebook for VarQITE, it should follow the same convention as other “noisy evaluation / sweep” notebooks: run the *parameter update* noiseless, then evaluate the converged circuit under noise.

---

## Recommended Reading Order

1. **Conceptual baseline**
   - `getting_started/H2_VQE_vs_QPE_From_Scratch.ipynb`

2. **VQE on H₂ (core workflows)**
   - `vqe/H2/Ansatz_Comparison.ipynb`
   - `vqe/H2/Bond_Length.ipynb`

3. **Noise robustness (statistical)**
   - `vqe/H2/Noise_Scan.ipynb`
   - `qpe/H2/Noisy.ipynb`

4. **Excited states**
   - `vqe/H2/QSE.ipynb` (post-VQE subspace expansion)
   - `vqe/H2/SSVQE.ipynb`, `vqe/H2/VQD.ipynb`
   - `vqe/H2/SSVQE_Comparisons.ipynb`, `vqe/H2/VQD_Comparisons.ipynb`

5. **Larger molecules / geometry**
   - `vqe/H3plus/Adapt.ipynb`, `vqe/H3plus/Noiseless.ipynb`, `vqe/H3plus/Noisy.ipynb`
   - `vqe/H2O/Bond_Angle.ipynb`

6. **VarQITE**
   - `qite/H2/Noiseless.ipynb`

---

## Outputs and Reproducibility

Running these notebooks generates plots and JSON records via the package’s caching and I/O utilities.

Default output locations:

- `results/vqe/`, `results/qpe/`, `results/qite/` — JSON run records
- `images/vqe/`, `images/qpe/`, `images/qite/` — saved plots

CLI workflows described in `USAGE.md` follow the same defaults.

---


📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
