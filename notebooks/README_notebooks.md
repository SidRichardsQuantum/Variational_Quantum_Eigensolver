# 📘 Notebooks — VQE & QPE

This directory contains curated Jupyter notebooks that demonstrate the **VQE** and **QPE** workflows using the packaged code in:

- `vqe/`
- `qpe/`
- `vqe_qpe_common/`

Most notebooks are written as **pure package clients** (i.e., they call `vqe.core` / `qpe.core` and do not define their own engines, devices, QNodes, caching, or plotting logic).
The **H₂ folder** also includes a small number of explicitly educational notebooks that build intuition by implementing selected components “from scratch”.

For background and CLI usage:

- **[THEORY.md](../THEORY.md)** — essential background
- **[USAGE.md](../USAGE.md)** — command-line usage and flags
- **[README.md](../README.md)** — project overview

---

## Directory Structure

```text
notebooks/
├── README_notebooks.md
│
├── getting_started/
│
├── vqe/
│   ├── H2/
│   ├── H2O/
│   ├── H3plus/
│   └── LiH/
│
└── qpe/
    └── H2/
```

---

## 🚀 Getting Started

If you are new to this repository, start here:

`notebooks/getting_started/H2_VQE_vs_QPE_From_Scratch.ipynb` → Conceptual, minimal implementations of VQE and QPE.

This notebook explains *what* the algorithms do exactly.

---

## ⚛️ VQE Notebooks

### H₂ (educational + production workflows)
Path: `notebooks/vqe/H2/`

H₂ is the primary educational benchmark: it is small enough to run quickly while still demonstrating the full VQE pipeline (ansatz choice, optimizers, geometry dependence, noise modelling, and excited-state methods).

| Notebook | Purpose | Style |
|---|---|---|
| `VQE_From_Scratch.ipynb` | Walkthrough of a minimal VQE implementation and concepts | Educational (no package usage) |
| `Ansatz_Comparison.ipynb` | Compare ansätze with an educational section plus a pure package-client workflow | Mixed (educational + package client) |
| `Bond_Length.ipynb` | H₂ bond-length scan using the package geometry-scan API | Package client |
| `Mapping_Comparison.ipynb` | Compare fermion-to-qubit mappings for H₂ | Package client |
| `Noise_Scan.ipynb` | **Multi-seed** noise statistics for H₂ (robustness under noise) | Package client |
| `Noisy_Ansatz_Comparison.ipynb` | Compare ansätze under noise (summary metrics / curves) | Package client |
| `Noisy_Ansatz_Convergence.ipynb` | Noisy convergence behaviour for ansatz choices | Package client |
| `Noisy_Optimizer_Comparison.ipynb` | Compare optimizers under noise (summary metrics / curves) | Package client |
| `Noisy_Optimizer_Convergence.ipynb` | Noisy convergence behaviour for optimizer choices | Package client |
| `SSVQE.ipynb` | Excited-state workflow via SSVQE-style objective using packaged building blocks | Package client |

Notes:
- `Noise_Scan.ipynb` is intentionally **multi-seed only** (statistical behaviour), not a single-seed demo notebook.
- `Ansatz_Comparison.ipynb` contains an explicitly educational section; the remainder of the notebook demonstrates the production workflow as a pure package client.

---

### H₃⁺ (larger system benchmarks)
Path: `notebooks/vqe/H3plus/`

H₃⁺ is used as the “next step up” from H₂ (more qubits, more structure), but notebooks here remain focused and practical.

| Notebook | Purpose | Style |
|---|---|---|
| `Noiseless.ipynb` | Noiseless VQE comparison for UCC-S / UCC-D / UCCSD on H₃⁺ | Package client |
| `Noisy.ipynb` | Noisy VQE comparison for UCC-S / UCC-D / UCCSD on H₃⁺ | Package client |

Note:
- The former H₃⁺ noise sweep/scan notebook was removed to keep runtimes reasonable; H₂ serves as the canonical noise-scan example.

---

### LiH (package client example)
Path: `notebooks/vqe/LiH/`

LiH demonstrates a larger chemistry system in a simple, reproducible way.

| Notebook | Purpose | Style |
|---|---|---|
| `Noiseless.ipynb` | Noiseless LiH ground-state VQE using **UCCSD** via `run_vqe` | Package client |

---

### H₂O (geometry example)
Path: `notebooks/vqe/H2O/`

H₂O is included primarily to demonstrate a **bond-angle scan** workflow.

| Notebook | Purpose | Style |
|---|---|---|
| `Bond_Angle.ipynb` | H–O–H angle scan using the package geometry-scan API | Package client |

---

## 🔷 QPE Notebooks

### H₂ (noiseless + noisy QPE)
Path: `notebooks/qpe/H2/`

These notebooks demonstrate the full QPE pipeline on H₂, including:
- controlled time evolution via `ApproxTimeEvolution`
- inverse QFT on ancillas
- phase → energy unwrapping using a Hartree–Fock reference
- optional noise models and parameter sweeps

They are kept intentionally minimal for runtime and clarity.

| Notebook | Purpose |
|---|---|
| `Noiseless.ipynb` | Noiseless QPE distribution for H₂ |
| `Noisy.ipynb` | Noisy QPE distribution for H₂ |

All QPE notebooks are now pure package clients, importing exclusively from qpe.core, qpe.hamiltonian, qpe.io_utils, and qpe.visualize. No notebook-level engines, devices, caching, or plotting utilities are used.

---

## Recommended Reading Order

1. **VQE on H₂ (core intuition + workflows)**
   - `VQE_From_Scratch.ipynb`
   - `Ansatz_Comparison.ipynb`
   - `Bond_Length.ipynb`
2. **Noise robustness (statistical)**
   - `Noise_Scan.ipynb`
3. **Larger molecules (package client usage)**
   - `LiH/Noiseless.ipynb`
   - `H3plus/Noiseless.ipynb` and `H3plus/Noisy.ipynb`
   - `H2O/Bond_Angle.ipynb`
4. **Excited states**
   - `H2/SSVQE.ipynb` and `H2/VQD.ipynb`
   - `H2/SSVQE_Comparisons.ipynb` and `H2/VQD_Comparisons.ipynb`
5. **QPE**
   - `qpe/H2/Noiseless.ipynb`
   - `qpe/H2/Noisy.ipynb`

---

## Outputs and Reproducibility

Running these notebooks will generate plots and JSON records via the package’s caching and I/O utilities.

Typical output locations in this repo layout:

- `results/vqe/` and `results/qpe/` — JSON run records
- `images/vqe/` and `images/qpe/` — saved plots

If you are using the CLI workflows described in `USAGE.md`, output locations may follow the package’s configured defaults for results/images.

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







# 📘 Notebooks — VQE & QPE

This directory contains curated Jupyter notebooks demonstrating **VQE** and **QPE** workflows using the packaged code in:

- `vqe/`
- `qpe/`
- `vqe_qpe_common/`

Most notebooks are written as **pure package clients** (i.e., they call `vqe.core` / `qpe.core` and do not define their own engines, devices, QNodes, caching, or plotting logic). The **H₂ folder** also includes a small number of explicitly educational notebooks that build intuition by implementing selected components “from scratch”.

For background and CLI usage:

- **[THEORY.md](../THEORY.md)** — algorithms and methodology
- **[USAGE.md](../USAGE.md)** — command-line usage and flags
- **[README.md](../README.md)** — project overview

---

## Directory Structure

```text
notebooks/
├── README_notebooks.md
│
├── getting_started/
│   ├── H2_VQE_vs_QPE_From_Scratch.ipynb
│   └── VQE_From_Scratch.ipynb
│
├── vqe/
│   ├── H2/
│   ├── H2O/
│   ├── H3plus/
│   └── LiH/
│
└── qpe/
    └── H2/
````

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

| Notebook                            | Purpose                                                                         | Style                                |
| ----------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------ |
| `Ansatz_Comparison.ipynb`           | Compare ansätze with an educational section plus a pure package-client workflow | Mixed (educational + package client) |
| `Bond_Length.ipynb`                 | H₂ bond-length scan using the package geometry-scan API                         | Package client                       |
| `Mapping_Comparison.ipynb`          | Compare fermion-to-qubit mappings for H₂                                        | Package client                       |
| `Noise_Scan.ipynb`                  | **Multi-seed** noise statistics for H₂ (robustness under noise)                 | Package client                       |
| `Noisy_Ansatz_Comparison.ipynb`     | Compare ansätze under noise (summary metrics / curves)                          | Package client                       |
| `Noisy_Ansatz_Convergence.ipynb`    | Noisy convergence behaviour for ansatz choices                                  | Package client                       |
| `Noisy_Optimizer_Comparison.ipynb`  | Compare optimizers under noise (summary metrics / curves)                       | Package client                       |
| `Noisy_Optimizer_Convergence.ipynb` | Noisy convergence behaviour for optimizer choices                               | Package client                       |
| `SSVQE.ipynb`                       | Excited states via **SSVQE** (multi-state joint optimisation)                   | Package client                       |
| `SSVQE_Comparisons.ipynb`           | Compare SSVQE performance across ansätze/optimizers + multi-seed validation     | Package client                       |
| `VQD.ipynb`                         | Excited states via **VQD** (sequential deflation)                               | Package client                       |
| `VQD_Comparisons.ipynb`             | Compare VQD performance across ansätze/optimizers + multi-seed validation       | Package client                       |

Notes:

* `Noise_Scan.ipynb` is intentionally **multi-seed** (statistical behaviour), not a single-seed demonstration notebook.
* `Ansatz_Comparison.ipynb` contains an explicitly educational section; the remainder of the notebook demonstrates the production workflow as a pure package client.

---

### H₃⁺ (larger system benchmarks)

Path: `notebooks/vqe/H3plus/`

H₃⁺ is used as the “next step up” from H₂ (more qubits, more structure), but notebooks here remain focused and practical.

| Notebook          | Purpose                                                   | Style          |
| ----------------- | --------------------------------------------------------- | -------------- |
| `Noiseless.ipynb` | Noiseless VQE comparison for UCC-S / UCC-D / UCCSD on H₃⁺ | Package client |
| `Noisy.ipynb`     | Noisy VQE comparison for UCC-S / UCC-D / UCCSD on H₃⁺     | Package client |

Note:

* H₂ is the canonical noise-scan benchmark; H₃⁺ notebooks are kept shorter to keep runtimes reasonable.

---

### LiH (package client example)

Path: `notebooks/vqe/LiH/`

LiH demonstrates a larger chemistry system in a simple, reproducible way.

| Notebook          | Purpose                                                      | Style          |
| ----------------- | ------------------------------------------------------------ | -------------- |
| `Noiseless.ipynb` | Noiseless LiH ground-state VQE using **UCCSD** via `run_vqe` | Package client |

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

These notebooks demonstrate the full QPE pipeline on H₂, including:

* controlled time evolution via `ApproxTimeEvolution`
* inverse QFT on ancillas
* phase → energy unwrapping using a Hartree–Fock reference
* optional noise models and parameter sweeps

They are kept intentionally minimal for runtime and clarity.

| Notebook          | Purpose                           |
| ----------------- | --------------------------------- |
| `Noiseless.ipynb` | Noiseless QPE distribution for H₂ |
| `Noisy.ipynb`     | Noisy QPE distribution for H₂     |

All QPE notebooks are pure package clients, importing exclusively from `qpe.core`, `qpe.hamiltonian`, `qpe.io_utils`, and `qpe.visualize`.

---

## Recommended Reading Order

1. **VQE on H₂ (core intuition + workflows)**
   * `Ansatz_Comparison.ipynb`
   * `Bond_Length.ipynb`

2. **Noise robustness (statistical)**
   * `Noise_Scan.ipynb`

3. **Larger molecules (package client usage)**
   * `LiH/Noiseless.ipynb`
   * `H3plus/Noiseless.ipynb` and `H3plus/Noisy.ipynb`
   * `H2O/Bond_Angle.ipynb`

4. **Excited states**
   * `H2/SSVQE.ipynb` and `H2/VQD.ipynb`
   * `H2/SSVQE_Comparisons.ipynb` and `H2/VQD_Comparisons.ipynb`

5. **QPE**
   * `qpe/H2/Noiseless.ipynb`
   * `qpe/H2/Noisy.ipynb`

---

## Outputs and Reproducibility

Running these notebooks generates plots and JSON records via the package’s caching and I/O utilities.

Output locations in this repo layout:

* `results/vqe/` and `results/qpe/` — JSON run records
* `images/vqe/` and `images/qpe/` — saved plots

If you are using the CLI workflows described in `USAGE.md`, output locations follow the same package defaults.

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
