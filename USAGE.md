# ⚛️ Variational Quantum Eigensolver & Quantum Phase Estimation Suite

This project implements and compares **Variational Quantum Eigensolver (VQE)** and **Quantum Phase Estimation (QPE)** algorithms using [PennyLane](https://pennylane.ai/).  
Both are modular, reproducible, and fully scriptable from the command line.

---

## Table of Contents
- [⚙️ Installation](#️-installation)
- [Directory Overview](#directory-overview)
- [Running VQE](#running-vqe)
- [Running QPE](#running-qpe)
- [Outputs & Caching](#outputs--caching)
- [🧪 Testing](#-testing)
- [Notes](#notes)
- [Citation](#citation)
- [Summary](#summary)

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

This installs both subpackages:
- `vqe` → Variational Quantum Eigensolver module
- `qpe` → Quantum Phase Estimation module

You can run either directly:
```bash
python -m vqe --molecule H2
python -m qpe --molecule H2
```

or use the entry points (if installed system-wide):
```bash
vqe --molecule H2
qpe --molecule H2
```

---

## Directory Overview

```
Variational_Quantum_Eigensolver/
├── vqe/               # VQE package (engine, CLI, visualization, optimizers)
├── qpe/               # QPE package (core, noise, CLI, visualization)
├── notebooks/         # Exploratory notebooks for molecules and noise tests
│   ├── vqe/
│   └── qpe/
├── package_results/   # Cached JSON results shared by both packages
├── vqe/images/        # VQE plots (convergence, scans, noise studies)
├── qpe/images/        # QPE plots (distributions, sweeps)
├── data/              # Optional molecule data
├── package_tests/     # Unit and reproducibility tests
│
├── LICENSE
├── README.md
├── THEORY.md
├── RESULTS.md
└── USAGE.md
```

---

## Running VQE

### Example: H₂ Ground-State Simulation
```bash
python -m vqe --molecule H2
```

Produces:
- Optimized ground-state energy
- Convergence plot → `vqe/images/`
- Cached result → `package_results/`

Other supported molecules:
```
--molecule H3+
--molecule LiH
--molecule H2O
```

### Optional flags:
```bash
--compare-mappings         # Compare Jordan–Wigner, Bravyi–Kitaev, and Parity mappings
--geometry-scan            # Bond length scans
--optimizer-comparison     # Compare classical optimizers
--noise-sweep              # Simulate different noise levels
--save-plot                # Save figures to vqe/images/
--no-plot                  # Run headless (no figure display)
```

Example:
```bash
python -m vqe --molecule LiH --geometry-scan --save-plot
```

---

## Running QPE

### Example: H₂ Phase Estimation
```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

Outputs:
- Probability histogram of ancilla states
- Estimated phase → energy conversion
- Cached JSON result → `package_results/`
- Optional plot → `qpe/images/`

Example output:
```
🔹 Running QPE for H2 (STO-3G)
▶️ Running QPE simulation...
💾 Saved QPE result → package_results/H2_QPE_<hash>.json

✅ QPE completed.
Most probable state: 0100
Estimated phase: 0.125000
Estimated energy: -0.78539816 Ha
Hartree–Fock energy: -0.88842304 Ha
ΔE (QPE - HF): +0.10302488 Ha
```

### Optional parameters:
```bash
--ancillas INT        # Number of ancilla qubits (default 4)
--t FLOAT             # Evolution time in exp(-i H t) (default 1.0)
--trotter-steps INT   # Trotterization steps (default 2)
--shots INT           # Number of samples (default 1000)
--noisy               # Enable noise model
--p_dep FLOAT         # Depolarizing probability
--p_amp FLOAT         # Amplitude damping probability
--save-plot           # Save figure to qpe/images/
--no-plot             # Disable plotting
```

**Example (noisy QPE):**
```bash
python -m qpe --molecule H2 --noisy --p_dep 0.05 --p_amp 0.02 --save-plot
```

---

## Outputs & Caching

| Type | Path | Description |
|------|------|-------------|
| **JSON Results** | `package_results/` | Shared cache for VQE and QPE results |
| **Plots** | `vqe/images/` or `qpe/images/` | Saved automatically with `--save-plot` |
| **Raw Data** | `data/` | Intermediate molecule data for notebooks |

Identical configurations automatically reuse cached runs.

---

## 🧪 Testing

To verify functionality:
```bash
pytest -v
```

Includes:
- Functional tests for VQE, SSVQE, and QPE runs
- Caching and reproducibility checks
- Plot generation and import smoke tests

---

## Notes

- **VQE** uses `default.qubit`; noisy simulations use `default.mixed`.
- **QPE** employs trotterized time evolution with optional depolarizing and amplitude damping noise.
- Both modules share a unified random seed and hashing mechanism for reproducibility.
- For open-shell systems (e.g. H₃⁺), install OpenFermion dependencies:
  ```bash
  pip install openfermion openfermionpyscf
  ```
  
---

## Summary

| Algorithm | Command | Outputs | Best for |
|------------|----------|----------|----------|
| **VQE** | `python -m vqe --molecule H2` | Convergence, geometry scans, noise sweeps | Larger molecules (LiH, H₂O) |
| **QPE** | `python -m qpe --molecule H2` | Phase histograms, eigenenergy extraction | Small molecules (H₂, H₃⁺) |

---

## Citation

If you use this project or its methods, please cite:
> Sid Richards (2025). *Variational Quantum Eigensolver and Quantum Phase Estimation Comparisons using PennyLane.*

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
