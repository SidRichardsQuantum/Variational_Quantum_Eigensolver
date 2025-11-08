# ⚛️ Variational Quantum Eigensolver & Quantum Phase Estimation

This project implements and compares **Variational Quantum Eigensolver (VQE)** and **Quantum Phase Estimation (QPE)** algorithms using [PennyLane](https://pennylane.ai/).  
Both are modular, reproducible, and fully scriptable from the command line.

---

## Table of Contents

- [⚙️ Installation](#️-installation)  
- [Directory Overview](#directory-overview)  
- [Running VQE](#running-vqe)  
  - [Example: H₂ Ground-State Simulation](#example-h₂-ground-state-simulation)  
  - [Supported Molecules](#other-supported-molecules)  
  - [Optional Flags](#optional-flags)  
- [Running QPE](#running-qpe)  
  - [Example: H₂ Phase Estimation (Noiseless)](#example-h₂-phase-estimation-noiseless)  
  - [Optional Parameters](#qpe-optional-parameters)  
- [Outputs & Caching](#outputs--caching)  
- [🧪 Testing](#-testing)  
- [Notes](#notes)
- [Citation](#citation)
- [Summary](#summary)

---

## ⚙️ Installation

1. Clone the repository and navigate to it:
   ```bash
   git clone https://github.com/<your-username>/Variational_Quantum_Eigensolver.git
   cd Variational_Quantum_Eigensolver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in editable (development) mode:
   ```bash
   pip install -e .
   ```

This makes both `vqe` and `qpe` executable as modules or scripts:
```bash
python -m vqe
python -m qpe
```

---

## Directory Overview

```
Variational_Quantum_Eigensolver/
├── vqe/                 # Packaged VQE module (CLI, engine, visualizations)
├── qpe/                 # Packaged QPE module (CLI, core logic, visualizations)
├── notebooks/           # Original research notebooks (for development)
│   ├── vqe/
│   └── qpe/
├── package results/     # Cached simulation results (JSON)
├── vqe/images/          # VQE plots and figures
├── qpe/images/          # QPE plots and figures
├── data/                # Raw molecule data / reference results
├── tests/               # Pytest regression tests
│
├── LICENSE              # MIT license
├── README.md            # Overview
├── THEORY.md            # Theoretical background and mathematical formulation
├── RESULTS.md           # Consolidated results and analysis
└── USAGE.md             # This file
```

---

## Running VQE

### Example: H₂ Ground-State Simulation
```bash
python -m vqe --molecule H2
```

**Output:**
- Optimized ground-state energy
- Convergence plots in `vqe/images/`
- Cached JSON results in `package results/`

### Other supported molecules:
```bash
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

### Example: H₂ Phase Estimation (noiseless)
```bash
python -m qpe --molecule H2
```

**Output:**
- QPE bitstring probability distribution
- Estimated eigenphase and corresponding energy
- Hartree–Fock comparison
- Cached results in `package results/`
- Saved plot in `qpe/images/`

Example output:
```
🔹 Running QPE for H2 (STO-3G)
▶️ Running QPE simulation...
💾 Saved QPE result → package results/H2_QPE_<hash>.json

✅ QPE completed.
Most probable state: 0100
Estimated phase: 0.125000
Estimated energy: -0.78539816 Ha
Hartree–Fock energy: -0.88842304 Ha
ΔE (QPE - HF): +0.10302488 Ha
```

### QPE optional parameters:
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
| **Numerical results** | `package results/` | JSON output with QPE or VQE parameters and energies |
| **Plots** | `vqe/images/` / `qpe/images/` | Figures saved automatically with `--save-plot` |
| **Data cache** | `data/vqe/` / `data/qpe/` | Intermediate molecule data for notebooks |

Cached results are reused automatically on reruns with identical parameters — skipping long recomputations.

---

## 🧪 Testing

To verify core functionality:
```bash
pytest -v
```

This runs lightweight reproducibility and structure tests for both VQE and QPE.

---

## Notes

- **VQE** scales well with system size; use it for LiH and H₂O.
- **QPE** grows rapidly in depth and qubits — best suited for H₂ or H₃⁺ in simulation.
- **OpenFermion Backend**: For open-shell systems (like H₃⁺), ensure you install:
  ```bash
  pip install openfermion openfermionpyscf
  ```
- **All random seeds** are fixed for reproducibility via `set_seed()`.

---

## Citation

If you use this project or its methods, please cite:
> Sid Richards (2025). *Variational Quantum Eigensolver and Quantum Phase Estimation Comparisons using PennyLane.*

---

## Summary

| Algorithm | Command | Outputs | Best for |
|------------|----------|----------|----------|
| **VQE** | `python -m vqe --molecule H2` | Convergence, mappings, geometry scans | Large molecules (LiH, H₂O) |
| **QPE** | `python -m qpe --molecule H2` | Phase distribution, eigenenergy extraction | Small systems (H₂, H₃⁺) |

Both frameworks share the same back-end chemistry and file structure, ensuring results are directly comparable.

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
