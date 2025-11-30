# ⚛️ VQE–QPE Quantum Simulation Suite — Usage Guide

This document provides **practical, example-driven instructions** for running the full VQE/QPE PennyLane simulation suite.  
It complements `README.md` by focusing on **how to run**, **what each command does**, and **where outputs go**.

---

## Installation

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

This installs the two Python packages:

- `vqe/` — Variational Quantum Eigensolver (+ SSVQE)
- `qpe/` — Quantum Phase Estimation (noiseless & noisy)

You may run them via:

```bash
python -m vqe --molecule H2
python -m qpe --molecule H2
```

or (if installed globally):

```bash
vqe --molecule H2
qpe --molecule H2
```

---

# Directory Overview

```
Variational_Quantum_Eigensolver/
├── README.md
├── THEORY.md
├── USAGE.md
├── LICENSE
├── pyproject.toml
│
├── vqe/                     # Variational Quantum Eigensolver package
├── qpe/                     # Quantum Phase Estimation package
├── vqe_qpe_common/          # Shared logic for VQE + QPE
│
├── results/                 # JSON outputs
├── images/                  # Saved plots (VQE + QPE)
├── data/                    # Optional molecule configs, external data
│
└── notebooks/               # Notebooks importing from the vqe/ and qpe/ packages
```

All VQE/QPE runs save:

- JSON output → `package_results/`
- Plots → `plots/`

These locations are **unified** across VQE and QPE.

---

# 🔹 Running VQE

VQE supports:
- Ground state VQE
- Excited-state SSVQE
- Geometry scans
- Noise sweeps
- Mapping comparisons
- Optimizer comparisons

### Basic run

```bash
python -m vqe --molecule H2
```

Outputs include:

- Optimized energy  
- Convergence plot saved under `plots/`  
- JSON record in `package_results/`

### Choose ansatz & optimizer

```bash
python -m vqe --molecule H2 -a UCCSD -o Adam
```

### Geometry scan (e.g., H₂ bond length)

```bash
python -m vqe   --scan-geometry H2_BOND   --range 0.5 1.5 7   --param-name bond   -a UCCSD
```

### Excited states (SSVQE)

```bash
python -m vqe --molecule H3+ --ssvqe --penalty-weight 10.0
```

### Noise sweep

```bash
python -m vqe --molecule LiH --noise-sweep --p-dep 0.02
```

---

# 🔹 Running QPE

QPE supports:

- Noiseless & noisy Quantum Phase Estimation
- Trotterized time evolution
- Arbitrary ancilla count
- Probability histograms
- Sweep plots (noise, time, ancillas)

### Basic QPE

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

### With plotting

```bash
python -m qpe --molecule H2 --plot
```

### Noisy QPE

```bash
python -m qpe --molecule H2 --noisy --p-dep 0.05 --p-amp 0.02 --plot
```

### Adjust time evolution and trotter steps

```bash
python -m qpe --molecule H2 --t 2.0 --trotter-steps 4 --ancillas 8
```

---

# Outputs & Caching

Every run produces:

| Type | Location | Description |
|------|----------|-------------|
| **JSON result** | `package_results/<run>.json` | energies, configuration, metadata |
| **Plots** | `plots/` | VQE convergence, QPE histograms, scans |
| **Hashed filenames** | Yes | Ensures filesystem-safe, collision-free naming |

Runs automatically reuse cached results unless `--force` is passed:

```bash
python -m qpe --molecule H2 --force
```

---

# 🧪 Testing

Run all tests:

```bash
pytest -v
```

Covers:

- VQE ground/SSVQE runs  
- QPE noiseless/noisy correctness  
- Plotting smoke tests  
- Caching integrity  

---

# Notes

- Devices used:
  - `default.qubit` for noiseless
  - `default.mixed` for noisy runs
- All geometry generation is shared through `vqe_qpe_common/geometry`
- All molecule data is unified in `vqe_qpe_common/molecules`
- All plotting uses a single interface in `vqe_qpe_common/plotting`

For open-shell systems:
```bash
pip install openfermion openfermionpyscf
```

---

# 🧾 Citation

If you use this project:

> Sid Richards (2025). *Variational Quantum Eigensolver and Quantum Phase Estimation Comparisons using PennyLane.*

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
