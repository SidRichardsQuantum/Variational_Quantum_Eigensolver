# ⚛️ VQE–QPE Quantum Simulation Suite — Usage Guide

This guide provides **practical, example-driven instructions** for running the full VQE/QPE PennyLane simulation suite.  
It complements `README.md` by focusing on **how to run**, **what each command does**, and **where outputs go**.

---

## 🚀 Installation

### Install from source (development mode)

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

This installs three Python packages:

- `vqe/` — Variational Quantum Eigensolver  
- `qpe/` — Quantum Phase Estimation  
- `vqe_qpe_common/` — shared utilities (geometry, Hamiltonians, plotting)

To verify installation:

```bash
python -c "import vqe, qpe; print('VQE+QPE imported successfully!')"
```

You can run via:

```bash
python -m vqe --molecule H2
python -m qpe --molecule H2
```

or, if installed globally:

```bash
vqe --molecule H2
qpe --molecule H2
```

---

# 📁 Directory Overview

```
Variational_Quantum_Eigensolver/
├── README.md
├── THEORY.md
├── USAGE.md
├── LICENSE
├── pyproject.toml
│
├── vqe/                     # Variational Quantum Eigensolver
├── qpe/                     # Quantum Phase Estimation
├── vqe_qpe_common/          # Shared logic (Hamiltonians, geometry, plotting)
│
├── results/                 # JSON outputs
├── plots/                   # Saved plots (VQE + QPE)
├── data/                    # Optional molecule configs or external data
│
└── notebooks/               # Example notebooks using the package APIs
```

All VQE/QPE runs save:

- JSON output → `package_results/`
- Plots → `plots/`

---

# 🔹 Running VQE

VQE supports:
- Ground-state VQE  
- Excited-state SSVQE  
- Geometry scans  
- Mapping comparisons  
- Noise sweeps  
- Optimizer comparisons  

### ▶ Basic run

```bash
python -m vqe --molecule H2
```

Outputs include:

- optimized energy  
- convergence plot → `plots/`  
- JSON record → `package_results/`  

### ▶ Choose ansatz & optimizer

```bash
python -m vqe --molecule H2 -a UCCSD -o Adam
```

### ▶ Geometry scan (bond stretch example)

```bash
python -m vqe --scan-geometry H2_BOND               --range 0.5 1.5 7               --param-name bond               -a UCCSD
```

### ▶ Excited states (SSVQE)

```bash
python -m vqe --molecule H3+ --ssvqe --penalty-weight 10.0
```

### ▶ Noise sweep

```bash
python -m vqe --molecule LiH --noise-sweep --p-dep 0.02
```

---

# 🔹 Running QPE

QPE supports:

- noiseless / noisy QPE  
- variable ancilla count  
- Trotterized time evolution  
- noise sweeps  
- probability histograms  

### ▶ Basic QPE

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

### ▶ With plotting

```bash
python -m qpe --molecule H2 --plot
```

### ▶ Noisy QPE

```bash
python -m qpe --molecule H2 --noisy --p-dep 0.05 --p-amp 0.02 --plot
```

### ▶ Adjust simulation parameters

```bash
python -m qpe --molecule H2 --t 2.0 --trotter-steps 4 --ancillas 8
```

---

# 📦 Outputs & Caching

Every run produces:

| Type | Location | Description |
|------|----------|-------------|
| **JSON result** | `package_results/` | energies, configs, metadata |
| **Plots** | `plots/` | convergence (VQE), histograms (QPE), scans |
| **Hashed filenames** | Yes | ensures collision-free caching |

To force a fresh run:

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

- VQE (ground + SSVQE)  
- QPE (noisy + noiseless)  
- CLI entry points  
- Plot generation  
- Caching  

---

# 📝 Notes

- Devices:
  - `default.qubit` (noisy/off)  
  - `default.mixed` (noisy simulations)  
- All geometry is defined in `vqe_qpe_common/geometry`  
- All Hamiltonians in `vqe_qpe_common/hamiltonian`  
- Plotting via `vqe_qpe_common/plotting`  

For open-shell simulations:

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
