# ⚛️ VQE–QPE Quantum Simulation Suite — Usage Guide

This guide explains how to run the VQE and QPE command-line tools, what the flags do, and where outputs are stored.  
It complements `README.md` (project overview) and `THEORY.md` (background concepts).

---

## ⚙️ Installation

```bash
pip install vqe-pennylane
```

### Install from source

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

This installs three packages:

- `vqe/` — Variational Quantum Eigensolver  
- `qpe/` — Quantum Phase Estimation  
- `vqe_qpe_common/` — Shared Hamiltonian, geometry, and plotting logic  

Quick check:

```bash
python -c "import vqe, qpe; print('VQE + QPE OK')"
```

---

## 📁 Directory Overview

```
├── vqe/                # VQE CLI, engine, ansatz, optimizers, plotting, caching
├── qpe/                # QPE CLI, engine, noise, trotter, plotting, caching
├── vqe_qpe_common/     # Shared molecule registry, Hamiltonians, utilities
│
├── images/             # Saved figures (auto-generated)
│   ├── vqe/            # VQE plots
│   └── qpe/            # QPE plots
│
├── results/            # Cached JSON run records
│   ├── vqe/            # VQE results
│   └── qpe/            # QPE results
│
└── notebooks/          # Example VQE / QPE workflows
```

All runs save:

| Output | Location |
|--------|----------|
| JSON results | `data/vqe/results/`, `data/qpe/results/` |
| Plots | `data/vqe/images/`, `data/qpe/images/` |
| Cached signatures | Included in hashed filenames |

---

# 🔷 Running VQE

Supported molecules (`vqe.__main__` presets):

```
H2, LiH, H2O, H3+
```

VQE supports:  
ground-state VQE, geometry scans, optimizer comparisons, mapping comparisons, noise sweeps, SSVQE excited states.

---

## ▶ Basic VQE run

```bash
python -m vqe --molecule H2
```

Outputs:

- convergence plot  
- JSON experiment record  

---

## ▶ Choose ansatz & optimizer

```bash
python -m vqe --molecule H2 -a UCCSD -o Adam
python -m vqe --molecule H2 -a RY-CZ  -o GradientDescent
python -m vqe --molecule H2 -a TwoQubit-RY-CNOT -o Momentum
```

---

## ▶ Geometry scans

H₂ bond-length scan:

```bash
python -m vqe --scan-geometry H2_BOND               --range 0.5 1.5 7               --param-name bond               -a UCCSD
```

LiH bond-length scan:

```bash
python -m vqe --scan-geometry LiH_BOND --range 1.2 2.5 7
```

Water bond-angle scan:

```bash
python -m vqe --scan-geometry H2O_ANGLE --range 100 115 7
```

---

## ▶ Excited states (SSVQE)

```bash
python -m vqe --molecule H3+ --ssvqe --penalty-weight 10.0
```

---

## ▶ Noise sweeps (multi-seed)

```bash
python -m vqe --molecule H2 --noise-sweep --p-dep 0.02
```

Note: Noise sweeps are intended for statistical analysis across multiple random seeds rather than single-shot demonstrations.

---

# 🔷 Running QPE

Supported molecules (`qpe.__main__` presets):

```
H2, LiH, H2O, H3+
```

QPE supports:  
noiseless/noisy execution, ancilla register size, first-order Trotterization, evolution time `t`, and probability histograms.

---

## ▶ Basic QPE run

```bash
python -m qpe --molecule H2 --ancillas 4
```

---

## ▶ Plotting

```bash
python -m qpe --molecule H2 --ancillas 3 --shots 2000 --plot
```

Save and show plot:

```bash
python -m qpe --molecule H2 --plot --save-plot
```

---

## ▶ Noisy QPE

```bash
python -m qpe --molecule H2 --noisy --p-dep 0.05 --p-amp 0.02
```

---

## ▶ Evolution & Trotter parameters

```bash
python -m qpe --molecule H2               --t 2.0               --trotter-steps 4               --ancillas 8               --shots 3000
```

---

# 🔁 Caching & Reproducibility

Every run creates a signature-keyed JSON file.  
Force recompute:

```bash
python -m vqe --molecule H2 --force
python -m qpe --molecule H2 --force
```

---

# 🧪 Testing

```bash
pytest -q
```

Covers:

- VQE & QPE engine smoke tests  
- molecule registry  
- shared Hamiltonian builder  
- CLI entry points  
- QPE sampling & normalization  

---

# Citation

If you use this software:

> Sid Richards (2025). *Variational Quantum Eigensolver and Quantum Phase Estimation Comparisons using PennyLane.*

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
