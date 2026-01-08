# ⚛️ VQE–QPE Quantum Simulation Suite — Usage Guide

This guide explains how to run the **VQE** and **QPE** command-line interfaces, what each mode does, and where outputs are stored.

It complements:

* **`README.md`** — project overview and structure
* **`THEORY.md`** — algorithmic and physical background

---

## ⚙️ Installation

### Install from PyPI

```bash
pip install vqe-pennylane
```

### Install from source (development mode)

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

This installs three Python packages:

* `vqe/` — Variational Quantum Eigensolver (VQE, SSVQE, VQD)
* `qpe/` — Quantum Phase Estimation
* `vqe_qpe_common/` — Shared Hamiltonians, molecules, geometry, plotting

Quick sanity check:

```bash
python -c "import vqe, qpe; print('VQE + QPE OK')"
```

---

## 📁 Output & Directory Layout

All executions automatically cache results and plots.

```
├── results/
│   ├── vqe/            # JSON records (VQE, SSVQE, VQD)
│   └── qpe/            # JSON records (QPE)
│
├── images/
│   ├── vqe/            # Convergence, scans, noise plots
│   └── qpe/            # Phase distributions, sweeps
```

Each run is keyed by a **hash of the full configuration** (molecule, ansatz, optimizer, noise, seed, etc.), ensuring:

* deterministic reproducibility
* safe caching
* no accidental overwrites

Use `--force` to ignore cached results.

---

# 🔷 Running VQE

Supported molecules (CLI presets):

```
H2, LiH, H2O, H3+
```

VQE supports:

* Ground-state VQE
* Geometry scans (bond length / bond angle)
* Optimizer comparisons
* Ansatz comparisons
* Fermion-to-qubit mapping comparisons
* Noise sweeps (single- and multi-seed)
* Excited states via **SSVQE** and **VQD**

---

## ▶ Basic ground-state VQE

```bash
python -m vqe --molecule H2
```

This performs a standard VQE run using defaults:

* Ansatz: `UCCSD`
* Optimizer: `Adam`
* Steps: `50`
* Mapping: `jordan_wigner`

Outputs:

* convergence plot (`images/vqe/`)
* JSON run record (`results/vqe/`)

---

## ▶ Choosing ansatz and optimizer

```bash
python -m vqe -m H2 -a UCCSD -o Adam
python -m vqe -m H2 -a RY-CZ -o GradientDescent
python -m vqe -m H2 -a StronglyEntanglingLayers -o Momentum
```

---

## ▶ Geometry scans

### H₂ bond-length scan

```bash
python -m vqe \
  --scan-geometry H2_BOND \
  --range 0.5 1.5 7 \
  --param-name bond \
  -a UCCSD
```

### LiH bond-length scan

```bash
python -m vqe \
  --scan-geometry LiH_BOND \
  --range 1.2 2.5 7
```

### H₂O bond-angle scan

```bash
python -m vqe \
  --scan-geometry H2O_ANGLE \
  --range 100 115 7
```

Each scan averages over seeds (if provided) and plots energy vs geometry.

---

## ▶ Optimizer comparison

```bash
python -m vqe \
  -m H2 \
  --compare-optimizers Adam GradientDescent Momentum
```

Produces convergence overlays and summary statistics.

---

## ▶ Ansatz comparison

```bash
python -m vqe \
  -m H2 \
  --compare-ansatzes UCCSD RY-CZ StronglyEntanglingLayers
```

---

## ▶ Fermion-to-qubit mapping comparison

```bash
python -m vqe \
  -m H2 \
  --mapping-comparison
```

Compares Jordan–Wigner, Bravyi–Kitaev, and parity mappings.

---

## ▶ Noise sweeps (single-seed)

```bash
python -m vqe \
  -m H2 \
  --noise-sweep
```

Computes ΔE and fidelity relative to a noiseless reference.

---

## ▶ Noise studies (multi-seed statistics)

```bash
python -m vqe \
  -m H2 \
  --multi-seed-noise \
  --noise-type depolarizing
```

This mode is intended for **statistical robustness analysis**, not demonstrations.

---

# 🔷 Excited-State Methods

## ▶ Subspace-Search VQE (SSVQE)

SSVQE optimizes multiple states **simultaneously**.

```bash
python -m vqe \
  -m H3+ \
  --ssvqe \
  --penalty-weight 10.0
```

Produces energy trajectories for each state and a multi-state convergence plot.

---

## ▶ Variational Quantum Deflation (VQD)

VQD finds excited states **sequentially** using deflation penalties.

VQD is currently exposed via the **Python API** and example notebooks:

```python
from vqe.vqd import run_vqd

res = run_vqd(
    molecule="H3+",
    num_states=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    noisy=True,
    depolarizing_prob=0.02,
)
```

CLI support for VQD is intentionally deferred to keep the interface explicit and controlled.

---

# 🔷 Running QPE

Supported molecules:

```
H2, LiH, H2O, H3+
```

QPE supports:

* Noiseless and noisy execution
* Configurable ancilla register
* Trotterized time evolution
* Histogram and sweep plots
* Result caching

---

## ▶ Basic QPE run

```bash
python -m qpe --molecule H2 --ancillas 4
```

---

## ▶ Plot phase distribution

```bash
python -m qpe \
  --molecule H2 \
  --ancillas 4 \
  --shots 2000 \
  --plot
```

---

## ▶ Noisy QPE

```bash
python -m qpe \
  --molecule H2 \
  --noisy \
  --p-dep 0.05 \
  --p-amp 0.02
```

---

## ▶ Evolution time & Trotter steps

```bash
python -m qpe \
  --molecule H2 \
  --t 2.0 \
  --trotter-steps 4 \
  --ancillas 8 \
  --shots 3000
```

---

# 🔁 Caching & Reproducibility

All runs are cached by a **full configuration hash**.

Force recomputation:

```bash
python -m vqe --molecule H2 --force
python -m qpe --molecule H2 --force
```

Seeds are always recorded in JSON outputs.

---

# 🧪 Testing

```bash
pytest -v
```

Covers:

* VQE & QPE engines
* Excited-state workflows
* Molecule registry
* Hamiltonian construction
* CLI entrypoints
* Noise handling

---

# Citation

If you use this software, please cite:

> Sid Richards (2025). *Variational Quantum Eigensolver and Quantum Phase Estimation using PennyLane.*

---

📘 **Author:** Sid Richards (SidRichardsQuantum)
LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

This project is licensed under the MIT License — see [LICENSE](LICENSE).
