# ⚛️ Usage Guide

This guide explains how to use the command-line interfaces for:

- **VQE** — Variational Quantum Eigensolver (ground & excited states)
- **QPE** — Quantum Phase Estimation
- **QITE** — Variational Quantum Imaginary Time Evolution (VarQITE)
- **common** — Unified Hamiltonian and molecule registry (internal)

It complements:

- **`README.md`** — project overview and architecture
- **`THEORY.md`** — algorithmic and physical background

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

This installs four tightly integrated packages:

| Package  | Purpose                                                         |
| -------- | --------------------------------------------------------------- |
| `vqe`    | Ground- and excited-state variational solvers (VQE, SSVQE, VQD) |
| `qpe`    | Quantum Phase Estimation                                        |
| `qite`   | Variational imaginary-time evolution (VarQITE)                  |
| `common` | Unified Hamiltonian, molecule registry, geometry, plotting      |

Quick sanity check:

```bash
python -c "import vqe, qpe, qite, common; print('All stacks OK')"
```

---

## 📁 Output & Directory Layout

All runs are **automatically cached** and **fully reproducible**.

```
├── results/
│   ├── vqe/            # VQE, SSVQE, VQD JSON records
│   ├── qpe/            # QPE JSON records
│   └── qite/           # VarQITE JSON records
│
├── images/
│   ├── vqe/            # Convergence, scans, noise plots
│   ├── qpe/            # Phase distributions, sweeps
│   └── qite/           # VarQITE convergence plots
```

Each run is keyed by a **hash of the full physical + numerical configuration**
(molecule, mapping, ansatz, optimizer, noise, seed, etc.).

To ignore cache:

```bash
--force
```

---

# 🔷 Running VQE

Supported molecule presets:

```
H2, LiH, H2O, H3+
```

VQE supports:

* Ground-state VQE
* Geometry scans (bond / angle)
* Ansatz, optimizer, and mapping comparisons
* Noise sweeps (single & multi-seed)
* Excited states (SSVQE, VQD)

---

## ▶ Basic ground-state VQE

```bash
vqe --molecule H2
```

Defaults:

* Ansatz: `UCCSD`
* Optimizer: `Adam`
* Steps: `50`
* Mapping: `jordan_wigner`

Outputs:

* `images/vqe/` — convergence plot
* `results/vqe/` — JSON record

---

## ▶ Choosing ansatz and optimizer

```bash
vqe -m H2 -a UCCSD -o Adam
vqe -m H2 -a RY-CZ -o GradientDescent
vqe -m H2 -a StronglyEntanglingLayers -o Momentum
```

---

## ▶ Geometry scans

### H₂ bond scan

```bash
vqe --scan-geometry H2_BOND --range 0.5 1.5 7
```

### H₂O angle scan

```bash
vqe --scan-geometry H2O_ANGLE --range 100 115 7
```

---

## ▶ Noise studies (statistics)

```bash
vqe -m H2 --multi-seed-noise --noise-type depolarizing
```

Designed for **robust noise analysis**, not demos.

---

# 🔷 Excited-State VQE

## ▶ Subspace-Search VQE (SSVQE)

```bash
vqe -m H3+ --ssvqe --penalty-weight 10.0
```

Optimizes multiple states **simultaneously**.

---

## ▶ Variational Quantum Deflation (VQD)

VQD is exposed via the Python API and notebooks:

```python
from vqe.vqd import run_vqd
res = run_vqd(molecule="H3+", num_states=3)
```

CLI exposure is intentionally deferred to keep workflows explicit.

---

# 🔷 Running QPE

QPE estimates energies via phase estimation.

---

## ▶ Basic QPE run

```bash
qpe --molecule H2 --ancillas 4
```

---

## ▶ Noisy QPE

```bash
qpe --molecule H2 --p-dep 0.05 --p-amp 0.02
```

---

## ▶ Trotterized evolution

```bash
qpe --molecule H2 --t 2.0 --trotter-steps 4 --ancillas 8
```

---

# 🔷 Running QITE (VarQITE)

QITE implements **variational imaginary-time evolution** using the McLachlan principle.

It is split into **two explicit modes**:

---

## ▶ True VarQITE (noiseless)

```bash
qite run --molecule H2 --steps 50 --dtau 0.2
```

* Pure-state evolution only
* Cached parameter trajectories
* Produces convergence plots and JSON records
* Uses `default.qubit` (statevector)

---

## ▶ Noisy evaluation of converged parameters

```bash
qite eval-noise --molecule H2 --dep 0.02 --amp 0.0 --pretty
```

* Evaluates **Tr[ρH]** on `default.mixed`
* Uses cached VarQITE parameters
* Does **not** re-optimize
* Supports noise sweeps and multi-seed statistics

---

## ▶ Depolarizing sweep (mean ± std)

```bash
qite eval-noise \
  --molecule H2 \
  --steps 50 \
  --sweep-dep 0,0.02,0.04 \
  --seeds 0,1,2
```

---

# 🔁 Caching & Reproducibility

All algorithms share:

* Unified Hamiltonian construction (`common.hamiltonian`)
* Deterministic run hashing
* Seed-safe caching
* JSON-first records
* Plot regeneration without recomputation

Force recomputation:

```bash
vqe --force
qpe --force
qite run --force
```

---

# 🧪 Testing

```bash
pytest -v
```

Covers:

* Hamiltonian registry & geometry
* VQE / QPE / QITE minimal runs
* Noise handling
* CLI entrypoints
* Matrix consistency across stacks

---

# 📚 Citation

If you use this software, please cite:

> Sid Richards (2026). *Unified Variational and Phase-Estimation Quantum Simulation Suite.*

---

**Author:** Sid Richards (SidRichardsQuantum)
LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

MIT License — see [LICENSE](LICENSE)
