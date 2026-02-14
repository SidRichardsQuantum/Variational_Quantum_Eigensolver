# ⚛️ Usage Guide

This guide explains how to use the command-line interfaces for:

- **VQE** — Variational Quantum Eigensolver (ground states, ADAPT-VQE, and excited-state workflows)
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
````

### Install from source (development mode)

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

This installs four tightly integrated packages:

| Package  | Purpose                                                       |
| -------- | ------------------------------------------------------------- |
| `vqe`    | Variational solvers (VQE, ADAPT-VQE, LR-VQE, QSE, SSVQE, VQD) |
| `qpe`    | Quantum Phase Estimation                                      |
| `qite`   | Variational imaginary-time evolution (VarQITE)                |
| `common` | Unified Hamiltonian, molecule registry, geometry, plotting    |

Quick sanity check:

```bash
python -c "import vqe, qpe, qite, common; print('All stacks OK')"
```

---

## 📁 Output & Directory Layout

All runs are **automatically cached** and **fully reproducible**.

```
├── results/
│   ├── vqe/            # VQE-family records (VQE, ADAPT-VQE, LR-VQE, QSE, SSVQE, VQD)
│   ├── qpe/            # QPE JSON records
│   └── qite/           # VarQITE JSON records
│
└── images/
    ├── vqe/            # Convergence, scans, noise plots, LR/QSE spectra
    ├── qpe/            # Phase distributions, sweeps
    └── qite/           # VarQITE convergence, diagnostics, noise plots
```

Each run is keyed by a **hash of the full physical + numerical configuration**
(molecule, mapping, ansatz, optimizer, noise, seed, etc.).

To ignore cache:

```bash
--force
```

---

## 🔷 Running VQE

All commands below can be invoked either as `vqe ...` **or** equivalently as `python -m vqe ...`
(recommended for reproducibility across environments).

VQE supports:

* Ground-state VQE
* ADAPT-VQE (adaptive ansatz construction)
* Geometry scans (bond / angle, VQE only)
* Ansatz, optimizer, and mapping comparisons
* Noise sweeps (single & multi-seed)
* Excited states:

  * **post-VQE**: LR-VQE, QSE
  * **variational**: SSVQE, VQD

Supported molecule presets:

```
H2, LiH, H2O, H3+
```

### ▶ Basic ground-state VQE

```bash
vqe --molecule H2
```

Defaults:

* Ansatz: `UCCSD`
* Optimizer: `Adam`
* Steps: `50`
* Mapping: `jordan_wigner`

Outputs:

* `images/vqe/` — convergence plot (if `--plot`)
* `results/vqe/` — JSON record

### ▶ Choosing ansatz and optimizer

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

## 🔷 Excited-State Methods

This project supports two classes of excited-state workflows:

* **Post-VQE** (no additional variational optimization):

  * **LR-VQE** (tangent-space TDA generalized EVP)
  * **QSE** (operator subspace expansion generalized EVP)

* **Variational excited-state solvers**:

  * **SSVQE** (multi-state objective with shared unitary)
  * **VQD** (sequential deflation)

### ▶ Linear-Response VQE (LR-VQE)

LR-VQE computes excitation energies by constructing the **tangent-space generalized eigenvalue problem**
around a converged **noiseless VQE reference state**.

Stage-1 implementation:

* Tangent-space TDA (Tamm–Dancoff approximation)
* Finite-difference parameter derivatives
* Generalized EVP: **A c = ω S c**
* **Noiseless-only** (statevector reference)

#### ▶ LR-VQE via CLI

```bash
# Run LR-VQE (no plot)
vqe -m H2 --lr-vqe --lr-k 4

# Plot spectrum (exact vs LR-VQE matched by nearest exact level index)
vqe -m H2 --lr-vqe --lr-k 4 --plot

# Save the plot to images/vqe/ (and show it)
vqe -m H2 --lr-vqe --lr-k 4 --save

# Control tangent numerics
vqe -m H2 --lr-vqe --lr-k 4 --lr-fd-eps 1e-3 --lr-eps 1e-10
```

#### ▶ LR-VQE via Python API

```python
from vqe.lr_vqe import run_lr_vqe

res = run_lr_vqe(
    molecule="H2",
    k=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=80,
    stepsize=0.2,
    mapping="jordan_wigner",
    fd_eps=1e-3,
    eps=1e-10,
)

print(res["excitations"])   # ω_i
print(res["eigenvalues"])   # E0 + ω_i
```

---

### ▶ Quantum Subspace Expansion (QSE)

QSE computes approximate excited-state energies by expanding a small operator
subspace around a converged noiseless VQE reference state.

#### ▶ QSE via CLI

```bash
vqe -m H2 --qse --qse-k 4 --qse-max-ops 24 --qse-eps 1e-8
```

#### ▶ QSE via Python API

```python
from vqe import run_qse

res = run_qse(
    molecule="H2",
    k=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=60,
    stepsize=0.2,
    mapping="jordan_wigner",
)
print(res["eigenvalues"])
```

---

### ▶ Subspace-Search VQE (SSVQE)

SSVQE optimizes multiple states **simultaneously** via a multi-state objective.

#### ▶ SSVQE via CLI

```bash
# Two states (default)
vqe -m H3+ --ssvqe --num-states 2

# Custom weights (must provide exactly --num-states values)
vqe -m H3+ --ssvqe --num-states 3 --weights 1 2 3
```

#### ▶ SSVQE via Python API

```python
from vqe.ssvqe import run_ssvqe
res = run_ssvqe(molecule="H3+", num_states=3)
print(res["energies_per_state"])
```

---

### ▶ Variational Quantum Deflation (VQD)

VQD computes excited states **sequentially** via deflation against previously converged states.

#### ▶ VQD via CLI

```bash
# Two states (default), beta controls deflation strength
vqe -m H3+ --vqd --num-states 2 --beta 10.0

# Optional beta schedule controls
vqe -m H3+ --vqd --num-states 3 --beta 10.0 --beta-start 0.0 --beta-ramp cosine --beta-hold-fraction 0.2
```

#### ▶ VQD via Python API

```python
from vqe.vqd import run_vqd
res = run_vqd(molecule="H3+", num_states=3)
print(res["energies_per_state"])
```

---

## 🔷 ADAPT-VQE

ADAPT-VQE constructs the variational ansatz adaptively by selecting operators from an excitation pool
using gradient scores, growing the circuit until a stopping tolerance is met.

### ▶ ADAPT-VQE via CLI

```bash
# Basic ADAPT-VQE run (defaults: pool=uccsd, max_ops=20, grad_tol=1e-3)
vqe -m H3+ --adapt

# Configure pool + stopping criteria
vqe -m H3+ --adapt --pool uccsd --max-ops 20 --grad-tol 1e-3

# Override ADAPT inner-loop optimization (defaults to --steps/--stepsize if omitted)
vqe -m H3+ --adapt --inner-steps 75 --inner-stepsize 0.2

# Noisy ADAPT-VQE (noise must be explicitly enabled)
vqe -m H3+ --adapt --noisy --depolarizing-prob 0.02 --amplitude-damping-prob 0.0
```

### ▶ ADAPT-VQE via Python API

```python
from vqe.adapt import run_adapt_vqe

res = run_adapt_vqe(
    molecule="H3+",
    pool="uccsd",
    max_ops=20,
    grad_tol=1e-3,
    inner_steps=75,
    inner_stepsize=0.2,
    optimizer_name="Adam",
    seed=0,
    mapping="jordan_wigner",
    plot=True,
)
print(res["energy"])
```

---

## 🔷 Running QPE

QPE estimates energies via phase estimation.

### ▶ Basic QPE run

```bash
qpe --molecule H2 --ancillas 4
```

### ▶ Noisy QPE

```bash
qpe --molecule H2 --noisy --p-dep 0.05 --p-amp 0.02
```

`--noisy` must be explicitly set, otherwise `--p-dep/--p-amp` are ignored.

### ▶ Trotterized evolution

```bash
qpe --molecule H2 --t 2.0 --trotter-steps 4 --ancillas 8
```

---

## 🔷 Running QITE (VarQITE / McLachlan)

QITE implements variational imaginary-time evolution using the McLachlan principle.

It is split into two explicit modes.

### ▶ True VarQITE (noiseless)

```bash
qite run --molecule H2 --steps 50 --dtau 0.2
```

* Pure-state evolution only
* Cached parameter trajectories
* Produces convergence plots and JSON records
* Uses `default.qubit` (statevector)

### ▶ Noisy evaluation of converged parameters

```bash
qite eval-noise --molecule H2 --dep 0.02 --amp 0.0 --pretty
```

* Evaluates Tr[ρH] on `default.mixed`
* Uses cached VarQITE parameters
* Does not re-optimize
* Supports noise sweeps and multi-seed statistics

### ▶ Depolarizing sweep (mean ± std)

```bash
qite eval-noise \
  --molecule H2 \
  --steps 50 \
  --sweep-dep 0,0.02,0.04 \
  --seeds 0,1,2 \
  --pretty
```

### ℹ️ QITE caching semantics

VarQITE cache keys include:

* Molecule + geometry
* Mapping + unit
* Ansatz
* Seed
* `dtau`, `steps`
* Numerical solver settings (`fd_eps`, `reg`, `solver`, `pinv_rcond`)

This guarantees that:

* changing numerics always triggers a recompute
* cached trajectories are physically and numerically consistent
* noisy evaluation never pollutes optimization caches

---

## 🔁 Caching & Reproducibility

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

## 🧪 Testing

```bash
pytest -q
```

Covers:

* Hamiltonian registry & geometry
* VQE / ADAPT-VQE / LR-VQE / QSE / QPE / QITE minimal runs
* Noise handling
* CLI entrypoints
* Matrix consistency across stacks

---

## 📚 Citation

If you use this software, please cite:

> Sid Richards (2026). *Unified Variational and Phase-Estimation Quantum Simulation Suite.*

---

**Author:** Sid Richards (SidRichardsQuantum)
LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

MIT License — see [LICENSE](LICENSE)
