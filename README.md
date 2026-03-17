# Quantum Simulation Suite — VQE, Excited-State Methods, QPE, and VarQITE (PennyLane)

<p align="center">

<a href="https://pypi.org/project/vqe-pennylane/">
<img src="https://img.shields.io/pypi/v/vqe-pennylane?style=flat-square" alt="PyPI Version">
</a>

<a href="https://pypi.org/project/vqe-pennylane/">
<img src="https://img.shields.io/pypi/dm/vqe-pennylane?style=flat-square" alt="PyPI Downloads">
</a>

<a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/actions/workflows/tests.yml">
<img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Variational_Quantum_Eigensolver/tests.yml?label=tests&style=flat-square" alt="Tests">
</a>

<img src="https://img.shields.io/pypi/pyversions/vqe-pennylane?style=flat-square" alt="Python Versions">

<img src="https://img.shields.io/github/license/SidRichardsQuantum/Variational_Quantum_Eigensolver?style=flat-square" alt="License">

</p>

A **modular quantum-chemistry simulation suite** built on **PennyLane**, combining:

- **Variational quantum algorithms** (VQE and excited-state extensions)
- **Phase-estimation algorithms** (QPE)
- **Imaginary-time evolution** (VarQITE)

The project provides a **reproducible research framework** with:

- unified molecule and Hamiltonian infrastructure
- consistent caching and run signatures
- shared plotting and output conventions
- CLI workflows and Python APIs
- curated notebooks for benchmarking and demonstrations

The repository consolidates many earlier exploratory notebooks into a **clean, versioned Python package** with a shared `common/` layer used by all solvers.

---

## Implemented Algorithms

### Variational methods

- **VQE** — ground-state variational eigensolver  
- **ADAPT-VQE** — adaptive ansatz growth using gradient-selected operators  

### Post-VQE excited-state methods

- **LR-VQE** — tangent-space linear response (Tamm–Dancoff approximation)
- **EOM-VQE** — full-response tangent-space equation of motion
- **QSE** — operator-subspace expansion
- **EOM-QSE** — commutator equation-of-motion formulation

### Variational excited-state solvers

- **SSVQE** — multi-state optimization in a shared unitary
- **VQD** — sequential excited-state deflation

### Non-variational algorithms

- **QPE** — phase-estimation energy extraction
- **QITE / VarQITE** — imaginary-time ground-state filtering

---

## Feature Overview

| Method | Category | Optimization | Noise Support |
|------|------|------|------|
| **VQE** | Variational | Classical optimizer | Yes |
| **ADAPT-VQE** | Variational | Adaptive + classical | Yes |
| **LR-VQE** | Post-VQE | Linear algebra | No (statevector only) |
| **EOM-VQE** | Post-VQE | Linear algebra | No (statevector only) |
| **QSE** | Post-VQE | Linear algebra | No (statevector only) |
| **EOM-QSE** | Post-VQE | Linear algebra | No (statevector only) |
| **SSVQE** | Variational | Simultaneous | Yes |
| **VQD** | Variational | Sequential | Yes |
| **QPE** | Phase estimation | None | Yes |
| **QITE / VarQITE** | Imaginary time | McLachlan updates | Noisy evaluation only |

---

## Documentation

The repository documentation is split across several files:

| File | Purpose |
|-----|------|
| **README.md** | Overview and quickstart |
| **USAGE.md** | CLI and Python usage guide |
| **THEORY.md** | Algorithms and methodology |
| **notebooks/README_notebooks.md** | Notebook index and workflow guide |

Start here if you are new to the project:

- 📘 **Algorithms and derivations:** [`THEORY.md`](THEORY.md)  
- ⚙️ **Command-line usage:** [`USAGE.md`](USAGE.md)  
- 📓 **Example notebooks:** [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md)

---

## Repository Structure

```
Variational_Quantum_Eigensolver/
├── README.md
├── THEORY.md
├── USAGE.md
├── pyproject.toml
│
├── vqe/        # Variational solvers (VQE, ADAPT-VQE, LR-VQE, EOM-VQE, QSE, EOM-QSE, SSVQE, VQD)
├── qpe/        # Quantum Phase Estimation
├── qite/       # Variational imaginary-time evolution (VarQITE)
│
├── common/     # Shared infrastructure
│   ├── molecules.py
│   ├── geometry.py
│   ├── hamiltonian.py
│   ├── plotting.py
│   ├── paths.py
│   └── persist.py
│
├── notebooks/  # Demonstrations and benchmarks
│
├── results/    # Cached run records (ignored in Git)
└── images/     # Generated plots (ignored in Git)
```

Design goals:
- Shared chemistry layer across all algorithms (`common/`)
- Deterministic caching and run signatures
- Unified plotting and output conventions
- CLI and Python APIs built on the same internal engines

---

## Installation

### Install from PyPI

```bash
pip install vqe-pennylane
```

### Install from source

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

### Verify installation

```bash
python -c "import vqe, qpe, qite, common; print('Quantum stacks imported successfully')"
```

---

## Common Core (Shared by VQE, QPE & QITE)

The following modules ensure full consistency between solvers:

| Module                          | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| `common/molecules.py`   | Canonical molecule definitions                  |
| `common/geometry.py`    | Bond/angle/coordinate generators                |
| `common/hamiltonian.py` | Hamiltonian construction + OpenFermion fallback |
| `common/plotting.py`    | Unified filename builder + PNG export           |

---

## VQE Package

The `vqe` module implements ground-state VQE together with multiple
excited-state workflows.

### Capabilities

- Ground-state **VQE**
- **ADAPT-VQE** adaptive ansatz construction
- Excited-state methods:

  - **LR-VQE**
  - **EOM-VQE**
  - **QSE**
  - **EOM-QSE**
  - **SSVQE**
  - **VQD**

- Geometry scans and mapping comparisons
- Optional noise models
- Deterministic caching and reproducible run signatures

### Minimal example

```python
from vqe.core import run_vqe

res = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", steps=50)
print(res["energy"])
```

Excited-state workflows and CLI usage are documented in [USAGE.md](USAGE.md).

---

## QPE Package

The `qpe` module implements **Quantum Phase Estimation** using the same
molecular Hamiltonians as the VQE stack.

Features:

- noiseless and noisy QPE
- Trotterized time evolution
- inverse quantum Fourier transform
- cached runs and reproducible plots

Example:

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, n_qubits, hf_state = build_hamiltonian("H2")

res = run_qpe(hamiltonian=H, hf_state=hf_state, n_ancilla=4)
print(res["energy"])
```

---

## QITE / VarQITE Package

The `qite` module implements **variational imaginary-time evolution**
using the **McLachlan variational principle**.

Capabilities:

- parameter-update imaginary-time evolution (noiseless)
- cached trajectories and convergence diagnostics
- noisy evaluation of converged parameters

Example:

```python
from qite.core import run_qite

res = run_qite(
    molecule="H2",
    ansatz_name="UCCSD",
    steps=50,
    dtau=0.2,
)

print(res["energy"])
```

---

### Capabilities

- **VarQITE (McLachlan)** imaginary-time parameter updates (noiseless, pure-state)
- Cached run records under `results/qite/` and convergence plots under `images/qite/`
- Explicit separation between optimization (`qite run`) and noisy evaluation (`qite eval-noise`)

### Example

```python
from qite.core import run_qite

res = run_qite(
    molecule="H2",
    ansatz_name="UCCSD",
    steps=50,
    dtau=0.2,
    seed=0,
    mapping="jordan_wigner",
    unit="angstrom",
    force=False,
)
print(res["energy"])
```

---

## Command-Line Interface

All algorithms can be executed via CLI entrypoints or via `python -m`.

### VQE

```bash
python -m vqe -m H2 -a UCCSD -o Adam --steps 50
```

### QPE

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

### QITE / VarQITE

```bash
python -m qite run --molecule H2 --steps 50 --dtau 0.2
```

For the full set of CLI workflows (including excited-state methods), see [USAGE.md](USAGE.md).

---

## Testing

Run the full test suite:

```bash
pytest -v
```

---

**Author:** Sid Richards (SidRichardsQuantum)
LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

Licensed under the **MIT License** — see [LICENSE](LICENSE).
