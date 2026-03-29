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

A **modular quantum simulation framework** built on **PennyLane**, combining:

- variational eigensolvers (ground and excited states)
- quantum phase estimation (QPE)
- variational imaginary-time evolution (VarQITE)

The project provides a **reproducible research environment** with:

- unified molecule and Hamiltonian infrastructure
- deterministic caching via run signatures
- shared plotting and output conventions
- CLI workflows and Python APIs
- curated notebooks for benchmarking and demonstrations

---

## Implemented Algorithms

### Variational methods

- **VQE** — ground-state eigensolver
- **ADAPT-VQE** — adaptive ansatz growth

### Post-VQE excited-state methods

- **LR-VQE**
- **EOM-VQE**
- **QSE**
- **EOM-QSE**

### Variational excited-state solvers

- **SSVQE**
- **VQD**

### Non-variational algorithms

- **QPE**
- **QITE / VarQITE**

---

## Documentation

Documentation is structured in **two layers**.

### Core (user-facing)

| File                              | Purpose                    |
| --------------------------------- | -------------------------- |
| **README.md**                     | overview and quickstart    |
| **USAGE.md**                      | CLI and Python workflows   |
| **THEORY.md**                     | algorithms and derivations |
| **notebooks/README_notebooks.md** | notebook guide             |

Start here:

- 📘 [`THEORY.md`](THEORY.md)
- ⚙️ [`USAGE.md`](USAGE.md)
- 📓 [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md)

---

### Extended documentation (`more_docs/`)

Deeper technical and architectural material.

| Path                        | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| `more_docs/architecture.md` | system design and module interactions    |
| `more_docs/vqe/`            | VQE workflows and implementation details |
| `more_docs/qpe/`            | time evolution and phase estimation      |
| `more_docs/qite/`           | VarQITE derivations and internals        |

Intended for:

- contributors
- advanced users
- algorithm deep dives beyond THEORY.md

---

## Repository Structure

```
Variational_Quantum_Eigensolver/
├── README.md
├── THEORY.md
├── USAGE.md
├── pyproject.toml
│
├── more_docs/
│   ├── architecture.md
│   ├── vqe/
│   ├── qpe/
│   └── qite/
│
├── vqe/        # variational + excited-state solvers
├── qpe/        # quantum phase estimation
├── qite/       # imaginary-time evolution
│
├── common/     # shared chemistry + infrastructure
│
├── notebooks/  # demonstrations and benchmarks
├── results/    # cached runs (gitignored)
└── images/     # generated plots (gitignored)
```

### Design principles

- shared chemistry + Hamiltonian layer (`common/`)
- consistent run signatures across all solvers
- deterministic caching for reproducibility
- CLI and Python APIs backed by identical core logic
- minimal configuration friction for common workflows

---

## Installation

### From PyPI

```bash
pip install vqe-pennylane
```

### From source

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

### Verify

```bash
python -c "import vqe, qpe, qite, common; print('Quantum stacks imported successfully')"
```

---

## Core Infrastructure

Shared modules used across all solvers.

| Module                  | Purpose                         |
| ----------------------- | ------------------------------- |
| `common/molecules.py`   | molecule definitions            |
| `common/geometry.py`    | coordinate generation           |
| `common/hamiltonian.py` | Hamiltonian construction        |
| `common/plotting.py`    | plot + filename standardisation |
| `common/persist.py`     | deterministic run hashing       |
| `common/paths.py`       | output directory structure      |

Provides:

- consistent Hamiltonians across algorithms
- reproducible experiment signatures
- standardised output locations

---

## VQE Package

Ground-state and excited-state workflows.

### Capabilities

- VQE, ADAPT-VQE
- LR-VQE, EOM-VQE
- QSE, EOM-QSE
- SSVQE, VQD
- noise support
- geometry scans

### Canonical entrypoint

```python
from vqe.core import run_vqe

res = run_vqe(
    molecule="H2",
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=50,
)

print(res["energy"])
```

---

## QPE Package

Quantum Phase Estimation using shared Hamiltonians.

### Features

- Trotterized time evolution
- inverse QFT
- noisy and noiseless execution
- cached runs

### Canonical entrypoint

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, n_qubits, hf_state = build_hamiltonian("H2")

res = run_qpe(
    hamiltonian=H,
    hf_state=hf_state,
    n_ancilla=4,
)

print(res["energy"])
```

---

## QITE / VarQITE Package

Imaginary-time evolution via the McLachlan variational principle.

### Capabilities

- variational imaginary-time updates
- convergence tracking
- cached trajectories
- optional noisy evaluation

### Canonical entrypoint

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

## Command-Line Interface

### VQE

```bash
python -m vqe -m H2 -a UCCSD -o Adam --steps 50
```

### QPE

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

### QITE

```bash
python -m qite run --molecule H2 --steps 50 --dtau 0.2
```

Full workflows:

⚙️ [`USAGE.md`](USAGE.md)

---

## Reproducibility

All solvers share:

- deterministic configuration hashing
- standardised result storage
- consistent naming conventions
- cached experiment reuse

Outputs are stored under:

```
results/<module>/
images/<module>/<MOLECULE>/
```

---

## Testing

```bash
pytest -v
```

Tests cover:

- core algorithm execution
- Hamiltonian construction
- CLI workflows
- smoke tests for packaged interfaces

---

## Author

**Sid Richards**

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](LICENSE)
