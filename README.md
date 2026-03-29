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

A **modular quantum simulation suite** built on **PennyLane**, combining:

- Variational quantum algorithms (VQE + excited states)
- Phase estimation (QPE)
- Imaginary-time evolution (VarQITE)

The project provides a **reproducible research framework** with:

- unified molecule and Hamiltonian infrastructure
- deterministic caching and run signatures
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

Documentation is structured in **two layers**:

### Core (user-facing)

| File | Purpose |
|-----|------|
| **README.md** | Overview and quickstart |
| **USAGE.md** | CLI and Python workflows |
| **THEORY.md** | Algorithms and derivations |
| **notebooks/README_notebooks.md** | Notebook guide |

Start here:

- 📘 [`THEORY.md`](THEORY.md)  
- ⚙️ [`USAGE.md`](USAGE.md)  
- 📓 [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md)

---

### Extended documentation (`more_docs/`)

The `more_docs/` directory contains **deeper technical and architectural material**:

| Path | Purpose |
|------|--------|
| `more_docs/architecture.md` | System design and module interactions |
| `more_docs/vqe/` | Detailed VQE workflows and internals |
| `more_docs/qpe/` | QPE time evolution and phase estimation details |
| `more_docs/qite/` | VarQITE derivations and implementation details |

These are intended for:
- contributors
- advanced users
- algorithm deep-dives beyond THEORY.md

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
├── vqe/        # Variational + excited-state solvers
├── qpe/        # Phase estimation
├── qite/       # Imaginary-time evolution
│
├── common/     # Shared chemistry + infrastructure
│
├── notebooks/  # Demonstrations and benchmarks
├── results/    # Cached runs (gitignored)
└── images/     # Generated plots (gitignored)

```

### Design principles

- Shared chemistry + Hamiltonian layer (`common/`)
- Identical run signatures across all solvers
- Deterministic caching for reproducibility
- CLI and Python APIs backed by the same core logic

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

## Common Core

Shared modules used across all solvers:

| Module                  | Purpose                         |
| ----------------------- | ------------------------------- |
| `common/molecules.py`   | Molecule definitions            |
| `common/geometry.py`    | Coordinate generation           |
| `common/hamiltonian.py` | Hamiltonian construction        |
| `common/plotting.py`    | Plot + filename standardisation |

---

## VQE Package

Implements ground-state and excited-state workflows.

### Capabilities

- VQE, ADAPT-VQE
- LR-VQE, EOM-VQE
- QSE, EOM-QSE
- SSVQE, VQD
- Noise support
- Geometry scans

### Example

```python
from vqe.core import run_vqe

res = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", steps=50)
print(res["energy"])
```

---

## QPE Package

Implements **Quantum Phase Estimation** with shared Hamiltonians.

### Features

- Trotterized time evolution
- Inverse QFT
- Noisy + noiseless execution
- Cached runs

### Example

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, n_qubits, hf_state = build_hamiltonian("H2")

res = run_qpe(hamiltonian=H, hf_state=hf_state, n_ancilla=4)
print(res["energy"])
```

---

## QITE / VarQITE Package

Implements **imaginary-time evolution** via the McLachlan principle.

### Capabilities

- Variational imaginary-time updates
- Convergence tracking
- Cached trajectories
- Optional noisy evaluation

### Example

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

See full workflows in [`USAGE.md`](USAGE.md).

---

## Testing

```bash
pytest -v
```

---

## Author

**Sid Richards**
LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

---

## License

MIT License — see [LICENSE](LICENSE)
