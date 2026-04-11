# Usage Guide

Workflows and APIs for:

- **VQE** — ground state, ADAPT-VQE, excited states
- **QPE** — quantum phase estimation
- **QITE / VarQITE / VarQRTE** — projected variational dynamics
- **common** — Hamiltonians, geometry, plotting, persistence

---

## Documentation Map

| File                        | Purpose                        |
| --------------------------- | ------------------------------ |
| `README.md`                 | overview and quickstart        |
| `USAGE.md`                  | workflows and APIs (this file) |
| `THEORY.md`                 | algorithms and derivations     |
| `more_docs/architecture.md` | system design                  |
| `more_docs/vqe/`            | VQE internals                  |
| `more_docs/qpe/`            | QPE details                    |
| `more_docs/qite/`           | QITE details                   |

---

## Core Execution Model

All stacks share the same pipeline:

```
molecule → geometry → Hamiltonian → algorithm → results + cache
```

Interface contract:

```python
H, n_qubits, hf_state = build_hamiltonian(...)
```

Implications:

- identical physics across all methods
- reproducible cross-algorithm comparisons
- unified caching and output structure

---

## Installation

### PyPI

```bash
pip install vqe-pennylane
```

### From source

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

Verify:

```bash
python -c "import vqe, qpe, qite, common; print('All stacks OK')"
```

---

## General Conventions

Output structure:

```
results/{vqe,qpe,qite}/
images/{vqe,qpe,qite}/
```

Execution behaviour:

- deterministic hashing defines run identity
- cached runs automatically reused
- `--force` bypasses cache
- identical Hamiltonians shared across algorithms

---

## Method Support Summary

| Method    | Family              | VQE reference required | Noise support |
| --------- | ------------------- | ---------------------: | ------------- |
| VQE       | variational         |                     no | yes           |
| ADAPT-VQE | variational         |                     no | yes           |
| LR-VQE    | post-VQE            |                    yes | no            |
| EOM-VQE   | post-VQE            |                    yes | no            |
| QSE       | post-VQE            |                    yes | no            |
| EOM-QSE   | post-VQE            |                    yes | no            |
| SSVQE     | variational excited |                     no | yes           |
| VQD       | variational excited |                     no | yes           |
| QPE       | phase estimation    |                     no | yes           |
| QITE      | imaginary time      |                     no | eval-only     |
| QRTE      | real-time dynamics  |                     no | no            |

---

## Quickstart

```bash
vqe --molecule H2

vqe -m H2 --lr-vqe --lr-k 4

qpe --molecule H2 --ancillas 4

qite run --molecule H2 --steps 50 --dtau 0.2

qite run-qrte --molecule H2 --steps 20 --dt 0.05
```

---

# VQE Workflows

Supports:

- ground-state optimisation
- ADAPT-VQE ansatz growth
- geometry scans
- noise studies
- excited-state solvers

Canonical entrypoint:

```python
from vqe.core import run_vqe
```

---

## Basic VQE

```bash
vqe --molecule H2
```

Defaults:

- ansatz → `UCCSD`
- optimizer → `Adam`
- steps → `50`

Equivalent Python:

```python
from vqe.core import run_vqe

res = run_vqe(molecule="H2")

print(res["energy"])
```

---

## Ansatz selection

```bash
vqe -m H2 -a UCCSD
vqe -m H2 -a RY-CZ
vqe -m H2 -a StronglyEntanglingLayers
```

Guidance:

| Ansatz                   | Typical use                |
| ------------------------ | -------------------------- |
| UCCSD                    | chemistry baseline         |
| RY-CZ                    | lightweight reference      |
| StronglyEntanglingLayers | expressive hardware ansatz |

See:

```
more_docs/vqe/ansatzes.md
```

---

## Optimizer selection

```bash
vqe -m H2 -o Adam
vqe -m H2 -o GradientDescent
vqe -m H2 -o NesterovMomentum
```

Guidance:

| Optimizer        | Behaviour                               |
| ---------------- | --------------------------------------- |
| Adam             | robust default                          |
| GradientDescent  | baseline                                |
| NesterovMomentum | faster convergence on smooth landscapes |

See:

```
more_docs/vqe/optimizers.md
```

---

## Geometry scans

```bash
vqe \
  --scan-geometry H2_BOND \
  --range 0.5 1.5 7
```

Produces:

- energy curves
- cached intermediate Hamiltonians
- reproducible scan identifiers

---

## Noise studies

```bash
vqe \
  -m H2 \
  --multi-seed-noise \
  --noise-type depolarizing
```

Noise options:

- depolarizing
- amplitude damping
- combined channels

---

# Excited-State Methods

Two categories are supported.

---

## Post-VQE methods

Require a converged noiseless VQE reference.

- LR-VQE
- EOM-VQE
- QSE
- EOM-QSE

Example:

```bash
vqe -m H2 --lr-vqe --lr-k 4
```

Typical workflow:

```
VQE → response problem → excitation energies
```

---

## Variational excited states

Optimised directly.

- SSVQE
- VQD

Example:

```bash
vqe -m H2 --vqd --k 3
```

---

# QPE

Quantum Phase Estimation using shared Hamiltonians.

Canonical entrypoint:

```python
from qpe.core import run_qpe
```

---

## Basic QPE

```bash
qpe --molecule H2 --ancillas 4
```

Equivalent Python:

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, _, hf_state = build_hamiltonian("H2")

res = run_qpe(
    hamiltonian=H,
    hf_state=hf_state,
    n_ancilla=4,
)

print(res["energy"])
```

---

## Noise

```bash
qpe \
  --molecule H2 \
  --noisy \
  --p-dep 0.05
```

---

## Time evolution controls

```bash
qpe \
  --molecule H2 \
  --t 2.0 \
  --trotter-steps 4
```

Controls:

- simulation time
- Trotter depth
- precision vs cost tradeoff

---

# QITE / Projected Dynamics

Imaginary-time and real-time projected evolution via McLachlan updates.

Canonical entrypoint:

```python
from qite.core import run_qite, run_qrte
```

---

## Execution modes

| Mode         | Purpose                       |
| ------------ | ----------------------------- |
| `run`        | noiseless parameter evolution |
| `eval-noise` | noisy measurement             |
| `sweep`      | multi-noise statistics        |

---

## Run

```bash
qite run \
  --molecule H2 \
  --steps 50 \
  --dtau 0.2
```

Equivalent Python:

```python
from qite.core import run_qite

res = run_qite(
    molecule="H2",
    steps=50,
    dtau=0.2,
)

print(res["energy"])
```

## Real-time run

```bash
qite run-qrte \
  --molecule H2 \
  --steps 50 \
  --dt 0.05
```

Equivalent Python:

```python
from qite.core import run_qrte

res = run_qrte(
    molecule="H2",
    steps=50,
    dt=0.05,
)

print(res["energy"])
print(res["times"])
```

Use `run_qrte()` after a relevant state has already been identified or prepared.
In practice that usually means:

- prepare a ground state with `run_vqe()` or `run_qite()`
- prepare an excited or approximate spectral reference with the excited-state tools
- evolve that prepared state in time and analyze observables rather than energy minimization

---

## Noisy evaluation

```bash
qite eval-noise \
  --molecule H2 \
  --dep 0.02
```

---

## Noise sweep

```bash
qite eval-noise \
  --molecule H2 \
  --sweep-dep 0,0.02,0.04 \
  --seeds 0,1,2
```

---

## Cache semantics

VarQITE / VarQRTE cache keys include:

- molecule, geometry
- ansatz
- `steps`, `dtau` or `dt`
- solver parameters
- seed
- initialization metadata for prepared-state VarQRTE runs

Ensures:

- reproducible optimisation trajectories
- noise evaluation does not invalidate optimisation cache

---

# Reproducibility

All stacks provide:

- deterministic hashing
- JSON-first outputs
- seed-aware execution
- identical Hamiltonians across algorithms

Force recomputation:

```bash
vqe --force

qpe --force

qite run --force

qite run-qrte --force
```

---

## Testing

```bash
pytest -q
```

---

## Citation

Sid Richards (2026)

Unified Variational and Phase-Estimation Quantum Simulation Suite

---

Author: Sid Richards
License: MIT
