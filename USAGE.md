# Usage Guide

This document covers **CLI workflows** and **Python APIs** for:

- **VQE** — ground-state, ADAPT-VQE, excited states
- **QPE** — Quantum Phase Estimation
- **QITE / VarQITE** — imaginary-time evolution
- **common** — shared Hamiltonian, geometry, plotting, persistence

---

## Documentation Map

| File | Purpose |
|------|--------|
| `README.md` | Overview and quickstart |
| `USAGE.md` | Workflows and APIs (this file) |
| `THEORY.md` | Algorithms and derivations |
| `more_docs/architecture.md` | System design |
| `more_docs/vqe/` | VQE internals |
| `more_docs/qpe/` | QPE details |
| `more_docs/qite/` | QITE details |

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

- results → `results/{vqe,qpe,qite}/`
- plots → `images/{vqe,qpe,qite}/`
- deterministic hashing defines run identity
- `--force` bypasses cache

---

## Method Support Summary

| Method    | Family              | VQE reference required | Noise     |
| --------- | ------------------- | ---------------------: | --------- |
| VQE       | Variational         |                     No | Yes       |
| ADAPT-VQE | Variational         |                     No | Yes       |
| LR-VQE    | Post-VQE            |                    Yes | No        |
| EOM-VQE   | Post-VQE            |                    Yes | No        |
| QSE       | Post-VQE            |                    Yes | No        |
| EOM-QSE   | Post-VQE            |                    Yes | No        |
| SSVQE     | Variational excited |                     No | Yes       |
| VQD       | Variational excited |                     No | Yes       |
| QPE       | Phase estimation    |                     No | Yes       |
| QITE      | Imaginary time      |                     No | Eval-only |

---

## Quickstart

```bash
vqe --molecule H2
vqe -m H2 --lr-vqe --lr-k 4
qpe --molecule H2 --ancillas 4
qite run --molecule H2 --steps 50 --dtau 0.2
```

---

# VQE Workflows

Supports:

- ground-state VQE
- ADAPT-VQE
- geometry scans
- noise studies
- excited states (post-VQE + variational)

---

## Basic VQE

```bash
vqe --molecule H2
```

Defaults:

- ansatz: `UCCSD`
- optimizer: `Adam`
- steps: `50`

---

## Ansatz selection

```bash
vqe -m H2 -a UCCSD -o Adam
vqe -m H2 -a RY-CZ -o GradientDescent
```

Guidance:

- **UCCSD** → chemistry default
- **RY-CZ** → lightweight baseline
- **StronglyEntanglingLayers** → expressive but harder to train

See: `more_docs/vqe/ansatzes.md`

---

## Optimizer selection

- **Adam** → default, robust
- **GradientDescent** → baseline
- **Momentum/Nesterov** → faster on smooth landscapes

See: `more_docs/vqe/optimizers.md`

---

## Geometry scans

```bash
vqe --scan-geometry H2_BOND --range 0.5 1.5 7
```

---

## Noise studies

```bash
vqe -m H2 --multi-seed-noise --noise-type depolarizing
```

---

## Python API

```python
from vqe.core import run_vqe

res = run_vqe(molecule="H2")
print(res["energy"])
```

---

# Excited States

## Post-VQE

- LR-VQE
- EOM-VQE
- QSE
- EOM-QSE

Require:

- converged **noiseless** VQE reference

---

## Variational

- SSVQE
- VQD

Do not require reference states.

---

(Sections below unchanged except formatting tightened)

---

# QPE

QPE uses the same Hamiltonian pipeline with **non-variational phase estimation**.

---

## Basic QPE

```bash
qpe --molecule H2 --ancillas 4
```

---

## Noisy QPE

```bash
qpe --molecule H2 --noisy --p-dep 0.05
```

---

## Time evolution control

```bash
qpe --molecule H2 --t 2.0 --trotter-steps 4
```

---

## Python API

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, _, hf_state = build_hamiltonian("H2")

res = run_qpe(hamiltonian=H, hf_state=hf_state)
print(res["energy"])
```

---

# QITE / VarQITE

VarQITE approximates imaginary-time evolution via McLachlan updates.

---

## Execution modes

| Mode         | Purpose                       |
| ------------ | ----------------------------- |
| `run`        | noiseless parameter evolution |
| `eval-noise` | noisy measurement             |

---

## Run

```bash
qite run --molecule H2 --steps 50 --dtau 0.2
```

---

## Noisy evaluation

```bash
qite eval-noise --molecule H2 --dep 0.02
```

---

## Sweep

```bash
qite eval-noise \
  --molecule H2 \
  --sweep-dep 0,0.02,0.04 \
  --seeds 0,1,2
```

---

## Python API

```python
from qite.core import run_qite

res = run_qite(molecule="H2", steps=50, dtau=0.2)
print(res["energy"])
```

---

## Caching Semantics

VarQITE cache keys include:

- molecule, geometry, mapping
- ansatz, seed
- `steps`, `dtau`
- solver settings (`fd_eps`, `reg`, etc.)

Ensures:

- numerically consistent reuse
- noise evaluation does not affect optimization cache

---

# Reproducibility

All stacks guarantee:

- deterministic hashing
- JSON-first outputs
- seed-aware execution

Force recompute:

```bash
vqe --force
qpe --force
qite run --force
```

---

## Testing

```bash
pytest -q
```

---

## Citation

> Sid Richards (2026). *Unified Variational and Phase-Estimation Quantum Simulation Suite.*

---

**Author:** Sid Richards
Licensed under the MIT License.
