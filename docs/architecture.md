# Architecture Overview

## Scope

This document describes the high-level architecture of the repository, including:

- module structure and responsibilities
- data and control flow across stacks
- shared infrastructure (Hamiltonians, devices, caching)
- design principles

The goal is to provide a clear mental model of how the VQE, QPE, and QITE stacks interact.

---

## High-Level Structure

The repository is organized into four primary packages:

```

common/   → shared chemistry, geometry, Hamiltonians, utilities
vqe/      → variational solvers and excited-state methods
qpe/      → Quantum Phase Estimation workflows
qite/     → variational imaginary-time evolution

```

### Responsibilities

| Package  | Role |
|----------|------|
| `common` | Single source of truth for physical systems and utilities |
| `vqe`    | Variational algorithms and excited-state methods |
| `qpe`    | Phase estimation and time-evolution workflows |
| `qite`   | Imaginary-time evolution (VarQITE) |
  
---

## Core Design Principle

> **All algorithm families operate on the same Hamiltonian and molecular definitions.**

This ensures:

- physically consistent comparisons
- no hidden differences between methods
- reproducible cross-algorithm results

---

## Data Flow

### End-to-End Workflow

```

molecule / geometry
↓
common.geometry
↓
common.hamiltonian
↓
(H, n_qubits, hf_state)
↓
algorithm (VQE / QPE / QITE)
↓
results + plots + cache

```

---

## The `common` Layer

### Purpose

The `common` package centralizes all physics and shared utilities.

### Key components

#### 1. Molecules

```

common/molecules.py

```

- defines supported molecules (H2, LiH, H2O, H3+, …)
- stores:
  - symbols
  - coordinates
  - charge
  - basis

---

#### 2. Geometry

```

common/geometry.py

```

- generates parametric geometries:
  - bond scans
  - angle scans
- used consistently across all algorithms

---

#### 3. Hamiltonian construction

```

common/hamiltonian.py

```

Single entrypoint:

```python
H, n_qubits, hf_state = build_hamiltonian(...)
```

Responsibilities:

* construct qubit Hamiltonian
* apply mapping (JW / BK / parity)
* provide Hartree–Fock reference state
* handle backend fallbacks (PennyLane / OpenFermion)

This is the **core shared interface** across all stacks.

---

#### 4. Utilities

```
common/
├── plotting.py
├── persist.py
```

* deterministic caching (JSON records)
* plot generation with consistent naming
* stable hashing of configurations

---

## The `vqe` Stack

### Purpose

Implements:

* ground-state VQE
* ADAPT-VQE
* excited-state methods (QSE, LR-VQE, etc.)

---

### Internal Structure

#### 1. Engine layer

```
vqe/engine.py
```

Responsibilities:

* device creation (`default.qubit` / `default.mixed`)
* ansatz construction
* optimizer creation
* QNode construction:

  * energy
  * state
  * overlaps
* noise application

This is the **core execution layer**.

---

#### 2. Algorithm layer

```
vqe/
├── core.py       (ground-state VQE)
├── adapt.py      (ADAPT-VQE)
├── lr_vqe.py
├── eom_vqe.py
├── qse.py
├── eom_qse.py
├── ssvqe.py
├── vqd.py
```

Each module:

* builds on the engine
* implements its own objective or eigenproblem
* returns structured results (JSON-compatible)

---

#### 3. Ansatz + optimizer

```
vqe/ansatz.py
vqe/optimizer.py
```

* ansatz factory + parameter initialization
* optimizer factory (PennyLane wrappers)

---

## The `qpe` Stack

### Purpose

Implements Quantum Phase Estimation using the same Hamiltonians.

### Flow

```
build_hamiltonian → prepare HF state → controlled evolution → phase estimation
```

Key characteristics:

* uses Trotterized time evolution
* supports noisy and noiseless simulation
* independent of variational optimization

---

## The `qite` Stack

### Purpose

Implements variational imaginary-time evolution (VarQITE).

### Design

Split into two phases:

1. **Parameter evolution (noiseless)**
2. **Noisy evaluation (post-processing)**

This avoids:

* instability from noisy linear solves
* contamination of cached parameter trajectories

---

## Device and Differentiation Strategy

### Device selection

| Mode      | Device          |
| --------- | --------------- |
| Noiseless | `default.qubit` |
| Noisy     | `default.mixed` |

---

### Differentiation

| Setting   | Method          |
| --------- | --------------- |
| Noiseless | parameter-shift |
| Noisy     | finite-diff     |

Handled automatically in the engine.

---

## Noise Integration

Noise is applied in the engine:

```
ansatz → noise → measurement
```

Features:

* legacy CLI interface
* extensible `noise_model(wires)`
* consistent across all VQE-family methods

---

## Caching and Reproducibility

### Design

All runs are:

* hashed deterministically
* stored as JSON records
* reproducible via configuration

### Key properties

* identical configs → identical cache keys
* parameter rounding for stability
* separation of:

  * compute
  * visualization

---

## CLI vs Python API

All stacks support both:

### CLI

```bash
vqe --molecule H2
qpe --molecule H2
qite run --molecule H2
```

### Python

```python
from vqe.core import run_vqe
res = run_vqe(...)
```

Both interfaces share the same underlying implementation.

---

## Design Patterns

### 1. Single source of truth

* Hamiltonians built once (`common`)
* reused everywhere

---

### 2. Thin algorithm layers

* algorithms do not rebuild infrastructure
* rely on engine + common modules

---

### 3. Separation of concerns

| Concern       | Location     |
| ------------- | ------------ |
| Physics       | `common`     |
| Execution     | `vqe.engine` |
| Algorithms    | `vqe/*.py`   |
| I/O + caching | `common`     |

---

### 4. Backwards compatibility

* legacy interfaces preserved (noise, mappings)
* graceful fallbacks (Hamiltonian construction)

---

### 5. Deterministic outputs

* stable hashing
* consistent directory structure
* JSON-first results

---

## Extending the System

### Add a new ansatz

* implement in `vqe/ansatz.py`
* ensure compatible signature

---

### Add a new optimizer

* register in `vqe/optimizer.py`

---

### Add a new algorithm

* create new module in `vqe/`
* use:

  * `build_hamiltonian`
  * engine QNodes
* return structured result dict

---

## Summary

This repository is structured around a **shared physical layer** (`common`) and multiple **algorithm stacks** (`vqe`, `qpe`, `qite`) built on top.

Key strengths:

* unified Hamiltonian construction
* modular execution engine
* consistent noise and device handling
* reproducible outputs
* extensible architecture

---

## Key Takeaway

> The architecture separates **physics**, **execution**, and **algorithms**, allowing each component to evolve independently while maintaining consistency across the entire system.
