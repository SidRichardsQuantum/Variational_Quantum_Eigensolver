# Variational Quantum Imaginary Time Evolution (VarQITE)

## Scope

This document describes the VarQITE implementation in this repository, including:

- the McLachlan variational principle
- the linear-system update rule
- numerical implementation details
- solver and regularization options
- design choices specific to this codebase

---

## Overview

Imaginary-time evolution drives a quantum state toward the ground state:

$$
|\psi(\tau)\rangle \propto e^{-H\tau} |\psi(0)\rangle
$$

Expanding in the energy eigenbasis:

$$
|\psi(0)\rangle = \sum_k c_k |E_k\rangle
$$

$$
e^{-H\tau}|\psi(0)\rangle
=
\sum_k c_k e^{-E_k \tau} |E_k\rangle
$$

Higher-energy components decay faster, leaving the ground state dominant.

---

## Variational Formulation (McLachlan Principle)

Because quantum circuits are unitary, imaginary-time evolution is approximated within a parameterized ansatz:

$$
|\psi(\theta)\rangle
$$

The McLachlan variational principle minimizes:

$$
\left\| \frac{\partial}{\partial \tau}|\psi(\theta)\rangle + H|\psi(\theta)\rangle \right\|
$$

This yields a linear system:

$$
A(\theta)\,\dot{\theta} = -C(\theta)
$$

---

## Matrix Definitions

### Overlap matrix

$$
A_{ij} = \Re \langle \partial_i \psi | \partial_j \psi \rangle
$$

### Force vector

$$
C_i = \Re \langle \partial_i \psi | (H - \langle H \rangle) | \psi \rangle
$$

---

## Parameter Update

A discrete step of size $\Delta\tau$:

$$
\theta \leftarrow \theta + \Delta\tau\,\dot{\theta}
$$

This update is applied iteratively.

---

## Implementation (This Repository)

### Core loop

The update is performed via:

```python
params = qite_step(...)
```

inside `run_qite(...)`.

At each iteration:

1. construct (A) and (C)
2. solve linear system
3. update parameters
4. evaluate energy

---

### Finite-difference derivatives

Tangent vectors are computed via finite differences:

- controlled by `fd_eps`
- used to approximate:

  - ( \partial_i |\psi\rangle )

Trade-off:

- small `fd_eps` → numerical noise
- large `fd_eps` → approximation error

---

## Linear System Solvers

The system:

[
A \dot{\theta} = -C
]

is solved using one of two methods.

---

### 1. Direct solve

```python
solver="solve"
```

- uses linear solve (`np.linalg.solve`)
- fast and accurate if (A) is well-conditioned
- fails if (A) is singular or ill-conditioned

---

### 2. Pseudoinverse

```python
solver="pinv"
```

- uses Moore–Penrose pseudoinverse
- controlled by `pinv_rcond`
- more stable for ill-conditioned systems
- slightly more expensive

---

## Regularization

To stabilize the system, a diagonal regularizer is added:

[
A \rightarrow A + \lambda I
]

controlled by:

```python
reg = 1e-6
```

Effects:

- improves conditioning
- reduces numerical instability
- introduces small bias

---

## Caching and Reproducibility

VarQITE runs are fully cached using:

- deterministic configuration hashing
- JSON run records

Cache keys include:

- molecule and geometry
- ansatz
- mapping
- `dtau`, `steps`
- numerical parameters:

  - `fd_eps`
  - `reg`
  - `solver`
  - `pinv_rcond`

Implication:

> Changing any numerical setting triggers recomputation.

---

## Device and State Requirements

VarQITE requires:

- **pure statevectors**
- exact overlap and derivative access

Therefore:

| Feature       | Support |
| ------------- | ------- |
| Noiseless run | Yes     |
| Noisy run     | No      |

---

## Noise Design (Important)

### Key design choice

> **VarQITE optimization is always noiseless.**

Noise is handled separately via:

```
qite eval-noise
```

This performs:

- noisy evaluation of converged parameters
- without re-running optimization

---

### Why this design?

The McLachlan update requires:

- accurate derivatives
- stable linear solves

Noise would:

- corrupt (A) and (C)
- destabilize the linear system
- break convergence

---

### Resulting workflow

```
Step 1: run_qite (noiseless)
    → converged parameters

Step 2: eval-noise
    → noisy energy evaluation
```

---

## Energy Convergence

Energy is tracked at each step:

```python
energies = [...]
```

Typical behaviour:

- monotonic decrease (approximate)
- eventual convergence plateau
- sensitivity to:

  - ansatz
  - dtau
  - conditioning of A

---

## Hyperparameters

### Time step

```python
dtau
```

- controls update size
- too large → instability
- too small → slow convergence

Typical values:

- `0.05 – 0.3`

---

### Steps

```python
steps
```

- number of iterations
- determines total imaginary time

---

### Finite difference

```python
fd_eps
```

- derivative resolution
- typical: `1e-3`

---

### Regularization

```python
reg
```

- stabilizes linear system
- typical: `1e-6`

---

### Solver

```python
solver = "solve" | "pinv"
```

Guidelines:

- use `"solve"` if stable
- use `"pinv"` if conditioning issues appear

---

## Practical Guidance

- start with:

  - `dtau = 0.1 – 0.2`
  - `solver="solve"`
- if unstable:

  - increase `reg`
  - switch to `"pinv"`
- monitor energy convergence
- compare with VQE baseline

---

## Limitations

- no noisy optimization
- finite-difference overhead
- scaling limited by parameter count
- sensitive to ill-conditioning of (A)

---

## Summary

| Feature            | Status                   |
| ------------------ | ------------------------ |
| McLachlan update   | Implemented              |
| Finite differences | Yes (`fd_eps`)           |
| Regularization     | Yes (`reg`)              |
| Solver control     | Yes (`solve` / `pinv`)   |
| Noisy optimization | Not supported            |
| Noisy evaluation   | Supported (CLI)          |
| Caching            | Deterministic + complete |

---

## Key Takeaway

> VarQITE in this repository is implemented as a **stable, reproducible linear-system solver with explicit numerical control**, and a deliberate separation between **optimization (noiseless)** and **evaluation (noisy)**.

This design prioritizes:

- numerical stability
- reproducibility
- clear separation of concerns
