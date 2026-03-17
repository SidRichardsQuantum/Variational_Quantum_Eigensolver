# VQE Ansatzes

This document describes the parameterized quantum circuits (ansatzes) used in the VQE workflows in this repository.

Implemented ansatz names (as accepted by `vqe.engine.build_ansatz`):

- `TwoQubit-RY-CNOT`
- `RY-CZ`
- `Minimal`
- `StronglyEntanglingLayers`
- `UCCSD` (`UCC-SD`)
- `UCC-D` (`UCCD`)
- `UCC-S` (`UCCS`)

These are constructed via:

```python
ansatz_fn, params0 = build_ansatz(...)
```

and used inside QNodes as:

```python
ansatz_fn(params, wires=range(num_wires), ...)
```

---

## Scope and Role in VQE

In VQE, the ansatz defines the trial state

$$
|\psi(\theta)\rangle,
$$

and the optimization problem is

$$
\min_{\theta \in \mathbb{R}^d} E(\theta)
$$

where

$$
E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle
$$

Definitions:

$\theta \in \mathbb{R}^d$: variational parameter vector
$d$: number of parameters (ansatz-dependent)
$H$: qubit Hamiltonian
$|\psi(\theta)\rangle$: quantum state prepared by the ansatz

---

## Common Notation

$n$: number of qubits (`num_wires`)
- wires: ordered list of qubit indices $\{0, \dots, n-1\}$
$\theta_i$: individual variational parameters
- layers: repeated circuit blocks (if applicable)

All ansatz functions have the interface:

```python
ansatz(params, wires, ...)
```

Additional arguments (`symbols`, `coordinates`, `basis`) are only used by chemistry ansatzes (UCC family).

---

## Ansatz Registry

```python
ANSATZES = {
    "TwoQubit-RY-CNOT": ...,
    "RY-CZ": ...,
    "Minimal": ...,
    "StronglyEntanglingLayers": ...,
    "UCCSD": ...,
    "UCC-SD": ...,
    "UCC-D": ...,
    "UCCD": ...,
    "UCC-S": ...,
    "UCCS": ...,
}
```

---

# Toy and Hardware-Efficient Ansatzes

These ansatzes:

- do **not** depend on molecular structure
- ignore `symbols`, `coordinates`, `basis`
- use simple gate patterns

---

## 1. Minimal Ansatz

### Definition

A 2-qubit circuit embedded in a larger register:

$$
|\psi(\theta)\rangle = \mathrm{CNOT}_{0,1}\, R_Y(\theta)_0\, |0\dots 0\rangle.
$$

### Circuit (first two wires only)

```
wire 0: ──RY(θ)────●──
                   │
wire 1: ───────────X──
```

Remaining wires (if any) are untouched.

### Parameters

$d = 1$
- params shape: `(1,)`

### Initialization

- random normal: $\theta \sim \mathcal{N}(0, \text{scale}^2)$

### Interpretation

- simplest nontrivial entangling circuit
- useful for debugging and visualization

---

## 2. TwoQubit-RY-CNOT

### Definition

Applies a 2-qubit motif to each adjacent pair:

For each pair $(i, i+1)$:

$$
R_Y(\theta_i)_i \rightarrow \text{CNOT}_{i,i+1} \rightarrow
R_Y(-\theta_i)_{i+1} \rightarrow \text{CNOT}_{i,i+1}
$$

### Circuit (for 3 wires)

```
wire 0: ──RY(θ₀)──●────────────●──
                  │            │
wire 1: ──────────X──RY(-θ₀)───X──RY(θ₁)──●──
                                          │
wire 2: ──────────────────────────────────X──RY(-θ₁)──●──
                                                      │
                                                     X──
```

### Parameters

$d = n - 1$
- one parameter per adjacent pair

### Initialization

- random normal

### Interpretation

- scalable extension of a 2-qubit motif
- introduces structured entanglement along a chain

---

## 3. RY-CZ Ansatz

### Definition

Single layer:

1. Local rotations
   $$
   \prod_{i=0}^{n-1} R_Y(\theta_i)_i
   $$

2. CZ chain
   $$
   \prod_{i=0}^{n-2} \text{CZ}_{i,i+1}
   $$

### Circuit (n wires)

```
wire 0: ──RY(θ₀)──●──────────────
                  │
wire 1: ──RY(θ₁)──●──●───────────
                     │
wire 2: ──RY(θ₂)─────●──●────────
                        │
...
```

### Parameters

$d = n$

### Initialization

- random normal

### Interpretation

- simple hardware-efficient ansatz
- separates local rotations and entangling layer

---

## 4. StronglyEntanglingLayers

### Definition

Uses PennyLane template:

```python
qml.templates.StronglyEntanglingLayers(params, wires)
```

In this repository:

- **fixed to 1 layer**
- each wire has 3 rotation parameters

### Parameter shape

$$
\text{params.shape} = (1, n, 3)
$$

### Conceptual structure

Each layer consists of:

- arbitrary single-qubit rotations
- dense entangling pattern

### Schematic

```
Layer:
  [Rotations on all qubits]
  [Entangling pattern across qubits]
```

### Initialization

- random normal with scale ~ $\pi$

### Interpretation

- expressive, hardware-efficient ansatz
- not chemistry-informed
- can be harder to optimize

---

# UCC Chemistry Ansatz Family

These ansatzes are **chemistry-inspired** and depend on:

- molecular symbols
- geometry (coordinates)
- basis set

They use excitation operators derived from quantum chemistry.

---

## General Form

All UCC ansatzes follow:

$$
|\psi(\theta)\rangle = e^{T(\theta) - T^\dagger(\theta)} |HF\rangle,
$$

where:

- $|HF\rangle$: Hartree–Fock reference state
- $T(\theta)$: excitation operator

---

## Excitation Construction

From:

```python
singles, doubles = qchem.excitations(electrons, spin_orbitals)
```

- singles: $a_a^\dagger a_i$
- doubles: $a_a^\dagger a_b^\dagger a_j a_i$

These are mapped to qubit operations via PennyLane templates:

- `qml.SingleExcitation`
- `qml.DoubleExcitation`

---

## Reference State Preparation

By default:

```python
qml.BasisState(hf_state, wires=wires)
```

So:

$$
|\psi_0\rangle = |HF\rangle
$$

This is handled **inside the ansatz**, not externally.

Optional overrides:

- `reference_state`
- `prepare_reference=False`

---

## Parameter Ordering (Important)

Parameters are ordered as:

```
[singles..., doubles...]
```

- first all single-excitation parameters
- then all double-excitation parameters

---

## 5. UCC-S (Singles Only)

### Definition

$$
T(\theta) = \sum_i \theta_i T_i^{(1)}
$$

### Parameters

$d = \text{number of single excitations}$

### Behavior

- applies only `qml.SingleExcitation`

---

## 6. UCC-D (Doubles Only)

### Definition

$$
T(\theta) = \sum_j \theta_j T_j^{(2)}
$$

### Parameters

$d = \text{number of double excitations}$

### Behavior

- applies only `qml.DoubleExcitation`

---

## 7. UCCSD (Singles + Doubles)

### Definition

$$
T(\theta) = \sum_i \theta_i T_i^{(1)} + \sum_j \theta_j T_j^{(2)}
$$

### Parameters

$d = \text{singles} + \text{doubles}$

### Behavior

- full chemistry ansatz used in most simulations

---

## Initialization (Critical)

For all UCC ansatzes:

$$
\theta = 0
$$

i.e.

```python
vals = np.zeros(...)
```

### Interpretation

- starts exactly at Hartree–Fock state
- consistent with quantum chemistry workflows
- gradients drive initial updates

---

## Dependence on Molecular System

For UCC ansatzes:

- parameter count depends on:

  - number of electrons
  - number of spin-orbitals
- therefore varies with:

  - molecule
  - basis
  - geometry

---

# Parameter Initialization Summary

| Ansatz                   | Initialization   |
| ------------------------ | ---------------- |
| Minimal                  | small random     |
| TwoQubit-RY-CNOT         | small random     |
| RY-CZ                    | small random     |
| StronglyEntanglingLayers | random ~ $\pi$ |
| UCC family               | **all zeros**    |

---

# When to Use Each Ansatz

### Minimal / Toy ansatzes

Use when:

- debugging
- visualizing energy landscapes
- testing optimizers

---

### RY-CZ / TwoQubit-RY-CNOT

Use when:

- comparing optimizers
- studying trainability
- hardware-oriented experiments

---

### StronglyEntanglingLayers

Use when:

- high expressibility is needed
- not focusing on chemistry accuracy
- exploring general variational behavior

---

### UCC-S / UCC-D / UCCSD

Use when:

- performing **quantum chemistry simulations**
- comparing with classical methods
- studying physically meaningful states

Typical hierarchy:

- UCC-S → simplest
- UCC-D → captures correlation
- UCCSD → most complete (default)

---

# Summary Table

| Ansatz                   | Type               | Parameters       | Uses HF reference | Chemistry-aware |
| ------------------------ | ------------------ | ---------------- | ----------------- | --------------- |
| Minimal                  | Toy                | $1$              | No                | No              |
| TwoQubit-RY-CNOT         | Toy / scalable     | $n-1$            | No                | No              |
| RY-CZ                    | Hardware-efficient | $n$              | No                | No              |
| StronglyEntanglingLayers | Hardware-efficient | $3n$             | No                | No              |
| UCC-S                    | Chemistry          | system-dependent | Yes               | Yes             |
| UCC-D                    | Chemistry          | system-dependent | Yes               | Yes             |
| UCCSD                    | Chemistry          | system-dependent | Yes               | Yes             |

---

# See Also

- [`THEORY.md`](../../THEORY.md) — conceptual overview of ansatz families
- [`docs/vqe/optimizers.md`](optimizers.md) — optimizer definitions
- `vqe/ansatz.py` — implementation details
- `vqe/engine.py` — ansatz construction and integration into QNodes
- `USAGE.md` — CLI and API usage examples
