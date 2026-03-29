# Fermion-to-Qubit Mappings

## Scope

This document describes the fermion-to-qubit mappings supported in this repository, their conceptual differences, and their practical impact on VQE, QPE, and related workflows.

Mappings are applied during Hamiltonian construction via `common.hamiltonian.build_hamiltonian(...)` and affect all downstream algorithms.

---

## Overview

Electronic-structure Hamiltonians are naturally expressed in terms of **fermionic creation and annihilation operators**. Quantum circuits, however, operate on **qubits**.

A fermion-to-qubit mapping transforms the fermionic Hamiltonian into an equivalent qubit operator:

$$
H_{\text{fermionic}} \;\longrightarrow\; H_{\text{qubit}}.
$$

This transformation preserves the spectrum but changes the **operator structure**, which directly impacts:

- Pauli string length and locality
- circuit depth
- measurement cost
- optimization behaviour

---

## Supported Mappings

This repository supports the following mappings (via PennyLane-qchem):

- `jordan_wigner` (default)
- `bravyi_kitaev`
- `parity`

Mapping selection is exposed through:

- CLI: `--mapping`
- Python API: `mapping="..."`

Example:

```bash
vqe -m H2 --mapping bravyi_kitaev
```

```python
from common.hamiltonian import build_hamiltonian

H, n_qubits, hf = build_hamiltonian("H2", mapping="parity")
```

---

## Jordan–Wigner (JW)

### Idea

Maps fermionic modes directly to qubits using occupation-number encoding.

Each fermionic operator is represented using strings of Pauli operators with **Z-chains** enforcing fermionic anti-commutation.

### Structure

Creation operator (schematically):

$$
a^\dagger_j ;\sim; X_j \prod_{k<j} Z_k
$$

### Properties

- simple and explicit mapping
- direct correspondence between orbitals and qubits
- produces **long Pauli strings** (length grows with system size)

### Practical impact

- deeper circuits due to long operator strings
- higher measurement cost
- very stable and widely used baseline

### Usage in this repository

- **default mapping**
- used as the baseline for comparisons
- always available via PennyLane backends

---

## Bravyi–Kitaev (BK)

### Idea

Encodes both **occupation** and **parity** information in a balanced way using a binary tree structure.

### Properties

- reduces average Pauli string length relative to JW
- logarithmic scaling of parity information
- more complex encoding than JW

### Practical impact

- shorter Pauli strings on average
- can reduce circuit depth
- may improve optimization stability in some cases

### Usage in this repository

- available via `mapping="bravyi_kitaev"`
- useful for comparing operator locality effects
- backend support may vary depending on PennyLane/qchem version

---

## Parity Mapping

### Idea

Encodes cumulative parity rather than direct occupation.

Closely related to Jordan–Wigner but reorganizes information to expose symmetry structure.

### Properties

- parity-based encoding
- can lead to more structured Hamiltonians
- often used in conjunction with symmetry reduction (not applied here)

### Practical impact

- may reduce effective operator complexity
- can change optimization landscape
- behaviour depends strongly on molecule and ansatz

### Usage in this repository

- available via `mapping="parity"`
- no explicit symmetry tapering is applied in this codebase
- primarily used for comparative studies

---

## Implementation Details (Repository-Specific)

### Mapping entrypoint

Mappings are applied in:

```python
common.hamiltonian.build_molecular_hamiltonian(...)
```

via:

```python
qml.qchem.molecular_hamiltonian(..., mapping=mapping)
```

### Default behaviour

- default mapping: `jordan_wigner`
- mapping string is normalized to lowercase
- passed through to PennyLane-qchem

### Backend fallback behaviour

If the installed PennyLane/qchem backend:

- does **not support the `mapping` argument**, or
- fails during Hamiltonian construction

the code will:

1. retry **without the mapping argument**
2. optionally retry using `method="openfermion"`

Implication:

> Mapping selection is **best-effort**, and actual behaviour may depend on the installed backend. If mapping-specific construction is unsupported, the Hamiltonian falls back to the backend default (typically Jordan–Wigner).

### No symmetry reduction

This repository:

- **does not perform qubit tapering**
- **does not exploit Z₂ symmetries**

Therefore:

- qubit counts are determined directly by the mapping + basis
- differences between mappings are limited to operator structure, not qubit reduction

---

## Effect on VQE and Other Algorithms

Mappings influence several aspects of algorithm performance:

### 1. Circuit depth

- longer Pauli strings → deeper circuits
- JW typically deepest
- BK / parity can be shallower

### 2. Measurement cost

- more non-local terms → more measurements
- affects shot-based simulations (less critical in analytic mode)

### 3. Optimization landscape

Mappings change:

- gradient magnitudes
- curvature of the loss surface
- presence of flat regions or oscillations

This can affect:

- convergence speed
- optimizer sensitivity
- stability of excited-state methods

### 4. Excited-state methods

For:

- QSE / EOM-QSE
- LR-VQE / EOM-VQE

mapping affects:

- operator matrices
- conditioning of overlap matrices
- numerical stability of eigenvalue problems

---

## Summary

| Mapping       | Locality | Pauli string length | Complexity | Typical role           |
| ------------- | -------- | ------------------- | ---------- | ---------------------- |
| Jordan–Wigner | Low      | Long                | Simple     | Default baseline       |
| Bravyi–Kitaev | Medium   | Medium              | Moderate   | Improved locality      |
| Parity        | Medium   | Medium              | Moderate   | Structural comparisons |

---

## Practical Guidance

- use **Jordan–Wigner** for:

  - baseline experiments
  - debugging
  - reproducibility across environments

- try **Bravyi–Kitaev** for:

  - reducing operator length
  - improving circuit structure

- try **Parity** for:

  - comparative studies
  - exploring different optimization behaviour

- if results appear identical across mappings:

  - your backend may have fallen back to the default mapping

---

## References

- Jordan and Wigner, *Über das Paulische Äquivalenzverbot*
- Bravyi and Kitaev, *Fermionic Quantum Computation*
- Seeley et al., *The Bravyi–Kitaev transformation for quantum computation of electronic structure*
