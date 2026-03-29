# Time Evolution in QPE

## Scope

This document describes how time evolution is implemented for Quantum Phase Estimation (QPE) in this repository, including:

- construction of the unitary $U = e^{-iHt}$
- Trotter decomposition
- controlled time evolution
- numerical trade-offs and errors

Time evolution is the **core computational primitive** underlying QPE.

---

## Overview

QPE requires implementing:

$$
U = e^{-iHt}
$$

where:

- $H$ is the molecular Hamiltonian
- $t$ is the evolution time

This unitary encodes the energy eigenvalues as phases.

---

## Hamiltonian Structure

After fermion-to-qubit mapping:

$$
H = \sum_j c_j P_j
$$

where:

- $c_j$ are real coefficients
- $P_j$ are Pauli strings (e.g. $XZIY$)

---

## Exact Evolution (Ideal)

If all terms commute:

$$
e^{-iHt} = \prod_j e^{-i c_j P_j t}
$$

However, in general:

$$
[P_i, P_j] \ne 0
$$

so exact factorization is not possible.

---

## Trotter Decomposition

To approximate evolution, this repository uses **Trotterization**.

### First-order Trotter formula

$$
e^{-iHt}
\approx
\left(
\prod_j e^{-i c_j P_j t / r}
\right)^r
$$

where:

- $r$ = number of Trotter steps

---

## Error Behaviour

Trotter error scales as:

$$
\mathcal{O}\left(\frac{t^2}{r}\right)
$$

Implications:

- increasing `trotter_steps` reduces error
- error grows with total evolution time $t$

---

## Implementation (This Repository)

### Hamiltonian source

Time evolution is built from:

```python
H, n_qubits, hf_state = build_hamiltonian(...)
```

All Pauli terms in (H) are used to construct evolution.

---

### Pauli exponentials

Each term:

[
e^{-i c_j P_j t}
]

is implemented using:

- basis rotations
- controlled rotations
- inverse basis rotations

---

### Controlled evolution

QPE requires:

[
U^{2^k} = e^{-iH t 2^k}
]

This is implemented by:

- scaling evolution time:

  - ( t \rightarrow t \cdot 2^k )
- applying Trotterized evolution

---

## Circuit Structure

For each ancilla qubit:

```
ancilla ──●────────────
           │
system  ───U^(2^k)────
```

This controlled evolution is repeated for each ancilla.

---

## Parameters

### Evolution time

```bash
--t
```

Controls:

- phase resolution
- energy scaling

Trade-offs:

- larger (t):

  - higher precision
  - risk of phase wrapping
- smaller (t):

  - safer range
  - lower resolution

---

### Trotter steps

```bash
--trotter-steps
```

Controls:

- approximation accuracy
- circuit depth

---

### Ancilla count

```bash
--ancillas
```

Controls:

- phase precision
- number of controlled evolutions

---

## Combined Trade-offs

| Parameter     | Effect                       |
| ------------- | ---------------------------- |
| (t)           | ↑ precision, ↑ wrapping risk |
| trotter_steps | ↓ error, ↑ circuit depth     |
| ancillas      | ↑ resolution, ↑ circuit size |

---

## Noise Effects

Under noise:

- each Trotter step introduces decoherence
- deeper circuits amplify errors
- controlled operations are especially sensitive

---

### Practical implication

- fewer Trotter steps → less noise, more approximation error
- more Trotter steps → better approximation, worse noise

---

## Practical Guidance

### Baseline settings

- `t = 1–2`
- `trotter_steps = 2–6`

---

### Improve accuracy

- increase `trotter_steps`
- increase `ancillas`
- ensure good input state

---

### Reduce noise impact

- reduce circuit depth (lower trotter_steps)
- reduce ancilla count
- use smaller (t)

---

## Limitations

- first-order Trotter only
- no higher-order Suzuki formulas
- no adaptive Trotterization
- no qubitization or advanced Hamiltonian simulation

---

## Summary

| Feature               | Status      |
| --------------------- | ----------- |
| Trotter decomposition | First-order |
| Controlled evolution  | Implemented |
| Shared Hamiltonian    | Yes         |
| Noise support         | Yes         |

---

## Key Takeaway

> Time evolution in this repository is implemented using **Trotterized exponentials of Pauli terms**, providing a practical and flexible approximation of (e^{-iHt}) suitable for QPE workflows.

The design prioritizes:

- simplicity
- compatibility with the shared Hamiltonian pipeline
- tunable accuracy via `trotter_steps`
