# Quantum Phase Estimation (QPE)

## Scope

This document describes the QPE implementation in this repository, including:

- the phase–energy relationship
- circuit structure and workflow
- Trotterized time evolution
- precision and resource trade-offs
- noise support and practical considerations

QPE operates on the same Hamiltonian pipeline as VQE and VarQITE.

---

## Overview

Quantum Phase Estimation extracts eigenvalues of a unitary operator.

For quantum chemistry, the unitary is:

$$
U = e^{-iHt}
$$

If:

$$
H|\psi\rangle = E|\psi\rangle
$$

then:

$$
U|\psi\rangle = e^{-iEt}|\psi\rangle = e^{2\pi i \theta}|\psi\rangle
$$

where:

$$
\theta = -\frac{Et}{2\pi}
$$

Thus:

$$
E = -\frac{2\pi \theta}{t}
$$

---

## QPE Circuit Structure

QPE uses two registers:

- **ancilla register** (phase estimation)
- **system register** (state preparation)

### Workflow

```

Ancilla:  |0⟩...|0⟩ ──H───●────●──── IQFT ── measure
│    │
System:   |ψ⟩ ──────────U────U²──────────────

```

Steps:

1. prepare ancillas in superposition
2. prepare system state (e.g. Hartree–Fock)
3. apply controlled powers of $U$
4. apply inverse QFT
5. measure ancillas → phase estimate

---

## Input State

QPE requires an approximate eigenstate:

$$
|\psi\rangle = \sum_k c_k |E_k\rangle
$$

Measurement yields eigenvalue $E_k$ with probability:

$$
|c_k|^2
$$

---

### In this repository

- default input: **Hartree–Fock state**
- optionally:
  - VQE-prepared states (via API workflows)

Implication:

> Accuracy depends strongly on overlap with the true eigenstate.

---

## Time Evolution

### Unitary

$$
U = e^{-iHt}
$$

### Implementation

Exact exponentiation is not available, so this repository uses:

> **Trotterized time evolution**

---

## Trotter Decomposition

If:

$$
H = \sum_j H_j
$$

then:

$$
e^{-iHt} \approx \left(\prod_j e^{-iH_j t / r}\right)^r
$$

where:

- $r$ = number of Trotter steps

---

### Error scaling

- Trotter error decreases with increasing $r$
- circuit depth increases linearly with $r$

---

### Practical trade-off

| Parameter         | Effect                          |
|------------------|----------------------------------|
| `trotter_steps`  | ↓ error, ↑ circuit depth         |
| `t`              | affects phase resolution         |

---

## Precision and Resources

### Number of ancillas

Let:

```bash
--ancillas n
```

Then:

- phase precision: ( \sim 1 / 2^n )
- energy precision:

[
\Delta E \sim \frac{2\pi}{t \cdot 2^n}
]

---

### Trade-offs

| Parameter  | Effect                                   |
| ---------- | ---------------------------------------- |
| ancillas ↑ | higher precision, more qubits            |
| t ↑        | finer resolution, risk of phase wrapping |
| trotter ↑  | better accuracy, deeper circuits         |

---

## Phase Wrapping

Because phase is modulo 1:

[
\theta \in [0,1)
]

Energy must satisfy:

[
E \in \left[-\frac{\pi}{t}, \frac{\pi}{t}\right]
]

---

### Implication

- large (t) → higher precision
- but:

  - risk of ambiguity (wrapping)
  - requires careful parameter choice

---

## Noise Support

QPE supports noisy simulation:

```bash
qpe --noisy --p-dep 0.05 --p-amp 0.02
```

---

### Noise effects

- decoherence reduces phase sharpness
- measurement distributions broaden
- peak identification becomes harder

---

### Device

| Mode      | Device          |
| --------- | --------------- |
| Noiseless | `default.qubit` |
| Noisy     | `default.mixed` |

---

## Measurement and Output

Measurement returns:

- bitstring from ancilla register
- interpreted as phase estimate

Converted to:

[
E = -\frac{2\pi \theta}{t}
]

---

## Implementation Details (This Repository)

### Shared Hamiltonian pipeline

QPE uses:

```python
H, n_qubits, hf_state = build_hamiltonian(...)
```

Same as VQE and QITE.

---

### Controlled evolution

- constructed via Trotterized exponentials
- applied as:

  - (U^{2^k}) for each ancilla

---

### CLI interface

Example:

```bash
qpe --molecule H2 --ancillas 4 --t 2.0 --trotter-steps 4
```

---

### Python API

```python
from qpe.core import run_qpe

res = run_qpe(
    hamiltonian=H,
    hf_state=hf_state,
    n_ancilla=4,
)
```

---

## Practical Guidance

### Choosing parameters

- start with:

  - `ancillas = 4–6`
  - `t = 1–2`
  - `trotter_steps = 2–6`

---

### Improving accuracy

- increase ancillas → better precision
- increase trotter_steps → lower Trotter error
- improve input state (e.g. VQE-prepared)

---

### Debugging

If results look incorrect:

- check phase wrapping (t too large)
- check input state overlap
- increase trotter steps
- reduce noise

---

## Comparison with VQE

| Feature        | QPE                       | VQE                      |
| -------------- | ------------------------- | ------------------------ |
| Type           | Phase estimation          | Variational optimization |
| Accuracy       | High (in principle exact) | Ansatz-limited           |
| Resources      | High (ancillas, depth)    | Lower                    |
| Noise          | Sensitive                 | More robust              |
| Input required | Good eigenstate           | None                     |

---

## Limitations

- requires good initial state
- deep circuits due to controlled evolution
- Trotter approximation error
- sensitive to noise
- scaling limited for large systems

---

## Summary

| Feature            | Status       |
| ------------------ | ------------ |
| Phase estimation   | Implemented  |
| Trotter evolution  | Implemented  |
| Noise support      | Yes          |
| Shared Hamiltonian | Yes          |
| Precision control  | Via ancillas |

---

## Key Takeaway

> QPE provides a **direct, non-variational route to eigenvalues**, trading increased circuit depth and resource requirements for high-precision energy estimation.

In this repository, it is implemented using:

- a shared Hamiltonian pipeline
- Trotterized time evolution
- flexible precision and noise controls
