# Noise Model

## Scope

This document describes how noise is modeled and applied in this repository, including:

- supported noise channels
- how noise is injected into circuits
- the dual interface (legacy vs extensible)
- effects on different algorithm families

Noise is implemented centrally in `vqe.engine` and is used by VQE and related workflows.

---

## Overview

Noise is applied **after the ansatz circuit** and before measurement.

Conceptually:

```

ansatz circuit
↓
apply noise channels
↓
measure observables

```

This models imperfect gate execution and environmental decoherence.

---

## Supported Noise Channels

Two built-in noise channels are supported:

### Depolarizing Noise

With probability $p_{\mathrm{dep}}$, the state is replaced by a uniformly mixed Pauli error:

$$
\mathcal{E}_{\mathrm{dep}}(\rho)
=
(1 - p_{\mathrm{dep}})\rho
+
\frac{p_{\mathrm{dep}}}{3}
\left(
X\rho X + Y\rho Y + Z\rho Z
\right)
$$

Properties:

- symmetric noise
- destroys coherence and entanglement
- commonly used as a generic error model

---

### Amplitude Damping

Models relaxation toward $|0\rangle$:

$$
\mathcal{E}_{\mathrm{amp}}(\rho)
=
E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger
$$

with:

$$
E_0 =
\begin{pmatrix}
1 & 0 \\
0 & \sqrt{1-p_{\mathrm{amp}}}
\end{pmatrix},
\quad
E_1 =
\begin{pmatrix}
0 & \sqrt{p_{\mathrm{amp}}} \\
0 & 0
\end{pmatrix}
$$

Properties:

- asymmetric noise
- drives population toward ground state
- physically motivated (energy relaxation)

---

## Noise Interfaces

This repository supports **two noise interfaces**.

---

### 1. Legacy interface (CLI-friendly)

Controlled via flags:

- `--noisy`
- `--depolarizing-prob`
- `--amplitude-damping-prob`

Example:

```bash
vqe -m H2 --noisy --depolarizing-prob 0.02
```

Behaviour:

* noise applied independently to each qubit
* both channels can be combined
* ignored if `--noisy` is not set

---

### 2. Extensible noise model

A user-defined callable:

```python
noise_model(wires: list[int]) -> None
```

Example:

```python
def my_noise(wires):
    for w in wires:
        qml.DepolarizingChannel(0.01, wires=w)
        qml.PhaseDamping(0.02, wires=w)
```

Passed into the engine:

```python
apply_optional_noise(..., noise_model=my_noise)
```

Behaviour:

* applied **after the ansatz**
* executed only if `noisy=True`
* can include arbitrary PennyLane channels

---

### Combined behaviour

If both interfaces are used:

* legacy depolarizing / amplitude damping are applied
* then `noise_model` is applied

This ensures:

* backward compatibility
* full extensibility

---

## Implementation Details

Noise is applied via:

```python
apply_optional_noise(...)
```

### Key properties

* no-op if `noisy=False`
* applied to all wires:

  ```python
  wires = list(range(num_wires))
  ```
* applied **inside QNodes**, after ansatz construction

---

## Device Selection

Noise requires a mixed-state simulator.

### Devices used

| Mode      | Device          |
| --------- | --------------- |
| Noiseless | `default.qubit` |
| Noisy     | `default.mixed` |

Selected via:

```python
make_device(num_wires, noisy=True/False)
```

Implication:

> Noise simulation uses density matrices, not statevectors.

---

## Effect on Differentiation

Differentiation method depends on noise:

| Setting   | Method            |
| --------- | ----------------- |
| Noiseless | `parameter-shift` |
| Noisy     | `finite-diff`     |

Reason:

* parameter-shift is not generally valid for noisy channels
* finite-difference is used as a fallback

---

## Algorithm Support

Noise affects different methods differently.

### Fully supported

* VQE
* ADAPT-VQE
* SSVQE
* VQD
* QPE (via noisy evolution)

---

### Partially supported

* QITE / VarQITE

  * optimization: noiseless
  * evaluation: noisy (post-processing)

---

### Not supported (noiseless-only)

* LR-VQE
* EOM-VQE
* QSE
* EOM-QSE

Reason:

* require statevector access
* rely on exact overlaps or tangent vectors

---

## Overlap Handling Under Noise

For methods like VQD:

* noiseless:
  [
  |\langle \psi_i | \psi_j \rangle|^2
  ]

* noisy:
  [
  \mathrm{Tr}(\rho_i \rho_j)
  ]

This enables excited-state workflows under noise.

---

## Practical Effects

Noise impacts:

### 1. Energy estimation

* increases variance
* introduces bias
* reduces achievable accuracy

---

### 2. Optimization

* flattens gradients
* introduces stochastic behaviour
* may slow convergence

---

### 3. Circuit expressibility

* reduces effective expressibility
* limits reachable states

---

### 4. Excited-state methods

* post-VQE methods typically break under noise
* variational methods remain usable

---

## Practical Guidance

* use small noise levels first:

  * e.g. `0.01 – 0.05`
* compare against noiseless baseline
* reduce stepsize when noise is high
* use multi-seed runs for statistical analysis:

  ```bash
  vqe --multi-seed-noise
  ```

---

## Limitations

* noise is applied **after the full ansatz**, not per gate
* no time-dependent noise modeling
* no hardware-specific calibration
* no error mitigation techniques included

---

## Summary

| Feature            | Status                  |
| ------------------ | ----------------------- |
| Depolarizing noise | Supported               |
| Amplitude damping  | Supported               |
| Custom noise model | Supported               |
| Mixed-state sim    | Yes (`default.mixed`)   |
| Gradient method    | Finite-diff under noise |
| Post-VQE methods   | Noiseless-only          |

---

## Key Takeaway

Noise is treated as a **modular, post-ansatz layer**:

* simple to use via CLI
* extensible via custom models
* integrated consistently across VQE workflows

This design enables controlled studies of noise effects without modifying algorithm implementations.
