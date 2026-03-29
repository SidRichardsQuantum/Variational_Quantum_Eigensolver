# Theory and Methodology

This document summarizes the main algorithms, physical assumptions, and implementation choices used in this project. It covers:

- molecular systems and shared chemistry infrastructure
- the Variational Quantum Eigensolver (VQE)
- ansatz families and optimizers
- fermion-to-qubit mappings
- excited-state methods
- ADAPT-VQE
- Quantum Phase Estimation (QPE)
- variational imaginary-time evolution (VarQITE)
- noise models

For practical workflows and CLI usage, see [`USAGE.md`](USAGE.md).
For overview and installation, see [`README.md`](README.md).

Detailed implementation references:

- [`more_docs/vqe/ansatzes.md`](more_docs/vqe/ansatzes.md)
- [`more_docs/vqe/optimizers.md`](more_docs/vqe/optimizers.md)

---

# Contents

- [Molecular Systems and Shared Chemistry Layer](#molecular-systems-and-shared-chemistry-layer)
- [Background](#background)

  - [Variational Principle](#variational-principle)
  - [Why qubit mappings are needed](#why-qubit-mappings-are-needed)
  - [Hartree-Fock Reference State](#hartree-fock-reference-state)
  - [From classical chemistry to quantum algorithms](#from-classical-chemistry-to-quantum-algorithms)
- [VQE Overview](#vqe-overview)

  - [Ansatz Families](#ansatz-families)
  - [Optimizers](#optimizers)
  - [Fermion-to-Qubit Mappings](#fermion-to-qubit-mappings)
- [Excited-State Methods](#excited-state-methods)
- [ADAPT-VQE](#adapt-vqe)
- [Quantum Phase Estimation](#quantum-phase-estimation)
- [Quantum Imaginary Time Evolution](#quantum-imaginary-time-evolution)
- [Noise Models](#noise-models)
- [References](#references)

---

# Molecular Systems and Shared Chemistry Layer

All algorithms operate on a unified molecular and Hamiltonian infrastructure to ensure physically consistent comparisons.

| Molecule | Typical uses                                      | Basis  | Approx. mapped qubits |
| -------- | ------------------------------------------------- | ------ | --------------------: |
| H₂       | ansatz studies, optimizer behaviour, QPE examples | STO-3G |                     4 |
| LiH      | bond-length scans                                 | STO-3G |                    12 |
| H₂O      | bond-angle scans                                  | STO-3G |                    14 |
| H₃⁺      | mapping comparisons, ADAPT-VQE, excited states    | STO-3G |                     6 |

Shared assumptions:

- molecular registry → `common/molecules.py`
- geometry generation → `common/geometry.py`
- Hamiltonian construction → `common/hamiltonian.py`
- STO-3G basis used for consistency across examples

Using a shared chemistry layer ensures that differences between VQE, QPE, and VarQITE reflect algorithmic behaviour rather than inconsistent physical inputs.

---

# Background

Quantum chemistry algorithms in this repository approximate eigenvalues and eigenstates of electronic Hamiltonians for small molecular systems.

Workflow:

1. choose molecule and geometry
2. construct electronic Hamiltonian
3. map fermionic operators to qubit operators
4. apply quantum algorithm to estimate spectral properties

Centralizing steps 1–3 ensures consistent physical inputs across algorithm families.

---

## Variational Principle

For normalized states:

$$
E_0 \le \langle \psi | H | \psi \rangle
$$

Minimizing the expectation value over a parameterized state family

$$
|\psi(\theta)\rangle
$$

produces the best approximation to the ground state accessible within the chosen ansatz manifold.

Variational methods rely on:

- differentiable parameterized circuits
- classical optimization loops
- expectation-value estimation

---

## Why qubit mappings are needed

Electronic Hamiltonians are expressed using fermionic creation and annihilation operators. Quantum circuits act on qubits, so the Hamiltonian must be mapped to qubit operators.

Common mappings:

- Jordan–Wigner
- Bravyi–Kitaev
- parity mapping

These mappings preserve physical observables but affect:

- Pauli-string locality
- circuit depth
- measurement grouping
- optimizer conditioning

---

## Hartree-Fock Reference State

Many workflows begin from a Hartree–Fock (HF) reference determinant.

$$
|HF\rangle = |\phi_1 \phi_2 \cdots \phi_n\rangle
$$

In qubit form, this becomes an occupation-number bitstring.

Example:

```
|111100000000⟩
```

Roles of the HF reference:

- initialization for chemistry-inspired ansätze
- reference state for ADAPT-VQE
- approximate eigenstate input for QPE
- starting state for imaginary-time evolution

---

## From classical chemistry to quantum algorithms

Once the qubit Hamiltonian is constructed, algorithm families differ in how they extract spectral information:

| method           | mechanism                               |
| ---------------- | --------------------------------------- |
| VQE              | variational energy minimization         |
| post-VQE methods | linear algebra in reduced manifolds     |
| QPE              | phase extraction from unitary evolution |
| VarQITE          | imaginary-time filtering                |

All share identical Hamiltonians, enabling consistent algorithm comparisons.

---

# VQE Overview

The Variational Quantum Eigensolver couples:

- parameterized quantum circuits
- classical optimization

Workflow:

1. prepare ansatz state
2. measure expectation value
3. update parameters
4. iterate to convergence

```
optimizer → parameters → circuit → expectation → update
```

Performance depends on:

- ansatz expressibility
- optimization landscape
- Hamiltonian structure

---

# Ansatz Families

Ansätze define the accessible variational manifold.

Tradeoffs:

- physical structure
- circuit depth
- parameter count
- trainability

---

## UCCSD

Unitary Coupled Cluster Singles and Doubles:

$$
|\psi(\theta)\rangle
====================

e^{T(\theta)-T^\dagger(\theta)}
|HF\rangle
$$

with

$$
T = T_1 + T_2
$$

Properties:

- chemistry motivated
- interpretable excitation structure
- strong performance for small molecules

---

## Hardware-efficient ansätze

Example: RY–CZ layered circuits.

Motivations:

- shallow depth
- tunable expressibility
- hardware compatibility
- useful for benchmarking optimizer behaviour

---

## Minimal ansätze

Low-parameter circuits used for:

- visualization
- pedagogical examples
- landscape analysis

---

# Optimizers

Optimization minimizes:

$$
E(\theta)
=========

\langle \psi(\theta) | H | \psi(\theta) \rangle
$$

Supported optimizers:

- Adam
- Gradient Descent
- RMSProp
- Adagrad
- Momentum / Nesterov

Differences:

- adaptive learning-rate scaling
- momentum accumulation
- noise robustness

Implementation uses a unified optimizer interface shared across ansätze.

---

# Fermion-to-Qubit Mappings

Mapping choice affects circuit structure and optimization behaviour.

### Jordan–Wigner

- direct occupation encoding
- simple construction
- longer Pauli strings

### Bravyi–Kitaev

- balanced parity/occupation encoding
- shorter average Pauli strings

### Parity mapping

- parity-based encoding
- can expose symmetries
- may reduce circuit depth

---

# Excited-State Methods

Excited states require additional structure beyond ground-state minimization.

Two main approaches are implemented.

---

## Post-VQE linear-algebra methods

Construct reduced eigenproblems around a converged reference state.

Methods:

- QSE
- EOM-QSE
- LR-VQE
- EOM-VQE

These rely on:

- high-quality ground-state reference
- well-conditioned reduced manifolds

Generally noiseless-only due to reliance on statevector information.

---

### QSE

Construct operator-generated subspace:

$$
|\phi_i\rangle = O_i|\psi\rangle
$$

Solve:

$$
Hc = ESc
$$

where:

$$
H_{ij}
======

\langle \psi|O_i^\dagger H O_j|\psi\rangle
$$

$$
S_{ij}
======

\langle \psi|O_i^\dagger O_j|\psi\rangle
$$

---

### EOM-QSE

Commutator-based reduced problem:

$$
A_{ij}
======

\langle \psi|O_i^\dagger[H,O_j]|\psi\rangle
$$

Solve:

$$
Ac = \omega Sc
$$

Typically non-Hermitian.

Positive real-dominant roots selected.

---

### LR-VQE

Tangent-space linear response around converged parameters:

$$
S_{ij}
======

\langle \partial_i\psi|\partial_j\psi\rangle
$$

$$
A_{ij}
======

\langle \partial_i\psi|(H-E_0)|\partial_j\psi\rangle
$$

Solve:

$$
Ac = \omega Sc
$$

Approximate excited energies:

$$
E_k = E_0 + \omega_k
$$

Corresponds to a Tamm–Dancoff style approximation.

---

### EOM-VQE

Full-response tangent-space formulation.

Produces paired roots:

$$
\pm \omega
$$

Positive solutions correspond to excitation energies.

More expressive but more numerically sensitive.

---

## Variational excited states

Solve excited states directly.

### SSVQE

Shared unitary applied to orthogonal inputs:

$$
|\psi_k(\theta)\rangle = U(\theta)|\phi_k\rangle
$$

Minimize:

$$
\sum_k w_k
\langle \psi_k|H|\psi_k\rangle
$$

---

### VQD

Sequential deflation:

$$
L_n =
\langle \psi_n|H|\psi_n\rangle
+
\beta \sum_{k<n}
|\langle \psi_k|\psi_n\rangle|^2
$$

Supports noisy evaluation using density matrices.

---

## Excited-state summary

| method  | category          | noise     |
| ------- | ----------------- | --------- |
| QSE     | operator subspace | noiseless |
| EOM-QSE | operator EOM      | noiseless |
| LR-VQE  | tangent response  | noiseless |
| EOM-VQE | full response     | noiseless |
| SSVQE   | variational       | supported |
| VQD     | deflation         | supported |

---

# ADAPT-VQE

Adaptive ansatz construction.

Instead of fixing the ansatz size, operators are added iteratively.

Ansatz:

$$
U_k(\theta)
===========

\prod_j
e^{\theta_j A_j}
$$

Operators selected by gradient magnitude.

Workflow:

1. optimize current parameters
2. evaluate gradients for operator pool
3. append best operator
4. repeat until convergence

Advantages:

- compact circuits
- interpretable operator growth
- convergence diagnostics

---

# Quantum Phase Estimation

QPE extracts eigenvalues from phase evolution:

$$
U = e^{-iHt}
$$

Eigenstate relation:

$$
U|\psi\rangle
=============

e^{-iEt}|\psi\rangle
$$

Energy recovered via:

$$
E = -\frac{2\pi \theta}{t}
$$

Registers:

- ancilla → phase precision
- system → approximate eigenstate

Tradeoffs:

- ancilla count vs precision
- Trotter depth vs error
- initial state overlap vs success probability

---

# Quantum Imaginary Time Evolution

Imaginary-time evolution:

$$
|\psi(\tau)\rangle
==================

e^{-H\tau}
|\psi(0)\rangle
$$

Suppresses higher-energy components.

VarQITE approximates evolution using McLachlan projection.

Linear system:

$$
A(\theta)\dot{\theta}
=====================

-C(\theta)
$$

with:

$$
A_{ij}
======

\Re
\langle \partial_i\psi|\partial_j\psi\rangle
$$

$$
C_i
===

\Re
\langle \partial_i\psi|
(H-\langle H\rangle)
|\psi\rangle
$$

Update:

$$
\theta
\leftarrow
\theta
+
\Delta\tau
\dot{\theta}
$$

Implementation features:

- noiseless parameter updates
- regularized linear solvers
- noise applied only during evaluation

---

# Noise Models

Noise channels implemented via PennyLane mixed-state simulation.

Supported channels:

- depolarizing noise
- amplitude damping

Noise placement:

- applied between circuit operations
- consistent across VQE, QPE, and evaluation stages

---

## Depolarizing channel

$$
\mathcal{E}(\rho)
=================

(1-p)\rho
+
\frac{p}{3}
(X\rho X
+
Y\rho Y
+
Z\rho Z)
$$

Produces isotropic decoherence.

---

## Amplitude damping

$$
E_0 =
\begin{pmatrix}
1 & 0 \
0 & \sqrt{1-p}
\end{pmatrix}
\quad
E_1 =
\begin{pmatrix}
0 & \sqrt{p} \
0 & 0
\end{pmatrix}
$$

Models relaxation toward ground state.

---

## Typical evaluation metrics

Noise studies examine:

- energy deviation
- convergence stability
- excitation ordering robustness
- optimizer sensitivity

---

# References

### Foundations

Aspuru-Guzik et al.
Simulated Quantum Computation of Molecular Energies

McArdle et al.
Quantum Computational Chemistry

Kitaev
Quantum Measurements and the Abelian Stabilizer Problem

---

### Excited states

Parrish et al.
Quantum Computation of Electronic Transitions using VQE

Higgott et al.
Variational Quantum Computation of Excited States

---

### ADAPT-VQE

Grimsley et al.
Adaptive variational algorithm for molecular simulation

---

### Imaginary-time simulation

McLachlan
Variational solution of the time-dependent Schrödinger equation

Yuan et al.
Theory of variational quantum simulation

---

### Fermion mappings

Seeley et al.
Bravyi–Kitaev transformation

---

### PennyLane

PennyLane documentation for:

- templates
- optimizers
- quantum chemistry tools

---

Author: Sid Richards
License: MIT
