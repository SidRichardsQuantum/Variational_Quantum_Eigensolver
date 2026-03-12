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

For practical usage and CLI examples, see [`USAGE.md`](USAGE.md).  
For a project overview and quickstart, see [`README.md`](README.md).

---

## Contents

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
  - [Why Excited States Are Harder Than Ground States](#why-excited-states-are-harder-than-ground-states)
  - [Two Families of Excited-State Methods in This Repository](#two-families-of-excited-state-methods-in-this-repository)
  - [QSE](#qse)
  - [EOM-QSE](#eom-qse)
  - [LR-VQE](#lr-vqe)
  - [EOM-VQE](#eom-vqe)
  - [SSVQE](#ssvqe)
  - [VQD](#vqd)
  - [Excited-State Comparison Summary](#excited-state-comparison-summary)
- [ADAPT-VQE](#adapt-vqe)
- [Quantum Phase Estimation](#quantum-phase-estimation)
- [Quantum Imaginary Time Evolution](#quantum-imaginary-time-evolution)
- [Noise Models](#noise-models)
- [References](#references)

---

## Molecular Systems and Shared Chemistry Layer

This repository uses a shared molecular registry and Hamiltonian construction layer so that all algorithm families operate on the same chemistry definitions.

| Molecule | Typical uses in this repository | Basis | Approx. mapped qubits |
|---|---|---|---:|
| **H2** | ansatz studies, optimizer studies, QPE, QSE-family examples | STO-3G | 4 |
| **LiH** | bond-length scans | STO-3G | 12 |
| **H2O** | bond-angle scans | STO-3G | 14 |
| **H3+** | mapping comparisons, ADAPT-VQE, variational excited states | STO-3G | 6 |

Shared chemistry assumptions:

- molecular definitions come from `common/molecules.py`
- geometry generation is centralized in `common/geometry.py`
- Hamiltonians are built through `common/hamiltonian.py`
- all examples in this repository use the **STO-3G** basis for consistency

This common layer is important because it ensures that comparisons between VQE, QPE, and VarQITE are physically aligned rather than being driven by different hidden chemistry choices.

---

## Background

Quantum chemistry algorithms in this repository aim to approximate eigenvalues and eigenstates of an electronic Hamiltonian for small molecular systems. After choosing a molecular geometry and basis set, the electronic-structure problem is mapped to a finite-dimensional operator acting on qubits. The different algorithm families in this project — VQE, excited-state extensions, QPE, and VarQITE — then provide different ways of extracting ground-state or excited-state information from that shared Hamiltonian.

At a high level, the workflow is:

1. choose a molecule and geometry
2. build the electronic Hamiltonian in a finite basis
3. map the fermionic problem to qubit operators
4. apply a quantum algorithm to estimate energies or state properties

This repository centralizes steps 1–3 in the shared `common` layer so that algorithmic comparisons are made on the same physical footing.

### Variational Principle

For any normalized trial state $ |\psi\rangle $, the expectation value of the Hamiltonian is an upper bound on the true ground-state energy:

$$
E_0 \le \langle \psi | H | \psi \rangle .
$$

This is the key idea behind variational methods such as VQE: if we choose a parameterized family of states $ |\psi(\theta)\rangle $, then minimizing the expectation value over $\theta$ produces the best approximation available within that ansatz family.

### Why qubit mappings are needed

Molecular Hamiltonians are naturally written in terms of **fermionic** creation and annihilation operators, while quantum circuits act on **qubits**. To run quantum algorithms on molecular systems, the fermionic Hamiltonian must therefore be transformed into an equivalent qubit Hamiltonian.

That transformation is carried out by a fermion-to-qubit mapping such as:

- Jordan-Wigner
- Bravyi-Kitaev
- Parity mapping

These mappings preserve the underlying physics but change the qubit-operator structure, which can affect circuit depth, Pauli-string locality, and optimization behaviour.

### Hartree-Fock Reference State

Many workflows in this repository begin from a Hartree-Fock (HF) reference configuration. Hartree-Fock provides a simple mean-field approximation to the molecular ground state and supplies a natural computational-basis starting point for chemistry-inspired quantum algorithms.

Abstractly, the Hartree-Fock state may be written as

$$
|HF\rangle = |\phi_1 \phi_2 \cdots \phi_n\rangle .
$$

In a qubit encoding, this becomes an occupation-number bitstring indicating which spin-orbitals are occupied. For example, in a minimal-basis LiH setup with 4 electrons in 12 spin-orbitals, the Hartree-Fock occupation pattern is

```text
|111100000000⟩
````

In this repository, the Hartree-Fock reference is used in several ways:

* as the starting determinant for chemistry-inspired VQE ansätze
* as the initial reference state for ADAPT-VQE
* as a practical approximate eigenstate input for QPE
* as a natural initial state for some imaginary-time workflows

### From classical chemistry to quantum algorithms

The algorithms in this project differ mainly in how they use the qubit Hamiltonian once it has been constructed:

* **VQE** minimizes the energy expectation value variationally
* **post-VQE excited-state methods** extract excitations from a converged VQE reference
* **QPE** estimates eigenvalues from phase information in time evolution
* **VarQITE** approximates imaginary-time ground-state filtering

Because all of these methods share the same molecule registry, geometry generation, and Hamiltonian builder, differences between results can be interpreted as algorithmic differences rather than inconsistencies in the chemistry setup.

---

## VQE Overview

The Variational Quantum Eigensolver combines a parameterized quantum circuit with a classical optimizer.

At a high level:

1. prepare a trial state ( |\psi(\theta)\rangle )
2. measure ( \langle \psi(\theta)|H|\psi(\theta)\rangle )
3. update ( \theta ) classically
4. repeat until convergence

```
VQE WORKFLOW
============

   Classical Optimizer                Quantum Circuit
   -------------------                ----------------
         │                                   │
         │  propose parameters θ             │
         ├──────────────────────────────────▶│
         │                                   │
         │                    prepare ansatz → measure energy
         │◀──────────────────────────────────┤
         │
   update θ ← minimize energy
         │
         └── repeat until convergence
```

The quality of VQE depends on three main ingredients:

* the ansatz family
* the optimizer
* the qubit encoding of the molecular Hamiltonian

---

## Ansatz Families

An ansatz specifies the family of trial states ( |\psi(\theta)\rangle ). In practice, different ansätze trade off physical bias, circuit depth, expressibility, and trainability.

### UCCSD

**Unitary Coupled Cluster Singles and Doubles (UCCSD)** is a chemistry-motivated ansatz derived from coupled-cluster theory.

Conceptually:

$$
|\psi_{\mathrm{UCCSD}}(\theta)\rangle = e^{T(\theta)-T^\dagger(\theta)} |HF\rangle,
$$

with

$$
T(\theta) = T_1(\theta) + T_2(\theta),
$$

where (T_1) contains single excitations and (T_2) contains double excitations.

In this repository, UCCSD is used because it:

* incorporates chemically meaningful excitation structure
* performs strongly on small molecules in minimal bases
* provides a natural reference point for comparing more hardware-oriented circuits

### RY-CZ Ansatz

This is a hardware-efficient ansatz built from layers of single-qubit (R_Y) rotations and entangling (CZ) gates.

Typical motivations:

* shallow, simple structure
* tunable depth
* useful for optimizer and trainability studies
* more hardware-oriented than UCC-style circuits

### Minimal / One-Parameter Ansatz

For very small educational examples, a deliberately simple ansatz can be useful. A one-parameter ansatz provides:

* shallow circuits
* intuitive energy landscapes
* easy visualization of optimization behaviour
* a useful pedagogical baseline

Such ansätze are not intended to be generally competitive, but they help illustrate how VQE behaves.

---

## Optimizers

The optimizer updates the circuit parameters in response to measured energy values and gradients or gradient estimates.

### Adam

Adam combines momentum and adaptive per-parameter learning rates. In practice it is often the default because it is:

* robust
* easy to tune
* effective on irregular objective landscapes

### Gradient Descent

A straightforward baseline method that updates directly along the negative gradient direction. It is useful for:

* pedagogical comparisons
* simple baselines
* testing sensitivity to step size

### Momentum and Nesterov Momentum

Momentum-based methods smooth updates and can help navigate shallow valleys or oscillatory directions. Nesterov momentum adds a look-ahead component that can improve convergence in smooth regimes.

### Adagrad

Adagrad adapts the learning rate using accumulated gradient history. It can help when different parameters evolve on very different scales, though it can become overly conservative over long runs.

### SPSA

**Simultaneous Perturbation Stochastic Approximation** is especially relevant for noisy settings because it estimates gradient information using very few function evaluations. It is useful when exact gradients are expensive or unstable.

---

## Fermion-to-Qubit Mappings

Electronic-structure Hamiltonians are naturally expressed in terms of fermionic creation and annihilation operators. To run them on qubits, they must be mapped to qubit operators.

This repository includes studies using three common mappings:

### Jordan-Wigner

* direct occupation-based encoding
* simple and explicit
* can produce long Pauli strings

### Bravyi-Kitaev

* balances occupation and parity information
* often reduces average Pauli-string length relative to Jordan-Wigner
* can improve circuit structure in some settings

### Parity Mapping

* encodes parity structure rather than direct occupation
* can reduce depth or expose useful symmetries
* may alter optimization behaviour and operator structure

These mappings change the qubit Hamiltonian representation and therefore can affect:

* circuit depth
* measurement structure
* optimization stability
* gradient magnitudes

---

## Excited-State Methods

Ground-state VQE is naturally variational. Excited states are more subtle because the variational minimum of a Hamiltonian is always biased toward the lowest eigenvalue.

### Why Excited States Are Harder Than Ground States

Standard VQE solves

$$
E_0 = \min_\theta \langle \psi(\theta)|H|\psi(\theta)\rangle .
$$

This directly targets the ground state, but not higher-energy eigenstates. To access excitations, one typically needs one of the following strategies:

* construct a reduced subspace around a converged ground-state reference
* impose orthogonality or deflation constraints
* optimize several states simultaneously

### Two Families of Excited-State Methods in This Repository

This repository contains two broad classes of excited-state methods.

#### 1. Post-VQE linear-algebra methods

These start from a converged **noiseless VQE reference state** and then solve a reduced eigenvalue problem:

* **QSE**
* **EOM-QSE**
* **LR-VQE**
* **EOM-VQE**

These methods do not perform a second variational optimization over excited states. Instead, they extract excitations from a local subspace or tangent-space construction around the reference state.

#### 2. Variational excited-state methods

These solve excited states directly through variational objectives:

* **SSVQE**
* **VQD**

These approaches are more flexible in noisy settings, but they involve additional optimization effort.

---

## QSE

**Quantum Subspace Expansion (QSE)** is a post-VQE method built around a converged reference state ( |\psi\rangle ). It constructs a small operator-generated subspace

$$
|\phi_i\rangle = O_i |\psi\rangle
$$

using a chosen operator set ( {O_i} ).

From this subspace, one forms

$$
H_{ij} = \langle \psi| O_i^\dagger H O_j |\psi\rangle,
\qquad
S_{ij} = \langle \psi| O_i^\dagger O_j |\psi\rangle,
$$

and solves the generalized eigenvalue problem

$$
Hc = ESc.
$$

Interpretation:

* (S) is the overlap matrix of the generated subspace
* (H) is the Hamiltonian projected into that subspace
* the resulting eigenvalues approximate low-lying energies accessible within the chosen manifold

In this repository, QSE is:

* a **post-VQE** workflow
* **noiseless-only**
* typically driven by a Hamiltonian-informed operator pool

Its performance depends strongly on the quality of both:

* the VQE reference state
* the selected operator manifold

---

## EOM-QSE

**Equation-of-Motion QSE (EOM-QSE)** keeps the same basic operator-manifold philosophy as QSE, but uses a commutator-based equation of motion.

Given a reference state ( |\psi\rangle ) and operator set ( {O_i} ), define

$$
A_{ij} = \langle \psi | O_i^\dagger [H, O_j] | \psi \rangle,
\qquad
S_{ij} = \langle \psi | O_i^\dagger O_j | \psi \rangle .
$$

The excitation energies are estimated from

$$
Ac = \omega Sc.
$$

Key differences relative to projection-QSE:

* the reduced problem is generally **non-Hermitian**
* eigenvalues may be complex
* physical roots are typically selected from **positive, real-dominant** solutions

In this implementation:

* the method is **noiseless-only**
* the operator pool is Hamiltonian-driven
* overlap filtering and simple root-selection heuristics are used to stabilize the result

EOM-QSE does not inherit a variational upper-bound property, so the usefulness of the extracted roots depends strongly on the chosen operator manifold and the quality of the reference state.

---

## LR-VQE

**Linear-Response VQE (LR-VQE)** is a post-VQE excited-state method based on the **tangent space of the variational manifold** at a converged ground-state point ( \theta^* ).

Let

$$
|\psi_0\rangle = |\psi(\theta^*)\rangle,
\qquad
E_0 = \langle \psi_0|H|\psi_0\rangle.
$$

Define tangent vectors

$$
|\partial_i \psi\rangle
=======================

\left.\frac{\partial}{\partial \theta_i} |\psi(\theta)\rangle\right|_{\theta=\theta^*}.
$$

LR-VQE constructs the tangent-space matrices

$$
S_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle,
\qquad
A_{ij} = \langle \partial_i \psi | (H - E_0) | \partial_j \psi \rangle.
$$

The excitation energies are obtained from the generalized eigenvalue problem

$$
Ac = \omega Sc.
$$

Approximate excited-state energies are then

$$
E_k \approx E_0 + \omega_k.
$$

In this repository, LR-VQE corresponds to a **Tamm-Dancoff approximation (TDA)** style tangent-space treatment. Practical implementation details include:

* finite-difference state derivatives
* explicit symmetrization / Hermitianization
* overlap filtering and rank truncation
* **noiseless-only** support because the method requires statevector tangents

This makes LR-VQE a numerically practical small-system excited-state tool, especially for benchmarking tangent-space ideas without introducing the full-response structure of EOM-VQE.

---

## EOM-VQE

**Equation-of-Motion VQE (EOM-VQE)** extends the tangent-space idea beyond the Tamm-Dancoff approximation by incorporating a **full-response** treatment.

Conceptually, the method augments the tangent-space description so that the resulting equation-of-motion problem captures a richer response structure than LR-VQE. In ideal form, the reduced problem yields paired roots ( \pm \omega ), and the physically relevant excitation energies are taken from the positive roots:

$$
E_k \approx E_0 + \omega_k,
\qquad \omega_k > 0.
$$

Relative to LR-VQE, EOM-VQE can capture additional response physics but is typically:

* more sensitive to the quality of the converged VQE reference
* more sensitive to ill-conditioning in the tangent-space metric
* numerically more delicate

In this repository, EOM-VQE is implemented using:

* a converged **noiseless** VQE reference
* finite-difference tangent construction
* overlap filtering / rank truncation
* stabilization via orthonormalization and Hermitianization steps
* positive-root selection from the reduced response spectrum

Because the workflow depends on statevector tangents, it is currently **noiseless-only**.

---

## SSVQE

**Subspace-Search VQE (SSVQE)** is a variational excited-state method that optimizes multiple states simultaneously using a shared-parameter unitary.

Choose orthogonal reference inputs ( |\phi_k\rangle ) and apply the same parameterized unitary (U(\theta)) to each:

$$
|\psi_k(\theta)\rangle = U(\theta),|\phi_k\rangle.
$$

Then optimize a weighted multi-state objective:

$$
\mathcal{L}(\theta)
===================

\sum_k w_k \langle \psi_k(\theta)|H|\psi_k(\theta)\rangle .
$$

The key idea is that orthogonality is enforced structurally by the choice of orthogonal input states rather than by an explicit overlap penalty.

Practical features:

* simultaneous multi-state optimization
* natural for small numbers of low-lying states
* compatible with noisy simulation in this repository

A limitation is that the optimization can become more difficult as the number of targeted states increases.

```
SSVQE IDEA
==========

Choose orthogonal inputs:
   |φ0⟩, |φ1⟩, ...

Apply shared unitary:
   |ψk(θ)⟩ = U(θ)|φk⟩

Optimize one joint loss:
   L(θ) = Σk wk ⟨ψk(θ)|H|ψk(θ)⟩
```

---

## VQD

**Variational Quantum Deflation (VQD)** computes excited states sequentially rather than simultaneously.

The workflow is:

1. solve a standard VQE problem for the ground state
2. optimize the next state with a deflation penalty that discourages overlap with previously found states
3. repeat for higher excited states

For the (n)-th state, the objective has the form

$$
\mathcal{L}_n(\theta_n)
=======================

\langle \psi(\theta_n)|H|\psi(\theta_n)\rangle
+
\beta \sum_{k<n} \mathcal{O}(\psi_k,\psi_n),
$$

where:

* ( \beta ) controls the strength of deflation
* ( \mathcal{O} ) is an overlap penalty term

Typical overlap models in this repository:

**Noiseless case**
$$
\mathcal{O}(\psi_k,\psi_n)=|\langle \psi_k|\psi_n\rangle|^2
$$

**Noisy case**
$$
\mathcal{O}(\rho_k,\rho_n)=\mathrm{Tr}(\rho_k\rho_n)
$$

VQD is attractive because it scales naturally from state to state and remains usable in noisy settings, but its performance depends on:

* appropriate deflation strength
* stable optimization
* good convergence of lower states before moving upward

This repository also includes optional **beta schedules** to ramp deflation strength gradually during optimization.

---

## Excited-State Comparison Summary

| Method      | Family      | Core idea                                          | Noise support  | Typical limitation                      |
| ----------- | ----------- | -------------------------------------------------- | -------------- | --------------------------------------- |
| **QSE**     | Post-VQE    | projection into operator subspace                  | Noiseless-only | quality depends on operator pool        |
| **EOM-QSE** | Post-VQE    | commutator equation of motion in operator manifold | Noiseless-only | non-Hermitian reduced problem           |
| **LR-VQE**  | Post-VQE    | tangent-space linear response (TDA)                | Noiseless-only | limited by tangent-space quality        |
| **EOM-VQE** | Post-VQE    | full-response tangent-space EOM                    | Noiseless-only | numerically more delicate               |
| **SSVQE**   | Variational | simultaneous multi-state optimization              | Supported      | harder as state count grows             |
| **VQD**     | Variational | sequential deflation                               | Supported      | depends on good lower-state convergence |

In this repository, the post-VQE methods are mainly intended for **small-system excited-state studies and benchmarking**, while the variational methods provide a more direct route to noisy excited-state experiments.

---

## ADAPT-VQE

**ADAPT-VQE** constructs the ansatz dynamically rather than fixing the full circuit structure in advance.

Instead of choosing a large ansatz such as UCCSD up front, the method starts from a reference state and repeatedly appends the operator from a predefined pool that appears most useful according to a gradient-based score.

This repository uses a chemistry-oriented ADAPT-VQE setup with:

* Hartree-Fock reference state
* excitation-operator pools based on UCC-style singles, doubles, or both

Let the adaptive ansatz after (k) selections be

$$
|\psi_k(\theta)\rangle = U_k(\theta)|HF\rangle,
$$

with

$$
U_k(\theta)=\prod_{j=1}^{k} e^{\theta_j A_j}.
$$

At each outer iteration, each remaining candidate operator (A) is scored using the energy gradient that would arise if it were appended with a zero-initialized new parameter. The next selected operator is the one with the largest gradient magnitude.

A simplified view of the workflow is:

1. optimize parameters for the current selected operator list
2. evaluate gradient scores for candidate pool operators
3. append the best operator
4. stop when the largest gradient falls below a tolerance or a maximum operator count is reached

```
ADAPT-VQE WORKFLOW
==================

Initialize:
   |ψ0⟩ = |HF⟩
   selected_ops = []

Repeat:

1. Optimize current ansatz parameters
2. Score candidate pool operators by gradient magnitude
3. Append the best operator

Stop when:
   max gradient < grad_tol
   or
   number of operators == max_ops
```

Relative to fixed-ansatz VQE, ADAPT-VQE can provide:

* shallower circuits for a target accuracy
* more interpretable operator selections
* a built-in convergence diagnostic through the gradient history

In this repository, ADAPT-VQE inherits the same caching, noise-handling, and output conventions as the broader VQE stack.

---

## Quantum Phase Estimation

**Quantum Phase Estimation (QPE)** estimates eigenvalues by extracting the phase of a unitary operator. In quantum chemistry, the relevant unitary is usually time evolution under the molecular Hamiltonian:

$$
U = e^{-iHt}.
$$

If ( |\psi\rangle ) is an eigenstate of (H) with energy (E), then

$$
U|\psi\rangle = e^{-iEt}|\psi\rangle = e^{2\pi i \theta} |\psi\rangle,
$$

so the phase ( \theta ) is related to the energy by

$$
\theta = -\frac{Et}{2\pi},
\qquad
E = -\frac{2\pi \theta}{t}.
$$

### QPE Workflow

QPE uses two registers:

* an **ancilla register** to encode the phase
* a **system register** prepared in an approximate eigenstate

The basic flow is:

1. initialize the ancillas in ( |0\rangle^{\otimes n} )
2. prepare the system state, often using the Hartree-Fock reference
3. apply Hadamards to create ancilla superposition
4. apply controlled powers of (U)
5. apply the inverse quantum Fourier transform
6. measure the ancillas and convert the phase estimate into an energy estimate

```
QPE OVERVIEW
============

Ancilla register:  |0⟩...|0⟩  --H-- controlled-U^{2^k} -- IQFT -- measure
System register:   |HF⟩   ----------------------------------------------
```

In this repository:

* QPE uses the same Hamiltonian pipeline as VQE
* controlled evolution is implemented through **trotterized time evolution**
* noisy and noiseless variants are both supported

Key trade-offs include:

* number of ancillas vs phase precision
* evolution time (t) vs phase resolution / wrapping
* Trotter error vs circuit cost
* quality of the input state overlap with the target eigenstate

Unlike VQE, QPE is not variational: it does not minimize an energy functional, but instead estimates eigenphases directly.

---

## Quantum Imaginary Time Evolution

Imaginary-time evolution suppresses excited-state components and drives a state toward the ground state:

$$
|\psi(\tau)\rangle \propto e^{-H\tau}|\psi(0)\rangle .
$$

If

$$
|\psi(0)\rangle = \sum_k c_k |E_k\rangle,
$$

then

$$
e^{-H\tau}|\psi(0)\rangle
=========================

\sum_k c_k e^{-E_k \tau}|E_k\rangle.
$$

After normalization, higher-energy components decay more rapidly than the ground-state contribution, provided (c_0 \neq 0).

Because quantum circuits are unitary while (e^{-H\tau}) is not, this repository uses a **variational approximation** based on the **McLachlan variational principle**, often called **VarQITE**.

### McLachlan Variational Update

Restrict the dynamics to a parameterized ansatz ( |\psi(\theta)\rangle ). The idea is to choose parameter updates so that the ansatz trajectory best matches imaginary-time evolution in a least-squares sense.

This leads to a linear system

$$
A(\theta),\dot{\theta} = -C(\theta),
$$

with

$$
A_{ij} = \Re \langle \partial_i \psi | \partial_j \psi \rangle,
\qquad
C_i = \Re \langle \partial_i \psi | (H-\langle H\rangle)|\psi\rangle.
$$

A discrete step of size ( \Delta\tau ) updates the parameters via

$$
\theta \leftarrow \theta + \Delta\tau,\dot{\theta}.
$$

In this repository:

* the parameter-update stage is **noiseless**
* multiple numerical linear solvers are supported
* optional regularization is available for ill-conditioned systems
* noisy studies are handled as **post-evaluation** of converged parameters, not noisy imaginary-time optimization

### Relationship to VQE and QPE

The three main algorithm families in this repository differ conceptually as follows:

* **VQE** — direct variational minimization of ( \langle \psi(\theta)|H|\psi(\theta)\rangle )
* **VarQITE** — variational approximation to imaginary-time ground-state filtering
* **QPE** — phase-based eigenvalue extraction from (e^{-iHt})

All three share the same chemistry layer and therefore can be compared on a consistent footing.

---

## Noise Models

This repository studies two main noise channels using PennyLane's mixed-state backend:

* **depolarizing noise**
* **amplitude damping**

Noise handling is algorithm-dependent:

* **VQE** — supported during circuit execution
* **QPE** — supported during controlled-evolution workflows
* **VarQITE** — parameter updates are noiseless; noise is applied only during post-evaluation
* **SSVQE / VQD** — noisy overlap handling is supported via density-matrix inner products

```
NOISE IN SIMULATIONS
====================

ideal gates
   ↓
apply noise
   ↓
next gates
   ↓
apply noise
   ↓
...
```

### Depolarizing Noise

Depolarizing noise drives a qubit toward a mixed state with probability (p_{\mathrm{dep}}):

$$
\mathcal{E}_{\mathrm{dep}}(\rho)
================================

(1-p_{\mathrm{dep}})\rho
+
\frac{p_{\mathrm{dep}}}{3}
\left(
X\rho X + Y\rho Y + Z\rho Z
\right).
$$

This models symmetric random errors and tends to reduce coherence and entanglement fidelity.

### Amplitude Damping

Amplitude damping models relaxation toward ( |0\rangle ) with probability (p_{\mathrm{amp}}):

$$
\mathcal{E}_{\mathrm{amp}}(\rho)
================================

E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger,
$$

where

$$
E_0 =
\begin{pmatrix}
1 & 0 \
0 & \sqrt{1-p_{\mathrm{amp}}}
\end{pmatrix},
\qquad
E_1 =
\begin{pmatrix}
0 & \sqrt{p_{\mathrm{amp}}} \
0 & 0
\end{pmatrix}.
$$

This is an asymmetric noise channel and tends to bias the system toward lower-energy computational-basis populations.

### Typical Evaluation Metrics

Noise studies in this repository commonly examine:

* **energy error** relative to noiseless baselines
* **fidelity-style overlap measures**
* optimizer / ansatz robustness under increasing noise strength
* sensitivity of phase-estimation distributions or imaginary-time post-evaluations

---

## References

### Foundations and Reviews

* Aspuru-Guzik et al., *Simulated Quantum Computation of Molecular Energies*.
* McArdle et al., *Quantum Computational Chemistry*.
* Kitaev, *Quantum Measurements and the Abelian Stabilizer Problem*.

### Excited-State Methods

* Parrish et al., *Quantum Computation of Electronic Transitions using a Variational Quantum Eigensolver*.
* Higgott et al., *Variational Quantum Computation of Excited States*.

### ADAPT-VQE

* Grimsley et al., *An adaptive variational algorithm for exact molecular simulations on a quantum computer*.

### Imaginary-Time / Variational Simulation

* McLachlan, *A variational solution of the time-dependent Schrödinger equation*.
* Yuan et al., *Theory of variational quantum simulation*.

### Fermion-to-Qubit Mappings and Chemistry Background

* Seeley et al., *The Bravyi-Kitaev transformation for quantum computation of electronic structure*.
* Standard Hartree-Fock and electronic-structure references.

### PennyLane Documentation

* PennyLane documentation for templates, optimizers, and quantum chemistry tooling.

---

**Author:** Sid Richards (SidRichardsQuantum)

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).