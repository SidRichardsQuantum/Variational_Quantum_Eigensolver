# VQE Excited-State Methods

## Scope

This document describes the excited-state methods implemented in the VQE stack, including:

- mathematical formulation
- differences between methods
- implementation details specific to this repository
- practical guidance

These methods operate on a shared Hamiltonian and ansatz infrastructure.

---

## Overview

Excited-state methods in this repository fall into two categories:

### 1. Post-VQE methods

Require a **converged noiseless VQE reference state**:

- LR-VQE
- EOM-VQE
- QSE
- EOM-QSE

They construct and solve a **reduced eigenvalue problem**.

---

### 2. Variational methods

Do **not** require a prior VQE run:

- SSVQE
- VQD

They directly optimize excited states using modified objectives.

---

## Comparison Summary

| Method      | Type        | Core object         | EVP type        | Noise | Key limitation                  |
|-------------|-------------|---------------------|------------------|-------|---------------------------------|
| LR-VQE      | Post-VQE    | Tangent space       | Hermitian        | No    | linearized approximation        |
| EOM-VQE     | Post-VQE    | Full response       | structured EVP   | No    | numerically delicate            |
| QSE         | Post-VQE    | Operator subspace   | Hermitian        | No    | operator pool dependence        |
| EOM-QSE     | Post-VQE    | Commutator manifold | Non-Hermitian    | No    | complex spectrum / filtering    |
| SSVQE       | Variational | Multi-state unitary | optimization     | Yes   | harder with many states         |
| VQD         | Variational | Deflated objective  | optimization     | Yes   | depends on prior convergence    |

---

# Post-VQE Methods

## LR-VQE

### Idea

Linear-response approximation using the **tangent space** of the ansatz at the optimum.

Let:

- $|\psi_0\rangle = |\psi(\theta^*)\rangle$
- $|\partial_i \psi\rangle = \partial_{\theta_i} |\psi(\theta)\rangle$

### Core problem

$$
A c = \omega S c
$$

where:

$$
S_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle
$$
$$
A_{ij} = \langle \partial_i \psi | (H - E_0) | \partial_j \psi \rangle
$$

Excitation energies:

$$
E_k \approx E_0 + \omega_k
$$

---

### Implementation details (this repo)

- tangent vectors computed via **finite differences** (`fd_eps`)
- matrices explicitly constructed via QNodes
- **Hermitianization** applied to stabilize:
  - $A \leftarrow (A + A^\dagger)/2$
  - $S \leftarrow (S + S^\dagger)/2$
- **overlap filtering / rank truncation**:
  - eigenvalues of $S$ below `eps` are removed
- generalized EVP solved on reduced subspace

---

### Properties

- Hermitian EVP
- numerically stable relative to EOM-VQE
- approximation limited to local linear response

---

## EOM-VQE

### Idea

Extends LR-VQE to a **full-response equation-of-motion** formulation.

### Core problem

Generalized eigenvalue problem with paired roots:

$$
\omega \in \{+\omega_k, -\omega_k\}
$$

Physical energies:

$$
E_k \approx E_0 + \omega_k, \quad \omega_k > 0
$$

---

### Implementation details

- same tangent construction as LR-VQE
- augmented response structure (full-response matrices)
- **positive-root selection** using `omega_eps`
- **overlap filtering / rank truncation**
- explicit stabilization:
  - orthonormalization
  - Hermitianization steps

---

### Properties

- richer physics than LR-VQE
- more sensitive to:
  - conditioning of $S$
  - numerical noise in derivatives
- can produce unstable or spurious roots

---

## QSE

### Idea

Construct a **subspace spanned by operators acting on the reference state**:

$$
|\phi_i\rangle = O_i |\psi\rangle
$$

### Core problem

$$
H c = E S c
$$

where:

$$
H_{ij} = \langle \psi | O_i^\dagger H O_j | \psi \rangle
$$
$$
S_{ij} = \langle \psi | O_i^\dagger O_j | \psi \rangle
$$

---

### Implementation details

- operator pool:
  - typically Hamiltonian-derived or truncated (`top-k`, `max_ops`)
- matrices built via QNodes
- **Hermitian EVP**
- optional truncation via:
  - operator selection
  - overlap threshold (`eps`)

---

### Properties

- physically intuitive subspace
- strongly dependent on operator pool quality
- relatively stable numerically

---

## EOM-QSE

### Idea

Uses a **commutator-based equation of motion** in operator space.

### Core problem

$$
A c = \omega S c
$$

where:

$$
A_{ij} = \langle \psi | O_i^\dagger [H, O_j] | \psi \rangle
$$

---

### Implementation details

- same operator pool as QSE
- **non-Hermitian matrix $A$**
- eigenvalues may be complex
- filtering applied:
  - discard roots with large imaginary parts (`imag_tol`)
  - keep positive real-dominant roots (`omega_eps`)

---

### Properties

- non-Hermitian EVP
- can produce richer spectra
- requires careful filtering

---

# Variational Methods

## SSVQE

### Idea

Optimize multiple states simultaneously using a shared unitary.

$$
|\psi_k(\theta)\rangle = U(\theta)|\phi_k\rangle
$$

### Objective

$$
\mathcal{L}(\theta) = \sum_k w_k \langle \psi_k(\theta)|H|\psi_k(\theta)\rangle
$$

---

### Implementation details

- orthogonality enforced via **orthogonal input states**
- single shared parameter vector
- weights control energy ordering

---

### Properties

- no explicit overlap penalty
- scales poorly with many states
- compatible with noisy simulation

---

## VQD

### Idea

Sequentially compute states with **deflation penalties**.

### Objective

$$
\mathcal{L}_n(\theta)
=
\langle \psi(\theta)|H|\psi(\theta)\rangle
+
\beta \sum_{k<n} \mathcal{O}(\psi_k,\psi)
$$

---

### Overlap models

Noiseless:

$$
|\langle \psi_k|\psi\rangle|^2
$$

Noisy:

$$
\mathrm{Tr}(\rho_k \rho)
$$

---

### Implementation details

- overlap computed via:
  - adjoint circuit (statevector)
  - density matrix overlap (noisy)
- optional **beta scheduling**:
  - ramp-up strategies
- sequential optimization

---

### Properties

- flexible and intuitive
- sensitive to:
  - choice of $\beta$
  - convergence of previous states

---

# Numerical Stability Considerations

Across all post-VQE methods:

### Ill-conditioning

- overlap matrix $S$ may be near-singular
- handled via:
  - eigenvalue thresholding (`eps`)
  - rank truncation

---

### Finite-difference sensitivity

- tangent methods depend on `fd_eps`
- too small → numerical noise  
- too large → approximation error

---

### Root selection

For EOM methods:

- discard:
  - complex roots (large imaginary part)
  - negative or near-zero roots
- keep:
  - positive, real-dominant roots

---

# Practical Guidance

- use **QSE** for:
  - stable, simple excited-state estimates

- use **LR-VQE** for:
  - tangent-space physics
  - controlled approximations

- use **EOM-VQE / EOM-QSE** for:
  - richer spectra (with care)

- use **SSVQE** for:
  - small multi-state problems
  - noisy simulations

- use **VQD** for:
  - scalable sequential workflows
  - noisy environments

---

# Key Takeaway

All methods in this repository:

- share a **common VQE reference and Hamiltonian**
- differ primarily in:
  - subspace construction
  - eigenproblem structure
  - numerical stability

Understanding these differences is critical for selecting the appropriate method for a given problem.
