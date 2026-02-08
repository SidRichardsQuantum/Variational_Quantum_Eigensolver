# 🧠 Theory & Methodology

This document provides a detailed explanation of the **Variational Quantum Eigensolver (VQE)**, the **molecules**, **ansatzes**, and **optimizers** used in this project.

---

## 📚 Table of Contents

- [Molecules Studied](#molecules-studied)
- [Background](#background)
- [VQE Algorithm Overview](#vqe-algorithm-overview)
   - [Ansatz Construction](#ansatz-construction)
   - [Optimizers](#optimizers)
   - [Fermion-to-Qubit Mappings](#fermion-to-qubit-mappings)
   - [Excited-State Methods](#excited-state-methods)
      - [Quantum Subspace Expansion](#quantum-subspace-expansion)
      - [Subspace-Search VQE](#subspace-search-vqe)
      - [Variational Quantum Deflation](#variational-quantum-deflation)
  - [ADAPT-VQE](#adapt-vqe)
- [Quantum Phase Estimation](#quantum-phase-estimation)
- [Quantum Imaginary Time Evolution](#quantum-imaginary-time-evolution)
- [Noise Types](#noise-types)
- [References](#references)

---

## Molecules Studied

| Molecule | Properties Investigated                         | Basis     | Qubits (mapped) |
|:--------:|:------------------------------------------------|:-----------|:----------------:|
| **H₂**    | Ansatz comparison, optimizer comparison, QPE, QSE | STO-3G     | 4 |
| **LiH**   | Bond-length scans (VQE)                         | STO-3G     | 12 |
| **H₂O**   | Bond-angle scans (VQE)                          | STO-3G     | 14 |
| **H₃⁺**   | Mapping comparisons, SSVQE excited states       | STO-3G     | 6 |

All molecular geometries now come from the **shared registry** in `common/molecules.py`.
All simulations use the **STO-3G** basis set for consistency.

Molecular Hamiltonians are constructed using PennyLane’s `qchem.molecular_hamiltonian` via the unified builder in `common/hamiltonian.py`, which sources symbols/coordinates/basis/charge from the shared registry (`common/molecules.py`).

---

## Background

### Variational Principle

The variational principle states that for any trial wavefunction $|\psi⟩$, the expectation value of the Hamiltonian is an upper bound to the true ground state energy:

$$E₀ ≤ ⟨\psi|H|\psi⟩$$

where:
- $E_0$ is the ground state energy
- $|\psi⟩$ is any normalized wavefunction
- $H$ is the Hamiltonian

### Hartree-Fock

The Hartree-Fock state is written as the tensor product of a qubit state $\phi$ for every electron orbital:

$$|HF⟩ = |φ₁φ₂...φₙ⟩$$

For LiH with $4$ electrons in $12$ orbitals, the HF reference state is ```|111100000000⟩```, which describes electrons occupying the four lowest energy orbital states.

## VQE Algorithm Overview

The VQE algorithm consists of:
1. **State Preparation**: Prepare parameterized quantum state $|\psi(\theta)⟩$
2. **Measurement**: Measure expectation value $⟨\psi(\theta)| H |\psi(\theta)⟩$
3. **Optimization**: Classically optimize parameters $\theta$ to minimize energy
4. **Iteration**: Repeat until convergence

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

---

### Ansatz Construction

An ansatz defines the functional form of the trial quantum state $|\psi(\theta)⟩$.
It determines how expressive, efficient, and trainable your VQE circuit is.
Different ansatze trade off physical accuracy, circuit depth, and compatibility with quantum hardware.

#### UCCSD (Unitary Coupled Cluster Singles and Doubles)

A chemistry-inspired ansatz derived from coupled-cluster theory. Includes single and double excitations applied as a unitary, Trotterized operator.

- Designed for capturing electron correlation from first principles
- Exact for small systems like H₂ or H₃⁺ in minimal basis sets (e.g., STO-3G)
- Used to compare excitation types (single vs. double vs. UCCSD) in **H₃⁺**

```
   |HF⟩ ── exp( T(θ) - T(θ)^† ) ──>  |ψ_UCCSD(θ)⟩

   where T(θ) = T₁(θ) + T₂(θ) for:
         T₁ singles  (a^†_p a_q)
         T₂ doubles  (a^†_p a^†_q a_r a_s)
```

#### $R_Y-C_Z$ Ansatz

A hardware-efficient ansatz composed of layers alternating single-qubit rotations and entangling gates.

- Uses $R_Y$ rotations followed by a chain of $C_Z$ gates
- Tunable number of layers (depth)
- Good expressibility for small and medium systems
- Easier to implement on near-term hardware

```
Layer k:
   ┌──────────── Ry rotations ─────────────┐   ┌──── CZ entanglers ───
   q0: ── Ry(θ₀,k) ────────────────────────────●─────────●────────────
                                               │         │
   q1: ── Ry(θ₁,k) ────────────────────────────X─────────●────────────
                                                         │
   q2: ── Ry(θ₂,k) ──────────────────────────────────────X────────────
```

#### Minimal / One-Parameter Ansatz

A manually constructed, problem-specific ansatz using very few parameters.

- Tailored for simple systems like H₂ in minimal basis
- Uses a single $R_Y$ rotation and one entangling gate (e.g., CNOT)
- Extremely shallow and interpretable
- Useful for testing optimizers, energy landscapes, or learning curves

```
   q0: ── Ry(θ) ──●────
                  │
   q1: ───────────X────   (single entangler)
```

---

### Optimizers

Classical optimizers are a critical component of the VQE algorithm, as they minimize the energy by adjusting circuit parameters $\theta$.

#### AdamOptimizer

Designed for fast, stable optimization by combining the benefits of momentum and adaptive learning rates.
- Automatically adjusts step size for each parameter
- Performs well in noisy or irregular energy landscapes
- Common default in VQE due to ease of use and robustness

#### GradientDescentOptimizer

The simplest optimizer, as it updates parameters in the direction of steepest descent.
- Useful for educational or baseline comparisons
- Very sensitive to step size
- Often slower and less reliable in quantum settings

#### MomentumOptimizer

Adds inertia to gradient descent to smooth parameter updates and help escape shallow local minima.
- Useful when gradients fluctuate heavily
- Reduces oscillations near minima
- Often used as a stepping stone toward more adaptive optimizers

#### NesterovMomentumOptimizer

An improvement over standard momentum optimizers that “looks ahead” before making updates.
- Accelerates convergence in smooth regions
- Helps avoid getting stuck in flat or gently curved regions
- Can be unstable if not tuned carefully

#### AdagradOptimizer

Adapts learning rates for each parameter based on past gradient history.
- Useful when some parameters require more aggressive updates than others
- Can become sluggish over time as it overcorrects

#### SPSAOptimizer

(Simultaneous Perturbation Stochastic Approximation)

Designed for noisy or hardware-executed circuits, where gradients are expensive or unreliable.
- Estimates the gradient using random perturbations
- Requires very few circuit evaluations per step
- Performs well in realistic noisy quantum environments

---

### Fermion-to-Qubit Mappings

To simulate molecular Hamiltonians on quantum computers, second-quantized fermionic operators must be mapped to qubit operators.  
This project compares three common mappings using the H₃⁺ molecule:

- **Jordan-Wigner (JW)**  
  Maps fermionic modes to qubits directly, preserving occupation order.  
  Simple but introduces long Pauli string chains for highly nonlocal interactions.

- **Bravyi-Kitaev (BK)**  
  Balances between local occupation and parity information.  
  Results in shorter average Pauli string lengths and fewer entangling gates in some cases.

- **Parity Mapping**  
  Encodes occupation parity rather than direct state, often reducing gate depth.  
  Can introduce nontrivial entanglement and symmetry behavior.

Each mapping transforms the Hamiltonian into a different structure of Pauli operators, which affects convergence, gradient norms, and optimization stability in VQE.

(The same ansatz and optimizers are applied across all mappings to isolate the impact of encoding alone.)

---

### Excited-State Methods

While the standard VQE algorithm is designed to find the **ground state** of a molecular Hamiltonian, many applications in quantum chemistry require access to **excited states** — for example, to predict **spectroscopic transitions**, **photoexcitation energies**, and **reaction pathways**.

#### Challenge

The original VQE formulation finds the **lowest eigenvalue** of the Hamiltonian by variationally minimizing the energy:

$$E_0 = \min_{\theta} ⟨\psi(\theta)| H |\psi(\theta)⟩$$

This process does not directly provide excited states, and repeating VQE with orthogonality constraints is non-trivial.

#### Quantum Subspace Expansion

Quantum Subspace Expansion (QSE) is a **post-VQE** method: it starts from a **converged
(noiseless) VQE reference state** $|\psi\rangle$ and builds a small linear subspace around it
using a chosen operator set $\{O_i\}$.

##### Subspace construction

Define the (generally non-orthonormal) basis states:

$$|\phi_i\rangle = O_i |\psi\rangle.$$

Then construct the subspace matrices

$$H_{ij} = \langle \psi| O_i^\dagger H O_j |\psi\rangle,\qquad
S_{ij} = \langle \psi| O_i^\dagger O_j |\psi\rangle.$$

The approximate spectrum in this subspace is obtained by solving the generalized eigenvalue problem:

$$Hc = ESc.$$

#### Subspace-Search VQE

Subspace-Search VQE (SSVQE) is a variational method that finds **multiple eigenstates simultaneously** by:

1. Preparing a **set of parameterized quantum states** $\{ |\psi_0(\theta_0)⟩, \ |\psi_1(\theta_1)⟩, \ \dots \}$

2. **Optimizing** all parameters to minimize a **weighted sum of expectation values**:

$$\mathcal{L} = \sum_i w_i ⟨\psi_i| H |\psi_i⟩$$

3. Orthogonality is enforced by choosing **orthogonal computational-basis input states** $|\phi_k\rangle$ and applying a **shared** parameterized unitary $U(\theta)$:

$$|\psi_k(\theta)\rangle = U(\theta)\,|\phi_k\rangle.$$

In this formulation, no explicit overlap penalties are required; distinct optimized states arise from the orthogonality of the inputs.

```
         θ(0)               θ(1)
      ┌─────────┐        ┌─────────┐
|0⟩──▶│  Ansatz │──▶|ψ₀⟩ │         │
      └─────────┘        └─────────┘
                           │
                           ▼
                       |ψ₁⟩, |ψ₂⟩, ...

Loss function:

   𝓛(θ) =
       Σᵢ wᵢ ⟨ψᵢ(θ)| H |ψᵢ(θ)⟩        (weighted energies)

OPTIMIZATION LOOP:

Initialize {θ(0), θ(1), …}
      │
      ▼
Prepare { |ψᵢ(θ(i))⟩ } on device
      │
      ▼
Measure ⟨ψᵢ|H|ψᵢ⟩ for each reference-indexed state
      │
      ▼
Compute 𝓛  → update all θ(i) with Adam
      │
      └── repeat until 𝓛 converges

Result: approximate low-lying spectrum {E₀, E₁, …} from a single joint optimization.
```

#### Variational Quantum Deflation

Variational Quantum Deflation (VQD) is an alternative variational approach for computing
**excited states sequentially**, rather than simultaneously.

Instead of optimizing multiple states in a single joint objective (as in SSVQE),
VQD proceeds by **iteratively deflating previously found eigenstates**.

---

### VQD Principle

1. First, solve a standard VQE problem to obtain the ground state:

$$E_0 = \min_{\theta_0} \langle \psi(\theta_0) | H | \psi(\theta_0) \rangle$$

2. For the $n$-th excited state, minimize the modified cost function:

$$\mathcal{L}_n(\theta_n) = \langle \psi(\theta_n) | H | \psi(\theta_n) \rangle$$

$$\beta \sum_{k < n} \mathcal{O}(\psi_k, \psi_n)$$

where:
- $\psi_k$ are **previously converged states**
- $\beta$ is a tunable deflation strength
- $\mathcal{O}$ is an overlap penalty enforcing orthogonality

#### Overlap Penalty

The overlap metric depends on whether the simulation is noiseless or noisy:

**Noiseless case**

$$\mathcal{O}(\psi_k, \psi_n) = |\langle \psi_k | \psi_n \rangle|^2$$

**Noisy case**

$$\mathcal{O}(\rho_k, \rho_n) = \mathrm{Tr}(\rho_k \rho_n)$$

This formulation allows VQD to remain valid when circuits are executed on
mixed-state simulators or noisy hardware.

#### k-State Generalization

VQD naturally generalizes to an arbitrary number of states:

- State 0: standard VQE
- State 1: deflated against state 0
- State 2: deflated against states 0 and 1
- …
- State $k-1$: deflated against all lower states

Each state is optimized **independently**, with its own parameter vector and
its own optimization loop.

#### Beta Scheduling

To improve stability, the deflation strength $\beta$ is typically **ramped** during optimization:

$$\beta(t) \in [\beta_{\text{start}}, \beta_{\text{end}}]$$

Common schedules include:
- Linear ramps
- Cosine ramps with smooth turn-on
- Optional warm-up periods with $\beta = 0$

This avoids early optimization being dominated by overlap penalties before
the energy landscape is sufficiently explored.

---

### Excited State Comparisons

| Method | Type | Optimization | Orthogonality mechanism | Noise support | Scaling |
|------|------|-------------|--------------------------|---------------|---------|
| **QSE** | Post-VQE | None (linear algebra) | Generalized EVP in $\{O_i|\psi\rangle\}$ subspace | Noiseless-only (statevector reference) | Pool-limited subspace |
| **SSVQE** | Variational | Simultaneous | Orthogonal reference inputs + shared $U(\theta)$ | Supported | Harder with many states |
| **VQD** | Variational | Sequential | Deflation against previously found states | Supported | Naturally $k$-state |

In this project:
- **SSVQE** is used for pedagogical demonstrations and small subspaces
- **VQD** is preferred for systematic, scalable excited-state studies

---

### ADAPT-VQE

ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter VQE) is a variational method that **constructs the ansatz on the fly**, rather than fixing a circuit structure in advance.

Instead of choosing a large, expressive ansatz up front (e.g., UCCSD), ADAPT-VQE grows a compact ansatz by repeatedly selecting the **most “useful” operator** from an excitation pool, based on an energy-gradient criterion.

This project implements a chemistry-oriented ADAPT-VQE variant:

- **Reference state:** Hartree–Fock determinant $|HF\rangle$
- **Operator pool:** UCC excitation generators (singles, doubles, or both)

#### Core idea

Let the current adaptive ansatz (after $k$ selections) be

$$|\psi_k(\theta)\rangle = U_k(\theta)\,|HF\rangle,$$

where $U_k(\theta)$ is a product of excitation unitaries chosen from a pool:

$$U_k(\theta) = \prod_{j=1}^{k} e^{\theta_j A_j}.$$

At each outer iteration, ADAPT-VQE scores every candidate operator $A$ in the remaining pool by asking:

> “If I appended this operator with parameter initialized at $\theta=0$, how large is the energy gradient?”

In this implementation the score is:

$$g(A) = \left|\frac{\partial E}{\partial \theta}\right|_{\theta=0}, \quad
E(\theta)=\langle \psi(\theta)|H|\psi(\theta)\rangle.$$

The operator with the **largest gradient magnitude** is appended:

$$A_{k+1} = \arg\max_{A\in\mathcal{P}\setminus\{A_1,\dots,A_k\}} g(A).$$

#### Two-loop workflow

ADAPT-VQE alternates between:

1. **Inner loop (standard VQE optimization)**  
   Optimize the parameters for the **current** selected operator list:

   $$\theta^{*}_k = \arg\min_{\theta}\;\langle \psi_k(\theta)|H|\psi_k(\theta)\rangle.$$

2. **Outer loop (operator selection)**  
   Evaluate $g(A)$ for each candidate operator $A$ when appended at zero,
   then append the best candidate and continue.

Stopping conditions:

- **Gradient tolerance:** stop when $\max_A g(A) < \varepsilon$ (called `grad_tol`)
- **Operator budget:** stop when $k$ reaches `max_ops`

```

# ADAPT-VQE WORKFLOW

Initialize:
|ψ₀⟩ = |HF⟩ , selected_ops = []

Repeat (outer loop):

1. Inner optimize current ansatz parameters θ
   → minimize E(θ) with Adam / GradientDescent / etc.

2. For each candidate operator A in pool:
   score g(A) = |∂E/∂θ_new| at θ_new = 0 (appended)

3. Append operator with largest g(A)

Stop if:
max g(A) < grad_tol  OR  len(selected_ops) == max_ops

```

#### Relationship to UCCSD and “fixed ansatz” VQE

- **Fixed ansatz VQE:** pick a circuit family (UCCSD, RY-CZ, HEA, …) and optimize its parameters.
- **ADAPT-VQE:** uses a pool (often UCC-like) but **selects only a subset** of operators, in an order chosen by gradient information.

In practice, this often yields:
- shallower circuits than “full UCCSD” at comparable accuracy (for small systems),
- more interpretable operator selections (which excitations mattered),
- a natural convergence diagnostic via the max-gradient stopping metric.

#### Notes on this implementation

- The excitation pool is generated from the same chemistry metadata used throughout the package (molecule registry + Hamiltonian builder).
- Noise handling and caching semantics match standard VQE:
  - noise parameters are canonicalized (non-effective noise does not pollute cache keys),
  - run records are hash-keyed by the full physical + numerical configuration.
- The solver records both:
  - outer-loop energies (post inner-optimization),
  - inner-loop energy trajectories per ADAPT iteration,
  - max-gradient history used for stopping,
  - the final ordered operator list and optimized parameters.

---

## Quantum Phase Estimation

The **Quantum Phase Estimation (QPE)** algorithm is a cornerstone of quantum computation for extracting eigenvalues of unitary operators.  
In the context of quantum chemistry, QPE can be used to determine the electronic ground-state energy of a molecule by estimating the eigenenergies of the time-evolution operator.

QPE is implemented for molecules defined in `common/molecules.py`, using the **same Hamiltonian pipeline as VQE**.  
This guarantees consistent chemistry and reproducible comparisons between VQE and QPE.
In contrast to VQE-based excited-state methods (SSVQE and VQD), QPE extracts eigenvalues directly via phase estimation, without variational optimization.

### QPE Background

For a given Hamiltonian $H$, we define the unitary operator:

$$U = e^{-i H t}$$

If $|\psi⟩$ is an eigenstate of $H$ with eigenvalue $E$, then:

$$U |\psi⟩ = e^{-iEt} |\psi⟩ = e^{2\pi i \theta} |\psi⟩,$$

where

$$\theta = -\frac{E t}{2\pi}.$$

The goal of QPE is to estimate the phase $\theta$, which directly encodes the **energy eigenvalue** through:

$$E = -\frac{2\pi \theta}{t}.$$

This approach differs fundamentally from VQE:
- **VQE**: Iteratively minimizes $⟨\psi(\theta)| H |\psi(\theta)⟩$ variationally.  
- **QPE**: Measures the eigenphase of $U = e^{-i H t}$ directly through interference.

### QPE Overview

QPE operates by coupling a register of $n$ qubits, which encodes the phase information, to a second register, which encodes the molecular information.

1. **Initialize**:
   - Prepare the second register in an approximate eigenstate, such as the Hartree–Fock state $|HF⟩$.
   - Initialize the first register in the state $|0⟩^{\otimes n}$.

2. **Create Superposition**:
   - Apply Hadamard gates to the first register to produce:

    $$\frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} |k⟩.$$

3. **Controlled Unitary Operations**:
   - Apply a sequence of controlled time-evolutions $U^{2^k}$, where each qubit in the first register controls a different power of $U$:

    $$\prod_{k=0}^{n-1} \text{C-}U^{2^k}.$$

   - These operations entangle the phase and molecular information.

4. **Inverse Quantum Fourier Transform (IQFT)**:
   - Apply the IQFT on the first register to convert the accumulated phase into a binary representation.

5. **Measurement**:
   - Measure the first register.  
   - The resulting bitstring corresponds to a binary fraction approximating the eigenphase $\theta$.

6. **Energy Recovery**:
   - The measured phase is converted to the molecular energy:
   
    $$E = -\frac{2\pi\theta}{t}.$$

In this implementation, each controlled-unitary block uses **trotterized time evolution**, with the number of Trotter steps configurable from the CLI or Python API.  
The initial state is the **Hartree–Fock reference** returned by the unified Hamiltonian pipeline; QPE performance depends on its overlap with the target eigenstate (HF is typically a good starting point for weakly correlated systems).

```
Ancilla register (phase qubits):    a₀  a₁  ...  a_{n-1}
System register (molecular state):  s₀  s₁  ...  s_{m-1}

1) INITIALIZATION:

Ancilla:  |0⟩^{⊗ n}  ──H──H── ... ──H──▶  (uniform superposition)
System:   |HF⟩       ────────────▶  approximate eigenstate of H

2) CONTROLLED TIME EVOLUTION:

For k = 0 .. n-1 (from least to most significant bit):

   a_k: ──●───────────────  applies   U^{2^k} = exp(-i H t 2^k)
           │
   sys:  U^{2^k}

Overall effect:
   Σ_k |k⟩_anc ⊗ U^k |HF⟩_sys
   → phase information e^{2π i θ k} encoded in ancillas

3) INVERSE QFT ON ANCIILA REGISTER:

   a₀: ── IQFT ──┐
   a₁: ──────────┼──▶ measurement → bitstring b_{n-1}…b₀
   ...           │
   a_{n-1}: ─────┘

Bitstring b ≈ binary fraction of phase θ:
   θ ≈ 0.b₁ b₂ … bₙ
Energy recovery:
   E ≈ -2π θ / t
```

#### Key Points

The inclusion of QPE in this project complements the variational studies by demonstrating:
- Exact phase–energy relationships
- Effects of decoherence on eigenvalue extraction
- Precision trade-offs between ancilla count, evolution time, and noise strength

---

## Quantum Imaginary Time Evolution

While VQE minimizes the energy directly via classical optimization, **imaginary-time evolution**
drives a state toward the ground state by evolving under a non-unitary propagator:

$$|\psi(\tau)\rangle \propto e^{-H\tau}|\psi(0)\rangle.$$

In practice, quantum hardware implements **unitary** circuits, so this project uses a
**variational** approximation to imaginary-time evolution via the **McLachlan variational principle**,
commonly referred to as **VarQITE**.

### Imaginary-time evolution and ground-state filtering

Let $H$ be a molecular Hamiltonian with eigenpairs $\{(E_k, |E_k\rangle)\}$.
Expanding the initial state as $|\psi(0)\rangle=\sum_k c_k |E_k\rangle$, we have

$$e^{-H\tau}|\psi(0)\rangle=\sum_k c_k e^{-E_k\tau}|E_k\rangle.$$

After normalization, higher-energy components are exponentially suppressed, so the state
approaches the ground state provided $c_0\neq 0$.

This is the core “ground-state filtering” mechanism that motivates QITE-style algorithms.

### McLachlan variational principle (VarQITE update rule)

We restrict dynamics to a parameterized family of states $|\psi(\theta)\rangle$
generated by an ansatz circuit. VarQITE chooses parameter updates so that the
ansatz trajectory best matches the imaginary-time flow in a least-squares sense:

$$\delta \left\| \left(\frac{d}{d\tau} + H - \langle H\rangle\right)|\psi(\theta)\rangle \right\| = 0.$$

With the tangent vectors $|\partial_i\psi\rangle = \partial |\psi(\theta)\rangle / \partial \theta_i$,
this yields a linear system for the parameter velocity $\dot{\theta}$:

$$A(\theta)\,\dot{\theta} = -C(\theta),$$

where the (real) matrix $A$ and vector $C$ are

$$A_{ij}=\Re\langle \partial_i\psi|\partial_j\psi\rangle, \qquad
C_i=\Re\langle \partial_i\psi|(H-\langle H\rangle)|\psi\rangle.$$

A discrete update with step size $\Delta\tau$ is then

$$\theta \leftarrow \theta + \Delta\tau\,\dot{\theta}.$$

**Implementation note.**
This project supports multiple numerical solvers for the linear system (direct solve, least-squares, pseudo-inverse),
with optional regularization to stabilize ill-conditioned $A$.

### Relationship to VQE and QPE in this repository

- **VQE**: direct energy minimization $E(\theta)=\langle \psi(\theta)|H|\psi(\theta)\rangle$ via classical optimization.
- **VarQITE**: parameter flow approximating imaginary-time evolution using a tangent-space linear solve.
- **QPE**: eigenvalue extraction via phase estimation of $e^{-iHt}$ (no variational optimization).

All three methods share a unified Hamiltonian and molecule source through the `common` layer,
ensuring that cross-method comparisons are chemically consistent.

---

## Noise Types

This project models two primary noise channels — **depolarizing** and **amplitude damping** — using PennyLane’s `default.mixed` backend.

Both VQE and QPE support these channels:
- VQE applies noise layer-by-layer inside the ansatz.
- QPE applies noise after each controlled-unitary evolution using `qpe.noise.apply_noise_all`.

**VarQITE (QITE)** parameter updates are **noiseless by design** (pure-state McLachlan update).  
Noise is supported only for **post-evaluation** of converged parameters (density-matrix expectation values on `default.mixed` via `qite eval-noise`).

```
NOISE IN SIMULATIONS
====================

Circuit execution with noise:

   ideal gates
       ↓
   apply noise channel(s)
       ↓
   next layer of gates
       ↓
   apply noise channel(s)
       ↓
   ...

Noise effects studied:
   • depolarizing noise (symmetric errors)
   • amplitude damping (relaxation toward |0⟩)

Used in:
   • VQE (after each ansatz layer)
   • QPE (after each controlled evolution step)
   • QITE / VarQITE (post-evaluation only; not during parameter updates)
```

For excited-state methods (SSVQE and VQD), noise is handled consistently by computing overlap penalties using density-matrix inner products rather than statevector overlaps.

### Depolarizing Noise

Models random qubit errors that drive each subsystem toward a mixed state with probability $p_{\text{dep}}$:

$$\mathcal{E}_{\text{dep}}(\rho) = (1 - p_{\text{dep}}) \rho + \frac{p_{\text{dep}}}{3}(X \rho X + Y \rho Y + Z \rho Z)$$

- Represents uniform gate and readout errors
- Applied independently to each qubit
- Causes global decoherence and loss of entanglement fidelity

### Amplitude Damping

Models **energy relaxation**, where excited states decay to the ground state $|0⟩$ with probability $p_{\text{amp}}$:

$$\mathcal{E}_{\text{amp}}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger$$

where

$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-p_{\text{amp}}} \end{pmatrix},
\quad
E_1 = \begin{pmatrix} 0 & \sqrt{p_{\text{amp}}} \\ 0 & 0 \end{pmatrix}$$

- Mimics spontaneous emission or thermal relaxation
- Applied independently to each qubit
- Introduces asymmetric noise and energy bias toward the ground state

### Evaluation Metrics

Noise strengths ($p_{\text{dep}}, p_{\text{amp}} \in [0, 0.1]$) are varied systematically to evaluate:

- **Energy error** — deviation from the noiseless ground-state energy
- **Fidelity** — overlap between noisy and noiseless final states: $F(|\psi_0⟩, \rho) = ⟨\psi_0| \rho | \psi_0⟩$

These metrics quantify the robustness of each **ansatz** and **optimizer** against realistic, per-qubit noise processes, independent of molecular size or qubit count.

---

## References

**Foundations**
- **Variational Quantum Eigensolver (VQE)** — overview  
  https://en.wikipedia.org/wiki/Variational_quantum_eigensolver  
- **Quantum Phase Estimation (QPE)** — overview  
  https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm

**Imaginary-Time / VarQITE**
- McLachlan, *A variational solution of the time-dependent Schrödinger equation* (1964).
- Yuan et al., *Theory of variational quantum simulation* (for McLachlan-based real/imaginary-time variational evolution).

**ADAPT-VQE**
- Grimsley et al., *An adaptive variational algorithm for exact molecular simulations on a quantum computer* (ADAPT-VQE, Nature Communications, 2019).

**Quantum Chemistry**
- **Hartree–Fock Method** — overview  
  https://en.wikipedia.org/wiki/Hartree–Fock_method  
- Seeley et al., *Fermion-to-Qubit Mappings* (Bravyi–Kitaev, JW)  
  https://arxiv.org/abs/1701.08213  
- Aspuru-Guzik et al., *Simulated Quantum Computation of Molecular Energies*  
  https://doi.org/10.1126/science.1113479  

**VQE Theory & Reviews**
- McArdle et al., *Quantum Computational Chemistry* (VQE review)  
  https://arxiv.org/abs/2001.03685  

**Quantum Algorithms**
- Kitaev, *Quantum Measurements and the Abelian Stabilizer Problem*  
  https://arxiv.org/abs/quant-ph/9511026  

**PennyLane Documentation**
- **Templates & Ansatzes**  
  https://docs.pennylane.ai/en/stable/code/qml.html  
- **Optimizers & Interfaces**  
  https://docs.pennylane.ai/en/stable/introduction/interfaces.html  

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
