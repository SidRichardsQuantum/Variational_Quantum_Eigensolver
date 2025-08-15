# 🧠 VQE Theory & Methodology

This document provides a detailed explanation of the **Variational Quantum Eigensolver (VQE)**, the **molecules**, **ansatzes**, and **optimizers** used in this project.

---

## 📚 Table of Contents

- [Molecules Studied](#molecules-studied)
- [Background](#background)
- [VQE Algorithm Overview](#vqe-algorithm-overview)
- [Ansatz Construction](#ansatz-construction)
- [Optimizers](#optimizers)
- [Fermion-to-Qubit Mappings](#fermion-to-qubit-mappings)
- [Excited State Methods in VQE](#excited-state-methods-in-vqe)
- [References](#references)

---

## Molecules Studied:

| Molecule |     Properties Scanned / Benchmarked     | Qubits Required |
|----------|------------------------------------------|-----------------|
|    H₂    |       Ansatz & Optimizer Comparison      |       $4$       |
|    LiH   |              Bond length scan            |       $12$      |
|    H₂O   |              Bond angle scan             |       $14$      |
|    H₃⁺   | Excitation & Mapping Comparisons & SSVQE |       $6$       |


All simulations use the **STO-3G** basis set for consistency.  
Molecular Hamiltonians are constructed using **second quantization** and mapped to qubit operators via the Jordan–Wigner, Bravyi–Kitaev or Parity transformations (via PennyLane's `qchem` module).

## Background

### Variational Principle

The variational principle states that for any trial wavefunction $|\psi⟩$, the expectation value of the Hamiltonian is an upper bound to the true ground state energy:
```
E₀ ≤ ⟨ψ|H|ψ⟩
```
where:
- $E_0$ is the ground state energy
- $|\psi⟩$ is any normalized wavefunction
- $H$ is the Hamiltonian

### Hartree-Fock

The Hartree-Fock state is written as the tensor product of a qubit state $\phi$ for every electron orbital:
```
|HF⟩ = |φ₁φ₂...φₙ⟩
```
For LiH with $4$ electrons in $12$ orbitals, the HF reference state is ```|111100000000⟩```, which describes electrons occupying the four lowest energy orbital states.

## VQE Algorithm Overview

The VQE algorithm consists of:
1. **State Preparation**: Prepare parameterized quantum state $|\psi(\theta)⟩$
2. **Measurement**: Measure expectation value $⟨\psi(\theta)| H |\psi(\theta)⟩$
3. **Optimization**: Classically optimize parameters $\theta$ to minimize energy
4. **Iteration**: Repeat until convergence

---

## Ansatz Construction:

An ansatz defines the functional form of the trial quantum state $|\psi(\theta)⟩$.
It determines how expressive, efficient, and trainable your VQE circuit is.
Different ansatze trade off physical accuracy, circuit depth, and compatibility with quantum hardware.

#### UCCSD (Unitary Coupled Cluster Singles and Doubles)

A chemistry-inspired ansatz derived from coupled-cluster theory. Includes single and double excitations applied in a unitary, Trotterized form.

- Designed for capturing electron correlation from first principles
- Exact for small systems like H₂ or H₃⁺ in minimal basis sets (e.g., STO-3G)
- Used to compare excitation types (single vs. double vs. UCCSD) in **H₃⁺**

#### $R_Y-C_Z$ Ansatz

A hardware-efficient ansatz composed of layers alternating single-qubit rotations and entangling gates.

- Uses $R_Y$ rotations followed by a chain of $C_Z$ gates
- Tunable number of layers (depth)
- Good expressibility for small and medium systems
- Easier to implement on near-term hardware

#### Minimal / One-Parameter Ansatz

A manually constructed, problem-specific ansatz using very few parameters.

- Tailored for simple systems like H₂ in minimal basis
- Uses a single $R_Y$ rotation and one entangling gate (e.g., CNOT)
- Extremely shallow and interpretable
- Useful for testing optimizers, energy landscapes, or learning curves

---

## Optimizers:

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

## Fermion-to-Qubit Mappings

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

## Excited State Methods in VQE

While the standard VQE algorithm is designed to find the **ground state** of a molecular Hamiltonian, many applications in quantum chemistry require access to **excited states** — for example, to predict **spectroscopic transitions**, **photoexcitation energies**, and **reaction pathways**.

### Challenge

The original VQE formulation finds the **lowest eigenvalue** of the Hamiltonian by variationally minimizing the energy:

$E_0 = \min_{\theta} ⟨\psi(\theta)| H |\psi(\theta)⟩$

This process does not directly provide excited states, and repeating VQE with orthogonality constraints is non-trivial.

### Subspace-Search VQE (SSVQE)

SSVQE is a variational method that finds **multiple eigenstates simultaneously** by:

1. Preparing a **set of parameterized quantum states**:

$\{ |\psi_0(\theta_0)⟩, \ |\psi_1(\theta_1)⟩, \ \dots \}$

2. **Optimizing** all parameters to minimize a **weighted sum of expectation values**:

$\mathcal{L} = \sum_i w_i ⟨\psi_i| H |\psi_i⟩$

3. Adding **orthogonality penalties** to ensure distinct states:

$\text{Penalty} \propto | ⟨\psi_i | \psi_j⟩ |^2$

This enforces that each optimized state corresponds to a different eigenvector of the Hamiltonian.

### Implementation Details for H₃⁺

- **Ansatz**: UCCSD with both single and double excitations.
- **States**: Two independent parameter sets (ψ₀ and ψ₁) initialized differently.
- **Penalty Term**: Proportional to $| ⟨\psi_0|\psi_1⟩ |^2$ with a tunable multiplier.
- **Optimizer**: Adam, step size tuned for stability and separation.
- **Outcome**: Variational estimates of both the ground-state and first excited-state energies, with an accurate excitation gap.

### Key Points

- Allows **simultaneous** calculation of ground and excited states.
- Optimization can become more challenging as the number of states increases.
- Choice of ansatz and penalty strength critically affects convergence and state separation.

---

## References

- [VQE](https://en.wikipedia.org/wiki/Variational_quantum_eigensolver)
- [Hartree-Fock Method](https://en.wikipedia.org/wiki/Hartree–Fock_method)
- [Ansatzes](https://docs.pennylane.ai/en/stable/code/qml.html)
- [Optimisers](https://docs.pennylane.ai/en/stable/introduction/interfaces.html)
- [Quantum Chemistry with Fermion-to-Qubit Mappings](https://arxiv.org/abs/1701.08213)
- [Variational Quantum Eigensolver Review](https://arxiv.org/abs/2001.03685)

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
