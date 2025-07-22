# 🧠 VQE Theory & Methodology

This document provides a detailed explanation of the **Variational Quantum Eigensolver (VQE)**, the **molecules**, **ansätze**, and **optimizers** used in this project.

---

## 📚 Table of Contents

- [Molecules Studied](#molecules-studied)
- [Background](#background)
- [VQE Algorithm Overview](#vqe-algorithm-overview)
- [Ansätz Construction](#ansätz-construction)
- [Optimizers](#optimizers)
- [References](#references)

---

## Molecules Studied:

| Molecule |   Properties Scanned  | Qubits Required |
|----------|-----------------------|-----------------|
|    H₂    |   Ansätz & Optimizer  |       $4$       |
|    LiH   | Bond length variation |       $12$      |
|    H₂O   | Bond angle variation  |       $14$      |

All simulations use the **STO-3G** basis set for consistency.
Molecular Hamiltonians are constructed using **second quantization** and mapped to qubit operators via the **Jordan-Wigner** transformation (via PennyLane's `qchem` module).

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

### Ansätz Construction:

An ansätze defines the functional form of the trial quantum state $|\psi(\theta)⟩$.
It determines how expressive, efficient, and trainable your VQE circuit is.
Different ansätze trade off physical accuracy, circuit depth, and compatibility with quantum hardware.

#### UCCSD (Unitary Coupled Cluster Singles and Doubles)

A chemistry-inspired ansätze derived from coupled-cluster theory. Includes single and double excitations applied in a unitary, Trotterized form.

- Designed for capturing electron correlation from first principles
- Exact for small systems like H₂ in minimal basis sets (e.g., STO-3G)
- Can become deep and resource-intensive for larger molecules

#### $R_Y-C_Z$ Ansätz

A hardware-efficient ansätze composed of layers alternating single-qubit rotations and entangling gates.

- Uses $R_Y$ rotations followed by a chain of $C_Z$ gates
- Tunable number of layers (depth)
- Good expressibility for small and medium systems
- Easier to implement on near-term hardware

#### Minimal / One-Parameter Ansätz

A manually constructed, problem-specific ansätze using very few parameters.

- Tailored for simple systems like H₂ in minimal basis
- Uses a single $R_Y$ rotation and one entangling gate (e.g., CNOT)
- Extremely shallow and interpretable
- Useful for testing optimizers, energy landscapes, or learning curves

### Optimizers:

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

## References

- [VQE](https://en.wikipedia.org/wiki/Variational_quantum_eigensolver)
- [Hartree-Fock Method](https://en.wikipedia.org/wiki/Hartree–Fock_method)
- [Ansätzes](https://docs.pennylane.ai/en/stable/code/qml.html)
- [Optimisers](https://docs.pennylane.ai/en/stable/introduction/interfaces.html)

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
