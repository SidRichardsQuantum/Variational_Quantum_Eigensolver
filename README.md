# Variational Quantum Eigensolver

This project implements a simulation of the **Variational Quantum Eigensolver (VQE)** algorithm for Lithium Hydride (LiH) and Water (H₂O) molecule using **PennyLane** and **Python**. The simulation demonstrates the calculation of the ground state energy and visualizes the optimized quantum state.

---

## Project Overview

The VQE is a hybrid quantum-classical algorithm used to estimate the ground state energy of molecular Hamiltonians. In this project:
- We model the molecule.
- We construct the molecular Hamiltonian.
- We define an ansatz using single and double excitations.
- We visualize both the energy convergence and the final quantum state.

We implement both noiseless and noisy runs on simulated quantum devices.

---

## Key Components

### Technologies Used
- Python
- PennyLane
- PennyLane.qchem
- Matplotlib
- NumPy

### Methodology
1. **Molecular Setup**
    - Atoms are positioned relative to each other in Angstroms (Å).
    - Molecular Hamiltonian is generated using PennyLane's `qchem` module.

2. **Hartree-Fock State**
    - The initial reference state is constructed with electrons in spin orbitals.

3. **Ansatz Construction**
    - The ansatz includes single and double excitations derived from the Hartree-Fock reference state.

4. **VQE Optimization**
    - Classical gradient descent optimization over 50 iterations.
    - The cost function minimizes the expectation value of the Hamiltonian.

5. **Result Visualization**
    - Energy convergence plot over iterations.
    - Bar plot showing significant amplitudes of the final quantum ground state.
    - Final quantum state expressed in ket notation.

---

## Files

- Jupyter notebooks containing the full simulation code, parameter optimization, and result visualization.
- ```images``` directory containing bar plots of final ground state amplitudes.

---

## Usage

### Prerequisites
Make sure the following Python packages are installed:
```bash
pip install pennylane pennylane-qchem matplotlib numpy

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
