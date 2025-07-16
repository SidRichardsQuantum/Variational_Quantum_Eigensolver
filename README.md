# Variational Quantum Eigensolver

This project implements a simulation of the **Variational Quantum Eigensolver (VQE)** algorithm for molecular systems such as **Lithium Hydride (LiH)** and **Water (H₂O)** using **PennyLane** and **Python**.
We demonstrate noisy and noiseless ground state energy calculations and eigenstate visualizations, using different ansatz approaches and optimisers.
The optimum bond-length of LiH and bond-angle of water are found, by using the VQE to calculate the ground state akin to molecules with variable geometry.

For detailed theoretical explanations, see [THEORY.md](THEORY.md).

---

## Project Overview

The VQE is a hybrid quantum-classical algorithm designed to find the ground state energy of molecular Hamiltonians on quantum devices.
This repository showcases:

- **Multiple molecular systems**: LiH, H₂O, H₂, etc...
- **Different ansatz approaches**: Single and/or double excitations
- **Hamiltonian construction**: Using quantum chemistry methods
- **Parameterized quantum circuits**: Excitation-based ansatze
- **Optimal geometry**: Bond-lengths and angles
- **Classical optimization**: Comparing different optimizers (Gradient Descent, Adam)
- **Comprehensive visualization**: Convergence and eigenstate plots

### Molecules Studied

- **Dihydrogen (H₂)**: Simplest molecule, pair of protons with a pair of electrons
  - 4 qubits required
  - Ansatz: Single + Double excitations (UCCSD)
  - Optimizer: Adam

- **Lithium Hydride (LiH)**: Simple molecule with 4 electrons
  - 12 qubits required
  - Ansatz: Double excitations only
  - Optimizers: Gradient Descent, Adam
  - VQE is ran for a range bond-lengths to find the optimum
  
- **Water (H₂O)**: Bent molecular geometry with 10 electrons
  - 14 qubits required
  - Ansatz: Single + Double excitations (UCCSD)
  - Optimizer: Adam
  - VQE is ran for a range bond-angles to find the optimum

### Optimization & Analysis

- Hartree-Fock state initialization for reference
- Gradient-based parameter optimization
- Energy convergence plots
- Quantum state amplitude plots and analysis
- Comparison with exact results

## Technologies Used

- **Python 3.8+**
- **[PennyLane](https://pennylane.ai/)**: Quantum machine learning library
- **[PennyLane-qchem](https://pennylane.ai/qml/demos/tutorial_qchem.html)**: Quantum chemistry extension
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **SciPy**: Scientific computing utilities

## Project Structure

```
variational_quantum_eigensolver/
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT license
├── README.md                 # This file 
├── THEORY.md                 # Detailed theoretical background
├── RESULTS.md                # Results and analysis
├── notebooks/                # Jupyter notebooks written in Python
│   ├── H2_Noiseless.ipynb    # Noiseless H₂ VQE implementation
│   ├── LiH_Noiseless.ipynb   # Noiseless LiH VQE implementation
│   ├── LiH_Bond_Length.ipynb  # Optimum bond-length of LiH
│   ├── H2O_Noiseless.ipynb   # Noiseless H₂O VQE implementation
│   └── H2O_Bond_Angle.ipynb   # Optimum bond-angle of H₂O
└── images/                   # Generated visualization plots
    ├── LiH_ground_state.png  # LiH ground state amplitude plot
    ├── H2O_ground_state.png  # H₂O ground state amplitude plot
    └── H2_ground_state.png    # H₂ ground state amplitude plot
```

## Usage

### Quick Start

1. Clone the repository:

```bash
git clone https://github.com/SidRichardsQuantum/vqe-project.git
cd variational_quantum_eigensolver
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the VQE simulations:

```bash
# Lithium Hydride
jupyter notebook LiH_Noiseless.ipynb

# Water
jupyter notebook H2O_Noiseless.ipynb
```

## Methodology

### 1. Molecular Setup

- Atomic positions specified in Angstroms (Å)
- Molecular Hamiltonian generated using PennyLane's `qchem` module
- Active space selection for computational tractability

### 2. Quantum State Preparation

- Hartree-Fock reference state as initial guess
- Jordan-Wigner transformation maps fermions to qubits
- State preparation circuit constructs the reference state

### 3. Ansatz Construction

Two different ansatz approaches are demonstrated:

- **Single excitations**: Allow orbital relaxation and electron correlation
- **Double excitations**: Capture dynamic correlation effects
- Combined ansatz provides more complete treatment of electron correlation

### 4. VQE Optimization

Different optimization approaches for each molecule:

#### LiH Optimization

- **Parameters**: 72 double excitation angles
- **Cost Function**: ⟨ψ(θ)| H |ψ(θ)⟩
- **Optimisers**: Gradient Descent, Adam
- **Convergence**: Energy tracked at each iteration for each optimiser

#### H₂ and H₂O Optimization

- **Parameters**: Single + double excitation angles
- **Cost Function**: ⟨ψ(θ_s, θ_d)| H |ψ(θ_s, θ_d)⟩
- **Optimizer**: Adam with $0.1$ step size
- **Convergence**: Energy tracked at each iteration

### 5. Analysis & Visualization

- Energy convergence plots
- Quantum state amplitude distributions
- Bond-lengths and angles against ground state energies

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
