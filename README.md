# Variational Quantum Eigensolver

This project implements a simulation of the **Variational Quantum Eigensolver (VQE)** algorithm for molecular systems including **Lithium Hydride (LiH)** and **Water (H₂O)** using **PennyLane** and **Python**. The simulations demonstrate ground state energy calculations and quantum state visualizations using different ansatz approaches.

---

## Project Overview

The VQE is a hybrid quantum-classical algorithm designed to find the ground state energy of molecular Hamiltonians on quantum devices. This repository showcases:

- **Two molecular systems**: LiH (4 electrons) and H₂O (10 electrons)
- **Different ansatz approaches**: Single and/or double excitations
- **Hamiltonian construction** using quantum chemistry methods
- **Parameterized quantum circuits** with excitation-based ansatze
- **Classical optimization** with different optimizers (Gradient Descent, Adam)
- **Comprehensive visualization** of results and quantum states

For detailed theoretical explanations, see [THEORY.md](THEORY.md)

### Molecules Studied

- **Lithium Hydride (LiH)**: Simple ionic molecule with 4 electrons
  - 12 qubits required
  - Ansatz: Double excitations only (72 parameters)
  - Optimizer: Gradient Descent
  
- **Water (H₂O)**: Bent molecular geometry with 10 electrons
  - 14 qubits required
  - Ansatz: Single + Double excitations (UCCSD)
  - Optimizer: Adam

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
vqe-project/
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT license
├── README.md                    # This file 
├── THEORY.md                    # Detailed theoretical background
├── RESULTS.md                   # Results and analysis
├── notebooks/                   # Jupyter notebooks written in Python
│   ├── LiH_VQE_Noiseless.ipynb  # Noiseless LiH VQE implementation
│   └── H20_VQE_Noiseless.ipynb  # Noiseless H₂O VQE implementation
└── images/                      # Generated visualization plots
    ├── LiH_ground_state.png     # LiH ground state amplitude plot
    └── H2O_ground_state.png     # H₂O ground state amplitude plot
```

## Usage

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/SidRichardsQuantum/vqe-project.git
cd vqe-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the VQE simulations:
```bash
# Lithium Hydride
jupyter notebook LiH_VQE_Noiseless.ipynb

# Water
jupyter notebook H2O_VQE_Noiseless.ipynb
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

#### LiH: Double Excitation Ansatz
- **72 parameters** for double excitation gates only
- Hartree-Fock reference state: |111100000000⟩
- Parameterized DoubleExcitation gates with optimizable angles

#### H₂O: UCCSD Ansatz (Singles + Doubles)
- **Single excitations**: Allow orbital relaxation and electron correlation
- **Double excitations**: Capture dynamic correlation effects
- Combined ansatz provides more complete treatment of electron correlation

### 4. VQE Optimization
Different optimization approaches for each molecule:

#### LiH Optimization
- **Optimizer**: Gradient Descent with 0.1 step size
- **Cost Function**: ⟨ψ(θ)|H|ψ(θ)⟩
- **Parameters**: 72 double excitation angles

#### H₂O Optimization  
- **Optimizer**: Adam with 0.1 step size
- **Cost Function**: ⟨ψ(θ_s, θ_d)|H|ψ(θ_s, θ_d)⟩
- **Parameters**: Single + double excitation angles
- **Convergence**: Energy tracked at each iteration

### 5. Analysis & Visualization
- Energy convergence plots
- Quantum state amplitude distributions
- Comparison with classical methods
- Error analysis and statistical metrics

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
