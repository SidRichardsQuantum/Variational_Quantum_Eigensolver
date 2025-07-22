# Variational Quantum Eigensolver

This project implements a simulation of the **Variational Quantum Eigensolver (VQE)** algorithm for molecular systems such as **Lithium Hydride (LiH)** and **Water (H₂O)** using **PennyLane** and **Python**.
We demonstrate noiseless ground state energy calculations and eigenstate visualizations, using different ansätze approaches and optimisers.
The optimum bond-length of LiH and bond-angle of water are found, by using the VQE to calculate the ground state akin to molecules with variable geometry.

For detailed theoretical explanations, see [THEORY.md](THEORY.md).

To read our main findings and results, see [RESULTS.md](RESULTS.md).

---

## Project Overview

VQE is a hybrid quantum-classical algorithm used to solve quantum chemistry problems.
This project implements VQE for:

- **H₂ (Dihydrogen)**: Optimizer and ansätze benchmarking
- **LiH (Lithium Hydride)**: Ground-state energy across bond lengths  
- **H₂O (Water)**: Energy variation with bond angle

## Technologies Used

- **Python 3.8+**
- **[PennyLane](https://pennylane.ai/)**: Quantum machine learning library
- **[PennyLane-qchem](https://pennylane.ai/qml/demos/tutorial_qchem.html)**: Quantum chemistry extension
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization

## Project Structure

```
variational_quantum_eigensolver/
├── LICENSE                         # MIT license
├── requirements.txt                # Dependencies
├── README.md                       # This file 
├── THEORY.md                       # Detailed theoretical background
├── RESULTS.md                      # Results and analysis
└── notebooks/                      # Jupyter notebooks written in Python
    ├── images/                     # Directory of generated visualization plots
    ├── H2_Noiseless.ipynb          # Noiseless H₂ VQE implementation
    ├── H2_Ansatz_Comparison.ipynb  # Comparing ansätze for H₂
    ├── LiH_Noiseless.ipynb         # Noiseless LiH VQE implementation
    ├── LiH_Bond_Length.ipynb       # Optimum bond-length of LiH
    ├── H2O_Noiseless.ipynb         # Noiseless H₂O VQE implementation
    └── H2O_Bond_Angle.ipynb        # Optimum bond-angle of H₂O
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
pip install pennylane matplotlib numpy scipy
```

3. Run the VQE simulations:

```bash
# Lithium Hydride
jupyter notebook LiH_Noiseless.ipynb
```

## Methodology Overview

1. Molecular Setup: generated using PennyLane's `qchem` module

2. Quantum State Preparation:

- Hartree-Fock reference state as initial guess
- Jordan-Wigner transformation maps fermions to qubits

3. Ansätze Construction: Single and double excitaions to maximise electron-electron correlation effects

4. VQE Optimization:

- Different optimisers Gradient Descent, Adam, etc...
- Energy tracked at each iteration

5. Analysis & Visualization

- Energy convergence plots
- Quantum state amplitude distributions
- Bond-lengths and angles against ground state energies

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
