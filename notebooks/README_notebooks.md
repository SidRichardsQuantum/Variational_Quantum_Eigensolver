# Variational Quantum Eigensolver

This project implements a simulation of the **Variational Quantum Eigensolver (VQE)** algorithm for molecular systems such as **Dihydrogen (H₂)**, **Lithium Hydride (LiH)**, **Water (H₂O)**, and the **Trihydrogen Cation (H₃⁺)** using **PennyLane** and **Python**.  
We demonstrate **noiseless ground state energy calculations** and **eigenstate visualizations**, using different **ansatzes** approaches, classical **optimisers** and **qubit mappings**.
The optimum bond-length of LiH and bond-angle of water are determined, and excitation strategies for H₃⁺ are evaluated.

For detailed theoretical explanations, see [THEORY.md](THEORY.md).

To read our main findings and results, see [RESULTS.md](RESULTS.md).

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology Overview](#methodology-overview)
- [Quantum Phase Estimation](#quantum-phase-estimation)

---

## Project Overview

VQE is a hybrid quantum-classical algorithm used to solve quantum chemistry problems.
This project implements VQE for:

- **H₂ (Dihydrogen)**: Optimizer and ansatzes benchmarking
- **H₃⁺ (Trihydrogen Cation)**: Excitation, mapping comparisons and Subspace-Search VQE
- **LiH (Lithium Hydride)**: Ground-state energy across bond lengths  
- **H₂O (Water)**: Energy variation with bond angle

## Technologies Used

- **Python 3.10+**
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **[PennyLane](https://pennylane.ai/)**: Quantum machine learning library
- **[PennyLane-qchem](https://pennylane.ai/qml/demos/tutorial_qchem.html)**: Quantum chemistry extension
- Scientific references: [arXiv papers](https://arxiv.org/search/?query=variational+quantum+eigensolver&searchtype=all)

## Project Structure

```
Variational_Quantum_Eigensolver/
├── LICENSE           # MIT license
├── README.md         # This file
├── THEORY.md         # Theoretical background and mathematical formulation
├── RESULTS.md        # Consolidated results and analysis
├── pyproject.toml    # For packaging
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
│
├── vqe/                  # Packaged VQE implementation
│   ├── __init__.py
│   ├── core.py
│   ├── engine.py
│   ├── optimizer.py
│   ├── io_utils.py
│   ├── visualize.py
│   ├── ansatz.py
│   ├── ssvqe.py
│   └── images/           # (May be temporary)
│
├── qpe/                  # (In progress) Packaged QPE implementation
│   └── __init__.py
│
├── notebooks/            # Jupyter notebooks for molecule-specific studies
│   ├── vqe/
│   │   ├── H2/           # H₂ simulations (noisy, noiseless, ansatz comparison, etc.)
│   │   ├── H2O/          # H₂O simulations (bond angle, noiseless runs)
│   │   ├── H3plus/       # H₃⁺ simulations (mappings, SSVQE, noise analysis)
│   │   ├── LiH/          # LiH simulations (bond length, noiseless runs)
│   │   └── vqe_utils.py  # Core VQE utility functions
│   └── qpe/
│       ├── H2/
│       └── qpe_utils.py  # Core QPE utility functions
│
├── data/                 # Stored numerical results and generated plots
│   ├── vqe/
│   │   ├── results/      # Saved numerical and energy JSON files
│   │   └── images/       # Generated visualization plots
│   └── qpe/
│       ├── results/      # Saved numerical and energy JSON files
│       └── images/       # Generated visualization plots
│
├── package_tests/        # Unit and reproducibility tests
│   ├── test_reproducibility.py
│   └── test_ssvqe_general.py
│
└── package_ results/      # Cached packaged output (JSON experiment records)
```

## Usage

### Quick Start

1. Clone the repository:

```bash
git clone https://github.com/SidRichardsQuantum/variational_quantum_eigensolver.git
cd variational_quantum_eigensolver
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the VQE simulations:

```bash
# Lithium Hydride (LiH)
jupyter notebook notebooks/vqe/LiH/LiH_Noiseless.ipynb

# Runs a noiseless VQE with a double–excitation UCC ansatz and the gradient descent optimizer
# Outputs convergence plots and the final ground state amplitudes
```

### Results Preview

Running `notebooks/vqe/LiH/LiH_Noiseless.ipynb` produces:

- **Convergence of VQE energy** (noiseless, LiH, double excitation ansatz)
- **Final ground state energy** close to the expected Hartree–Fock reference
- **Ground state amplitudes** plotted as a bar chart

![LiH VQE Convergence](/data/vqe/images/LiH_GradientDescent.png)

![LiH Ground State](/data/vqe/images/LiH_GroundState_UCCSD_Adam_s0.png)

## Methodology Overview

1. Molecular Setup: generated using PennyLane's `qchem` module

2. Quantum State Preparation:

- Hartree-Fock reference state as initial guess
- Jordan-Wigner transformation maps fermions to qubits
- Mapping comparison for H₃⁺

3. Ansatzes Construction:

- Single and/or double excitation circuits
- Ansatzes comparison for H₂
- Full excitation comparison analysis for H₃⁺

4. VQE Optimization:

- Classical optimisers such as Adam, Gradient Descent, Nesterov, SPSA, etc...
- Energy tracked at each iteration
- Different optimisers compared for H₂

5. Analysis & Visualization

- Energy convergence plots
- Quantum state amplitude distributions
- Quantum circuit diagrams
- Bond-lengths or angles against ground state energies

## Quantum Phase Estimation

The **Quantum Phase Estimation (QPE)** algorithm is implemented in this project as a complementary approach to the VQE.  
While VQE variationally minimizes the energy using hybrid quantum–classical optimization, QPE directly extracts eigenenergies from the phase of the unitary time-evolution operator  
$U = e^{-iHt}$.

This implementation includes:

- **Noiseless and noisy simulations** of H₂  
- **Parameter sweeps** over evolution time and ancilla qubit count  
- **Noise models** for depolarizing and amplitude damping channels  
- **Caching** of results and figures for reproducibility  
- **Phase-to-energy reconstruction** with automatic aliasing correction

Example notebooks:

- `notebooks/qpe/H2/H2_QPE_Noiseless.ipynb`
- `notebooks/qpe/H2/H2_QPE_Noisy.ipynb`

Output plots (saved in `/data/qpe/images/`) visualize ancilla distributions and how measured energies depend on evolution time or noise level.

![H₂ QPE Distribution](/data/qpe/images/H2_QPE_NoiseDep_PeakMeanStd_s0.png)

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
