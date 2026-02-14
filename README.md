# Quantum Simulation Suite — VQE + ADAPT-VQE + LR-VQE + QSE + SSVQE + VQD + QPE + QITE (PennyLane)

<p align="center">

  <a href="https://pypi.org/project/vqe-pennylane/">
    <img src="https://img.shields.io/pypi/v/vqe-pennylane?style=flat-square" alt="PyPI Version">
  </a>

  <a href="https://pypi.org/project/vqe-pennylane/">
    <img src="https://img.shields.io/pypi/dm/vqe-pennylane?style=flat-square" alt="PyPI Downloads">
  </a>

  <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/actions/workflows/tests.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Variational_Quantum_Eigensolver/tests.yml?label=tests&style=flat-square" alt="Tests">
  </a>

  <img src="https://img.shields.io/pypi/pyversions/vqe-pennylane?style=flat-square" alt="Python Versions">

  <img src="https://img.shields.io/github/license/SidRichardsQuantum/Variational_Quantum_Eigensolver?style=flat-square" alt="License">

</p>

A modern, modular, and fully reproducible **quantum-chemistry simulation suite** built on **PennyLane**, featuring:

- **Variational Quantum Eigensolver (VQE)** (ground state)
- **Linear-Response VQE (LR-VQE)** (post-VQE tangent-space excited states via TDA)
- **Quantum Subspace Expansion (QSE)** (post-VQE operator-subspace spectrum around a reference state)
- **Subspace-Search VQE (SSVQE)** (multiple low-lying states, subspace objective)
- **Variational Quantum Deflation (VQD)** (excited states via deflation)
- **Adaptive Derivative-Assembled Pseudo-Trotter VQE (ADAPT-VQE)** (adaptive ansatz growth)
- **Quantum Phase Estimation (QPE)** (phase-based energy estimation)
- **Quantum Imaginary Time Evolution (QITE / VarQITE)** (imaginary-time ground-state filtering via McLachlan updates)
- **Unified molecule registry, geometry generators, and plotting tools**
- **Consistent caching and reproducibility across all solvers**

This project refactors all previous notebooks into a clean, versioned Python package with a shared `common/` layer for Hamiltonians, molecules, geometry, plotting, paths, and persistence.

## How to get started

- For the background and derivations: see [THEORY.md](THEORY.md)
- For CLI usage and automation: see [USAGE.md](USAGE.md)
- For example notebooks: see [notebooks/README_notebooks.md](notebooks/README_notebooks.md)

These documents complement this `README.md` and provide the theoretical foundation and hands-on execution details.

---

## Project Structure

```

Variational_Quantum_Eigensolver/
├── README.md
├── THEORY.md
├── USAGE.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── vqe/                     # VQE package
│   ├── __main__.py          # CLI: python -m vqe
│   ├── core.py              # VQE orchestration (runs, scans, sweeps)
│   ├── engine.py            # Devices, noise, ansatz/optimizer plumbing
│   ├── ansatz.py            # UCCSD, RY-CZ, HEA, minimal ansatz
│   ├── adapt.py             # ADAPT-VQE (adaptive ansatz growth)
│   ├── optimizer.py         # Adam, GD, Momentum, SPSA, etc.
│   ├── hamiltonian.py       # VQE wrapper → uses common.hamiltonian
│   ├── io_utils.py          # JSON caching, run signatures
│   ├── visualize.py         # Convergence, scans, noise plots
│   ├── lr_vqe.py            # LR-VQE (tangent-space linear response / TDA)
│   ├── qse.py               # QSE (post-VQE operator-subspace expansion)
│   ├── vqd.py               # VQD (excited states)
│   └── ssvqe.py             # SSVQE (excited states)
│
├── qpe/                     # QPE package
│   ├── __main__.py          # CLI: python -m qpe
│   ├── core.py              # Controlled-U, trotterized dynamics, iQFT
│   ├── hamiltonian.py       # QPE wrapper → uses common.hamiltonian
│   ├── io_utils.py          # JSON caching, run signatures
│   ├── noise.py             # Depolarizing + amplitude damping channels
│   └── visualize.py         # Phase histograms + sweep plots
│
├── qite/                    # QITE / VarQITE package
│   ├── __main__.py          # CLI: python -m qite  (subcommands: run, eval-noise)
│   ├── core.py              # VarQITE orchestration (cached runs)
│   ├── engine.py            # Ansatz plumbing + energy/state QNodes for noiseless/noisy evaluation
│   ├── hamiltonian.py       # QITE wrapper → uses common.hamiltonian
│   ├── io_utils.py          # Run config hashing + JSON run records
│   └── visualize.py         # Convergence + diagnostics/noise plots
│
├── common/                  # Shared logic for VQE + QPE + QITE
│   ├── geometry.py          # Bond/angle geometry generators
│   ├── hamiltonian.py       # Unified Hamiltonian builder
│   ├── molecules.py         # Molecule registry
│   ├── naming.py            # Stable filename / token formatting
│   ├── paths.py             # Results/images root resolution
│   ├── persist.py           # Atomic JSON read/write
│   ├── plotting.py          # Unified plotting + filename builders
│   └── molecule_viz.py      # Molecule visualization helpers
│
├── images/                  # Saved png files. In .gitignore
├── results/                 # JSON outputs.    In .gitignore
│
└── notebooks/
    ├── README_notebooks.md  # Notebook index
    ├── getting_started/     # Intro notebook implementing VQE and QPE from scratch
    ├── vqe/                 # Package-client notebooks for VQE/SSVQE/VQD
    └── qpe/                 # Package-client notebooks for QPE

```

This structure ensures:

- **VQE, QPE, and QITE share the same chemistry** (`common/`)
- **All results are cached consistently** (`results/`)
- **All plots use one naming system** (`common/plotting.py`)
- **CLI tools are production-ready** (`python -m vqe`, `python -m qpe`)

---

## Installation

### Install from PyPI

```bash
pip install vqe-pennylane
```

### Install from source (development mode)

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

### Confirm installation

```bash
python -c "import vqe, qpe, qite, common; print('All stacks imported successfully!')"
```

---

## Common Core (Shared by VQE, QPE & QITE)

The following modules ensure full consistency between solvers:

| Module                          | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| `common/molecules.py`   | Canonical molecule definitions                  |
| `common/geometry.py`    | Bond/angle/coordinate generators                |
| `common/hamiltonian.py` | Hamiltonian construction + OpenFermion fallback |
| `common/plotting.py`    | Unified filename builder + PNG export           |

---

## VQE package

### Capabilities

* Ground-state VQE
* Excited-state tooling via **LR-VQE**, **QSE**, **SSVQE**, and **VQD**
* ADAPT-VQE (adaptive operator selection from an excitation pool)
* Geometry scans and mapping comparisons
* Optional noise models (depolarizing / amplitude damping and custom noise callables)
* Result caching (hash-based signatures) and unified plot naming

### VQE example

```python
from vqe.core import run_vqe

result = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", steps=50)
print(result["energy"])
```

### Energy ordering policy (important)

For excited-state workflows (`SSVQE`, `VQD`), the package reports energies in a consistent way:

* `energies_per_state[k]` is the trajectory for the *k-th reported energy*.
* **Final energies are ordered ascending (lowest → highest)** for stable reporting in notebooks/tables.

This avoids “state swap” confusion when a particular optimization run lands in a different eigenstate ordering.

### Operator-Subspace Methods (Post-VQE)

#### QSE overview

Quantum Subspace Expansion (QSE) is a **post-processing** method built on top of a converged VQE reference state $|\psi\rangle$.

Given an operator pool $\{O_i\}$ (default: Hamiltonian-driven top-$k$ Pauli terms), QSE forms the subspace matrices

$$H_{ij}=\langle\psi|O_i^\dagger H O_j|\psi\rangle,\qquad
S_{ij}=\langle\psi|O_i^\dagger O_j|\psi\rangle$$

and solves the generalized eigenvalue problem

$$Hc = ESc$$

#### LR-VQE overview

Linear-Response VQE (LR-VQE) extracts excited-state information from a **converged VQE reference state** by analysing the **ansatz tangent space**.

Rather than expanding a discrete operator pool (as in QSE), LR-VQE constructs the generalized eigenvalue problem

A c = ω S c

where

• S encodes overlaps between parameter-derivative (tangent) states  
• A encodes Hamiltonian projections in the same tangent space  
• ω are excitation energies relative to the VQE reference energy E₀

Excited energies are then

Eᵢ = E₀ + ωᵢ

This implementation uses a **Tamm–Dancoff approximation (TDA)**, which is numerically stable,
noise-resistant, and sufficient for small-molecule benchmarks.

### SSVQE (excited-state) overview

SSVQE targets multiple low-lying states in a single shared-parameter optimization:

* Choose orthogonal computational-basis reference states (|\phi_k\rangle)
* Apply a shared parameterized unitary (U(\theta)) to each reference:
  $$|\psi_k(\theta)\rangle = U(\theta)\,|\phi_k\rangle$$
* Minimize a weighted sum of energies:
  $$\mathcal{L}(\theta) = \sum_k w_k \langle \psi_k(\theta)|H|\psi_k(\theta)\rangle$$
  Orthogonality is enforced by the orthogonality of the inputs (|\phi_k\rangle), not by overlap penalties.

### VQD (excited-state) overview

VQD computes excited states sequentially:

* First optimize a ground state $|\psi_0(\theta_0)\rangle$
* Then optimize an excited state $|\psi_1(\theta_1)\rangle$ using a deflation term:
  $$\mathcal{L}(\theta_1) = E(\theta_1) + \beta \cdot \text{Overlap}(\psi_0,\psi_1)$$
  In the noiseless case, overlap approximates $|\langle \psi_0|\psi_1\rangle|^2$; with noise it can be implemented using a density-matrix similarity proxy.

### ADAPT-VQE overview

ADAPT-VQE (adaptive ansatz) constructs the variational ansatz **on the fly**, rather than fixing it in advance.

The workflow alternates between two loops:

* **Inner loop:** optimize the parameters of the current ansatz using standard VQE.
* **Outer loop:** evaluate the energy gradient that would result from appending each candidate operator from a predefined pool (e.g. UCC singles/doubles), and **append the operator with the largest gradient magnitude**.

This procedure continues until the maximum gradient falls below a tolerance or a maximum operator budget is reached.

Key features of this implementation:

* Chemistry-motivated operator pools (**UCC-S / UCC-D / UCCSD**)
* Hartree–Fock reference state
* Deterministic operator selection via $|\partial E / \partial \theta|$ at $\theta=0$
* Fully compatible with the existing VQE engine, noise handling, caching, and plotting infrastructure

---

## QPE package

### Capabilities

* Noiseless and noisy QPE
* Trotterized $e^{-iHt}$
* Inverse QFT
* Noise channels
* Cached results

### Example

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, n_qubits, hf_state = build_hamiltonian(
    molecule="H2",
    mapping="jordan_wigner",
    unit="angstrom",
)
result = run_qpe(hamiltonian=H, hf_state=hf_state, n_ancilla=4)
```

---

## QITE / VarQITE package

### Capabilities

* **VarQITE (McLachlan)** imaginary-time parameter updates (noiseless, pure-state)
* Cached run records under `results/qite/` and convergence plots under `images/qite/`
* Explicit separation between optimization (`qite run`) and noisy evaluation (`qite eval-noise`)

### Example

```python
from qite.core import run_qite

res = run_qite(
    molecule="H2",
    ansatz_name="UCCSD",
    steps=50,
    dtau=0.2,
    seed=0,
    mapping="jordan_wigner",
    unit="angstrom",
    force=False,
)
print(res["energy"])
```

---

## CLI usage

All CLIs are installed as entrypoints (`vqe`, `qpe`, `qite`) or can be invoked via `python -m`.

### VQE

```bash
python -m vqe -m H2 -a UCCSD -o Adam --steps 50
```

### QPE

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

### QITE / VarQITE

```bash
# Noiseless VarQITE run (cached)
python -m qite run --molecule H2 --steps 50 --dtau 0.2 --seed 0

# Noisy evaluation of cached parameters
python -m qite eval-noise --molecule H2 --steps 50 --seed 0 --dep 0.02 --amp 0.0 --pretty

# Depolarizing sweep averaged across seeds
python -m qite eval-noise --molecule H2 --steps 50 --sweep-dep 0,0.02,0.04 --seeds 0,1,2 --pretty
```

For full CLI coverage (including excited-state workflows), see [USAGE.md](USAGE.md).

---

## Tests

```bash
pytest -v
```

---

Author: Sid Richards (SidRichardsQuantum)

LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
