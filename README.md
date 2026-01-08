# Quantum Simulation Suite — VQE + VQD + SSVQE + QPE (PennyLane)

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
- **Subspace-Search VQE (SSVQE)** (multiple low-lying states, subspace objective)
- **Variational Quantum Deflation (VQD)** (excited states via deflation)
- **Quantum Phase Estimation (QPE)** (phase-based energy estimation)
- **Unified molecule registry, geometry generators, and plotting tools**
- **Consistent caching and reproducibility across all solvers**

This project refactors all previous notebooks into a clean Python package with a shared `vqe_qpe_common/` layer for Hamiltonians, molecules, geometry, and plotting.

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
│   ├── ansatz.py            # UCCSD, RY-CZ, HEA, minimal ansätze
│   ├── optimizer.py         # Adam, GD, Momentum, SPSA, etc.
│   ├── hamiltonian.py       # VQE wrapper → uses vqe_qpe_common.hamiltonian
│   ├── io_utils.py          # JSON caching, run signatures
│   ├── visualize.py         # Convergence, scans, noise plots
│   ├── vqd.py               # VQD (excited states)
│   └── ssvqe.py             # SSVQE (excited states)
│
├── qpe/                     # QPE package
│   ├── __main__.py          # CLI: python -m qpe
│   ├── core.py              # Controlled-U, trotterized dynamics, iQFT
│   ├── hamiltonian.py       # QPE wrapper → uses vqe_qpe_common.hamiltonian
│   ├── io_utils.py          # JSON caching, run signatures
│   ├── noise.py             # Depolarizing + amplitude damping channels
│   └── visualize.py         # Phase histograms + sweep plots
│
├── vqe_qpe_common/          # Shared logic for VQE + QPE
│   ├── geometry.py          # Bond/angle geometry generators
│   ├── hamiltonian.py       # Unified Hamiltonian builder (PennyLane/OpenFermion)
│   ├── molecules.py         # Unified molecule registry
│   ├── molecule_viz.py      # Draw molecules
│   └── plotting.py          # Shared plotting + filename builders
│
├── images/                  # Saved png files
│   ├── vqe/
│   └── qpe/
│
├── results/                 # JSON outputs
│   ├── vqe/
│   └── qpe/
│
└── notebooks/
├── README_notebooks.md  # Notebook index
├── getting_started/     # Intro notebook implementing VQE and QPE from scratch
├── vqe/                 # Package-client notebooks for VQE/SSVQE/VQD
└── qpe/                 # Package-client notebooks for QPE

````

This structure ensures:

- **VQE and QPE share the same chemistry** (`vqe_qpe_common/`)
- **All results are cached consistently** (`results/`)
- **All plots use one naming system** (`vqe_qpe_common/plotting.py`)
- **CLI tools are production-ready** (`python -m vqe`, `python -m qpe`)

---

## Installation

### Install from PyPI

```bash
pip install vqe-pennylane
````

### Install from source (development mode)

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

### Confirm installation

```bash
python -c "import vqe, qpe; print('VQE+QPE imported successfully!')"
```

---

## Common Core (Shared by VQE & QPE)

The following modules ensure full consistency between solvers:

| Module                          | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| `vqe_qpe_common/molecules.py`   | Canonical molecule definitions                  |
| `vqe_qpe_common/geometry.py`    | Bond/angle/coordinate generators                |
| `vqe_qpe_common/hamiltonian.py` | Hamiltonian construction + OpenFermion fallback |
| `vqe_qpe_common/plotting.py`    | Unified filename builder + PNG export           |

---

## VQE package

### Capabilities

* Ground-state VQE
* Excited states via **SSVQE** and **VQD**
* Geometry scans and mapping comparisons
* Optional noise models (depolarizing / amplitude damping and custom noise callables)
* Result caching (hash-based signatures) and unified plot naming

### Energy ordering policy (important)

For excited-state workflows (`SSVQE`, `VQD`), the package reports energies in a consistent way:

* `energies_per_state[k]` is the trajectory for the *k-th reported energy*.
* **Final energies are ordered ascending (lowest → highest)** for stable reporting in notebooks/tables.

This avoids “state swap” confusion when a particular optimization run lands in a different eigenstate ordering.

### VQE example

```python
from vqe.core import run_vqe

result = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", steps=50)
print(result["energy"])
```

### SSVQE (excited-state) overview

SSVQE targets multiple low-lying states in a single shared-parameter optimization:

* Choose orthogonal computational-basis reference states (|\phi_k\rangle)
* Apply a shared parameterized unitary (U(\theta)) to each reference:
  $$|\psi_k(\theta)\rangle = U(\theta),|\phi_k\rangle$$
* Minimize a weighted sum of energies:
  $$\mathcal{L}(\theta) = \sum_k w_k \langle \psi_k(\theta)|H|\psi_k(\theta)\rangle$$
  Orthogonality is enforced by the orthogonality of the inputs (|\phi_k\rangle), not by overlap penalties.

### VQD (excited-state) overview

VQD computes excited states sequentially:

* First optimize a ground state $|\psi_0(\theta_0)\rangle$
* Then optimize an excited state $|\psi_1(\theta_1)\rangle$ using a deflation term:
  $$\mathcal{L}(\theta_1) = E(\theta_1) + \beta \cdot \text{Overlap}(\psi_0,\psi_1)$$
  In the noiseless case, overlap approximates $|\langle \psi_0|\psi_1\rangle|^2$; with noise it can be implemented using a density-matrix similarity proxy.

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
from vqe_qpe_common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

symbols = ["H", "H"]
coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]]

H, n_qubits, hf_state = build_hamiltonian(symbols, coords, charge=0, basis="STO-3G")
result = run_qpe(hamiltonian=H, hf_state=hf_state, n_ancilla=4)
```

---

## CLI usage

### VQE

```bash
python -m vqe -m H2 -a UCCSD -o Adam --steps 50
```

### QPE

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
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
