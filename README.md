# Quantum Simulation Suite — VQE + QPE (PennyLane)

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

A modern, modular, and fully reproducible **quantum-chemistry simulation suite** built on  
**PennyLane**, featuring:

- **Variational Quantum Eigensolver (VQE)**  
- **State-Specific VQE (SSVQE)**  
- **Quantum Phase Estimation (QPE)**  
- **Unified molecule registry, geometry generators, and plotting tools**  
- **Consistent caching and reproducibility across all solvers**

This project refactors all previous notebooks into a clean Python package with  
a shared `vqe_qpe_common/` layer for Hamiltonians, molecules, geometry, and plotting.

- **[THEORY.md](THEORY.md)** — Full background on VQE, QPE, ansatzes, optimizers, mappings, and noise models.  
- **[USAGE.md](USAGE.md)** — Practical guide for installation, CLI usage, running simulations, caching, and plotting.

These documents complement the README and provide both the *theoretical foundation* and the *hands-on execution details* of the VQE/QPE suite.

---

# Project Structure

```
Variational_Quantum_Eigensolver/
├── README.md
├── THEORY.md
├── USAGE.md
├── LICENSE
├── pyproject.toml
│
├── vqe_qpe_common/          # Shared logic for VQE + QPE
│   ├── molecules.py         # Unified molecule registry
│   ├── geometry.py          # Bond/angle geometry generators
│   ├── hamiltonian.py       # Unified Hamiltonian builder (PennyLane/OpenFermion)
│   └── plotting.py          # Shared plotting + filename builders
│
├── vqe/                     # Variational Quantum Eigensolver package
│   ├── __main__.py          # CLI: python -m vqe
│   ├── core.py              # VQE orchestration (runs, scans, sweeps)
│   ├── engine.py            # Devices, noise, ansatz/optimizer plumbing
│   ├── ansatz.py            # UCCSD, RY-CZ, HEA, minimal ansätze
│   ├── optimizer.py         # Adam, GD, Momentum, SPSA, etc.
│   ├── hamiltonian.py       # VQE-specific wrapper → uses vqe_qpe_common.hamiltonian
│   ├── io_utils.py          # JSON caching, run signatures
│   ├── visualize.py         # Convergence, scans, noise plots
│   └── ssvqe.py             # Subspace-search VQE (excited states)
│
├── qpe/                     # Quantum Phase Estimation package
│   ├── __main__.py          # CLI: python -m qpe
│   ├── core.py              # Controlled-U, trotterized dynamics, iQFT
│   ├── hamiltonian.py       # QPE-specific wrapper → uses vqe_qpe_common.hamiltonian
│   ├── io_utils.py          # JSON caching, run signatures
│   ├── noise.py             # Depolarizing + amplitude damping channels
│   └── visualize.py         # Phase histograms + sweep plots
│
├── results/                 # JSON outputs
├── images/                  # Saved plots (VQE + QPE)
├── data/                    # Optional molecule configs, external data
│
└── notebooks/               # Notebooks importing from the vqe/ and qpe/ packages
```

This structure ensures:

- **VQE and QPE share the same chemistry** (`vqe_qpe_common/`)
- **All results are cached consistently** (`results/`)
- **All plots use one naming system** (`vqe_qpe_common/plotting.py`)
- **Notebooks import from the real package** (no duplicated code)
- **CLI tools are production-ready** (`python -m vqe`, `python -m qpe`)

---

# ⚙️ Installation

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
python -c "import vqe, qpe; print('VQE+QPE imported successfully!')"
```

---

# Common Core (Shared by VQE & QPE)

The following modules ensure full consistency between solvers:

| Module | Purpose |
|--------|---------|
| `vqe_qpe_common/molecules.py` | Canonical molecule definitions |
| `vqe_qpe_common/geometry.py` | Bond/angle/coordinate generators |
| `vqe_qpe_common/hamiltonian.py` | Hamiltonian construction + OpenFermion fallback |
| `vqe_qpe_common/plotting.py` | Unified filename builder + PNG export |

---

# 🔹 VQE Package

Features:
- Ground-state VQE
- Excited-state SSVQE
- Geometry scans
- Noise sweeps
- Mapping comparisons
- Optimizer registry
- Result caching

Run example:

```python
from vqe.core import run_vqe
result = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", n_steps=50)
print(result["energy"])
```

---

# 🔹 QPE Package

Features:
- Noiseless & noisy QPE  
- Trotterized exp(-iHt)  
- Inverse QFT  
- Noise channels  
- Cached results  

Example:

```python
from vqe_qpe_common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, n_qubits, hf_state = build_hamiltonian(["H","H"], coords, 0, "STO-3G")
result = run_qpe(H, hf_state, n_ancilla=4)
```

---

# CLI Usage

### VQE  
```bash
python -m vqe -m H2 -a UCCSD -o Adam --steps 50
```

### QPE  
```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

---

# 🧪 Tests

```bash
pytest -v
```

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
