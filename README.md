# Quantum Simulation Suite — VQE + QPE

**Variational Quantum Eigensolver (VQE) and Quantum Phase Estimation (QPE)**  
Built with [PennyLane](https://pennylane.ai) for quantum chemistry, variational optimization, and phase-based energy estimation.

This project refactors the original notebook experiments into a **modular, reproducible Python package**.

---

## 🧠 Overview

This repository provides two complementary simulation subpackages:

| Package | Description |
|----------|--------------|
| **`vqe`** | Variational Quantum Eigensolver + SSVQE for excited states. |
| **`qpe`** | Quantum Phase Estimation with noiseless/noisy channels and trotterized dynamics. |

All runs are cached and reproducible.  
Results and plots are stored under:
```
package_results/     ← JSON run records (configs, results)
vqe/images/          ← VQE plots and convergence graphs
qpe/images/          ← QPE histograms and sweeps
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -e .
```

**Dependencies:**
- `pennylane >= 0.42`
- `numpy`
- `matplotlib`
- `scipy`
- `openfermionpyscf` (for open-shell chemistry)

---

# 🔹 VQE Module

**Variational Quantum Eigensolver (VQE) and SSVQE Toolkit**

### Features
- Ground- and excited-state simulation (VQE & SSVQE)
- Noise-aware studies (depolarizing, amplitude damping)
- Optimizer and ansatz comparison utilities
- Geometry scans and mapping comparisons
- JSON result caching and reproducible configuration hashes

### Structure

| Module | Purpose |
|---------|----------|
| `vqe/core.py` | Main orchestration (VQE runs, sweeps, scans, comparisons) |
| `vqe/engine.py` | Core engine (device, noise, ansatz, optimizer plumbing) |
| `vqe/ansatz.py` | Defines UCC, RY-CZ, and hardware-efficient ansatzes |
| `vqe/hamiltonian.py` | Molecular Hamiltonians and geometry builders |
| `vqe/optimizer.py` | Optimizer registry and minimization loop |
| `vqe/io_utils.py` | JSON I/O, configuration hashing, directory setup |
| `vqe/visualize.py` | Plotting for convergence, noise sweeps, comparisons |
| `vqe/ssvqe.py` | State-specific VQE (excited states) |

---

## 🚀 Quick Start: VQE

### 1️⃣ Ground-state energy
```python
from vqe.core import run_vqe

result = run_vqe(
    molecule="H2",
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    n_steps=50,
    stepsize=0.2,
)
print("Ground-state energy:", result["energy"])
```

### 2️⃣ Optimizer comparison
```python
from vqe.core import run_vqe_optimizer_comparison
run_vqe_optimizer_comparison("H2", ansatz_name="RY-CZ")
```

### 3️⃣ Noisy VQE
```python
from vqe.core import run_vqe_noise_sweep
run_vqe_noise_sweep("LiH", ansatz_name="UCC-D")
```

### 4️⃣ Geometry scans
```python
import numpy as np
from vqe.core import run_vqe_geometry_scan

run_vqe_geometry_scan(
    molecule="H2_BOND",
    param_name="bond",
    param_values=np.linspace(0.5, 2.5, 8),
)
```

### 5️⃣ SSVQE (Excited States)
```python
from vqe.ssvqe import run_ssvqe

res = run_ssvqe(
    molecule="H3+",
    num_states=2,
    ansatz_name="UCCSD",
    steps=80,
    stepsize=0.4,
)
```

---

# 🔹 QPE Module

**Quantum Phase Estimation (QPE) Simulation Suite**

Implements noiseless and noisy QPE using **trotterized time evolution**, inverse QFT, and optional noise channels.

### Features
- Standard and noisy QPE (depolarizing / amplitude damping)
- Hartree–Fock state preparation
- Caching and reproducibility identical to VQE
- Phase histograms and parameter sweep plots
- CLI runner for quick experiments (`python -m qpe --molecule H2`)

### Structure

| Module | Purpose |
|---------|----------|
| `qpe/core.py` | Core QPE implementation (controlled evolutions, iQFT) |
| `qpe/noise.py` | Noise models (depolarizing & amplitude damping) |
| `qpe/io_utils.py` | Result saving, loading, and signature hashing |
| `qpe/visualize.py` | Plotting utilities for histograms and sweeps |
| `qpe/__main__.py` | CLI interface for running simulations |
| `package_results/` | Shared cache between VQE and QPE |

---

## 🚀 Quick Start: QPE

### 1️⃣ Run a simple QPE

```python
from qpe.core import run_qpe
import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H"]
coordinates = np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.7414]])
H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=0, basis="STO-3G")
hf_state = qml.qchem.hf_state(2, n_qubits)

result = run_qpe(H, hf_state, n_ancilla=4, t=1.0, shots=2000, molecule_name="H2")
print("Estimated energy:", result["energy"])
```

### 2️⃣ Add noise
```python
from qpe.core import run_qpe
result = run_qpe(
    hamiltonian=H,
    hf_state=hf_state,
    n_ancilla=4,
    t=1.0,
    noise_params={"p_dep": 0.02, "p_amp": 0.01},
    shots=2000,
)
```

### 3️⃣ Plot results
```python
from qpe.visualize import plot_qpe_distribution
plot_qpe_distribution(result)
```

---

## 🧩 CLI Usage

You can also run QPE directly from the command line:

```bash
python -m qpe --molecule H2 --ancillas 4 --shots 2000
```

Add noise:
```bash
python -m qpe --molecule H3+ --noisy --p_dep 0.02 --p_amp 0.01 --save-plot
```

All plots are saved under:
```
qpe/images/
```

---

## 📊 Reproducibility (VQE & QPE)

Each run produces a unique **MD5 hash signature** derived from its configuration:
- molecule, geometry, and basis
- ansatz / optimizer or number of ancillas
- time parameter `t`
- noise settings
- shot count

Results are stored under `package_results/` as:
```json
{
  "molecule": "H2",
  "energy": -1.137,
  "phase": 0.125,
  "n_ancilla": 4,
  "t": 1.0,
  "noise": null,
  "shots": 2000
}
```

---

## 🧪 Tests

All package-level tests are located under:
```
package_tests/
```

To run:
```bash
pytest -v
```

Includes:
- **Functional tests** for VQE, SSVQE, and QPE runs
- **Caching and reproducibility** checks
- **Plot generation** and import smoke tests

---

## 📘 Notebooks

Exploratory notebooks remain in:
```
notebooks/
```
They now import from the packaged modules rather than duplicating code.

---

## ⚙️ Extending the Framework

- Add ansatz → `vqe/ansatz.py`
- Add optimizer → `vqe/optimizer.py`
- Add geometry generator → `vqe/hamiltonian.py`
- Add noise model → `vqe/engine.py` or `qpe/noise.py`

---

## Citation

If you use this project or its methods, please cite:
> Sid Richards (2025). *Variational Quantum Eigensolver and Quantum Phase Estimation Comparisons using PennyLane.*

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
