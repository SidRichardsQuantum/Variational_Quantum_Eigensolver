# Quantum Simulation Suite

<p align="center">

<a href="https://pypi.org/project/vqe-pennylane/">
<img src="https://img.shields.io/pypi/v/vqe-pennylane?style=flat-square" alt="PyPI Version">
</a>

<a href="https://pypi.org/project/vqe-pennylane/">
<img src="https://img.shields.io/pypi/pyversions/vqe-pennylane?style=flat-square" alt="Python Versions">
</a>

<a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/actions/workflows/tests.yml">
<img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Variational_Quantum_Eigensolver/tests.yml?label=tests&style=flat-square" alt="Tests">
</a>

<a href="https://sidrichardsquantum.github.io/Variational_Quantum_Eigensolver/">
<img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Variational_Quantum_Eigensolver/pages.yml?label=docs&style=flat-square" alt="Docs">
</a>

<a href="LICENSE">
<img src="https://img.shields.io/github/license/SidRichardsQuantum/Variational_Quantum_Eigensolver?style=flat-square" alt="License">
</a>

<a href="https://github.com/sponsors/SidRichardsQuantum">
<img src="https://img.shields.io/badge/sponsor-GitHub-ea4aaa?style=flat-square&logo=githubsponsors" alt="Sponsor">
</a>

</p>

PyPI: [https://pypi.org/project/vqe-pennylane/](https://pypi.org/project/vqe-pennylane/)

Docs: [https://sidrichardsquantum.github.io/Variational_Quantum_Eigensolver/](https://sidrichardsquantum.github.io/Variational_Quantum_Eigensolver/)

PennyLane-based workflows for:

- ground-state VQE
- excited-state methods
- quantum phase estimation
- variational imaginary-time evolution
- variational real-time evolution

Implemented packages:

- `vqe` for ground-state and excited-state solvers
- `qpe` for phase-estimation workflows
- `qite` for projected variational dynamics (`VarQITE`, `VarQRTE`)
- `common` for shared chemistry, Hamiltonians, caching, and plotting

## Table of Contents

- [What This Repo Is Good For](#what-this-repo-is-good-for)
- [Choose A Method](#choose-a-method)
- [Install](#install)
- [Quickstart](#quickstart)
- [Typical Workflow](#typical-workflow)
- [Package Overview](#package-overview)
  - [vqe](#vqe)
  - [qpe](#qpe)
  - [qite](#qite)
- [Shared Infrastructure](#shared-infrastructure)
- [Supported Molecule Inputs](#supported-molecule-inputs)
- [Non-Molecule Mode](#non-molecule-mode)
- [Outputs And Reproducibility](#outputs-and-reproducibility)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Repository Layout](#repository-layout)
- [Testing](#testing)
- [Support Development](#support-development)
- [Author](#author)
- [License](#license)

## What This Repo Is Good For

Use this repo if you want:

- one Hamiltonian pipeline shared across VQE, QPE, and QITE/QRTE
- one shared problem-resolution layer for molecule, explicit-geometry, and expert-mode inputs
- reproducible runs with stable cache keys and JSON outputs
- both Python APIs and CLI workflows
- notebooks that separate demos from benchmarks

It is optimized for small-molecule algorithm development and comparison, not large-scale production chemistry.

## Choose A Method

Use `VQE` when you want a ground-state energy or a good reference state.

Use `ADAPT-VQE` when you want an adaptive ansatz rather than a fixed one.

Use `QSE`, `EOM-QSE`, `LR-VQE`, or `EOM-VQE` when you already have a converged VQE reference and want excited-state information.

Use `SSVQE` or `VQD` when you want variational excited-state solvers directly.

Use `QPE` when you want spectral / phase information rather than a compact variational state.

Use `VarQITE` when you want imaginary-time relaxation toward a low-energy state.

Use `VarQRTE` when you already have a relevant state and want to evolve it in real time and analyze observables.

Use expert-mode Hamiltonian inputs when you want to benchmark algorithms on non-chemistry qubit models without going through the molecule / geometry pipeline.

## Install

From PyPI:

```bash
pip install vqe-pennylane
```

From source:

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

Verify:

```bash
python -c "import vqe, qpe, qite, common; print('Quantum stacks imported successfully')"
```

## Quickstart

Python:

```python
import pennylane as qml

from vqe import run_vqe
from qite import run_qite, run_qrte

vqe_res = run_vqe(molecule="H2")
print("VQE:", vqe_res["energy"])

qite_res = run_qite(molecule="H2", steps=75, dtau=0.2)
print("VarQITE:", qite_res["energy"])

qrte_res = run_qrte(molecule="H2", steps=20, dt=0.05)
print("VarQRTE final energy:", qrte_res["energy"])

H_model = qml.Hamiltonian([1.0, 0.5], [qml.PauliZ(0), qml.PauliX(0)])
model_res = run_vqe(
    hamiltonian=H_model,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    steps=10,
    plot=False,
)
print("Model VQE:", model_res["energy"])
```

`run_vqe()` uses the calibrated default stepsize for the selected optimizer when `stepsize` is omitted.
The `vqe` CLI does the same when `--stepsize` is omitted.

CLI:

```bash
python -m vqe -m H2
python -m qpe --molecule H2 --ancillas 4 --shots 1000 --trotter-steps 2
python -m qite run --molecule H2 --steps 75 --dtau 0.2
python -m qite run-qrte --molecule H2 --steps 20 --dt 0.05
```

## Typical Workflow

For stationary-state work:

1. run `VQE` to get a ground-state reference
2. run post-VQE or excited-state solvers if needed
3. compare against exact small-system references where possible

For dynamics:

1. prepare a state with `VQE`, `VarQITE`, or an excited-state workflow
2. use `VarQRTE` for real-time dynamics
3. benchmark against exact evolution on small systems when possible

## Package Overview

### `vqe`

Includes:

- `run_vqe`
- `run_vqe_low_qubit_benchmark`
- `run_adapt_vqe`
- `run_lr_vqe`
- `run_eom_vqe`
- `run_qse`
- `run_eom_qse`
- `run_ssvqe`
- `run_vqd`

Use it for:

- ground-state optimization
- low-qubit multi-molecule benchmarking
- excited-state studies
- ansatz / optimizer / mapping comparisons
- geometry scans
- noisy VQE experiments

### `qpe`

Includes:

- `run_qpe`

Use it for:

- phase-to-energy estimation
- ancilla / shot studies
- controlled time-evolution experiments

Default QPE settings are H2-calibrated baseline defaults:

- `n_ancilla=4`
- `t=1.0`
- `trotter_steps=2`
- `shots=1000`

They are intended as good small-molecule starting values, not globally optimized settings for every molecule.

### `qite`

Includes:

- `run_qite`
- `run_qrte`

Use it for:

- imaginary-time relaxation with `VarQITE`
- real-time dynamics with `VarQRTE`
- prepared-state quench workflows

Important:

- `VarQITE` is a state-finding / relaxation method
- `VarQRTE` is a dynamics method, not an eigensolver
- for time-independent Hamiltonians, `VarQRTE` should usually conserve energy up to numerical error

## Shared Infrastructure

All solver packages use the same chemistry layer in `common/`.

That gives you:

- consistent Hamiltonians across algorithms
- comparable outputs across methods
- shared run signatures and cache reuse
- standardized result and image paths

Main shared modules:

- `common/molecules.py`
- `common/geometry.py`
- `common/hamiltonian.py`
- `common/problem.py`
- `common/persist.py`
- `common/plotting.py`
- `common/paths.py`

Useful shared helpers:

- `build_hamiltonian(...)`
- `get_exact_spectrum(...)`
- `summarize_registry_coverage(...)`

## Supported Molecule Inputs

For standard chemistry workflows, prefer the shared molecule pipeline over expert-mode Hamiltonian inputs.

Built-in registry molecules currently include:

- `H`, `H-`, `H2`, `H2+`, `H2-`, `H3`, `H3+`, `H4`, `H4+`, `H5+`, `H6`
- `He`, `He+`, `He2`, `HeH+`
- `Li`, `Li+`, `LiH`
- `Be`, `Be+`, `BeH2`
- `B`, `B+`, `C`, `C+`, `N`, `N+`, `O`, `O+`, `F`, `F+`, `Ne`
- `H2O`

If a target system is not in the registry, use explicit geometry inputs such as `symbols=...`, `coordinates=...`, `charge=...`, `multiplicity=...`, and `basis=...`.
That keeps the run on the standard chemistry path and avoids expert mode unless you already have a prebuilt qubit Hamiltonian.

For a ready-made inventory of the built-in registry, use:

```python
from common import summarize_registry_coverage

rows = summarize_registry_coverage()
```

## Non-Molecule Mode

The Python APIs also support a direct qubit-Hamiltonian mode for algorithm benchmarking outside chemistry.

Currently supported:

- `run_vqe(..., hamiltonian=H, num_qubits=..., reference_state=...)`
- `run_qite(..., hamiltonian=H, num_qubits=..., reference_state=...)`
- `run_qrte(..., hamiltonian=H, num_qubits=..., reference_state=...)`
- `run_qpe(..., hamiltonian=H, hf_state=..., system_qubits=...)`

Notes:

- this is a Python expert-mode API, not a CLI feature
- arbitrary qubit wire labels are normalized internally
- `run_qpe(...)` expert mode requires both `hamiltonian` and `hf_state`
- `ansatz_name="auto"` can select a conservative model ansatz for recognized Pauli structures such as TFIM, XXZ/Heisenberg, and SSH-like hopping chains
- chemistry-specific ansatzes like `UCCSD` still require chemistry metadata; for generic or unclassified model Hamiltonians use `ansatz_name="auto"` or explicit non-chemistry ansatzes such as `RY-CZ`, `Minimal`, or `StronglyEntanglingLayers`

## Outputs And Reproducibility

Generated outputs are written under:

```text
results/vqe/
results/qpe/
results/qite/
images/vqe/
images/qpe/
images/qite/
```

General behavior:

- run configurations are hashed deterministically
- matching runs reuse cached JSON records
- expert-mode Hamiltonian runs cache by canonical Pauli-term fingerprint, reference bitstring, resolved ansatz, and solver settings
- cached records missing runtime metadata are treated as stale and recomputed automatically
- `--force` recomputes instead of loading cache

For sampled QPE runs with finite `shots`, `seed` still matters because the measured bitstring distribution is stochastic.
In analytic mode (`shots=None`), the seed is effectively irrelevant.

## Notebooks

Notebook guide:

- [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md)

Layout:

- `notebooks/getting_started/` for usage-oriented demos
- `notebooks/benchmarks/` for exact-reference, comparison, and scan workflows

Recommended starting points:

- [`notebooks/getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb`](notebooks/getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb)
- [`notebooks/getting_started/07_getting_started_qite_h2.ipynb`](notebooks/getting_started/07_getting_started_qite_h2.ipynb)
- [`notebooks/getting_started/11_getting_started_qrte_h2.ipynb`](notebooks/getting_started/11_getting_started_qrte_h2.ipynb)
- [`notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb`](notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb)

## Documentation

The repository documentation can be built as a navigable Sphinx site:

```bash
pip install -e ".[docs]"
python -m sphinx -W -b html docs docs/_build/html
```

Use these in order:

1. [`README.md`](README.md) for orientation and quickstart
2. [`USAGE.md`](USAGE.md) for CLI and Python entrypoints
3. [`THEORY.md`](THEORY.md) for algorithmic background
4. [`RESEARCH.md`](RESEARCH.md) for benchmark evidence standards
5. [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md) for notebook navigation

Deeper implementation notes:

- [`more_docs/architecture.md`](more_docs/architecture.md)
- `more_docs/vqe/`
- `more_docs/qpe/`
- `more_docs/qite/`

## Repository Layout

```text
Variational_Quantum_Eigensolver/
├── vqe/
├── qpe/
├── qite/
├── common/
├── notebooks/
│   ├── getting_started/
│   └── benchmarks/
├── results/
├── images/
├── README.md
├── USAGE.md
├── THEORY.md
└── pyproject.toml
```

## Testing

The default pytest target is the fast development suite.
Slow chemistry checks and selected subprocess CLI integration tests are marked separately.

```bash
pytest -q
```

Run the full suite, including slow integration coverage, with:

```bash
pytest -q -o addopts=''
```

Run a registered benchmark suite and write reproducible artifacts with:

```bash
python -m common.benchmarks run --suite expert-z-cross-method --out benchmark_runs
```

List available suites with:

```bash
python -m common.benchmarks list
```

Compare two benchmark runs with:

```bash
python -m common.benchmarks compare --base old_run/h2-cross-method --head new_run/h2-cross-method
```

---

## Support development

If this repository is useful for research, learning, or experimentation, you can support continued development via GitHub Sponsors:

https://github.com/sponsors/SidRichardsQuantum

Sponsorship helps support ongoing work on open-source implementations of quantum algorithms, including improvements to documentation, reproducible workflows, and example notebooks.

Support helps maintain and expand practical tooling for variational quantum methods, quantum simulation workflows, and related experimentation.

## Citation

Sid Richards (2026)

Unified Variational and Phase-Estimation Quantum Simulation Suite

## Author

Sid Richards

- LinkedIn: [sid-richards-21374b30b](https://www.linkedin.com/in/sid-richards-21374b30b/)
- GitHub: [SidRichardsQuantum](https://github.com/SidRichardsQuantum)

## License

MIT. See [LICENSE](LICENSE).
