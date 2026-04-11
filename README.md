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
<a href="LICENSE">
<img src="https://img.shields.io/github/license/SidRichardsQuantum/Variational_Quantum_Eigensolver?style=flat-square" alt="License">
</a>

</p>

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

## What This Repo Is Good For

Use this repo if you want:

- one Hamiltonian pipeline shared across VQE, QPE, and QITE/QRTE
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
from vqe import run_vqe
from qite import run_qite, run_qrte

vqe_res = run_vqe(molecule="H2")
print("VQE:", vqe_res["energy"])

qite_res = run_qite(molecule="H2", steps=50, dtau=0.2)
print("VarQITE:", qite_res["energy"])

qrte_res = run_qrte(molecule="H2", steps=20, dt=0.05)
print("VarQRTE final energy:", qrte_res["energy"])
```

CLI:

```bash
python -m vqe -m H2
python -m qpe --molecule H2 --ancillas 4 --shots 2000
python -m qite run --molecule H2 --steps 50 --dtau 0.2
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
- `run_adapt_vqe`
- `run_lr_vqe`
- `run_eom_vqe`
- `run_qse`
- `run_eom_qse`
- `run_ssvqe`
- `run_vqd`

Use it for:

- ground-state optimization
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
- `common/persist.py`
- `common/plotting.py`
- `common/paths.py`

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
- `--force` recomputes instead of loading cache

## Notebooks

Notebook guide:

- [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md)

Layout:

- `notebooks/getting_started/` for usage-oriented demos
- `notebooks/benchmarks/` for exact-reference, comparison, and scan workflows

Recommended starting points:

- [`notebooks/getting_started/vqe_vs_qpe_from_scratch_h2.ipynb`](notebooks/getting_started/vqe_vs_qpe_from_scratch_h2.ipynb)
- [`notebooks/getting_started/06_getting_started_qite_h2.ipynb`](notebooks/getting_started/06_getting_started_qite_h2.ipynb)
- [`notebooks/getting_started/13_getting_started_qrte_h2.ipynb`](notebooks/getting_started/13_getting_started_qrte_h2.ipynb)
- [`notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb`](notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb)

## Documentation

Use these in order:

1. [`README.md`](README.md) for orientation and quickstart
2. [`USAGE.md`](USAGE.md) for CLI and Python entrypoints
3. [`THEORY.md`](THEORY.md) for algorithmic background
4. [`notebooks/README_notebooks.md`](notebooks/README_notebooks.md) for notebook navigation

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

```bash
pytest -q
```

## Author

Sid Richards

- LinkedIn: [sid-richards-21374b30b](https://www.linkedin.com/in/sid-richards-21374b30b/)
- GitHub: [SidRichardsQuantum](https://github.com/SidRichardsQuantum)

## License

MIT. See [LICENSE](LICENSE).
