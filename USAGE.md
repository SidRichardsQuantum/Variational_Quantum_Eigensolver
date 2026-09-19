# Usage Guide

Workflows and APIs for:

- **VQE** — ground state, ADAPT-VQE, excited states
- **QPE** — quantum phase estimation
- **QITE / VarQITE / VarQRTE** — projected variational dynamics
- **common** — Hamiltonians, geometry, plotting, persistence

---

## Table of Contents

- [Documentation Map](#documentation-map)
- [Core Execution Model](#core-execution-model)
- [Supported Molecule Inputs](#supported-molecule-inputs)
- [Installation](#installation)
- [General Conventions](#general-conventions)
- [Method Support Summary](#method-support-summary)
- [Quickstart](#quickstart)

- [VQE Workflows](#vqe-workflows)
  - [Basic VQE](#basic-vqe)
  - [Non-molecule expert mode](#non-molecule-expert-mode)
  - [Ansatz selection](#ansatz-selection)
  - [Optimizer selection](#optimizer-selection)
  - [Geometry scans](#geometry-scans)
  - [Noise studies](#noise-studies)
  - [Low-qubit benchmark](#low-qubit-benchmark)

- [Excited-State Methods](#excited-state-methods)
  - [Post-VQE methods](#post-vqe-methods)
  - [Variational excited states](#variational-excited-states)

- [QPE](#qpe)
  - [Basic QPE](#basic-qpe)
  - [Noise](#noise)
  - [Time evolution controls](#time-evolution-controls)

- [QITE / Projected Dynamics](#qite--projected-dynamics)
  - [Execution modes](#execution-modes)
  - [Run](#run)
  - [Real-time run](#real-time-run)
  - [Noisy evaluation](#noisy-evaluation)
  - [Noise sweep](#noise-sweep)
  - [Cache semantics](#cache-semantics)

- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

---

## Documentation Map

| File                        | Purpose                        |
| --------------------------- | ------------------------------ |
| `README.md`                 | overview and quickstart        |
| `USAGE.md`                  | workflows and APIs (this file) |
| `THEORY.md`                 | algorithms and derivations     |
| `more_docs/architecture.md` | system design                  |
| `more_docs/vqe/`            | VQE internals                  |
| `more_docs/qpe/`            | QPE details                    |
| `more_docs/qite/`           | QITE details                   |

---

## Core Execution Model

All stacks share the same pipeline:

```
problem spec → resolved problem → algorithm → results + cache
```

Lower-level chemistry contract:

```python
H, n_qubits, hf_state = build_hamiltonian(...)
```

Registry-inventory helper:

```python
rows = summarize_registry_coverage(...)
```

High-level shared resolver:

```python
problem = resolve_problem(...)
```

from:

```python
from common.problem import resolve_problem
from common import summarize_registry_coverage
```

Implications:

- identical physics across all methods
- one normalization path for molecule, explicit-geometry, and expert-mode inputs
- reproducible cross-algorithm comparisons
- unified caching and output structure

There is also an expert-mode path for prebuilt qubit Hamiltonians when you do not want molecule or geometry inputs.

## Supported Molecule Inputs

The shared chemistry pipeline accepts three input styles.

### 1. Registry molecule names

Use `molecule="..."` with the built-in molecule registry when a named system is already supported.

Current registry molecules:

| Category | Molecules |
| --- | --- |
| Hydrogen atoms and ions | `H`, `H-` |
| Helium systems | `He`, `He+`, `He2`, `HeH+` |
| First-row atoms and ions | `Li`, `Li+`, `Be`, `Be+`, `B`, `B+`, `C`, `C+`, `N`, `N+`, `O`, `O+`, `F`, `F+`, `Ne` |
| Hydrogen clusters | `H2`, `H2+`, `H2-`, `H3`, `H3+`, `H4`, `H4+`, `H5+`, `H6` |
| Small molecules | `LiH`, `H2O`, `BeH2` |

Several common aliases are normalized automatically, for example:

- `h2` -> `H2`
- `H3PLUS` -> `H3+`
- `H2_PLUS` -> `H2+`

### 2. Parametric geometry tags

Use `molecule="..."` with a geometry tag when you want a generated structure rather than a fixed registry entry.

Supported geometry tags:

- `H2_BOND`
- `H3+_BOND`
- `LiH_BOND`
- `H2O_ANGLE`

These are intended for scans and geometry studies.

### 3. Explicit geometry mode

Use explicit molecular data when the target system is not in the registry:

```python
from common.hamiltonian import build_hamiltonian

H, n_qubits, hf_state = build_hamiltonian(
    symbols=["H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    charge=0,
    multiplicity=1,
    basis="sto-3g",
)
```

This same explicit-geometry input style is supported by the high-level runners such as `run_vqe(...)`, `run_qpe(...)`, `run_qite(...)`, and `run_qrte(...)`.

If `molecule="..."` is not a supported registry key or geometry tag, the builder raises a `KeyError`.
In that case, use explicit geometry mode instead of expert mode whenever possible.

To inspect the currently supported built-in registry programmatically:

```python
from common import summarize_registry_coverage

rows = summarize_registry_coverage()
```

---

## Installation

Supported Python versions are **3.10–3.12**, with `numpy>=1.23,<2.0` and
`pennylane>=0.42,<0.45`. Python 3.13 support is deferred until the NumPy 2
compatibility migration is complete.

### PyPI

```bash
pip install vqe-pennylane
```

### From source

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

Verify:

```bash
python -c "import vqe, qpe, qite, common; print('All stacks OK')"
```

---

## Experiment Studio and Progress Observers

Version 0.3.28 adds an optional browser interface for VQE and ADAPT-VQE.
From a source checkout with the package installed, run `python -m studio` and
open `http://127.0.0.1:8000`. Studio is excluded from the wheel and source
distribution. See the [Studio guide](docs/studio.md) for Codespaces setup,
comparisons, JSON exports, cancellation, and persistence behavior. Queued and
running experiments can be cancelled from their cards or detail views; Ctrl-C
cancels unfinished work and cleans up isolated workers.

Python callers can observe actual energy samples without using Studio:

```python
from vqe import run_vqe

result = run_vqe(steps=10, plot=False, progress_callback=print)
```

`run_adapt_vqe()` accepts the same optional callback. Observers receive fresh
status dictionaries synchronously; a cache hit emits a single `cache_hit` event.
Callbacks are excluded from cache identity, and their exceptions propagate.
A completed iteration budget is not a convergence or accuracy certificate.

---

## General Conventions

Output structure, relative to the runtime data root:

```
results/{vqe,qpe,qite}/
images/{vqe,qpe,qite}/
```

The default root is `~/.local/share/vqe-pennylane` on Linux (or
`$XDG_DATA_HOME/vqe-pennylane`), `~/Library/Application Support/vqe-pennylane` on
macOS, and `%LOCALAPPDATA%/vqe-pennylane` on Windows. Generated artifacts are
stored outside the installed package.

Set `VQE_PENNYLANE_DATA_DIR` to override the root. To retain the old checkout-local
layout, run `export VQE_PENNYLANE_DATA_DIR="$PWD"` from the repository. Python
code can also set this environment variable after importing the solvers.

Version 0.3.27 versions cache signatures to invalidate results from earlier
numerical implementations, including spin-reference and QPE term-order fixes.
Old files remain available for inspection; matching runs are recomputed once.

Execution behaviour:

- deterministic hashing defines run identity
- cached runs automatically reused
- VQE, QPE, VarQITE, and VarQRTE refresh cached records missing required runtime
  metadata; ADAPT retains legacy results and omits unavailable compute timings
- `--force` bypasses cache
- identical Hamiltonians shared across algorithms

---

## Method Support Summary

| Method    | Family              | VQE reference required | Noise support |
| --------- | ------------------- | ---------------------: | ------------- |
| VQE       | variational         |                     no | yes           |
| ADAPT-VQE | variational         |                     no | yes           |
| LR-VQE    | post-VQE            |                    yes | no            |
| EOM-VQE   | post-VQE            |                    yes | no            |
| QSE       | post-VQE            |                    yes | no            |
| EOM-QSE   | post-VQE            |                    yes | no            |
| SSVQE     | variational excited |                     no | yes           |
| VQD       | variational excited |                     no | yes           |
| QPE       | phase estimation    |                     no | yes           |
| QITE      | imaginary time      |                     no | eval-only     |
| QRTE      | real-time dynamics  |                     no | no            |

---

## Quickstart

```bash
vqe --molecule H2

vqe -m H2 --lr-vqe --lr-k 4

qpe --molecule H2 --ancillas 4 --shots 1000 --trotter-steps 2

qite run --molecule H2 --steps 75 --dtau 0.2

qite run-qrte --molecule H2 --steps 20 --dt 0.05
```

Python expert-mode example:

```python
import pennylane as qml

from qite.core import run_qite
from vqe.core import run_vqe

H_model = qml.Hamiltonian(
    [1.0, 0.4],
    [qml.PauliZ(0), qml.PauliX(0)],
)

vqe_res = run_vqe(
    hamiltonian=H_model,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    optimizer_name="Adam",
    steps=10,
    plot=False,
)

qite_res = run_qite(
    hamiltonian=H_model,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    steps=10,
    dtau=0.1,
    plot=False,
    show=False,
)
```

---

# VQE Workflows

Supports:

- ground-state optimisation
- ADAPT-VQE ansatz growth
- geometry scans
- noise studies
- excited-state solvers

Canonical entrypoint:

```python
from vqe.core import run_vqe
```

---

## Basic VQE

```bash
vqe --molecule H2
```

Defaults:

- ansatz → `UCCSD`
- optimizer → `Adam`
- steps → `75`

Equivalent Python:

```python
from vqe.core import run_vqe

res = run_vqe(molecule="H2")

print(res["energy"])
```

### Non-molecule expert mode

Use this when you already have a qubit Hamiltonian and want to benchmark VQE directly.

```python
import pennylane as qml

from vqe.core import run_vqe

H_model = qml.Hamiltonian(
    [1.0, -0.7],
    [qml.PauliZ(0), qml.PauliX(0)],
)

res = run_vqe(
    hamiltonian=H_model,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="auto",
    optimizer_name="Adam",
    steps=20,
    plot=False,
)
```

Notes:

- expert-mode VQE is Python-only
- prebuilt-Hamiltonian runs use cache keys based on a canonical Pauli-term fingerprint, `num_qubits`, `reference_state`, resolved ansatz, solver settings, and seed
- `reference_state` should be a computational-basis bitstring of length `num_qubits`
- `ansatz_name="auto"` can select TFIM, XXZ/Heisenberg, SSH-like hopping, or generic fallback ansatzes from Hamiltonian structure
- generic model Hamiltonians should use non-chemistry ansatzes unless you also provide chemistry metadata

---

## Ansatz selection

```bash
vqe -m H2 -a UCCSD
vqe -m H2 -a RY-CZ
vqe -m H2 -a StronglyEntanglingLayers
```

Guidance:

| Ansatz                   | Typical use                |
| ------------------------ | -------------------------- |
| UCCSD                    | chemistry baseline         |
| RY-CZ                    | lightweight reference      |
| StronglyEntanglingLayers | expressive hardware ansatz |

See:

```
more_docs/vqe/ansatzes.md
```

---

## Optimizer selection

```bash
vqe -m H2 -o Adam
vqe -m H2 -o GradientDescent
vqe -m H2 -o NesterovMomentum
```

Guidance:

| Optimizer        | Behaviour                               |
| ---------------- | --------------------------------------- |
| Adam             | robust default                          |
| GradientDescent  | baseline                                |
| NesterovMomentum | faster convergence on smooth landscapes |

If `stepsize` / `--stepsize` is omitted for VQE workflows, the calibrated default for the selected optimizer is used.

See:

```
more_docs/vqe/optimizers.md
```

---

## Geometry scans

```bash
vqe \
  --scan-geometry H2_BOND \
  --range 0.5 1.5 7
```

Produces:

- energy curves
- cached intermediate Hamiltonians
- reproducible scan identifiers

---

## Noise studies

```bash
vqe \
  -m H2 \
  --multi-seed-noise \
  --noise-type depolarizing
```

Noise options:

- depolarizing
- amplitude damping
- combined channels

---

## Low-qubit benchmark

Use this when you want one decision-grade VQE summary across the supported small molecules instead of a single-molecule sweep.

Python:

```python
from vqe import run_vqe_low_qubit_benchmark

bench = run_vqe_low_qubit_benchmark(
    max_qubits=10,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    seeds=[0, 1, 2],
    show=False,
)

for row in bench["rows"]:
    print(
        row["molecule"],
        row["num_qubits"],
        row["abs_error_mean"],
        row["runtime_mean_s"],
    )
```

Reported per molecule:

- resolved qubit count
- Hamiltonian term count
- exact ground-state reference energy
- mean / standard deviation of final VQE energy across seeds
- mean / standard deviation of absolute error against exact diagonalization
- mean / standard deviation of original compute runtime

By default, molecules that cannot be run with the selected ansatz are skipped and reported under `bench["skipped"]`.
Set `skip_failures=False` if you want the first incompatible case to raise immediately.

When cached artifacts already exist, the benchmark prefers each run's stored `compute_runtime_s` value over the current cache-hit wall time, so runtime tables still reflect the original compute cost.

---

# Excited-State Methods

Two categories are supported.

---

## Post-VQE methods

Require a converged noiseless VQE reference.

- LR-VQE
- EOM-VQE
- QSE
- EOM-QSE

Example:

```bash
vqe -m H2 --lr-vqe --lr-k 4
```

Typical workflow:

```
VQE → response problem → excitation energies
```

---

## Variational excited states

Optimised directly.

- SSVQE
- VQD

Example:

```bash
vqe -m H2 --vqd --k 3
```

---

# QPE

Quantum Phase Estimation using shared Hamiltonians.

Canonical entrypoint:

```python
from qpe.core import run_qpe
```

---

## Basic QPE

```bash
qpe --molecule H2 --ancillas 4
```

Baseline defaults:

- `n_ancilla=4`
- `t=1.0`
- `trotter_steps=2`
- `shots=1000`

These defaults are calibrated against `H2` and should be treated as baseline small-molecule settings, not universally optimized values for every chemistry problem.

Equivalent Python:

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, _, hf_state = build_hamiltonian("H2")

res = run_qpe(
    hamiltonian=H,
    hf_state=hf_state,
    n_ancilla=4,
)

print(res["energy"])
```

Expert-mode QPE also supports direct qubit-Hamiltonian input:

```python
import pennylane as qml

from qpe.core import run_qpe

H_model = qml.Hamiltonian(
    [1.0, 0.5],
    [qml.PauliZ(0), qml.PauliX(0)],
)

res = run_qpe(
    hamiltonian=H_model,
    hf_state=[1],
    system_qubits=1,
    n_ancilla=4,
    shots=2000,
    plot=False,
)
```

Notes:

- QPE expert mode requires both `hamiltonian` and `hf_state`
- `system_qubits` defaults to the `hf_state` length when omitted, but cannot be smaller than the Hamiltonian wire count
- prebuilt-Hamiltonian QPE runs use cache keys based on a canonical Pauli-term fingerprint, `hf_state`, system-qubit count, phase-estimation settings, seed, shots, and noise settings
- for finite-shot QPE, `seed` is still meaningful because sampling is stochastic; in analytic mode (`shots=None`) it is effectively irrelevant

---

## Noise

```bash
qpe \
  --molecule H2 \
  --noisy \
  --p-dep 0.05
```

---

## Time evolution controls

```bash
qpe \
  --molecule H2 \
  --t 2.0 \
  --trotter-steps 4
```

Controls:

- simulation time
- Trotter depth
- precision vs cost tradeoff

---

# QITE / Projected Dynamics

Imaginary-time and real-time projected evolution via McLachlan updates.

Canonical entrypoint:

```python
from qite.core import run_qite, run_qrte
```

---

## Execution modes

| Mode         | Purpose                       |
| ------------ | ----------------------------- |
| `run`        | noiseless parameter evolution |
| `eval-noise` | noisy measurement             |
| `sweep`      | multi-noise statistics        |

---

## Run

```bash
qite run \
  --molecule H2 \
  --steps 75 \
  --dtau 0.2
```

Equivalent Python:

```python
from qite.core import run_qite

res = run_qite(
    molecule="H2",
    steps=75,
    dtau=0.2,
)

print(res["energy"])
```

### Non-molecule expert mode

`run_qite(...)` and `run_qrte(...)` also accept a prebuilt qubit Hamiltonian:

```python
import pennylane as qml

from qite.core import run_qite, run_qrte

H_model = qml.Hamiltonian(
    [1.0, 0.25],
    [qml.PauliZ(0), qml.PauliX(0)],
)

qite_res = run_qite(
    hamiltonian=H_model,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    steps=10,
    dtau=0.1,
    plot=False,
    show=False,
)

qrte_res = run_qrte(
    hamiltonian=H_model,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    steps=10,
    dt=0.05,
    plot=False,
    show=False,
)
```

## Real-time run

```bash
qite run-qrte \
  --molecule H2 \
  --steps 50 \
  --dt 0.05
```

Equivalent Python:

```python
from qite.core import run_qrte

res = run_qrte(
    molecule="H2",
    steps=50,
    dt=0.05,
)

print(res["energy"])
print(res["times"])
```

Use `run_qrte()` after a relevant state has already been identified or prepared.
In practice that usually means:

- prepare a ground state with `run_vqe()` or `run_qite()`
- prepare an excited or approximate spectral reference with the excited-state tools
- evolve that prepared state in time and analyze observables rather than energy minimization

---

## Noisy evaluation

```bash
qite eval-noise \
  --molecule H2 \
  --dep 0.02
```

---

## Noise sweep

```bash
qite eval-noise \
  --molecule H2 \
  --sweep-dep 0,0.02,0.04 \
  --seeds 0,1,2
```

---

## Cache semantics

VarQITE / VarQRTE cache keys include:

- molecule, geometry
- canonical Pauli-term Hamiltonian fingerprints for prebuilt-Hamiltonian expert-mode runs
- ansatz
- `ansatz_kwargs`
- `steps`, `dtau` or `dt`
- solver parameters
- seed
- reference bitstrings for expert-mode Hamiltonian runs
- initialization metadata for prepared-state VarQRTE runs

Ensures:

- reproducible optimisation trajectories
- noise evaluation does not invalidate optimisation cache
- runtime-based benchmarks require recorded compute timings; legacy ADAPT cache hits may omit them

---

# Reproducibility

All stacks provide:

- deterministic hashing
- JSON-first outputs
- seed-aware execution
- identical Hamiltonians across algorithms

Force recomputation:

```bash
vqe --force

qpe --force

qite run --force

qite run-qrte --force
```

---

## Testing

The default pytest target runs the fast development suite.
Slow chemistry checks and selected subprocess CLI integration tests are available behind pytest markers.

```bash
pytest -q
```

Run every test, including slow integration coverage:

```bash
pytest -q -o addopts=''
```

---

## Citation

Sid Richards (2026)

Unified Variational and Phase-Estimation Quantum Simulation Suite

## Author

Sid Richards

- LinkedIn: [sid-richards-21374b30b](https://www.linkedin.com/in/sid-richards-21374b30b/)
- GitHub: [SidRichardsQuantum](https://github.com/SidRichardsQuantum)

## License

MIT. See [LICENSE](LICENSE).
