# Usage Guide

This document covers the main **CLI workflows** and selected **Python APIs** for:

- **VQE** — ground-state VQE, ADAPT-VQE, and excited-state workflows
- **QPE** — Quantum Phase Estimation
- **QITE / VarQITE** — variational imaginary-time evolution
- **common** — shared Hamiltonian, molecule, geometry, plotting, and persistence utilities

It complements:

- **`README.md`** — overview and quickstart
- **`THEORY.md`** — algorithms and methodology
- **`docs/vqe/ansatzes.md`** — VQE ansatz reference
- **`docs/vqe/optimizers.md`** — VQE optimizer reference

---

## Installation

### Install from PyPI

```bash
pip install vqe-pennylane
```

### Install from source

```bash
git clone https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver.git
cd Variational_Quantum_Eigensolver
pip install -e .
```

Installed packages:

| Package  | Purpose                                               |
| -------- | ----------------------------------------------------- |
| `vqe`    | Variational solvers and excited-state workflows       |
| `qpe`    | Quantum Phase Estimation                              |
| `qite`   | Variational imaginary-time evolution                  |
| `common` | Shared chemistry, geometry, plotting, and persistence |

Verify installation:

```bash
python -c "import vqe, qpe, qite, common; print('All stacks OK')"
```

---

## General Conventions

All workflows share the same core conventions:

- **Cached outputs** are written under `results/`
- **Generated figures** are written under `images/`
- **Run identity** is determined from the full physical and numerical configuration
- **`--force`** bypasses cache and recomputes the run
- CLI commands can be run either as installed entrypoints (`vqe`, `qpe`, `qite`) or via `python -m ...`

Default output layout:

```
results/
├── vqe/
├── qpe/
└── qite/

images/
├── vqe/
├── qpe/
└── qite/
```

---

## Method Support Summary

| Method             | Family                    | Requires converged VQE reference | Noise Support         |
| ------------------ | ------------------------- | -------------------------------: | --------------------- |
| **VQE**            | Variational               |                               No | Yes                   |
| **ADAPT-VQE**      | Variational               |                               No | Yes                   |
| **LR-VQE**         | Post-VQE                  |                              Yes | No                    |
| **EOM-VQE**        | Post-VQE                  |                              Yes | No                    |
| **QSE**            | Post-VQE                  |                              Yes | No                    |
| **EOM-QSE**        | Post-VQE                  |                              Yes | No                    |
| **SSVQE**          | Variational excited state |                               No | Yes                   |
| **VQD**            | Variational excited state |                               No | Yes                   |
| **QPE**            | Phase estimation          |                               No | Yes                   |
| **QITE / VarQITE** | Imaginary time            |                               No | Noisy evaluation only |

---

## Quickstart Commands

### Ground-state VQE

```bash
vqe --molecule H2
```

### LR-VQE

```bash
vqe -m H2 --lr-vqe --lr-k 4
```

### QPE

```bash
qpe --molecule H2 --ancillas 4
```

### VarQITE

```bash
qite run --molecule H2 --steps 50 --dtau 0.2
```

---

## Running VQE

All VQE-family commands can be invoked as either:

```bash
vqe ...
```

or

```bash
python -m vqe ...
```

The `vqe` stack supports:

- ground-state VQE
- ADAPT-VQE
- geometry scans
- ansatz / optimizer / mapping comparisons
- noise studies
- excited-state workflows:

  - **post-VQE:** LR-VQE, EOM-VQE, QSE, EOM-QSE
  - **variational:** SSVQE, VQD

Supported molecule presets:

```
H2, LiH, H2O, H3+
```

### Basic ground-state VQE

```bash
vqe --molecule H2
```

Default settings:

- ansatz: `UCCSD`
- optimizer: `Adam`
- steps: `50`
- mapping: `jordan_wigner`

Typical outputs:

- `results/vqe/` — JSON record
- `images/vqe/` — plot output when plotting/saving is enabled

### Choosing ansatz and optimizer

```bash
vqe -m H2 -a UCCSD -o Adam
vqe -m H2 -a RY-CZ -o GradientDescent
vqe -m H2 -a StronglyEntanglingLayers -o Momentum
```

### Ansatz guidance

The best ansatz depends on whether your priority is chemistry structure,
hardware efficiency, or simple debugging.

Practical recommendations for this repository:

- **UCCSD (default)**
  - best starting point for small-molecule quantum chemistry
  - chemistry-informed and Hartree–Fock based
  - usually the most physically meaningful default choice

- **UCC-S / UCC-D**
  - useful for ablations or smaller chemistry-inspired tests
  - `UCC-S` is simpler but less expressive
  - `UCC-D` often captures more correlation than singles-only

- **RY-CZ**
  - simple hardware-efficient baseline
  - useful for optimizer comparisons and trainability studies
  - lighter-weight than UCC-style ansatzes

- **TwoQubit-RY-CNOT**
  - structured toy/scalable baseline
  - useful for small experiments and debugging

- **StronglyEntanglingLayers**
  - more expressive hardware-efficient template
  - useful for general variational experiments
  - can be harder to optimize cleanly

- **Minimal**
  - pedagogical/debugging ansatz only
  - useful for very small demonstrations, not realistic chemistry studies

General tips:

- start with `UCCSD` for chemistry-oriented VQE
- start with `RY-CZ` for lightweight optimizer comparisons
- use `Minimal` or `TwoQubit-RY-CNOT` for debugging and sanity checks
- prefer simpler ansatzes when diagnosing optimizer or noise behaviour

For full ansatz definitions, parameter conventions, and initialization details, see:

→ `docs/vqe/ansatzes.md`

### Optimizer guidance

The choice of optimizer can significantly affect VQE convergence speed and stability.

Practical recommendations for this repository:

- **Adam (default)**
  - good general-purpose choice
  - robust across most molecules and ansätze
  - typical stepsize: `0.1 – 0.3`

- **GradientDescent**
  - useful baseline for comparison
  - requires smaller stepsizes (e.g. `0.01 – 0.1`)
  - may converge slowly or oscillate

- **Momentum / Nesterov**
  - can accelerate convergence relative to GradientDescent
  - useful for smoother energy landscapes
  - still sensitive to stepsize

- **RMSProp / Adagrad**
  - adaptive per-parameter scaling
  - can help when parameters evolve on different scales
  - Adagrad may become overly conservative over long runs

General tips:

- if convergence is unstable → reduce `--stepsize`
- if convergence is too slow → increase `--stepsize` slightly
- for small molecules (e.g. `H2`), moderately larger stepsizes often work well
- for larger systems or deeper ansätze, smaller stepsizes are usually safer

For full mathematical definitions and update rules, see:

→ `docs/vqe/optimizers.md`

### Geometry scans

#### H2 bond scan

```bash
vqe --scan-geometry H2_BOND --range 0.5 1.5 7
```

#### H2O angle scan

```bash
vqe --scan-geometry H2O_ANGLE --range 100 115 7
```

### Noise studies

```bash
vqe -m H2 --multi-seed-noise --noise-type depolarizing
```

This mode is intended for **statistical noise analysis**, not just single-run demonstration.

### Python API example

```python
from vqe.core import run_vqe

res = run_vqe(
    molecule="H2",
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=50,
)

print(res["energy"])
```

---

## Excited-State Workflows

The repository supports two classes of excited-state methods.

### Post-VQE methods

These operate on top of a converged **noiseless** VQE reference state:

- **LR-VQE** — tangent-space linear response (TDA)
- **EOM-VQE** — full-response tangent-space equation of motion
- **QSE** — projection-based operator subspace expansion
- **EOM-QSE** — commutator equation of motion in an operator manifold

### Variational excited-state methods

These solve excited states directly:

- **SSVQE** — simultaneous multi-state optimization
- **VQD** — sequential deflation

---

## LR-VQE

LR-VQE constructs a tangent-space generalized eigenvalue problem around a converged
**noiseless** VQE reference state.

Properties:

- tangent-space Tamm–Dancoff approximation
- finite-difference parameter derivatives
- generalized EVP
- **statevector-only**

### CLI

```bash
# Basic LR-VQE run
vqe -m H2 --lr-vqe --lr-k 4

# Plot spectrum
vqe -m H2 --lr-vqe --lr-k 4 --plot

# Save plot output
vqe -m H2 --lr-vqe --lr-k 4 --save

# Control tangent numerics
vqe -m H2 --lr-vqe --lr-k 4 --lr-fd-eps 1e-3 --lr-eps 1e-10
```

### Python API

```python
from vqe.lr_vqe import run_lr_vqe

res = run_lr_vqe(
    molecule="H2",
    k=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=80,
    stepsize=0.2,
    mapping="jordan_wigner",
    fd_eps=1e-3,
    eps=1e-10,
)

print(res["excitations"])
print(res["eigenvalues"])
```

---

## EOM-VQE

EOM-VQE solves the **full-response** tangent-space equation-of-motion problem around a
converged **noiseless** VQE reference state.

Properties:

- full-response tangent-space solve
- positive-root selection
- overlap filtering / rank truncation
- **statevector-only**

### CLI

```bash
# Basic EOM-VQE run
vqe -m H2 --eom-vqe --eom-k 4

# Plot spectrum
vqe -m H2 --eom-vqe --eom-k 4 --plot

# Save plot output
vqe -m H2 --eom-vqe --eom-k 4 --save

# Control tangent numerics
vqe -m H2 --eom-vqe --eom-k 4 --eom-fd-eps 1e-3 --eom-eps 1e-10 --eom-omega-eps 1e-12
```

### Python API

```python
from vqe.eom_vqe import run_eom_vqe

res = run_eom_vqe(
    molecule="H2",
    k=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=80,
    stepsize=0.2,
    mapping="jordan_wigner",
    fd_eps=1e-3,
    eps=1e-10,
    omega_eps=1e-12,
)

print(res["excitations"])
print(res["eigenvalues"])
```

---

## QSE

QSE computes approximate excited-state energies by expanding an operator subspace around a
converged **noiseless** VQE reference state.

### CLI

```bash
vqe -m H2 --qse --qse-k 4 --qse-max-ops 24 --qse-eps 1e-8
```

### Python API

```python
from vqe.qse import run_qse

res = run_qse(
    molecule="H2",
    k=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=60,
    stepsize=0.2,
    mapping="jordan_wigner",
)

print(res["eigenvalues"])
```

---

## EOM-QSE

EOM-QSE computes excitation energies from the **commutator equation of motion**
in an operator manifold around a converged **noiseless** VQE reference state.

Properties:

- generally **non-Hermitian**
- eigenvalues may be complex
- returns positive, real-dominant roots
- **statevector-only**

### CLI

```bash
# Basic EOM-QSE run
vqe -m H2 --eom-qse --eom-qse-k 4 --eom-qse-max-ops 24 --eom-qse-eps 1e-10

# Control root filtering
vqe -m H2 --eom-qse --eom-qse-k 4 --eom-qse-imag-tol 1e-10 --eom-qse-omega-eps 1e-12
```

### Python API

```python
from vqe.eom_qse import run_eom_qse

res = run_eom_qse(
    molecule="H2",
    k=3,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=80,
    stepsize=0.2,
    mapping="jordan_wigner",
    pool="hamiltonian_topk",
    max_ops=24,
    eps=1e-10,
    imag_tol=1e-10,
    omega_eps=1e-12,
)

print(res["excitations"])
print(res["eigenvalues"])
```

---

## SSVQE

SSVQE optimizes multiple states **simultaneously** using a shared-parameter unitary.

### CLI

```bash
# Two states (default)
vqe -m H3+ --ssvqe --num-states 2

# Custom weights
vqe -m H3+ --ssvqe --num-states 3 --weights 1 2 3
```

### Python API

```python
from vqe.ssvqe import run_ssvqe

res = run_ssvqe(
    molecule="H3+",
    num_states=3,
)

print(res["energies_per_state"])
```

---

## VQD

VQD computes excited states **sequentially** using deflation against previously converged states.

### CLI

```bash
# Two states (default)
vqe -m H3+ --vqd --num-states 2 --beta 10.0

# Optional beta scheduling controls
vqe -m H3+ --vqd --num-states 3 --beta 10.0 --beta-start 0.0 --beta-ramp cosine --beta-hold-fraction 0.2
```

### Python API

```python
from vqe.vqd import run_vqd

res = run_vqd(
    molecule="H3+",
    num_states=3,
)

print(res["energies_per_state"])
```

---

## ADAPT-VQE

ADAPT-VQE grows the ansatz adaptively by selecting operators from a pool using
gradient-based scores until a stopping criterion is met.

### CLI

```bash
# Basic ADAPT-VQE run
vqe -m H3+ --adapt

# Configure pool and stopping criteria
vqe -m H3+ --adapt --pool uccsd --max-ops 20 --grad-tol 1e-3

# Override inner-loop optimization
vqe -m H3+ --adapt --inner-steps 75 --inner-stepsize 0.2

# Noisy ADAPT-VQE
vqe -m H3+ --adapt --noisy --depolarizing-prob 0.02 --amplitude-damping-prob 0.0
```

### Python API

```python
from vqe.adapt import run_adapt_vqe

res = run_adapt_vqe(
    molecule="H3+",
    pool="uccsd",
    max_ops=20,
    grad_tol=1e-3,
    inner_steps=75,
    inner_stepsize=0.2,
    optimizer_name="Adam",
    seed=0,
    mapping="jordan_wigner",
    plot=True,
)

print(res["energy"])
```

---

## Running QPE

QPE estimates energies by phase estimation using the same Hamiltonian pipeline as the VQE stack.

### Basic QPE

```bash
qpe --molecule H2 --ancillas 4
```

### Noisy QPE

```bash
qpe --molecule H2 --noisy --p-dep 0.05 --p-amp 0.02
```

`--noisy` must be explicitly enabled; otherwise noise parameters are ignored.

### Trotterized evolution

```bash
qpe --molecule H2 --t 2.0 --trotter-steps 4 --ancillas 8
```

### Python API

```python
from common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe

H, n_qubits, hf_state = build_hamiltonian("H2")

res = run_qpe(
    hamiltonian=H,
    hf_state=hf_state,
    n_ancilla=4,
)

print(res["energy"])
```

---

## Running QITE / VarQITE

QITE uses the McLachlan variational principle to approximate imaginary-time evolution.

The implementation is split into two explicit modes:

- **`qite run`** — noiseless parameter evolution
- **`qite eval-noise`** — noisy evaluation of converged parameters

### Noiseless VarQITE run

```bash
qite run --molecule H2 --steps 50 --dtau 0.2
```

Properties:

- pure-state evolution
- cached parameter trajectories
- convergence diagnostics and JSON outputs
- `default.qubit` reference workflow

### Noisy evaluation of converged parameters

```bash
qite eval-noise --molecule H2 --dep 0.02 --amp 0.0 --pretty
```

Properties:

- evaluates `Tr[ρH]` on `default.mixed`
- reuses cached VarQITE parameters
- does not re-optimize
- supports sweeps and multi-seed summaries

### Depolarizing sweep

```bash
qite eval-noise \
  --molecule H2 \
  --steps 50 \
  --sweep-dep 0,0.02,0.04 \
  --seeds 0,1,2 \
  --pretty
```

### Python API

```python
from qite.core import run_qite

res = run_qite(
    molecule="H2",
    ansatz_name="UCCSD",
    steps=50,
    dtau=0.2,
    seed=0,
)

print(res["energy"])
```

### QITE caching semantics

VarQITE cache keys include:

- molecule and geometry
- mapping and unit
- ansatz
- seed
- `dtau` and `steps`
- numerical solver settings such as `fd_eps`, `reg`, `solver`, and `pinv_rcond`

This ensures:

- changes in numerics trigger recomputation
- cached trajectories remain physically and numerically consistent
- noisy evaluation does not pollute optimization caches

---

## Caching and Reproducibility

All algorithm families share:

- unified Hamiltonian construction via `common.hamiltonian`
- deterministic run hashing
- JSON-first records
- seed-aware caching
- plot regeneration without recomputing core runs

Force recomputation with:

```bash
vqe --force
qpe --force
qite run --force
```

---

## Testing

Run the test suite with:

```bash
pytest -q
```

Coverage includes:

- Hamiltonian and molecule utilities
- minimal VQE / ADAPT-VQE / LR-VQE / EOM-VQE / QSE / EOM-QSE / QPE / QITE runs
- noise handling
- CLI entrypoints
- cross-stack consistency checks

---

## Citation

If you use this software, please cite:

> Sid Richards (2026). *Unified Variational and Phase-Estimation Quantum Simulation Suite.*

---

**Author:** Sid Richards (SidRichardsQuantum)
LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

Licensed under the **MIT License** — see [LICENSE](LICENSE).
