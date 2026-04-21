# VQE Defaults

This page documents the default choices used by the main VQE workflow and how
to override them intentionally.

Defaults are chosen for small, reproducible examples and benchmark baselines.
They are not universal chemistry recommendations.

## Main Defaults

`vqe.run_vqe(...)` defaults to:

| Parameter | Default |
| --------- | ------- |
| `molecule` | `"H2"` |
| `seed` | `0` |
| `steps` | `75` |
| `ansatz_name` | `"UCCSD"` |
| `optimizer_name` | `"Adam"` |
| `stepsize` | calibrated per optimizer when omitted |
| `mapping` | `"jordan_wigner"` |
| `basis` | `"sto-3g"` |
| `unit` | `"angstrom"` |
| `noisy` | `False` |
| `plot` | `True` |
| `force` | `False` |

The CLI follows the same default intent. When `--stepsize` is omitted, it
passes through the omitted value so the selected optimizer can resolve its
calibrated default.

## Optimizer Stepsizes

The optimizer registry in `vqe.optimizer` defines calibrated default
stepsizes:

| Optimizer | Default stepsize |
| --------- | ---------------- |
| `Adam` | `0.15` |
| `GradientDescent` | `0.10` |
| `Momentum` | `0.10` |
| `NesterovMomentum` | `0.20` |
| `RMSProp` | `0.01` |
| `Adagrad` | `0.10` |

Use an explicit `stepsize` when you are doing optimizer calibration or when a
benchmark needs a fixed learning rate:

```python
from vqe import run_vqe

result = run_vqe(
    molecule="H2",
    optimizer_name="Adam",
    stepsize=0.1,
    steps=100,
    plot=False,
)
```

## Ansatz Defaults

The default ansatz is `UCCSD`, matching the small chemistry focus of the
package examples. It uses the shared ansatz construction path and respects
charge-aware and active-space settings.

For quick smoke tests or compact model Hamiltonians, use a lighter ansatz:

```python
result = run_vqe(
    molecule="H2",
    ansatz_name="RY-CZ",
    steps=25,
    plot=False,
)
```

For expert-mode Hamiltonians, `ansatz_name="auto"` inspects the Pauli structure
and chooses a conservative ansatz such as `TFIM-HVA`, `XXZ-HVA`,
`NumberPreservingGivens`, or `StronglyEntanglingLayers`.

## Problem Defaults

Registry mode defaults to `molecule="H2"`. The registry supplies geometry,
charge, multiplicity, basis, and coordinate unit. For custom chemistry inputs,
use explicit geometry mode:

```python
result = run_vqe(
    symbols=["H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    unit="angstrom",
    plot=False,
)
```

For larger systems, use active spaces deliberately:

```python
result = run_vqe(
    molecule="LiH",
    active_electrons=2,
    active_orbitals=2,
    plot=False,
)
```

## Noise Defaults

VQE is noiseless by default. Set `noisy=True` and one or more built-in noise
probabilities to evaluate or optimize under noise:

```python
result = run_vqe(
    molecule="H2",
    noisy=True,
    depolarizing_prob=0.02,
    steps=75,
    plot=False,
)
```

Supported built-in probabilities are:

- `depolarizing_prob`
- `amplitude_damping_prob`
- `phase_damping_prob`
- `bit_flip_prob`
- `phase_flip_prob`

When `noisy=False`, noise settings are ignored by the QNode construction path.

## Caching Defaults

VQE caches chemistry and expert-mode runs by default. The cache key includes
the resolved problem, ansatz, optimizer, steps, stepsize, seed, mapping,
active-space settings, noise settings, and expert-mode Hamiltonian fingerprint
when applicable.

Use `force=True` or `--force` to recompute:

```python
result = run_vqe(molecule="H2", force=True, plot=False)
```

## Plot Defaults

`run_vqe(...)` plots convergence by default. Disable plotting in scripts,
tests, and docs examples:

```python
result = run_vqe(molecule="H2", plot=False)
```

## Comparison Helpers

The package includes helper workflows for common default-sensitivity studies:

- ansatz comparisons
- optimizer comparisons
- mapping comparisons
- noise scans
- low-qubit multi-molecule benchmarks
- geometry scans

Use these helpers or the benchmark notebooks when changing defaults. A default
change should be justified by a reproducible comparison rather than a single
run.

## Benchmark Evidence

The main default-calibration notebook is:

- `notebooks/benchmarks/defaults/VQE_Default_Calibration.ipynb`

Benchmark-facing evidence is split between source notebooks and curated result
surfaces. For default decisions, start with:

- `notebooks/benchmarks/SUMMARY.md`
- `notebooks/benchmarks/RESULTS.md`
- `docs/benchmarks/summary.md`
- `docs/benchmarks/results.md`

The most relevant curated artifacts for VQE defaults are:

| Evidence | Artifact |
| -------- | -------- |
| H2 ansatz behavior | `notebooks/benchmarks/_artifacts/figures/h2_ansatz_comparison.png` |
| H2 mapping behavior | `notebooks/benchmarks/_artifacts/figures/h2_mapping_comparison_uccsd.png` |
| Low-qubit VQE scaling | `notebooks/benchmarks/_artifacts/figures/low_qubit_vqe.png` |
| Low-qubit VQE table | `notebooks/benchmarks/_artifacts/tables/low_qubit_vqe_summary.csv` |
| H2 noise reference seeds | `notebooks/benchmarks/_artifacts/tables/h2_noise_reference.csv` |

Use curated artifacts when documenting or reviewing defaults. Use raw
`results/` and `images/` only as local generated output. The artifact policy is
documented in `docs/common/caching_and_artifacts.md`.

When changing a default, update or rerun the relevant benchmark notebook, export
artifacts with:

```bash
python scripts/export_benchmark_artifacts.py
```

Then review `notebooks/benchmarks/RESULTS.md` and
`notebooks/benchmarks/_artifacts/` together.

## Recommended Override Strategy

Start from the defaults when:

- checking installation
- running H2 examples
- comparing against existing notebooks
- creating a small benchmark baseline

Override defaults when:

- the molecule is not H2
- the ansatz is not chemistry-oriented
- the optimizer trace is unstable
- a noisy run needs fixed channel probabilities
- an active space is required for runtime
- a paper or benchmark requires exact reproducibility
