# Variational Quantum Real-Time Evolution

Variational Quantum Real-Time Evolution, or VarQRTE, approximates real-time
Schrodinger evolution inside a parameterized quantum circuit. In this package
it is exposed through `qite.run_qrte(...)` and the `qite run-qrte` command.

Use VarQRTE when you already have a relevant initial state and want projected
real-time dynamics without leaving the variational ansatz manifold. It is a
dynamics workflow, not a ground-state optimizer. For relaxation toward a
low-energy state, use VarQITE or VQE first.

## Quickstart

Python:

```python
from qite import run_qrte

result = run_qrte(
    molecule="H2",
    ansatz_name="UCCSD",
    steps=50,
    dt=0.05,
    plot=False,
)

print(result["energy"])
print(result["times"][-5:])
```

CLI:

```bash
python -m qite run-qrte --molecule H2 --ansatz UCCSD --steps 50 --dt 0.05 --no-show
```

## What The Solver Does

Real-time evolution applies:

```{math}
|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle
```

VarQRTE approximates that trajectory with an ansatz state `|psi(theta)>`.
Instead of exactly applying the full unitary evolution, the solver projects the
real-time derivative into the tangent space of the ansatz at each step.

The implementation uses a McLachlan projected update for real time. At each
iteration it builds a tangent-space linear system, solves for `dot(theta)`, and
advances the parameters by a time step `dt`.

## Inputs

`run_qrte(...)` uses the same shared problem-resolution layer as VQE, QPE, and
VarQITE. It supports built-in molecules, explicit geometries, active-space
chemistry runs, and expert-mode qubit Hamiltonians.

### Registry Molecule

```python
from qite import run_qrte

result = run_qrte(molecule="H2", steps=50, dt=0.05, plot=False)
```

### Explicit Geometry

```python
from qite import run_qrte

result = run_qrte(
    symbols=["H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    basis="sto-3g",
    charge=0,
    unit="angstrom",
    steps=50,
    dt=0.05,
    plot=False,
)
```

CLI equivalent:

```bash
python -m qite run-qrte \
  --symbols H,H \
  --coordinates "0,0,0; 0,0,0.7414" \
  --basis sto-3g \
  --steps 50 \
  --dt 0.05
```

### Active Space

```python
from qite import run_qrte

result = run_qrte(
    molecule="LiH",
    active_electrons=2,
    active_orbitals=2,
    steps=50,
    dt=0.025,
    plot=False,
)
```

### Expert-Mode Hamiltonian

```python
import pennylane as qml

from qite import run_qrte

hamiltonian = qml.Hamiltonian(
    [1.0, -0.5, 0.25],
    [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)],
)

result = run_qrte(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ansatz_name="auto",
    steps=40,
    dt=0.05,
    plot=False,
)
```

Expert-mode runs are cacheable. The cache key includes a canonical
Hamiltonian fingerprint, the qubit count, the reference state, the time step,
and the numerical controls.

## Initial State And Prepared Parameters

By default, VarQRTE initializes the selected ansatz from the same seeded
parameter initialization used by the QITE ansatz builder. For dynamics studies,
you often want to start from a prepared state instead.

The Python API accepts `initial_params`:

```python
import numpy as np

from qite import run_qite, run_qrte

prepared = run_qite(molecule="H2", steps=75, dtau=0.2, plot=False)
initial_params = np.array(prepared["final_params"]).reshape(
    prepared["final_params_shape"]
)

trajectory = run_qrte(
    molecule="H2",
    initial_params=initial_params,
    steps=50,
    dt=0.05,
    plot=False,
)
```

When `initial_params` is provided, its flattened size must match the parameter
count for the selected ansatz. The returned result records
`"initialization": "provided"`.

## Ansatz Selection

VarQRTE delegates ansatz construction to the same shared QITE/VQE ansatz
plumbing used by VarQITE. The default is `UCCSD` for small chemistry examples.

Common choices:

| Ansatz | Use when |
| ------ | -------- |
| `UCCSD` | Small chemistry problems with a chemistry reference state |
| `RY-CZ` | Lightweight smoke tests and compact demos |
| `StronglyEntanglingLayers` | Generic variational baselines |
| `auto` | Expert-mode qubit Hamiltonians where the package should choose a conservative ansatz |

Additional ansatz options can be passed through `ansatz_kwargs` from Python:

```python
result = run_qrte(
    molecule="H2",
    ansatz_name="StronglyEntanglingLayers",
    ansatz_kwargs={"layers": 2},
    steps=40,
    dt=0.025,
    plot=False,
)
```

## Numerical Controls

The main VarQRTE controls are:

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `steps` | `50` | Number of real-time updates |
| `dt` | `0.05` | Real-time step size |
| `fd_eps` | `1e-3` | Central finite-difference step for tangent vectors |
| `reg` | `1e-6` | Diagonal regularization added to the tangent-space matrix |
| `solver` | `"solve"` | Linear-system backend: `"solve"`, `"lstsq"`, or `"pinv"` |
| `pinv_rcond` | `1e-10` | Cutoff used by the pseudoinverse solver |

Use smaller `dt` values when the trajectory is unstable or the energy drifts
more than expected for the chosen ansatz. Increase `reg` or use
`solver="pinv"` when the tangent-space solve is ill-conditioned.

CLI example:

```bash
python -m qite run-qrte \
  --molecule H2 \
  --steps 100 \
  --dt 0.025 \
  --solver pinv \
  --reg 1e-5 \
  --pinv-rcond 1e-9
```

## Outputs

`run_qrte(...)` returns a dictionary with the final energy, time grid, energy
trace, final state, final parameters, parameter history, timing metadata, and
VarQRTE numerical settings.

Important fields:

| Field | Meaning |
| ----- | ------- |
| `energy` | Final energy in Hartree |
| `energies` | Energy after initialization and each real-time step |
| `times` | Time grid matching `energies` |
| `final_params` | Flattened final ansatz parameters |
| `final_params_shape` | Shape needed to reconstruct the parameter array |
| `params_history` | Flattened parameter vector after each recorded time |
| `final_state_real`, `final_state_imag` | Final statevector components |
| `initialization` | `"default"` or `"provided"` |
| `varqrte` | `fd_eps`, `reg`, `solver`, and `pinv_rcond` used for the run |
| `compute_runtime_s` | Time spent computing or original cached compute time |
| `runtime_s` | Wall time for this call, including cache lookup |
| `cache_hit` | Whether the returned result came from cache |

Reconstructing the final parameters:

```python
import numpy as np

params = np.array(result["final_params"]).reshape(result["final_params_shape"])
```

## Caching

VarQRTE stores JSON run records under the package's standard `results/qite/`
location. Cache keys include the resolved problem, ansatz, seed, step count,
`dt`, initialization mode, active-space settings, and numerical controls such
as `fd_eps`, `reg`, `solver`, and `pinv_rcond`.

Use `force=True` or `--force` to ignore an existing cache record:

```python
result = run_qrte(molecule="H2", force=True, plot=False)
```

```bash
python -m qite run-qrte --molecule H2 --force
```

## Noise Policy

VarQRTE projected updates require pure statevectors. Noisy or mixed-state
real-time parameter updates are intentionally rejected by `run_qrte(...)`.

For noisy post-evaluation workflows, use VarQITE's `qite eval-noise` path for
converged VarQITE circuits. VarQRTE does not currently expose an equivalent
noisy trajectory-evaluation CLI.

## Plotting

By default, `run_qrte(...)` plots and saves the energy trace against time step.
In scripts, CI, or docs examples, disable plotting:

```python
result = run_qrte(molecule="H2", plot=False)
```

For CLI runs, `--no-show` is useful in headless environments:

```bash
python -m qite run-qrte --molecule H2 --no-show
```

## Method Selection

Use VarQRTE when:

- you want projected real-time dynamics on a prepared variational state
- you need a time-resolved comparison against exact evolution on small systems
- you want to study how an ansatz represents real-time trajectories
- the problem is small enough for statevector derivative calculations

Prefer VarQITE when:

- you want imaginary-time relaxation toward a low-energy state
- you need converged parameters before dynamics

Prefer VQE when:

- the main goal is a stationary ground-state energy
- optimizer and ansatz benchmarking is the primary question

Prefer QPE when:

- phase, spectral, or eigenvalue-estimation behavior is the target

## Troubleshooting

| Symptom | Likely cause | Try |
| ------- | ------------ | --- |
| Energy drifts strongly | `dt` is too large or the ansatz cannot represent the trajectory well | Lower `dt`, increase ansatz expressivity, or compare against exact evolution |
| Direct solve fails | Tangent-space matrix is singular or ill-conditioned | Use `solver="lstsq"` or `solver="pinv"` |
| Run is slow | Finite-difference state derivatives scale with parameter count | Use a smaller ansatz, fewer steps, or a smaller active space |
| `initial_params` raises a size error | Prepared parameters do not match the selected ansatz | Use the same ansatz/problem settings as the preparation run |
| `ValueError` about noise | VarQRTE updates are pure-state only | Run noiseless projected dynamics |
| Repeated run returns immediately | Cache hit | Use `force=True` or `--force` to recompute |

## Further Reading

The implementation details live in:

- `qite.core.run_qrte`
- `qite.engine.qrte_step`
- `qite.engine.make_state_qnode`
- `qite.engine.make_energy_qnode`

Related notebooks:

- `notebooks/getting_started/11_getting_started_qrte_h2.ipynb`
- `notebooks/qite/H2/Real_Time.ipynb`
- `notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb`
