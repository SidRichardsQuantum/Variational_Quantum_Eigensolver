# Variational Quantum Imaginary-Time Evolution

Variational Quantum Imaginary-Time Evolution, or VarQITE, approximates
imaginary-time evolution inside a parameterized quantum circuit. In this
package it is exposed through `qite.run_qite(...)` and the `qite run` command.

Use VarQITE when you want a variational route toward a low-energy state that is
different from optimizer-based VQE. It is most useful as a ground-state
relaxation method, as a point of comparison against VQE and QPE, or as a state
preparation step before real-time projected dynamics.

## Quickstart

Python:

```python
from qite import run_qite

result = run_qite(
    molecule="H2",
    ansatz_name="UCCSD",
    steps=75,
    dtau=0.2,
    plot=False,
)

print(result["energy"])
print(result["energies"][-5:])
```

CLI:

```bash
python -m qite run --molecule H2 --ansatz UCCSD --steps 75 --dtau 0.2 --no-show
```

The default command is also `run`, so this is equivalent:

```bash
python -m qite --molecule H2 --steps 75 --dtau 0.2
```

## What The Solver Does

Imaginary-time evolution applies:

```{math}
|\psi(\tau)\rangle \propto e^{-H\tau} |\psi(0)\rangle
```

Higher-energy components decay faster than lower-energy components. VarQITE
does not apply this non-unitary operator directly. Instead, it projects the
imaginary-time derivative back into the tangent space of an ansatz
`|psi(theta)>`.

The implementation uses McLachlan's variational principle, which gives a
linear system at each step:

```{math}
A(\theta)\dot{\theta} = -C(\theta)
```

with:

```{math}
A_{ij} = \operatorname{Re}\langle \partial_i \psi | \partial_j \psi \rangle
```

and:

```{math}
C_i = \operatorname{Re}\langle \partial_i \psi |
(H - \langle H \rangle) | \psi \rangle
```

The discrete parameter update is:

```{math}
\theta \leftarrow \theta + \Delta\tau\,\dot{\theta}
```

## Inputs

`run_qite(...)` uses the same shared problem-resolution layer as the VQE and
QPE modules. You can run against a built-in molecule, an explicit geometry, an
active-space chemistry problem, or an expert-mode qubit Hamiltonian.

### Registry Molecule

```python
from qite import run_qite

result = run_qite(molecule="H2", steps=75, dtau=0.2, plot=False)
```

### Explicit Geometry

```python
from qite import run_qite

result = run_qite(
    symbols=["H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    basis="sto-3g",
    charge=0,
    unit="angstrom",
    steps=75,
    dtau=0.2,
    plot=False,
)
```

CLI equivalent:

```bash
python -m qite run \
  --symbols H,H \
  --coordinates "0,0,0; 0,0,0.7414" \
  --basis sto-3g \
  --steps 75 \
  --dtau 0.2
```

### Active Space

```python
from qite import run_qite

result = run_qite(
    molecule="LiH",
    active_electrons=2,
    active_orbitals=2,
    steps=75,
    dtau=0.15,
    plot=False,
)
```

### Expert-Mode Hamiltonian

```python
import pennylane as qml

from qite import run_qite

hamiltonian = qml.Hamiltonian(
    [1.0, -0.7, 0.25],
    [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)],
)

result = run_qite(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ansatz_name="auto",
    steps=40,
    dtau=0.1,
    plot=False,
)
```

Expert-mode runs are cacheable. The cache key includes a canonical
Hamiltonian fingerprint, the qubit count, and the reference state.

## Ansatz Selection

The default ansatz is `UCCSD`, which is appropriate for the small chemistry
examples used throughout the repository. VarQITE delegates ansatz construction
to the VQE ansatz plumbing where possible, so charge-aware and active-space
chemistry ansatz behavior stays consistent across the packages.

Common choices:

| Ansatz | Use when |
| ------ | -------- |
| `UCCSD` | Small chemistry problems with a chemistry reference state |
| `RY-CZ` | Lightweight hardware-efficient smoke tests |
| `StronglyEntanglingLayers` | Generic variational baselines |
| `auto` | Expert-mode qubit Hamiltonians where the package should pick a conservative ansatz |

Additional ansatz options can be passed with `ansatz_kwargs`:

```python
result = run_qite(
    molecule="H2",
    ansatz_name="StronglyEntanglingLayers",
    ansatz_kwargs={"layers": 2},
    steps=50,
    dtau=0.1,
    plot=False,
)
```

## Numerical Controls

The main VarQITE controls are:

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `steps` | `75` | Number of imaginary-time updates |
| `dtau` | `0.2` | Imaginary-time step size |
| `fd_eps` | `1e-3` | Central finite-difference step for tangent vectors |
| `reg` | `1e-6` | Diagonal regularization added to the McLachlan matrix |
| `solver` | `"solve"` | Linear-system backend: `"solve"`, `"lstsq"`, or `"pinv"` |
| `pinv_rcond` | `1e-10` | Cutoff used by the pseudoinverse solver |

Use smaller `dtau` values when the energy trace oscillates or diverges. Increase
`steps` when the trace is still decreasing at the end of a run. Use
`solver="pinv"` or `solver="lstsq"` when the direct solve is unstable for an
ill-conditioned ansatz tangent space.

CLI example:

```bash
python -m qite run \
  --molecule H2 \
  --steps 100 \
  --dtau 0.1 \
  --solver pinv \
  --reg 1e-5 \
  --pinv-rcond 1e-9
```

## Outputs

`run_qite(...)` returns a dictionary with the final energy, convergence trace,
final state, final parameters, timing metadata, and VarQITE numerical settings.

Important fields:

| Field | Meaning |
| ----- | ------- |
| `energy` | Final energy in Hartree |
| `energies` | Energy after initialization and each VarQITE step |
| `final_params` | Flattened final ansatz parameters |
| `final_params_shape` | Shape needed to reconstruct the parameter array |
| `final_state_real`, `final_state_imag` | Final statevector components |
| `varqite` | `fd_eps`, `reg`, `solver`, and `pinv_rcond` used for the run |
| `compute_runtime_s` | Time spent computing or original cached compute time |
| `runtime_s` | Wall time for this call, including cache lookup |
| `cache_hit` | Whether the returned result came from cache |

Reconstructing parameters:

```python
import numpy as np

params = np.array(result["final_params"]).reshape(result["final_params_shape"])
```

## Caching

VarQITE stores JSON run records under the package's standard `results/qite/`
location. Cache keys include the resolved problem, ansatz, seed, step count,
`dtau`, active-space settings, and numerical controls such as `fd_eps`, `reg`,
`solver`, and `pinv_rcond`.

Use `force=True` or `--force` to ignore an existing cache record:

```python
result = run_qite(molecule="H2", force=True, plot=False)
```

```bash
python -m qite run --molecule H2 --force
```

## Noise Policy

VarQITE parameter updates require pure statevectors and stable derivatives.
Noisy or mixed-state optimization is therefore intentionally rejected by
`run_qite(...)`.

If you want to evaluate a converged VarQITE circuit under noise, use the CLI
post-evaluation workflow:

```bash
python -m qite eval-noise \
  --molecule H2 \
  --steps 75 \
  --dtau 0.2 \
  --depolarizing-prob 0.02 \
  --pretty
```

For noise sweeps:

```bash
python -m qite eval-noise \
  --molecule H2 \
  --steps 75 \
  --dtau 0.2 \
  --sweep-noise-type depolarizing \
  --sweep-levels 0,0.01,0.02,0.04 \
  --seeds 0,1,2,3,4 \
  --json
```

This workflow first obtains the noiseless converged parameters, using the cache
when available, then evaluates the resulting circuit on `default.mixed`.

## Plotting

By default, `run_qite(...)` plots and saves the convergence curve. In scripts,
CI, or docs examples, disable plotting:

```python
result = run_qite(molecule="H2", plot=False)
```

For CLI runs, `--no-show` is useful in headless environments:

```bash
python -m qite run --molecule H2 --no-show
```

## Method Selection

Use VarQITE when:

- you want a projected imaginary-time relaxation workflow
- you need a comparison point against VQE optimizer behavior
- you want converged parameters for post-evaluation or projected dynamics
- the problem is small enough for statevector derivative calculations

Prefer VQE when:

- you want direct optimizer control and broader noisy-evaluation comparisons
- you are primarily benchmarking ansatz and optimizer choices
- you do not need imaginary-time dynamics diagnostics

Prefer QPE when:

- phase or spectral information is the main target
- you want to study ancilla, shot, and time-evolution tradeoffs

## Troubleshooting

| Symptom | Likely cause | Try |
| ------- | ------------ | --- |
| Energy oscillates or increases sharply | `dtau` is too large or the tangent-space solve is unstable | Lower `dtau`, increase `reg`, or use `solver="pinv"` |
| Direct solve fails | McLachlan matrix is singular or ill-conditioned | Use `solver="lstsq"` or `solver="pinv"` |
| Run is slow | Finite-difference state derivatives scale with parameter count | Use a smaller ansatz, fewer steps, or a smaller active space |
| `ValueError` about noise | VarQITE updates are pure-state only | Use `qite eval-noise` for post-evaluation |
| Repeated run returns immediately | Cache hit | Use `force=True` or `--force` to recompute |

## Further Reading

The implementation details live in:

- `qite.core.run_qite`
- `qite.engine.qite_step`
- `qite.engine.make_state_qnode`
- `qite.engine.make_energy_qnode`

Related notebooks:

- `notebooks/getting_started/07_getting_started_qite_h2.ipynb`
- `notebooks/benchmarks/defaults/VarQITE_Default_Calibration.ipynb`
- `notebooks/benchmarks/comparisons/H2/Cross_Method_Comparison.ipynb`
- `notebooks/benchmarks/comparisons/LiH/Cross_Method_Comparison.ipynb`
