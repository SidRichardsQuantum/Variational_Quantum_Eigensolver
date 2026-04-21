# Expert-Mode Hamiltonians

Expert mode lets solvers run on a caller-provided `pennylane.Hamiltonian`
instead of building a chemistry Hamiltonian from the molecule registry or an
explicit geometry.

Use expert mode for compact spin models, custom Pauli Hamiltonians, and
non-chemistry benchmarks such as Ising, Heisenberg, or SSH chains.

## Basic Pattern

Pass at least:

- `hamiltonian`
- `num_qubits`

For VQE, VarQITE, and VarQRTE, also pass `reference_state` when the selected
ansatz needs an initial computational-basis state.

```python
import pennylane as qml

from vqe import run_vqe

hamiltonian = qml.Hamiltonian(
    [1.0, -0.5, 0.25],
    [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)],
)

result = run_vqe(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ansatz_name="auto",
    steps=50,
    plot=False,
)
```

The same expert-mode problem can be passed to QPE and QITE routines:

```python
from qpe import run_qpe
from qite import run_qite, run_qrte

qpe_result = run_qpe(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ancillas=4,
    shots=1000,
)

qite_result = run_qite(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ansatz_name="auto",
    steps=40,
    dtau=0.1,
    plot=False,
)

qrte_result = run_qrte(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ansatz_name="auto",
    steps=40,
    dt=0.05,
    plot=False,
)
```

## Wire Handling

The shared problem resolver inspects the Hamiltonian wires. If the Hamiltonian
uses non-contiguous or non-integer wire labels, it maps them onto contiguous
wires starting at zero.

`num_qubits` cannot be smaller than the number of wires used by the provided
Hamiltonian. It can be larger if you intentionally want idle wires in the
ansatz or reference state.

## Reference States

`reference_state` is a computational-basis bitstring with length
`num_qubits`.

```python
reference_state = [1, 0, 0, 1]
```

VQE accepts expert-mode Hamiltonians without a reference state if the selected
ansatz can operate without one. VarQITE and VarQRTE default to the all-zero
reference state when a reference is not supplied. QPE requires an initial state
for phase estimation.

For reproducible comparisons, pass the reference state explicitly.

## Automatic Ansatz Selection

Expert mode supports `ansatz_name="auto"` for VQE, VarQITE, and VarQRTE. The
selector inspects the Pauli-term structure and chooses a conservative ansatz:

| Detected structure | Selected ansatz |
| ------------------ | --------------- |
| Nearest-neighbor `ZZ` couplings with transverse `X` fields | `TFIM-HVA` |
| Nearest-neighbor `XX + YY + ZZ` exchange | `XXZ-HVA` |
| Nearest-neighbor `XX + YY` exchange without `ZZ` terms | `NumberPreservingGivens` |
| No confident model match | `StronglyEntanglingLayers` |

Auto-selected defaults include modest layer counts. Caller-provided
`ansatz_kwargs` override those defaults:

```python
result = run_vqe(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    ansatz_name="auto",
    ansatz_kwargs={"layers": 6},
    plot=False,
)
```

Results include an `ansatz_selection` entry when automatic selection is used.
It records the requested mode, selected ansatz, reason, and resolved ansatz
keyword arguments.

## Caching

Expert-mode runs are cacheable. The run signature includes a canonical
Hamiltonian fingerprint, `num_qubits`, `reference_state`, ansatz settings,
solver settings, and relevant algorithm controls.

This means two separately constructed but Pauli-equivalent Hamiltonians can
reuse the same cache record when the rest of the configuration matches.

Use `force=True` to recompute:

```python
result = run_vqe(
    hamiltonian=hamiltonian,
    num_qubits=2,
    reference_state=[1, 0],
    force=True,
    plot=False,
)
```

## Noise Support

VQE supports noisy evaluation and noisy optimization through the built-in noise
probability arguments and `default.mixed` device path.

VarQITE and VarQRTE updates are pure-state projected-dynamics methods. They
reject noisy or mixed-state parameter updates. For VarQITE, the CLI provides
`qite eval-noise` to post-evaluate converged VarQITE parameters under noise.

QPE has its own noise utilities and should be configured through the QPE API.

## Model-Hamiltonian Benchmarks

The repository includes expert-mode benchmark notebooks for non-chemistry
Hamiltonians:

- `notebooks/benchmarks/non_molecule/TFIM_Cross_Method_Benchmark.ipynb`
- `notebooks/benchmarks/non_molecule/Heisenberg_Chain_Benchmark.ipynb`
- `notebooks/benchmarks/non_molecule/SSH_Chain_Benchmark.ipynb`

These notebooks compare exact diagonalization, VQE, VarQITE, and QPE on
prebuilt `qml.Hamiltonian` inputs and are the best examples of the intended
expert-mode workflow.

## Common Errors

| Error or symptom | Cause | Fix |
| ---------------- | ----- | --- |
| `num_qubits cannot be smaller...` | Hamiltonian uses more wires than `num_qubits` | Increase `num_qubits` or simplify the Hamiltonian |
| `reference_state length must match num_qubits` | Bitstring length is wrong | Pass one bit per qubit |
| Unexpected fallback ansatz | `auto` could not classify the Pauli structure | Choose an explicit ansatz or pass `ansatz_kwargs` |
| Cache hit when experimenting | Same canonical Hamiltonian and config | Use `force=True` |

