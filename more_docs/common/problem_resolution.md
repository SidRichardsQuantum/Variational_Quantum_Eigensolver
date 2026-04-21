# Problem Resolution

Problem resolution is the shared path that turns user inputs into a Hamiltonian,
qubit count, reference state, metadata, and cache policy.

It is implemented in `common.problem.resolve_problem(...)` and used by VQE,
QPE, VarQITE, and VarQRTE. This shared layer keeps chemistry inputs,
expert-mode Hamiltonians, active spaces, units, and cache behavior consistent
across solver packages.

## Resolution Outputs

The resolver returns a `ResolvedProblem` with:

- `hamiltonian`
- `num_qubits`
- `reference_state`
- `molecule_label`
- `symbols`
- `coordinates`
- `basis`
- `charge`
- `multiplicity`
- `mapping`
- `unit`
- `active_electrons`
- `active_orbitals`
- `cacheable`

Solver entrypoints use this metadata to build devices, ansatzes, cache keys,
plots, and JSON result records.

## Input Modes

There are three main input modes:

| Mode | Required inputs | Use when |
| ---- | --------------- | -------- |
| Registry molecule | `molecule="H2"` | You want a stable built-in chemistry problem |
| Explicit geometry | `symbols`, `coordinates` | You need custom geometry, basis, charge, or multiplicity |
| Expert Hamiltonian | `hamiltonian`, `num_qubits` | You already have a `qml.Hamiltonian` or non-chemistry model |

Do not mix expert-mode controls into chemistry mode. `num_qubits` and
`reference_state` are only accepted when `hamiltonian` is provided.

## Registry Molecules

Registry mode is the default:

```python
from vqe import run_vqe

result = run_vqe(molecule="H2", plot=False)
```

The molecule registry supplies symbols, coordinates, charge, multiplicity,
basis, and stored coordinate unit. If the caller does not explicitly request a
non-singlet multiplicity, the registry multiplicity is preserved.

Registry mode is cacheable.

## Explicit Geometry

Explicit geometry mode is selected when both `symbols` and `coordinates` are
provided:

```python
from qite import run_qite

result = run_qite(
    symbols=["H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    unit="angstrom",
    plot=False,
)
```

The resolver passes these values to the shared Hamiltonian builder and records
the resolved metadata for cache signatures and result output.

Explicit geometry mode is cacheable.

## Active Spaces

Both registry and explicit-geometry chemistry modes accept:

- `active_electrons`
- `active_orbitals`

```python
from vqe import run_vqe

result = run_vqe(
    molecule="LiH",
    active_electrons=2,
    active_orbitals=2,
    plot=False,
)
```

The resolved active-space settings are returned in `ResolvedProblem` and
included in cache keys. This prevents full-space and active-space runs from
colliding.

## Mapping And Units

The default fermion-to-qubit mapping is:

```python
mapping = "jordan_wigner"
```

The default coordinate unit is:

```python
unit = "angstrom"
```

The resolver normalizes string settings such as mapping, basis, and unit. The
Hamiltonian builder handles supported coordinate unit conversion. Energies are
reported in Hartree.

## Expert Mode

Expert mode is selected when `hamiltonian` is provided:

```python
import pennylane as qml

from vqe import run_vqe

hamiltonian = qml.Hamiltonian(
    [1.0, -0.5],
    [qml.PauliZ(0), qml.PauliX(0)],
)

result = run_vqe(
    hamiltonian=hamiltonian,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    plot=False,
)
```

The resolver inspects Hamiltonian wires and maps them to contiguous integer
wires if necessary. `num_qubits` must be at least the number of Hamiltonian
wires. `reference_state`, when supplied, must have length `num_qubits`.

Expert-mode cache records use a canonical Hamiltonian fingerprint so equivalent
Pauli Hamiltonians can reuse cache entries when the rest of the configuration
matches.

## Reference-State Policy

In chemistry modes, the Hamiltonian builder returns the Hartree-Fock reference
state and the resolver passes it to downstream solvers.

In expert mode:

- VQE accepts `reference_state` when the ansatz needs one.
- QPE requires a reference or initial state for phase estimation workflows.
- VarQITE and VarQRTE default to an all-zero reference state when none is
  provided by their entrypoints.

For reproducible expert-mode studies, pass `reference_state` explicitly.

## Cacheability

The resolver marks registry and explicit-geometry chemistry problems as
cacheable. Expert-mode `ResolvedProblem.cacheable` is `False` because the
problem itself is external, but solver entrypoints still enable caching by
adding the canonical Hamiltonian, qubit count, and reference state to their run
configuration.

In practice, all high-level chemistry and expert-mode solver runs can use
caches when enough metadata is available to build a stable signature.

## Common Validation Errors

| Error | Cause | Fix |
| ----- | ----- | --- |
| `num_qubits is only supported when hamiltonian is provided` | Expert-mode argument passed to chemistry mode | Remove `num_qubits` or provide `hamiltonian` |
| `reference_state is only supported when hamiltonian is provided` | Reference bitstring passed to chemistry mode | Use registry or explicit geometry HF state, or switch to expert mode |
| `num_qubits cannot be smaller...` | Expert Hamiltonian uses more wires than requested | Increase `num_qubits` |
| `reference_state length must match num_qubits` | Bitstring length mismatch | Pass one bit per qubit |
| Unknown molecule | Registry key is not present | Use a supported registry name or explicit geometry |

## Related Pages

Read these alongside this page:

- `docs/common/molecule_registry.md`
- `docs/common/expert_mode.md`
- `docs/common/caching_and_artifacts.md`
- `docs/vqe/defaults.md`

