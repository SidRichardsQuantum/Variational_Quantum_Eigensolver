# Molecule Registry

The molecule registry is the shared source of built-in chemistry problems used
by VQE, QPE, VarQITE, VarQRTE, notebooks, and benchmark helpers.

It lives in `common.molecules.MOLECULES`. Each entry stores:

- atomic symbols
- Cartesian coordinates
- charge
- spin multiplicity
- basis
- coordinate unit

Registry entries are intentionally small, mostly STO-3G systems. They are meant
for algorithm development, examples, smoke tests, and reproducible benchmark
panels, not high-accuracy production chemistry.

## Supported Entries

Current registry names:

| Family | Entries |
| ------ | ------- |
| Hydrogen atoms and ions | `H`, `H-` |
| Hydrogen molecules and chains | `H2`, `H2+`, `H2-`, `H3`, `H3+`, `H4`, `H4+`, `H5+`, `H6` |
| Noble gas atoms and ions | `He`, `He+`, `Ne` |
| Small atoms and cations | `Li`, `Li+`, `Be`, `Be+`, `B`, `B+`, `C`, `C+`, `N`, `N+`, `O`, `O+`, `F`, `F+` |
| Small molecules | `He2`, `HeH+`, `LiH`, `BeH2`, `H2O` |

The registry stores coordinates in angstrom by default. Public APIs accept a
`unit` argument so callers can request geometry handling in `angstrom` or
`bohr`; energies are reported in Hartree.

## Registry Mode

Registry mode is selected by passing `molecule=...` without explicit symbols
and coordinates:

```python
from vqe import run_vqe

result = run_vqe(molecule="H2", steps=75, plot=False)
```

The same pattern works across the solver packages:

```python
from qpe import run_qpe
from qite import run_qite, run_qrte

qpe_result = run_qpe(molecule="H2", ancillas=4, shots=1000)
qite_result = run_qite(molecule="H2", steps=75, dtau=0.2, plot=False)
qrte_result = run_qrte(molecule="H2", steps=50, dt=0.05, plot=False)
```

In registry mode, the stored charge, multiplicity, basis, and coordinates define
the problem. This keeps benchmark inputs stable and makes cache keys
reproducible.

## Charge And Multiplicity

Charged and open-shell systems should normally be selected by their registry
name, for example `H2+`, `Li+`, or `O+`. The registry entry carries the
charge and multiplicity together.

For built-in registry molecules, avoid overriding chemistry metadata such as
`charge` or `basis` through the Hamiltonian builder. Use explicit geometry mode
when you need custom chemistry settings.

The shared problem-resolution layer preserves registry multiplicities. If a
caller asks for `molecule="H2+"`, downstream solvers see the cation charge and
doublet multiplicity from the registry.

## Explicit Geometry Mode

Use explicit geometry mode when the geometry, basis, charge, or multiplicity is
part of the experiment:

```python
from vqe import run_vqe

result = run_vqe(
    symbols=["H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    unit="angstrom",
    steps=75,
    plot=False,
)
```

Explicit geometry mode requires both `symbols` and `coordinates`. Coordinates
must be an `N x 3` array-like object matching the number of symbols.

## Active Spaces

Larger registry molecules can be restricted with `active_electrons` and
`active_orbitals`:

```python
from vqe import run_vqe

result = run_vqe(
    molecule="LiH",
    active_electrons=2,
    active_orbitals=2,
    steps=75,
    plot=False,
)
```

Active spaces are resolved by the shared Hamiltonian pipeline and are included
in run signatures. Changing active-space settings creates a distinct cache
record.

Use active spaces when:

- the full STO-3G problem is too large for a quick benchmark
- you need a fair cross-method comparison on a fixed qubit count
- you want a compact problem for noisy or multi-seed runs

## Cache Behavior

Registry and explicit-geometry chemistry runs are cacheable. The cache key
includes the resolved molecule or geometry, charge, basis, mapping, unit,
active-space settings, ansatz, optimizer or evolution settings, seed, and solver
specific numerical controls.

Use `force=True` or `--force` when you want to recompute instead of loading a
matching JSON run record.

## Benchmark Coverage

The registry is used directly by benchmark notebooks such as:

- `notebooks/benchmarks/comparisons/multi_molecule/Registry_Coverage.ipynb`
- `notebooks/benchmarks/comparisons/multi_molecule/Low_Qubit_VQE_Benchmark.ipynb`
- `notebooks/benchmarks/comparisons/multi_molecule/Hydrogen_Family_VQE_Benchmark.ipynb`
- `notebooks/benchmarks/comparisons/multi_molecule/Atomic_Ionization_Energy_Benchmark.ipynb`

These notebooks are the best way to check which entries are cheap enough for
heavier solver benchmarks and which ones should remain exact-reference or
metadata coverage cases.

## When To Add A Registry Entry

Add a registry entry when the geometry and quantum numbers are useful as a
stable, named benchmark input. Prefer explicit geometry mode when the point is a
one-off scan, a tutorial-specific structure, or a user-provided molecule.

Good registry candidates:

- small systems used in multiple notebooks or tests
- charged or open-shell systems where charge and multiplicity should stay paired
- benchmark panel entries that need stable labels

Avoid adding entries that are too expensive for routine CI or examples unless
they are clearly marked as benchmark-only usage.

