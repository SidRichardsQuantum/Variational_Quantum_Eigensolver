# Solving Real Problems With This Repo

This repository solves a practical quantum-algorithm decision problem:

> Given a small molecule or low-qubit Hamiltonian, determine which quantum simulation method, ansatz, mapping, optimizer, noise model, or runtime setting gives the most useful result for the available qubit budget.

The repo is not positioned as production-scale chemistry software.
Its real value is reproducible small-system experimentation: comparing near-term quantum simulation workflows, documenting their tradeoffs, and turning those comparisons into decision-ready examples.

---

## The Problems We Solve

Researchers, students, and quantum software engineers often need to answer:

- Which algorithm should I try first for this small molecule or model system?
- How close does VQE, QPE, or VarQITE get to an exact small-system reference?
- Which ansatz, optimizer, or fermion-to-qubit mapping is most reliable?
- How sensitive is a workflow to noise, shots, seeds, and calibration choices?
- Which systems are small enough to benchmark quickly on a laptop or simulator?
- Can the same Hamiltonian input be reused across VQE, QPE, and QITE?

This repo answers those questions through a shared Hamiltonian pipeline, callable Python APIs, CLI entrypoints, notebooks, benchmark artifacts, and documented defaults.

---

## Who This Is For

Use this repo if you are:

- Learning how VQE, QPE, QITE, and QRTE behave on concrete systems.
- Building small-molecule quantum algorithm demonstrations.
- Comparing ansatzes, optimizers, mappings, and noise channels.
- Creating benchmark panels for low-qubit molecules or model Hamiltonians.
- Teaching quantum simulation with reproducible notebooks and JSON outputs.
- Prototyping workflows before moving to hardware-oriented tooling.

This repo is not the right tool if you need:

- Large-molecule production chemistry.
- Industrial accuracy guarantees.
- Full hardware execution management.
- General-purpose classical quantum chemistry replacement.
- Curated public benchmark datasets from raw local cache files.

---

## Real Problem Workflows

### 1. Choose A Ground-State Method For A Small Molecule

Problem:

> I have a small molecule such as `H2`, `H3+`, `H4`, or active-space `LiH`.
> Which method gives the best ground-state estimate for the cost?

Use:

- `vqe.run_vqe(...)` for a fast variational baseline.
- `qite.run_qite(...)` for imaginary-time relaxation.
- `qpe.run_qpe(...)` for phase-estimation comparison.
- Exact references where available for small systems.

Useful notebooks:

- `notebooks/benchmarks/comparisons/H2/Cross_Method_Comparison.ipynb`
- `notebooks/benchmarks/comparisons/LiH/Cross_Method_Comparison.ipynb`
- `notebooks/benchmarks/comparisons/multi_molecule/Low_Qubit_VQE_Benchmark.ipynb`

Outcome:

- Energy estimate.
- Absolute error against exact diagonalization where feasible.
- Runtime and cache behavior.
- Recommended method for that system size.

### 2. Pick An Ansatz, Optimizer, Or Mapping

Problem:

> VQE works, but I do not know which ansatz, optimizer, or mapping to use.

Use:

- `vqe.run_vqe_ansatz_comparison(...)`
- `vqe.run_vqe_optimizer_comparison(...)`
- `vqe.run_vqe_mapping_comparison(...)`
- `vqe.run_adapt_vqe(...)`

Useful notebooks:

- `notebooks/benchmarks/vqe/H2/Ansatz_Comparison.ipynb`
- `notebooks/benchmarks/vqe/H2/Mapping_Comparison.ipynb`
- `notebooks/getting_started/10_adapt_vqe_h3plus.ipynb`

Outcome:

- Convergence traces.
- Final energies and errors.
- Parameter-count and runtime tradeoffs.
- A documented default or recommended configuration.

### 3. Study Noise Sensitivity

Problem:

> I need to know whether a VQE or QPE configuration is robust under simple simulated noise channels.

Use:

- VQE noisy runs and multi-seed noise sweeps.
- QPE shot and noise calibration workflows.
- Shared result records for reproducibility.

Useful notebooks:

- `notebooks/benchmarks/vqe/H2/Noise_Robustness_Benchmark.ipynb`
- `notebooks/benchmarks/vqe/H2/Noise_Scan.ipynb`
- `notebooks/benchmarks/qpe/H2/Noisy.ipynb`
- `notebooks/benchmarks/qpe/H2/Calibration_Decision_Map.ipynb`

Outcome:

- Noise-channel sensitivity ranking.
- Seed spread.
- Shot sensitivity.
- Failure modes such as branch-selection or dominant-bin errors in QPE.

### 4. Estimate Excited States

Problem:

> I have a converged or usable reference state and want excited-state information for a small system.

Use:

- `vqe.run_qse(...)`
- `vqe.run_eom_qse(...)`
- `vqe.run_lr_vqe(...)`
- `vqe.run_eom_vqe(...)`
- `vqe.run_ssvqe(...)`
- `vqe.run_vqd(...)`

Useful notebooks:

- `notebooks/getting_started/05_excited_states_h2.ipynb`
- `notebooks/benchmarks/vqe/H2/SSVQE_Comparisons.ipynb`
- `notebooks/benchmarks/vqe/H2/VQD_Comparisons.ipynb`

Outcome:

- Excited-state estimates.
- Comparison between post-VQE and direct variational excited-state methods.
- Diagnostics on whether the reference state is good enough.

### 5. Simulate Dynamics

Problem:

> I want to compare variational real-time or imaginary-time dynamics against an exact small-system reference.

Use:

- `qite.run_qite(...)` for imaginary-time relaxation.
- `qite.run_qrte(...)` for real-time evolution.

Useful notebooks:

- `notebooks/getting_started/07_getting_started_qite_h2.ipynb`
- `notebooks/getting_started/11_getting_started_qrte_h2.ipynb`
- `notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb`

Outcome:

- Energy trajectory.
- Observable trajectory where configured.
- Difference from exact evolution for small systems.

### 6. Benchmark Non-Chemistry Hamiltonians

Problem:

> I want to test VQE, QPE, or QITE on a qubit Hamiltonian that is not a molecule.

Use expert-mode Hamiltonian inputs:

```python
import pennylane as qml
from vqe import run_vqe

H = qml.Hamiltonian([1.0, 0.5], [qml.PauliZ(0), qml.PauliX(0)])

result = run_vqe(
    hamiltonian=H,
    num_qubits=1,
    reference_state=[1],
    ansatz_name="RY-CZ",
    steps=10,
    plot=False,
)

print(result["energy"])
```

Useful notebooks:

- `notebooks/benchmarks/non_molecule/TFIM_Cross_Method_Benchmark.ipynb`
- `notebooks/benchmarks/non_molecule/Heisenberg_Chain_Benchmark.ipynb`
- `notebooks/benchmarks/non_molecule/SSH_Chain_Benchmark.ipynb`

Outcome:

- Algorithm comparisons outside the chemistry pipeline.
- Small-model validation before larger custom experiments.

---

## Supported Inputs

The repo supports three main input styles.

### Registry Molecules

Use named molecules such as:

- `H`, `H-`
- `He`, `He+`
- `H2`, `H2+`, `H2-`
- `H3`, `H3+`
- `H4`, `H4+`, `H5+`, `H6`
- `Li`, `Li+`, `LiH`
- `Be`, `Be+`, `BeH2`
- `B`, `B+`, `C`, `C+`, `N`, `N+`, `O`, `O+`, `F`, `F+`, `Ne`
- `He2`, `HeH+`
- `H2O`

### Parametric Geometry Tags

Use generated geometry families for scans:

- `H2_BOND`
- `H3+_BOND`
- `LiH_BOND`
- `H2O_ANGLE`

### Explicit Geometry

Use symbols, coordinates, charge, multiplicity, and basis when the target system is not already in the registry.

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

---

## Decision Guide

| Need | Start With | Reason |
| --- | --- | --- |
| Fast ground-state estimate | VQE | Lowest workflow overhead and broad benchmark coverage. |
| Adaptive ansatz construction | ADAPT-VQE | Useful when fixed ansatz choice is uncertain. |
| Post-reference excited states | QSE, EOM-QSE, LR-VQE, EOM-VQE | Reuses a converged VQE reference. |
| Direct variational excited states | SSVQE, VQD | Optimizes excited states directly. |
| Phase or spectral information | QPE | Estimates phase/energy distribution rather than only a variational minimum. |
| Imaginary-time relaxation | VarQITE | Provides an alternative relaxation path toward low-energy states. |
| Real-time dynamics | VarQRTE | Evolves a prepared state and tracks dynamics. |
| Non-molecule qubit model | Expert mode | Bypasses chemistry inputs and accepts a prebuilt Hamiltonian. |

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
