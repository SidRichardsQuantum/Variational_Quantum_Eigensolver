# Changelog

All notable changes to this project will be documented in this file.

---

## [0.3.14] - April 13, 2026

### Added

- **Expanded small-system molecule registry**

  Added curated atom and atomic-ion entries for small benchmarkable systems in the
  shared molecule registry:

  - `He`, `He+`
  - `Li`, `Li+`
  - `Be`, `Be+`
  - `B`, `B+`
  - `C`, `C+`
  - `N`, `N+`
  - `O`, `O+`
  - `F`, `F+`
  - `Ne`

  This extends the standard chemistry path for low-qubit studies without requiring
  explicit geometry setup or expert-mode Hamiltonian inputs.

- **Registry coverage notebook**

  Added `notebooks/benchmarks/comparisons/multi_molecule/Registry_Coverage.ipynb`
  to summarize the built-in chemistry registry with resolved qubit counts, term
  counts, charge and multiplicity defaults, and exact-ground reference energies.

- **Shared registry-coverage helper**

  Added `common.summarize_registry_coverage(...)` so notebooks and scripts can
  reuse one standard table-building path for registry molecule coverage.

### Changed

- **Open-shell standard chemistry builds now use OpenFermion directly**

  Open-shell registry and explicit-geometry builds now route straight to the
  OpenFermion backend instead of first failing through the default PennyLane-qchem
  backend, removing repeated fallback warnings in notebooks such as the hydrogen-family benchmark.

- **Input documentation now describes supported molecule names explicitly**

  Added a `Supported Molecule Inputs` section to `USAGE.md` and summarized the same
  guidance in `README.md` so users can see when to use registry names, geometry tags,
  explicit geometry mode, or expert mode.

- **Hydrogen-family benchmark exact references corrected**

  Updated the hydrogen-family benchmark to compare against same-electron-sector
  exact ground energies rather than the unrestricted global minimum of the full
  qubit Hamiltonian.

## [0.3.13] - April 13, 2026

### Added

- **Low-qubit multi-molecule VQE benchmark helper**

  Added `run_vqe_low_qubit_benchmark(...)` to benchmark supported registry
  molecules up to a chosen qubit limit, aggregating runtime and exact-ground
  error statistics across seeds for small-system comparison work.

- **Cached compute-runtime metadata across public runners**

  `run_vqe(...)`, `run_qite(...)`, `run_qrte(...)`, and `run_qpe(...)` now
  return `runtime_s`, `compute_runtime_s`, and `cache_hit` so benchmark code
  can distinguish cache-load latency from original compute cost.

- **Low-qubit VQE benchmark notebook**

  Added `notebooks/benchmarks/comparisons/multi_molecule/Low_Qubit_VQE_Benchmark.ipynb`
  as the notebook-facing companion for the low-qubit benchmark helper.

### Changed

- **VQE CLI now matches calibrated optimizer-default behaviour**

  `vqe --stepsize` is now optional. When omitted, the CLI preserves the same
  calibrated per-optimizer default stepsize behaviour as the Python APIs
  instead of forcing a legacy fixed `0.2` learning rate.

### Fixed

- Fixed stale VQE CLI argument forwarding that still cast `stepsize` eagerly in
  standard VQE, compare-noise, and post-VQE excited-state workflows, preventing
  the new optimizer-default resolution path from taking effect.
- Fixed optimizer documentation to reflect the canonical
  `NesterovMomentum` registry entry, the alias model, and the current
  calibrated default-stepsize registry layout.
- Fixed notebook and documentation path references so the renumbered
  `getting_started/` notebook names are referenced consistently.

## [0.3.12] - April 13, 2026

### Added

- **Reproducibility and scaling benchmark notebooks**

  Added benchmark notebooks for:

  - H2 reproducibility, including seed spread, cache timing, and noisy-vs-noiseless comparisons
  - multi-molecule scaling across H2, LiH, and BeH2

- **Default-calibration benchmark notebooks**

  Added dedicated notebooks to calibrate package defaults for:

  - VQE
  - VarQITE
  - QPE

  These notebooks are intended to be kept as benchmark artifacts so future default changes can be justified from recorded sweeps rather than one-off intuition.

### Changed

- **Solver defaults are now benchmark-backed**

  Updated default settings to match the current calibration work:

  - `run_vqe(...)`: `ansatz_name="UCCSD"`, `optimizer_name="Adam"`, `stepsize=0.2`, `steps=75`
  - `run_qite(...)`: `ansatz_name="UCCSD"`, `dtau=0.2`, `steps=75`
  - `run_qpe(...)`: `n_ancilla=4`, `t=1.0`, `trotter_steps=2`, `shots=1000`

  QPE defaults are documented as H2-calibrated baseline defaults: good small-molecule starting values, not globally optimized settings for every chemistry problem.

- **Notebook organization and guidance refined**

  Updated notebook indexes, roadmap notes, and comparison notebooks to reflect:

  - the new benchmark / default-calibration coverage
  - pruning of redundant notebooks
  - clearer distinctions between introductory, benchmark, and specialized workflows

- **Canonical ansatz registry exposed**

  `vqe.ansatz.ANSATZES` now exposes only canonical UCC names (`UCCS`, `UCCD`, `UCCSD`) instead of duplicate alias entries.

### Fixed

- Fixed invalid ansatz handling so unsupported ansatz names fail immediately instead of silently falling back in VarQITE-related workflows.
- Fixed package-side Matplotlib environment setup to reduce writable-config warnings and avoid eager Matplotlib imports on ordinary package import paths.
- Fixed the H2 ansatz-comparison notebook to use the canonical ansatz registry cleanly and improved its plotting readability.

---

## [0.3.11] - April 13, 2026

### Added

- **Non-molecule / expert-mode qubit Hamiltonian support**

  Added direct Hamiltonian workflows across the public Python APIs so algorithm benchmarking no longer has to go through the chemistry pipeline:

  - `run_vqe(..., hamiltonian=H, num_qubits=..., reference_state=...)`
  - `run_qite(..., hamiltonian=H, num_qubits=..., reference_state=...)`
  - `run_qrte(..., hamiltonian=H, num_qubits=..., reference_state=...)`
  - `run_qpe(..., hamiltonian=H, hf_state=..., system_qubits=...)`

- **Shared problem-resolution layer**

  Added `common.problem.resolve_problem(...)` as the central input-normalization path for molecule, explicit-geometry, and expert-mode runs.

- **Result-path override support**

  Added `VQE_PENNYLANE_DATA_DIR` support so tests and scripted runs can redirect generated `results/` and `images/` artifacts to a separate writable root.

### Changed

- **Documentation updated for expert mode**

  `README.md`, `USAGE.md`, and architecture notes now describe the non-molecule workflows and the shared problem-resolution model explicitly.

- **Solver internals simplified around shared inputs**

  VQE, QPE, VarQITE, and VarQRTE now rely more consistently on the common resolution / metadata path instead of maintaining parallel per-solver input handling.

### Fixed

- Fixed wire-normalization and reference-state handling for direct Hamiltonian workflows so non-consecutive / non-integer operator wires are accepted consistently.
- Fixed cache-signature and output-metadata handling to include the resolved shared problem fields consistently across supported workflows.
- Expanded regression coverage for expert-mode execution, run signatures, cache behavior, and public API expectations.

---

## [0.3.10] – April 12, 2026

### Added

- **Benchmark roadmap for notebooks**

  Added a notebook-facing benchmark roadmap to clarify the highest-value next additions for:

  - reproducibility studies
  - scaling comparisons
  - QPE calibration sweeps
  - cross-method comparisons
  - noise robustness studies

- **New benchmark notebooks**

  - `notebooks/benchmarks/qpe/H2/Calibration_Sweep.ipynb` — calibrates QPE energy quality against ancilla count, evolution time, Trotter steps, and shots, with explicit phase / aliasing diagnostics
  - `notebooks/benchmarks/comparisons/H2/Cross_Method_Comparison.ipynb` — compares VQE, QPE, and VarQITE on a shared H2 Hamiltonian using public package APIs

- **Improved analysis notebooks**

  - `notebooks/vqe/H2O/Bond_Angle.ipynb` now performs a coarse-to-refined angle scan, shows the scan tables directly, and estimates the optimum angle from a local quadratic fit instead of only choosing the lowest sampled point

### Fixed

- Fixed `run_qpe(..., shots=None)` so analytic QPE mode now returns probabilities instead of failing on sample-based measurement requests.
- Fixed `python -m qpe` to return a nonzero exit code on failure so CI and shell workflows can detect errors reliably.
- Fixed `python -m qpe` explicit-geometry CLI support to match the documented Python API / changelog contract.
- Fixed the QPE CLI qubit summary to report the stored system-qubit field correctly.
- Fixed duplicated solver-side input resolution by routing VQE, QPE, VarQITE, and VarQRTE through a shared `common.problem.resolve_problem(...)` layer.
- Fixed QPE expert mode to reject partial expert inputs instead of silently falling back when only `hamiltonian` or only `hf_state` is provided.

### Internal

- Expanded QPE regression coverage for:

  - analytic `shots=None` execution
  - explicit-geometry CLI invocation
  - nonzero CLI failure status
- Added direct tests for shared problem resolution, including expert-mode wire normalization and explicit-geometry metadata handling.

---

## [0.3.8] – April 11, 2026

### Added

- **Variational Quantum Real-Time Evolution (VarQRTE)** as a first-class projected-dynamics workflow in the `qite` package:

  - `run_qrte(...)` Python API
  - `qite run-qrte` CLI entrypoint
  - real-time McLachlan update support alongside VarQITE
  - parameter-history output for observable reconstruction and benchmarking

- **Prepared-state VarQRTE support**

  `run_qrte(...)` now accepts `initial_params`, enabling real-time evolution from states prepared by prior VQE / QITE workflows instead of only from default ansatz initialization.

- **New VarQRTE notebooks**

  - `notebooks/getting_started/11_getting_started_qrte_h2.ipynb` — prepared-state VarQRTE usage demo
  - `notebooks/qite/H2/Real_Time.ipynb` — package-client VarQRTE workflow
  - `notebooks/benchmarks/qite/H2/Exact_QRTE_Benchmark.ipynb` — exact-vs-VarQRTE H2 quench benchmark

- **Dedicated notebook benchmark layout**

  benchmark / comparison notebooks now live under:

  - `notebooks/benchmarks/vqe/`
  - `notebooks/benchmarks/qpe/`
  - `notebooks/benchmarks/qite/`

### Changed

- **Excited-state geometry handling unified**

  `SSVQE` and `VQD` now route explicit-geometry workflows through the shared Hamiltonian builder rather than ad hoc Hamiltonian construction paths.

- **Projected-dynamics framing updated**

  Documentation now treats `qite/` as the projected-dynamics package for:

  - VarQITE
  - VarQRTE

  rather than describing it only as imaginary-time evolution.

- **Notebook organization clarified**

  tutorial / getting-started notebooks remain under `notebooks/getting_started/`, while reproducible comparison and validation workflows have been separated into `notebooks/benchmarks/`.

### Fixed

- Fixed broken explicit-geometry support in QPE where `run_qpe(...)` forwarded geometry kwargs that `qpe.hamiltonian.build_hamiltonian(...)` did not accept.
- Fixed missing fermion-to-qubit `mapping` propagation in:

  - SSVQE execution
  - SSVQE cache metadata
  - VQD CLI forwarding

- Fixed charge propagation through chemistry-inspired UCC ansatz plumbing so charged systems no longer silently use neutral excitation bookkeeping.
- Fixed the remaining QITE chemistry-ansatz delegation issue by routing through the shared VQE ansatz builder while preserving a true VarQITE implementation.
- Fixed a QITE UCCSD stagnation bug caused by double reference-state preparation in delegated chemistry ansatzes.
- Fixed malformed QRTE notebook labels and stale notebook path references after the benchmark reorganization.

### Internal

- Expanded regression coverage for:

  - QPE explicit geometry
  - excited-state mapping forwarding
  - charged UCC ansatz construction
  - QITE charge-aware ansatz delegation
  - VarQITE UCCSD descent on H2
  - VarQRTE prepared-state initialization

- Added `.codex` and `.codex/` to `.gitignore`.

---

## [0.3.7] – April 9, 2026

### Added

- **Explicit geometry support across VQE, QPE, and QITE workflows**

  All major ground-state and imaginary-time solvers can now operate either:

  - from a molecule registry entry (e.g. `"H2"`)
  - from fully user-specified molecular geometry

  Explicit geometry inputs include:

  - `symbols`
  - `coordinates`
  - `basis`
  - `charge`
  - `unit`
  - `mapping`

  enabling workflows such as:

  - custom bond lengths
  - non-registry molecules
  - geometry overrides in tutorial notebooks
  - consistent benchmarking across VQE, QPE, and QITE

- **Unified Hamiltonian metadata interface**

  All algorithm-facing Hamiltonian builders now return:

  ```
  (H, n_qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out)
  ```

  ensuring a single consistent contract between:

  - `vqe.hamiltonian`
  - `qpe.hamiltonian`
  - `qite.hamiltonian`
  - `common.hamiltonian`

- **New geometry override tutorial**

  ```
  notebooks/getting_started/08_geometry_override_h2.ipynb
  ```

  demonstrates:

  - registry vs explicit geometry workflows
  - direct control of molecular structure
  - consistent solver behaviour across representations

- **Improved QPE visualisation**

  - bitstring probability plots now render correctly for multi-bit ancilla registers
  - axis label overlap resolved using rotated tick labels and tight layout
  - consistent plotting behaviour aligned with `common.plotting`

- **CLI support for explicit geometry**

  `python -m vqe`, `python -m qpe`, and `python -m qite` now accept:

  ```
  --symbols
  --coordinates
  --basis
  --charge
  --unit
  ```

  allowing command-line execution of custom molecular systems without modifying source code.

---

### Changed

- **run_vqe(), run_qpe(), and run_qite() APIs standardized**

  Solver entrypoints now share a consistent geometry interface:

  ```python
  symbols=None
  coordinates=None
  basis="sto-3g"
  charge=0
  unit="angstrom"
  mapping="jordan_wigner"
  ```

  allowing uniform usage patterns across algorithms.

- **common.hamiltonian promoted as single source of truth**

  All solver-specific Hamiltonian wrappers now delegate to:

  ```
  common.hamiltonian.build_hamiltonian(...)
  ```

  eliminating duplicated logic and preventing drift between algorithm implementations.

- **Configuration hashing extended**

  geometry metadata (`symbols`, `coordinates`, `basis`, `charge`, `unit`) now contributes to:

  - run configuration dictionaries
  - cache keys
  - JSON metadata records

  ensuring reproducibility across geometry overrides.

- **Notebooks updated to use public APIs only**

  tutorial notebooks now rely exclusively on:

  ```
  run_vqe
  run_qpe
  run_qite
  build_hamiltonian
  ```

  avoiding internal imports and improving long-term maintainability.

---

### Fixed

- Fixed unpacking errors caused by inconsistent Hamiltonian return signatures.
- Fixed missing `charge` propagation through VQE configuration dictionaries.
- Fixed QPE CLI import error where `build_hamiltonian` was not defined in `qpe.core`.
- Fixed incompatibilities between registry-based and explicit-geometry workflows.
- Fixed QPE plotting readability issues due to overlapping ancilla bitstring labels.
- Fixed inconsistent metadata propagation between QPE, QITE, and VQE pipelines.

---

### Internal

- Harmonized solver interfaces across:

  - VQE
  - QPE
  - QITE

  enabling future algorithms to integrate with minimal infrastructure changes.

- Reduced implicit coupling between solver packages by routing all molecular construction through `common`.

- Improved forward compatibility for:

  - geometry scans
  - basis comparisons
  - molecular dataset benchmarking
  - future chemistry-oriented tutorials

---

## [0.3.5] – March 12, 2026

### Changed

- **Comprehensive documentation refresh** across the project:

  - `README.md` reorganized for clearer separation between project overview, solver capabilities, and package structure.
  - `USAGE.md` expanded and clarified to document all CLI workflows (VQE, ADAPT-VQE, LR-VQE, EOM-VQE, QSE, EOM-QSE, SSVQE, VQD, QPE, QITE) with consistent CLI/API examples.
  - `THEORY.md` substantially expanded and reorganized:

    - clearer background section connecting quantum chemistry to VQE/QPE/QITE,
    - structured explanation of excited-state methods,
    - improved comparison of post-VQE vs variational excited-state approaches,
    - additional discussion of mappings, ansatz design, and imaginary-time evolution.

- **Documentation structure made consistent across the repository**:

  - Theory, usage, and architecture content now separated cleanly between
    `README.md`, `USAGE.md`, and `THEORY.md`.
  - Table-of-contents anchors and section hierarchy aligned across documents.

### Fixed

- Corrected outdated or inconsistent documentation references after the introduction of
  LR-VQE, EOM-VQE, QSE, and EOM-QSE.
- Fixed stale table-of-contents anchors and section names in `THEORY.md`.

### Internal

- **Repository hygiene improvements**:

  - `.gitignore` simplified and standardized for Python packaging, notebooks, and generated artifacts.
  - Ensured generated outputs (`results/`, `images/`, build artifacts) remain excluded from version control.

- Minor consistency improvements across documentation and project metadata in preparation for the **0.3.5 release**.

---

## [0.3.4] – February 26, 2026

### Added

- **Equation-of-Motion VQE (EOM-VQE)** as a first-class post-VQE excited-state method in the `vqe` package:

  - Implements **tangent-space full-response EOM-VQE** (positive-root spectrum) as a sister method to LR-VQE.
  - Builds tangent vectors using finite-difference parameter derivatives evaluated at a **converged VQE reference**.
  - Solves a stabilized full-response eigenproblem using overlap-spectrum filtering + whitening,
    returning excitation energies $\omega_i$ and excited-state energies $E_i = E_0 + \omega_i$.
  - Designed explicitly as **noiseless-only** (statevector reference), consistent with tangent-space response theory.

- **Equation-of-Motion QSE (EOM-QSE)** as a first-class post-VQE excited-state method in the `vqe` package:

  - Implements commutator-based EOM in an operator manifold via the generalized eigenvalue problem
    $A c = \omega S c$ with $A_{ij}=\langle\psi|O_i^\dagger[H,O_j]|\psi\rangle$ and $S_{ij}=\langle\psi|O_i^\dagger O_j|\psi\rangle$.
  - Handles the generally **non-Hermitian** reduced problem by selecting **positive, real-dominant** roots
    using `imag_tol` and `omega_eps` filtering.
  - Uses a Hamiltonian-driven Pauli pool (plus identity) with overlap filtering for numerical stability.
  - Designed explicitly as **noiseless-only** (statevector reference), consistent with the current post-VQE workflow.

- **EOM spectrum plotting utilities**:

  - `vqe.visualize.plot_eom_vqe_spectrum(...)`
  - `vqe.visualize.plot_eom_qse_spectrum(...)`
  - Mirrors the LR-VQE plotting conventions: exact spectrum vs post-VQE roots matched by nearest exact level index,
    with per-root $|\Delta E|$ annotations and molecule pretty titles (subscripts) while keeping filenames ASCII-safe.

- **New EOM example notebooks** (pure package clients):

  - `notebooks/vqe/H2/EOM_VQE.ipynb` — exact-spectrum benchmark + EOM-VQE excited energies vs nearest exact eigenvalues.
  - `notebooks/vqe/H2/EOM_QSE.ipynb` — exact-spectrum benchmark + EOM-QSE roots (real-dominant selection) vs nearest exact eigenvalues.

- **EOM CLI support** via `python -m vqe --eom-vqe` and `python -m vqe --eom-qse`:

  - Runs EOM-VQE / EOM-QSE using the same molecule/ansatz/optimizer configuration flags as VQE.
  - Optional plotting / saving of spectrum figures using the standard `--plot` / `--save` semantics.

- **New minimal EOM tests**:

  - End-to-end pytest coverage for EOM-VQE and EOM-QSE on H₂, including deterministic behavior given seed + forced recomputation,
    sorted/finite roots, and presence/structure of diagnostics + configuration metadata.

### Changed

- **EOM caching prefixes made solver-specific (no legacy behavior):**

  - EOM-VQE results now use the prefix token **`eom_vqe`** (previously ambiguous `eom`).
  - EOM-QSE results now use the prefix token **`eom_qse`** (previously ambiguous `eom`).

  This guarantees EOM-VQE and EOM-QSE caches cannot collide and ensures strict reproducibility.

- **Documentation updates across the project**:

  - `README.md` — EOM-VQE and EOM-QSE added to solver overview and project capabilities.
  - `USAGE.md` — EOM-VQE and EOM-QSE documented alongside other post-VQE excited-state workflows (CLI + API).
  - `THEORY.md` — excited-state discussion extended to include full-response tangent-space EOM-VQE and commutator EOM-QSE.
  - `notebooks/README_notebooks.md` — EOM notebooks added alongside LR-VQE and QSE in the H₂ section.

### Internal

- EOM methods implemented **without introducing new infrastructure layers**:

  - reuse the existing VQE engine for devices/ansatz/state QNodes,
  - reuse `common` plotting/I/O conventions via existing utilities,
  - maintain deterministic hashing and JSON-first run records consistent with the broader suite.

---

## [0.3.3] – February 14, 2026

### Added

- **Linear-Response VQE (LR-VQE)** as a first-class post-VQE excited-state method in the `vqe` package:

  - Implements **tangent-space TDA LR-VQE** via the generalized eigenvalue problem
    $A c = \omega S c$, solved by **overlap-spectrum filtering + whitening** for numerical stability.
  - Builds tangent vectors using finite-difference parameter derivatives evaluated at a **converged VQE reference**.
  - Returns excitation energies $\omega_i$ and excited-state energies $E_i = E_0 + \omega_i$, plus diagnostics
    (overlap spectrum, kept rank, conditioning, and thresholds).
  - Designed explicitly as **noiseless-only** (statevector reference), consistent with tangent-space LR theory.

- **LR-VQE spectrum plotting utility**:

  - `vqe.visualize.plot_lr_vqe_spectrum(...)`
  - Plots the exact spectrum and LR-VQE roots **matched to nearest exact level index** (no horizontal jitter),
    with per-root $|\Delta E|$ annotations.
  - Uses molecule **pretty titles** (subscripts) while keeping filenames **ASCII-safe** via shared plotting utilities.

- **New LR-VQE example notebook**:

  - `notebooks/vqe/H2/LR_VQE.ipynb`
  - Demonstrates end-to-end LR-VQE workflow on H₂:

    - exact-spectrum benchmark,
    - deterministic VQE reference $E_0$ and parameters $\theta^\*$ (used to build the tangent space),
    - tangent-space generalized EVP (TDA),
    - comparison against nearest exact eigenvalues with $|\Delta E|$ reporting.
  - Written as a **pure package client**, using only public APIs.

- **LR-VQE CLI support** via `python -m vqe --lr-vqe`:

  - Runs LR-VQE using the same molecule/ansatz/optimizer configuration flags as VQE.
  - Optional plotting / saving of the LR spectrum figure using the standard `--plot` / `--save` semantics.

- **New LR-VQE deterministic test**:

  - Minimal end-to-end pytest covering:

    - successful execution on H₂,
    - deterministic results given seed and forced recomputation,
    - sorted, finite excitation energies and eigenvalues,
    - presence and structure of diagnostics and configuration metadata.

### Changed

- **`run_vqe()` result schema extended** to support post-VQE methods cleanly:

  - Adds `final_params` (optimized parameter vector) and `params_history` (parameter trajectory).
  - Enables true post-VQE methods (e.g., LR-VQE) without rebuilding optimization logic externally.

- **Documentation updates across the project**:

  - `README.md` — LR-VQE added to solver overview and project capabilities.
  - `USAGE.md` — LR-VQE documented alongside other post-VQE excited-state workflows (CLI + API).
  - `THEORY.md` — excited-state discussion extended to include tangent-space linear response (TDA).
  - `notebooks/README_notebooks.md` — LR-VQE notebook added in the H₂ section.

### Fixed

- Standardized access to VQE optimized parameters for post-processing workflows by introducing `final_params`
  (eliminates brittle reliance on implicit/legacy parameter keys).

### Internal

- LR-VQE implemented **without introducing new infrastructure layers**:

  - reuses the existing VQE engine for devices/ansatz/state QNodes,
  - reuses `common` plotting/I/O conventions via existing utilities.

## [0.3.2] – February 8, 2026

### Added

- **Quantum Subspace Expansion (QSE)** as a first-class post-VQE excited-state method in the `vqe` package:

  - Implements standard QSE via subspace projection around a converged VQE reference state.
  - Operator pools constructed from top-|coeff| Pauli terms of the molecular Hamiltonian (plus identity).
  - Generalized eigenvalue problem $Hc = ESc$ solved with overlap-matrix eigenvalue filtering for numerical stability.
  - Returns lowest-*k- approximate eigenvalues with detailed diagnostics (subspace rank, conditioning, kept modes).
  - Designed explicitly as **noiseless-only**, consistent with statevector-based QSE theory.

- **New QSE example notebook**:

  - `notebooks/vqe/H2/QSE.ipynb`
  - Demonstrates end-to-end QSE workflow on H₂:

    - cached VQE reference,
    - subspace construction,
    - comparison against the exact qubit Hamiltonian spectrum,
    - visualization of QSE vs exact energies.
  - Written as a **pure package client**, using only public APIs and shared plotting utilities.

- **New QSE smoke + sanity tests**:

  - Minimal end-to-end pytest covering:

    - successful execution on H₂,
    - correct handling of reduced-rank subspaces,
    - sorted, finite eigenvalues,
    - presence and structure of diagnostics and configuration metadata.
  - Integrated into the existing test suite without increasing runtime instability.

### Changed

- **Documentation updates across the project**:

  - `README.md` — QSE added to the solver overview and project capabilities.
  - `USAGE.md` — clarified solver scope and excited-state landscape.
  - `THEORY.md` — extended excited-state discussion to include subspace-based (post-VQE) methods.
  - `notebooks/README_notebooks.md` — QSE notebook added, positioned alongside SSVQE and VQD.

- **Internal linear-algebra handling hardened**:

  - Explicit use of complex-valued Hamiltonian matrices where required to avoid silent imaginary-part truncation.
  - Improves numerical correctness without altering existing solver semantics.

### Fixed

- Prevented silent truncation of requested QSE eigenvalues when subspace rank is reduced by overlap filtering.
- Eliminated ambiguous casting of complex Hamiltonians to real arrays in internal linear-algebra paths.

### Internal

- QSE implemented **without introducing new infrastructure layers**:

  - reuses `common.hamiltonian`, `common.persist`, `common.plotting`, and existing VQE caching semantics.
- Version bumped from **0.3.1 → 0.3.2**.

---

## [0.3.1] – February 7, 2026

### Added

- **Full ADAPT-VQE implementation** as a first-class solver in the `vqe` package:

  - Chemistry-oriented ADAPT-VQE with Hartree–Fock reference state.
  - Operator pools based on **UCC singles / doubles / singles+doubles** (`uccs`, `uccd`, `uccsd`).
  - Deterministic operator selection via maximum energy gradient
    $|\partial E / \partial \theta|$ evaluated at zero initialization.
  - Explicit inner/outer loop structure with configurable:

    - inner optimizer steps and step size,
    - maximum operator budget,
    - gradient stopping tolerance.
  - Fully compatible with existing VQE infrastructure:

    - device selection,
    - noise handling,
    - caching,
    - plotting,
    - run hashing.

- **ADAPT-VQE CLI support** via `python -m vqe --adapt`:

  - Unified with existing VQE / SSVQE / VQD CLI dispatcher.
  - Supports operator pool selection, stopping criteria, noise flags, plotting, and cache control.
  - Results cached under the same deterministic hashing and filesystem conventions as standard VQE.

- **ADAPT-VQE result schema** standardized and persisted:

  - Outer-loop energies.
  - Inner-loop convergence trajectories per ADAPT iteration.
  - Maximum gradient history used for stopping.
  - Ordered list of selected operators with wire indices.
  - Final optimized parameter vector.
  - Full run configuration embedded in JSON records.

- **New ADAPT-VQE smoke test**:

  - Ensures end-to-end execution, caching, and deterministic behavior.
  - Integrated into the existing pytest suite alongside VQE / QPE / QITE tests.

### Changed

- **`vqe` CLI generalized** to act as a unified driver for:

  - ground-state VQE,
  - excited-state solvers (SSVQE, VQD),
  - adaptive ansatz construction (ADAPT-VQE).

- Documentation updates across:

  - `README.md` — ADAPT-VQE promoted to a first-class solver with conceptual and CLI overview.
  - `USAGE.md` — explicit ADAPT-VQE CLI usage and Python API examples.

- Plotting for ADAPT-VQE integrated with `common.plotting`:

  - unified filename construction,
  - consistent molecule labeling,
  - reproducible image paths.

### Fixed

- Ensured ADAPT-VQE noise flags are canonicalized consistently with standard VQE
  (non-effective noise parameters no longer pollute cache keys or filenames).
- Prevented silent operator/parameter mismatches by enforcing strict length checks
  between selected operator lists and parameter vectors.

### Internal

- ADAPT-VQE implemented without introducing any new infrastructure layers:

  - reuses `common.hamiltonian`, `common.plotting`, `common.persist`, and existing VQE engine utilities.
- Test suite expanded to cover adaptive workflows without increasing runtime instability.

---

## [0.3.0] – January 25, 2026

### Added

- **Unified infrastructure layer (`common/`)** as the single source of truth for:

  - Hamiltonian construction (`common.hamiltonian`)
  - Filesystem layout (`common.paths`)
  - Naming and ASCII-safe identifiers (`common.naming`)
  - Plot routing and filenames (`common.plotting`)
  - Atomic JSON persistence and stable hashing (`common.persist`)

- Full **VarQITE (McLachlan) workflow** promoted to a first-class package:

  - Noiseless imaginary-time evolution with cached parameter trajectories.
  - Post-hoc noisy evaluation on `default.mixed` using density-matrix expectation values.
  - Depolarizing sweeps with multi-seed averaging and statistics.
  - Deterministic, seed-safe caching keyed on physical *and- numerical parameters.

- New **QITE CLI** with explicit command separation:

  - `qite run` for true VarQITE (pure-state, noiseless).
  - `qite eval-noise` for noisy evaluation and noise sweeps.

- **Round-trip caching tests** and **public API smoke tests** covering VQE, QPE, and QITE.
- ASCII-safe path guarantees for all result and image outputs (titles vs filenames formally separated).

### Changed

- **Major internal refactor of VQE, QPE, and QITE** to fully delegate:

  - Hamiltonians to `common.hamiltonian`
  - Paths to `common.paths`
  - Plot naming and routing to `common.plotting`
  - Hashing and persistence to `common.persist`

- Removed legacy `vqe_qpe_common` and replaced it with explicit, testable modules.
- Hardened all CLIs (VQE / QPE / QITE):

  - Deterministic run signatures
  - Identical caching semantics
  - Strict separation of computation, I/O, and plotting

- Standardized metadata returned by all Hamiltonian builders to ensure cross-algorithm compatibility.
- Notebooks updated to use **pure package APIs only** (no internal imports).

### Fixed

- Cache degeneracy and seed-collision bugs across QITE and QPE.
- Inconsistent molecule naming between paths and plot titles.
- Silent mismatches between Hamiltonian wire orderings in mixed stacks.
- Import-order and packaging errors revealed by full test isolation.

### Internal

- Repository architecture flattened and made fully explicit.
- All algorithms now share the same:

  - filesystem layout
  - naming rules
  - hashing logic
  - persistence model

- Test suite expanded to enforce architectural invariants, not just numerical correctness.

---

## [0.2.5] – January 12, 2026

### Added
- **Variational Quantum Deflation (VQD)** implementation for excited-state calculations:
  - Sequential k-state VQD workflow with deflation against previously converged states.
  - Noise-aware overlap penalties using density-matrix inner products.
  - Configurable deflation strength with support for ramped beta schedules.
  - Dedicated convergence plotting for multi-state VQD runs.
- Fully refactored **Subspace-Search VQE (SSVQE)** workflow:
  - Unified API consistent with the core VQE engine.
  - Explicit handling of k-state objectives with reproducible ordering of energies.
  - Improved noise support and plotting via shared visualization utilities.
- New VQE excited-state example notebooks:
  - `SSVQE.ipynb` and `SSVQE_Comparisons.ipynb`
  - `VQD.ipynb` and `VQD_Comparisons.ipynb`
- Public API exposure of excited-state solvers:
  - `run_ssvqe` and `run_vqd` available directly via `vqe` package imports.

### Changed
- Updated CLI (`python -m vqe`) to support:
  - Explicit SSVQE execution mode.
  - Clear separation between ground-state, SSVQE, and VQD workflows.
- Documentation updates across `README.md`, `USAGE.md`, and `THEORY.md`:
  - Excited-state methods promoted to first-class features.
  - Formal theoretical treatment of both SSVQE and VQD, including noise-aware formulations.
- Version bumped from **0.2.4 → 0.2.5**.

### Internal
- Refactored excited-state logic to reuse the shared VQE engine (devices, ansatz, noise, caching).
- Ensured deterministic ordering of excited-state energies independent of weight choices.

---

## [0.2.3] – December 22, 2025

### Added
- QPE CLI stability improvements
- Shared VQE/QPE common layer refinements

### Changed
- Notebook structure finalized (educational vs package-client split)
- Linting and CI workflows added (Black, Ruff, pytest)

### Fixed
- QPE CLI argument handling
- Import ordering and unused variables

---

## [0.2.2] – December 12, 2025

### Fixed
- Resolved GitHub Actions CI failures caused by invalid `pyproject.toml` license configuration.
- Corrected `project.license` to a valid SPDX string to satisfy PEP 621 validation.
- Removed deprecated and conflicting license classifiers that broke editable installs on CI.
- Fixed CI failures on Python 3.9 by aligning supported Python versions with PennyLane requirements.
- Ensured `pip install -e .` works reliably in clean CI environments.

### Changed
- Restricted supported Python versions to **>=3.10**, matching PennyLane ≥0.42 compatibility.
- Updated dependency constraints to prevent incompatible PennyLane versions being selected on older Python runtimes.
- Improved CI robustness by testing only supported Python versions.
- Bumped package version to **0.2.2**.

### Internal
- Verified full test suite passes locally and on GitHub Actions.
- Stabilised packaging and metadata to support future releases without CI regressions.

---

## [0.2.1] – November 30, 2025

### Fixed
- Resolved QPE sampling bug where 0-D arrays caused CLI crashes.
- Corrected `run_qpe()` handling to accept only keyword arguments.
- Updated `run_vqe()` test usage to match refactored API (removed deprecated arguments).
- Improved error handling in QPE CLI.
- Ensured QPE bitstring extraction works consistently across deterministic/nondeterministic outputs.

### Added
- Complete test suite overhaul for both VQE and QPE.
- GitHub Actions CI workflow (`tests.yml`) with Python 3.12 support.
- New minimal tests for VQE and QPE using unified Hamiltonian builder.
- Updated `USAGE.md` with accurate commands, outputs, and unified directory structure.

### Changed
- Bumped package version to **0.2.1**.
- Improved installation documentation.
- Unified plot and JSON output directories across VQE and QPE.
- Cleaned internal APIs for `run_vqe()` and `run_qpe()` to match the refactored package design.

---

## [0.2.0] – November 29, 2025

### Added
- First PyPI release of `vqe-pennylane`.
- Modularized `vqe/` and `qpe/` packages with shared logic under `vqe_qpe_common/`.
- Command-line interfaces for both VQE & QPE.
- Caching, plotting, and reproducible run hashing.
- Full molecule support (H₂, LiH, H₃⁺, H₂O).
- Noisy QPE via `default.mixed`.
- Initial example notebooks.
- Full repository refactor: new modules, documentation, directory structure.

---
