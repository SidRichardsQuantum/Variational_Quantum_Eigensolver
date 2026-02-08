# Changelog

All notable changes to this project will be documented in this file.

---

## [0.3.2] – 2026-02-08

### Added

* **Quantum Subspace Expansion (QSE)** as a first-class post-VQE excited-state method in the `vqe` package:

  * Implements standard QSE via subspace projection around a converged VQE reference state.
  * Operator pools constructed from top-|coeff| Pauli terms of the molecular Hamiltonian (plus identity).
  * Generalized eigenvalue problem $Hc = ESc$ solved with overlap-matrix eigenvalue filtering for numerical stability.
  * Returns lowest-*k* approximate eigenvalues with detailed diagnostics (subspace rank, conditioning, kept modes).
  * Designed explicitly as **noiseless-only**, consistent with statevector-based QSE theory.

* **New QSE example notebook**:

  * `notebooks/vqe/H2/QSE.ipynb`
  * Demonstrates end-to-end QSE workflow on H₂:

    * cached VQE reference,
    * subspace construction,
    * comparison against the exact qubit Hamiltonian spectrum,
    * visualization of QSE vs exact energies.
  * Written as a **pure package client**, using only public APIs and shared plotting utilities.

* **New QSE smoke + sanity tests**:

  * Minimal end-to-end pytest covering:

    * successful execution on H₂,
    * correct handling of reduced-rank subspaces,
    * sorted, finite eigenvalues,
    * presence and structure of diagnostics and configuration metadata.
  * Integrated into the existing test suite without increasing runtime instability.

### Changed

* **Documentation updates across the project**:

  * `README.md` — QSE added to the solver overview and project capabilities.
  * `USAGE.md` — clarified solver scope and excited-state landscape.
  * `THEORY.md` — extended excited-state discussion to include subspace-based (post-VQE) methods.
  * `notebooks/README_notebooks.md` — QSE notebook added, positioned alongside SSVQE and VQD.

* **Internal linear-algebra handling hardened**:

  * Explicit use of complex-valued Hamiltonian matrices where required to avoid silent imaginary-part truncation.
  * Improves numerical correctness without altering existing solver semantics.

### Fixed

* Prevented silent truncation of requested QSE eigenvalues when subspace rank is reduced by overlap filtering.
* Eliminated ambiguous casting of complex Hamiltonians to real arrays in internal linear-algebra paths.

### Internal

* QSE implemented **without introducing new infrastructure layers**:

  * reuses `common.hamiltonian`, `common.persist`, `common.plotting`, and existing VQE caching semantics.
* Maintains strict backward compatibility with all existing VQE, SSVQE, VQD, QPE, and QITE APIs.
* Version bumped from **0.3.1 → 0.3.2**.

---

## [0.3.1] - 2026-02-07

### Added

* **Full ADAPT-VQE implementation** as a first-class solver in the `vqe` package:

  * Chemistry-oriented ADAPT-VQE with Hartree–Fock reference state.
  * Operator pools based on **UCC singles / doubles / singles+doubles** (`uccs`, `uccd`, `uccsd`).
  * Deterministic operator selection via maximum energy gradient
    $|\partial E / \partial \theta|$ evaluated at zero initialization.
  * Explicit inner/outer loop structure with configurable:

    * inner optimizer steps and step size,
    * maximum operator budget,
    * gradient stopping tolerance.
  * Fully compatible with existing VQE infrastructure:

    * device selection,
    * noise handling,
    * caching,
    * plotting,
    * run hashing.

* **ADAPT-VQE CLI support** via `python -m vqe --adapt`:

  * Unified with existing VQE / SSVQE / VQD CLI dispatcher.
  * Supports operator pool selection, stopping criteria, noise flags, plotting, and cache control.
  * Results cached under the same deterministic hashing and filesystem conventions as standard VQE.

* **ADAPT-VQE result schema** standardized and persisted:

  * Outer-loop energies.
  * Inner-loop convergence trajectories per ADAPT iteration.
  * Maximum gradient history used for stopping.
  * Ordered list of selected operators with wire indices.
  * Final optimized parameter vector.
  * Full run configuration embedded in JSON records.

* **New ADAPT-VQE smoke test**:

  * Ensures end-to-end execution, caching, and deterministic behavior.
  * Integrated into the existing pytest suite alongside VQE / QPE / QITE tests.

### Changed

* **`vqe` CLI generalized** to act as a unified driver for:

  * ground-state VQE,
  * excited-state solvers (SSVQE, VQD),
  * adaptive ansatz construction (ADAPT-VQE).
* Documentation updates across:

  * `README.md` — ADAPT-VQE promoted to a first-class solver with conceptual and CLI overview.
  * `USAGE.md` — explicit ADAPT-VQE CLI usage and Python API examples.
* Plotting for ADAPT-VQE integrated with `common.plotting`:

  * unified filename construction,
  * consistent molecule labeling,
  * reproducible image paths.

### Fixed

* Ensured ADAPT-VQE noise flags are canonicalized consistently with standard VQE
  (non-effective noise parameters no longer pollute cache keys or filenames).
* Prevented silent operator/parameter mismatches by enforcing strict length checks
  between selected operator lists and parameter vectors.

### Internal

* ADAPT-VQE implemented without introducing any new infrastructure layers:

  * reuses `common.hamiltonian`, `common.plotting`, `common.persist`, and existing VQE engine utilities.
* Test suite expanded to cover adaptive workflows without increasing runtime instability.
* Codebase now supports **fixed-ansatz, adaptive-ansatz, variational imaginary-time, and phase-estimation workflows** under a single, coherent architecture.

---

## [0.3.0] – 2026-01-25

### Added

* **Unified infrastructure layer (`common/`)** as the single source of truth for:

  * Hamiltonian construction (`common.hamiltonian`)
  * Filesystem layout (`common.paths`)
  * Naming and ASCII-safe identifiers (`common.naming`)
  * Plot routing and filenames (`common.plotting`)
  * Atomic JSON persistence and stable hashing (`common.persist`)
* Full **VarQITE (McLachlan) workflow** promoted to a first-class package:

  * Noiseless imaginary-time evolution with cached parameter trajectories.
  * Post-hoc noisy evaluation on `default.mixed` using density-matrix expectation values.
  * Depolarizing sweeps with multi-seed averaging and statistics.
  * Deterministic, seed-safe caching keyed on physical *and* numerical parameters.
* New **QITE CLI** with explicit command separation:

  * `qite run` for true VarQITE (pure-state, noiseless).
  * `qite eval-noise` for noisy evaluation and noise sweeps.
* **Round-trip caching tests** and **public API smoke tests** covering VQE, QPE, and QITE.
* ASCII-safe path guarantees for all result and image outputs (titles vs filenames formally separated).

### Changed

* **Major internal refactor of VQE, QPE, and QITE** to fully delegate:

  * Hamiltonians to `common.hamiltonian`
  * Paths to `common.paths`
  * Plot naming and routing to `common.plotting`
  * Hashing and persistence to `common.persist`
* Removed legacy `vqe_qpe_common` and replaced it with explicit, testable modules.
* Hardened all CLIs (VQE / QPE / QITE):

  * Deterministic run signatures
  * Identical caching semantics
  * Strict separation of computation, I/O, and plotting
* Standardized metadata returned by all Hamiltonian builders to ensure cross-algorithm compatibility.
* Notebooks updated to use **pure package APIs only** (no internal imports).

### Fixed

* Cache degeneracy and seed-collision bugs across QITE and QPE.
* Inconsistent molecule naming between paths and plot titles.
* Silent mismatches between Hamiltonian wire orderings in mixed stacks.
* Import-order and packaging errors revealed by full test isolation.

### Internal

* Repository architecture flattened and made fully explicit.
* All algorithms now share the same:

  * filesystem layout
  * naming rules
  * hashing logic
  * persistence model
* Test suite expanded to enforce architectural invariants, not just numerical correctness.
* Project is now structurally ready for QITE extensions, QSVT, and QML modules.

---

## [0.2.5] – 2026-01-12

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
- Maintained full backwards compatibility with existing VQE and QPE APIs.

---

## [0.2.3] – 2025-12-22

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

## [0.2.2] - 2025-12-12
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

## [0.2.1] - 2025-11-30
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

## [0.2.0] - 2025-11-29
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
