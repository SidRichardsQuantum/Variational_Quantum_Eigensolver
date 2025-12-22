# Changelog

All notable changes to this project will be documented in this file.

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
