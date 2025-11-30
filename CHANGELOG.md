# Changelog

All notable changes to this project will be documented in this file.

---

## [0.2.1] - 2025-11-30
### Fixed
- Resolved QPE sampling bug where 0‑D arrays caused CLI crashes.
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

## [0.1.0] - 2025-11-09
### Added
- Early VQE prototype.
- Prototype QPE implementation.
- Legacy notebooks and utilities.
- Initial Hamiltonian generation and ansatz support.

---

