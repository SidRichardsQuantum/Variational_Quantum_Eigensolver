# Package Engineering Roadmap

This roadmap tracks repository and package reliability work. Research questions
and proposed benchmark notebooks remain in
[`notebooks/BENCHMARK_ROADMAP.md`](notebooks/BENCHMARK_ROADMAP.md).

Open items are ordered by their impact on installation reliability,
reproducibility, and release confidence. Completed work is marked with its release.

## Priority 0: Installation And Supported Environments

### Move generated data outside the installed package — completed in v0.3.27

Runtime artifacts now use platform user data directories. The explicit
`VQE_PENNYLANE_DATA_DIR` override is resolved when files are accessed, including
after imports. VQE, QPE, VarQITE, and VarQRTE have save/reuse regression coverage,
and `README.md` and `USAGE.md` document the defaults and migration.

### Align supported Python and NumPy versions — completed in v0.3.27

Package metadata and the required CI matrix support Python 3.10–3.12 with
NumPy 1.x. Python 3.13 is excluded from `requires-python` and the classifiers.
The optional NumPy 2 probe runs on Python 3.12, installs its intended dependency
set first, then installs the project without re-resolving NumPy 1.x.

Remaining migration work:

- migrate the supported PennyLane/NumPy range together
- validate Python 3.13 before restoring it to package metadata and CI
- make the NumPy 2 job required once that range is supported

## Priority 1: Numerical API Correctness

### Validate solvers and record fallback — completed in v0.3.27

VarQITE and VarQRTE reject unknown solver names before cache lookup or circuit
evaluation. Only linear-algebra failures trigger fallback; result metadata
records the actual solver at every step, including after cache reuse.

### Correct spin references and QPE cache equivalence — completed in v0.3.27

References reflect multiplicity and UCC/ADAPT pools conserve their spin
projection. QPE canonicalizes its Trotter ordering to match its cache identity.
Regression tests cover both behaviors and invalidate earlier numerical caches.
Exact total-spin constraints remain outside the current excitation ansatz's
contract and require separate validation when scientifically necessary.

## Priority 1: CI And Merge Protection

### Restore pull-request test coverage — completed in v0.3.27

Pull requests and pushes to main run the fast Python 3.10–3.12 matrix. Pushes
to main and manual dispatches also run the full integration suite and the
optional NumPy 2 probe. Superseded runs retain concurrency cancellation.

Remaining repository administration:

- configure branch protection/rulesets to require the fast test jobs before merge

This is a GitHub repository setting; workflow changes alone do not enforce it.

## Priority 2: Source Distribution Completeness

### Keep packaged documentation links valid

Version 0.3.27 includes `PROBLEM.md`, `ROADMAP.md`, and `more_docs/` in the source
distribution. A broader audit of repository-relative documentation links remains.

Planned work:

- remove/replace any remaining links that are intended to be repository-only
- add a package-content check for documentation targets referenced from included
  Markdown files

Completion criteria:

- documentation shipped in the sdist contains no repository-relative links to
  omitted files
- wheel and sdist metadata continue to pass `twine check`
