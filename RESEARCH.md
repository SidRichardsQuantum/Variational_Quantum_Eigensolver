# Research Use

This repository is most useful as a reproducible evidence generator for
small-system quantum simulation studies. It is not a claim that any one method
is universally best, and it is not production chemistry software.

For the problem scope and user-facing workflows, read `PROBLEM.md`. For the
current benchmark inventory, read `notebooks/benchmarks/SUMMARY.md`. For
published tables and figures, read `notebooks/benchmarks/RESULTS.md`.

## Research Claims This Repo Can Support

Use this repo to support claims of these forms:

- for a named small molecule or low-qubit Hamiltonian, method A was more
  accurate, faster, or more stable than method B under a stated configuration
- a solver default is reasonable for a documented calibration panel
- a configuration is sensitive to seed, shots, noise channel, mapping, ansatz,
  optimizer, or active-space choice
- a non-chemistry Hamiltonian can be run through the same expert-mode API as the
  chemistry benchmarks

Do not use this repo, by itself, to claim:

- chemical accuracy for large molecules
- hardware performance or device-readiness
- universal algorithm rankings across problem classes
- production-quality quantum chemistry results

## Evidence Standard

A result should be treated as research evidence only when it records:

- the resolved problem: molecule or model, geometry, charge, basis, mapping,
  active space, qubit count, and Hamiltonian-term count where available
- the reference: exact diagonalization, Hartree-Fock, analytical model result,
  or an explicit statement that no reference is used
- the solver configuration: method, ansatz, optimizer, step counts, stepsizes,
  QPE evolution settings, shots, seeds, and noise model
- the metrics: energy, absolute error against the reference where meaningful,
  runtime, cache-hit state, and method-specific diagnostics
- the statistical design: seed list, shot list, repetitions, failure criteria,
  and aggregation method for stochastic or optimizer-sensitive studies
- the environment: package version and relevant dependency versions for
  release-grade benchmark artifacts

The benchmark row contract is documented in
`notebooks/benchmarks/SCHEMA.md`.

## Claim Levels

| Level | Meaning | Minimum evidence |
| --- | --- | --- |
| Smoke | The API path runs. | One tiny deterministic case. |
| Case study | A method behaves as reported on one problem. | One problem, fixed config, reference where available. |
| Benchmark | A comparison is decision-useful. | Multiple methods or settings, common reference, runtime/cache metadata, documented metrics. |
| Reproducibility study | Stability is measured. | Multiple seeds or shots, aggregate statistics, failure-rate notes. |
| Release-grade evidence | A result can be cited. | Curated artifact export, versioned code, clean validation checks, and documented limitations. |

## Benchmark Acceptance Checklist

Before treating a new notebook as a benchmark, verify that it:

- asks one explicit research or method-selection question
- avoids duplicating an existing notebook unless it replaces or generalizes it
- follows `notebooks/benchmarks/TEMPLATE.md` for question, scope, reference,
  metrics, aggregation, limitations, and artifact-export notes
- uses the shared problem-resolution and Hamiltonian pipeline
- compares against an exact or clearly documented reference when feasible
- reports cache hits separately from compute runtime
- exports any published table or figure through
  `scripts/export_benchmark_artifacts.py`
- states limitations in the notebook or nearby docs when a method is known to
  be noiseless-only, calibration-specific, or small-system-only

## Release Protocol

For a release intended to be useful in research:

1. Run the default and full test suites.
2. Run registered benchmark suites that should become release evidence. Use
   `python -m common.benchmarks list` to inspect available suites, then run a
   selected suite, for example
   `python -m common.benchmarks run --suite h2-cross-method`.
3. Compare refreshed suite outputs against the previous evidence set with
   `python -m common.benchmarks compare --base old_run --head new_run`.
4. Rerun benchmark notebooks whose results are being refreshed.
5. Export curated artifacts with `python scripts/export_benchmark_artifacts.py`.
6. Confirm `notebooks/benchmarks/_artifacts/benchmark_manifest.json` describes
   the published artifact set.
7. Build docs with `python -m sphinx -W -b html docs docs/_build/html`.
8. Tag the code and attach or archive the curated artifact set if the release is
   meant to be cited directly.

## Document Boundaries

The markdown files intentionally have separate jobs:

- `README.md`: installation, orientation, and quickstart
- `PROBLEM.md`: what practical problems the repo is for
- `THEORY.md`: algorithm background
- `USAGE.md`: API and CLI usage
- `RESEARCH.md`: evidence standards and benchmark acceptance rules
- `notebooks/benchmarks/SUMMARY.md`: benchmark inventory
- `notebooks/benchmarks/RESULTS.md`: curated result surfaces
- `notebooks/benchmarks/SCHEMA.md`: benchmark row and artifact metadata fields
