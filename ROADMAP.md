# Package Engineering Roadmap

This roadmap tracks repository and package reliability, completion of existing
solver workflows, and documentation clarity.
Research questions and proposed benchmark notebooks remain in
[`notebooks/BENCHMARK_ROADMAP.md`](notebooks/BENCHMARK_ROADMAP.md).

Open items are ordered by their impact on installation reliability,
reproducibility, and release confidence.

## Current Baseline And Next Milestones

The v0.3.28 Studio runs from a source checkout and supports VQE/ADAPT-VQE,
live observations, comparisons, JSON export, cache-backed history, and re-run
configuration restoration, queued/running cancellation, and isolated worker
processes with private artifact staging. It remains outside the wheel and source distribution;
see the [Studio guide](docs/studio.md) for setup and current limitations.

The working tree now adds explicit termination, supplied initial parameters,
shared inputs across existing solver workflows, Studio VarQITE, and compatible
VQE → VarQITE refinement with source comparison and combined compute costs.
The next scientific milestone is sector-aware references/state diagnostics.
Prepared-state QPE, geometry-scan continuation, and reusable trajectory analysis
remain planned. Python/NumPy compatibility is a separate installation milestone.

## Priority 0: Installation And Supported Environments

### Migrate supported Python and NumPy versions

Planned work:

- migrate the supported PennyLane/NumPy range together
- validate Python 3.13 before restoring it to package metadata and CI
- make the NumPy 2 job required once that range is supported

## Priority 1: CI And Merge Protection

### Optionally enforce checks through repository rulesets

Planned work:

- activate the prepared `.github/rulesets/main.json` ruleset to require fast
  tests, lint, package, and Studio checks before merge; the current integration
  token returned HTTP 403 when attempting activation. See the
  [release guide](docs/releasing.md) for the exact administrator command.

This is a GitHub repository setting; workflow changes alone do not enforce it.

## Completed In v0.3.28: Studio Execution Reliability

- Queued and running cancellation with persisted `cancelled` state and timestamps,
  card/detail actions, and idempotent cancellation requests.
- One isolated worker process at a time, with copied cache inputs and private
  scientific output until successful publication. Solver defaults, seeds, and
  scientific cache identity remain unchanged.
- Completion and cancellation share a lock: the first terminal transition wins.
  Shutdown cancels pending work; abrupt parent exit stops the worker. Restart
  marks interrupted manifests failed and removes stale staging directories.
- Lifecycle tests cover cancellation, cache preservation, completion races,
  shutdown, parent exit, and restart. Browser checks cover both cancellation
  actions, restored statuses, and successful execution afterwards.

## Priority 1: Scientific References And State Diagnostics

### Match exact references to the physical sector being studied

Extend the existing small-system reference helpers so chemistry comparisons can
distinguish the full qubit spectrum from the spectrum accessible in a specified
electron-number and spin-projection sector.

Planned work:

- Accept the same resolved geometry, basis, active space, and mapping as the
  solver, with explicit full-space or sector-restricted reference selection.
  Preserve the current full-space behavior by default and label the selection
  in reference metadata and benchmark outputs.
- Add shared state diagnostics for particle number, spin projection, total
  spin `S²`, and energy variance, supporting statevectors and density matrices
  where applicable. Report sector weight or leakage when a sector is selected.
- Surface available references, reference errors, sector leakage, and state
  diagnostics in Studio with the selected physical sector and reference scope
  clearly labelled. Never infer missing diagnostics from completion status.
- Respect qubit encodings when constructing sector projectors and observables.
  Distinguish conservation of particle number and `M_s` from a pure total-spin
  sector; measuring `S²` does not impose that constraint.

Completion criteria:

- small reference cases verify sector membership and consistent results across
  supported mappings, including a case where the full-space and selected-sector
  spectra differ
- diagnostics identify deliberately introduced sector leakage and distinguish
  an eigenstate from a superposition; low variance alone is not labelled proof
  of ground-state accuracy
- references and diagnostics remain bounded to small-system validation; generic
  symmetry tapering and production-scale diagonalization are outside this item

## Priority 2: Complete Existing Solver Workflows

Shared problem inputs, VQE/VarQITE termination, and supplied parameters are implemented.
Prepared-state QPE, scan continuation, and reusable trajectory analysis remain open. Research validation is
tracked in the [notebook roadmap](notebooks/BENCHMARK_ROADMAP.md#validation-of-existing-solver-workflows).

### Implemented: shared inputs across ADAPT and excited-state methods

- ADAPT and existing excited-state entrypoints use shared problem resolution for
  explicit geometry, active spaces, and multiplicity. QPE accepts multiplicity.
- Post-VQE methods reuse the resolved Hamiltonian and reference. Expert inputs
  work with supported ansatzes/pools; ADAPT explicitly requires its chemistry pool.
- Resolved inputs participate in cache identity and result metadata; supported
  chemistry options are forwarded by the CLIs. Expert Hamiltonians remain Python-only.
- Regression coverage includes active-space ADAPT, spin references, sparse expert
  registers, cache identity, and existing molecule-name workflows.

### Accept prepared states in QPE

Planned work:

- Support VQE- and VarQITE-prepared inputs through a reproducible preparation
  circuit specification or an explicitly simulator-only statevector input,
  preserving basis-state preparation as the default.
- Validate register size, normalization, wire order, and mapping. Include the
  preparation and bound parameters in cache identity; do not cache arbitrary
  callables by name alone.
- Keep phase-unwrapping reference selection explicit and separate from state
  preparation. Record preparation provenance and document how its noise is
  handled. Correct the current documentation claim of prepared-state API support
  as part of delivering and documenting the supported path.

Completion criteria:

- tiny eigenstate and superposition inputs produce the expected analytic
  ancilla distributions, with finite-shot checks treated statistically
- VQE → QPE and VarQITE → QPE examples reuse the same resolved problem;
  preparation changes invalidate cache reuse, and existing HF calls still work

### Implemented: explicit termination reporting

- VQE and VarQITE retain fixed budgets by default and optionally stop after a
  specified number of consecutive small energy changes. Results record reasons,
  thresholds, diagnostics, actual updates, and attempted updates. Numerical failure
  retains only finite iterates; cache hits preserve the outcome.
- Studio displays these outcomes, marks numerical failures failed, and distinguishes
  ADAPT operator budget, pool exhaustion, and gradient tolerance. ADAPT non-finite
  calculations raise rather than publishing a completed result.
- Benchmark rows retain termination and failure status. QRTE retains its requested
  time horizon; optimizer stopping is separate from physical accuracy.

### Supplied parameters implemented; geometry-scan continuation remains planned

VQE and VarQITE accept finite, shape-compatible `initial_params`, record shape
and provenance, and include initialization in cache identity. Studio verifies
same-problem, same-state VQE → VarQITE transfer. The `h2-refinement` suite compares
this workflow with independent VarQITE across three seeds, including source cost.
This is parameter transfer, not optimizer checkpoint/resume.

Remaining work:

- Allow opt-in continuation along geometry scans and VQE → VarQITE refinement.
  Preserve independent seeded starts as the default and keep separate
  continuation chains for each seed.
- Check ansatz, parameter ordering, reference, mapping, and active-space
  compatibility before transferring parameters. Document that transfer between
  molecular geometries is an initialization heuristic, not an assertion that
  orbital bases or physical states are identical.

Completion criteria:

- supplied parameters reproduce the intended initial ansatz state, incompatible
  transfers fail clearly, and cache hits preserve initialization metadata
- scan outputs record direction and predecessor provenance; parameter warm
  starts are distinguished from exact optimizer checkpoint/resume, which remains
  deferred

### Reuse observable-trajectory analysis outside notebooks

Planned work:

- Extract the reusable analysis in the existing exact QRTE benchmark into small
  shared helpers for reconstructing states from parameter histories, evaluating
  named Pauli observables, and comparing against optional reference trajectories.
- Preserve time grids, wire order, parameter shape, and state representation.
  Support energy/conservation diagnostics and fidelities where the supplied
  reference permits them, with explicit storage and reconstruction costs.

Completion criteria:

- the existing QRTE benchmark uses the helpers and retains its observable and
  fidelity results within declared numerical tolerances
- package users can analyze a trajectory without copying notebook internals;
  general time-dependent Hamiltonians and open-system dynamics remain outside
  this item

## Priority 2: Incremental Studio Workflow Expansion

Deliver these after execution reliability and the shared solver contracts above.

Planned work:

- Implemented: VarQITE, VQE → VarQITE refinement with source digest and prepared-state
  checks, independent seeded starts, source comparison, initialization provenance,
  combined compute costs, cache reuse, export, and browser coverage.
- Add explicit geometry and active-space controls only through the shared problem
  resolver, respecting each method's supported inputs. Enable recreation of
  compatible Python/CLI artifacts only when their original inputs can be
  recovered and verified; otherwise retain view/export access.
- Follow with prepared-state QPE after its preparation API is implemented, then
  excited-state methods (VQD/SSVQE and QSE-family views). Add VarQRTE with a time
  axis and observable trajectories, not an energy-minimization presentation.
- Index/paginate large histories and add reproducibility bundles containing
  immutable artifacts, resolved inputs, preparation/initialization provenance,
  and environment metadata with compatibility checks.

Completion criteria:

- each adapter reuses the Python solver and authoritative artifacts, with
  method-specific validation, cache reuse, export, and browser coverage
- refinement and prepared-state examples use the same resolved physical problem;
  unsupported transfers fail clearly before submission
- restored/imported runs retain provenance; bundles identify missing or
  incompatible environments without silently changing the experiment

## Completed In v0.3.28: Source Distribution Documentation

Repository-only notebook links point to their hosted sources. The generated
benchmark report stays outside the sdist alongside its figures; research notes
link to the hosted report. Packaged relative Markdown targets are checked against actual sdist members in package CI
with `scripts/check_sdist_links.py`. Wheel and sdist metadata are checked with
`twine check`; Studio remains excluded from both distributions.

## Priority 2: Plot And Diagram Clarity

### Improve existing scientific visuals without expanding their scope

These changes support the package's small-system method comparisons and explain
its existing execution paths. Keep the current figure and panel counts, preserve
the underlying results, and use the existing Matplotlib and Mermaid tooling.

Planned work:

- Low-qubit VQE benchmark: replace absolute-error bars with dots and error bars
  on a logarithmic axis so the smaller errors remain visible. Handle zero errors
  and intervals reaching zero explicitly. Shorten panel titles to "Runtime" and
  "Absolute error", and identify "mean ± SD across seeds" once. Update both the
  package helper in `vqe/core.py` and the existing benchmark notebook.
- QPE calibration decision map: choose annotation text colors for contrast with
  each heatmap cell's background in
  `notebooks/benchmarks/qpe/H2/Calibration_Decision_Map.ipynb`.
- H2 ansatz and mapping comparisons: add an exact ground-energy reference for
  the same resolved problem and distinguish overlapping curves with line styles
  in `vqe/core.py`. Give the mapping figure a short caption explaining that it
  reports the configured VQE runs; the plot alone does not establish the cause
  of different final energies.
- Excited-state spectra: replace vertical exact-energy stems with short
  horizontal level marks in the LR-VQE, EOM-VQE, and EOM-QSE plotting helpers in
  `vqe/visualize.py`, preserving level indices, estimated energies, and error
  annotations while making the lowest exact level visible.
- Architecture diagrams in `more_docs/architecture.md`: route cache hits to
  "Return result" and misses through "Compute → Store → Return"; add the VQE
  optimizer-to-ansatz feedback arrow labelled "Updated parameters"; and show
  prebuilt Hamiltonians bypassing Hamiltonian construction during resolution.

Completion criteria:

- existing plots remain readable at notebook and documentation display sizes,
  including overlapping traces, small errors, and heatmap annotations
- exact references match the plotted problem, and uncertainty labels and scale
  choices preserve the meaning of the data
- affected notebook outputs are regenerated, curated figures are refreshed with
  `scripts/export_benchmark_artifacts.py`, and both are visually checked for
  consistency
- rendered architecture diagrams agree with the implemented execution paths

## Priority 3: Optional Cross-SDK Validation And Benchmarking

### Validate this package's circuits with quantum-backend-bench

Use [quantum-backend-bench](https://github.com/SidRichardsQuantum/Quantum_Backend_Bench)
to check numerical agreement and measure execution costs for the small-system
VQE and QPE workloads produced here. Keep chemistry, optimization, and scientific
result interpretation in this package; reuse the other package's neutral schemas,
SDK translation, and execution reports through an optional adapter.

Planned work, in implementation order:

- Add an optional dependency extra and a small bridge in `common` for resolved
  Pauli Hamiltonians and captured state-preparation circuits. Preserve reference
  preparation, wire order, qubit mapping, identity terms, and parameter bindings.
  Decompose templates into supported gates and report unsupported operations
  explicitly. Validate a compatible Python/NumPy/PennyLane dependency set without
  narrowing the base package's supported environments.
- Add a frozen-parameter VQE validation suite through `common/benchmarks.py`:
  start with a small explicit Hamiltonian and `RY-CZ`, then cover H2 and decomposed
  UCCSD. Compare the same prepared state's energy on PennyLane and one additional
  local SDK, using the same resolved Hamiltonian and parameters. Distinguish
  cross-SDK disagreement from variational error against the exact ground energy.
- Export versioned circuit, Hamiltonian, and supported workflow artifacts with
  parameter bindings and environment metadata alongside existing benchmark
  outputs. Verify Hamiltonian round trips and execute generated SDK workflows;
  neutral semantic verification alone does not establish native SDK agreement.
- Benchmark those validated circuits with repeated execution timings and
  structural metrics. Separate execution, translation/setup, and full solver
  runtime; identify cache hits and use matching shots and measurement requests.
  Extend to decomposed QPE circuits once controlled evolution and ancilla bit
  ordering are validated, comparing ancilla distributions and derived energies.

Completion criteria:

- base installation, imports, and normal solver runs work without the extra;
  optional integration checks exercise a documented compatible dependency set
- small noiseless cases agree in energy within declared numerical tolerances,
  with statevector checks up to global phase where supported; matching
  computational-basis probabilities alone is insufficient for VQE validation
- integration checks cover reference preparation, wire/mapping conventions, and
  rejected operations; sampled comparisons report shot uncertainty separately
  from deterministic numerical tolerances
- saved artifacts reproduce the frozen workload and identify SDK versions,
  execution settings, timing scope, and whether native execution was verified

Scope is local simulator validation and workload profiling. Generic SDK
translation belongs upstream in quantum-backend-bench. Cloud/QPU execution,
automatic backend selection, optimizer migration, and replacement of the VQE or
VarQITE/VarQRTE engines are outside this integration. Measurement-cost research is
tracked in the [notebook benchmark roadmap](notebooks/BENCHMARK_ROADMAP.md#cross-sdk-validation-and-measurement-cost).
