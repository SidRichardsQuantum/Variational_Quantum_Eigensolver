# Package Engineering Roadmap

This roadmap tracks repository and package reliability, completion of existing
solver workflows, and documentation clarity.
Research questions and proposed benchmark notebooks remain in
[`notebooks/BENCHMARK_ROADMAP.md`](notebooks/BENCHMARK_ROADMAP.md).

Optional qml-pennylane studies for sparse energy-curve interpolation and the
effect of solver error on physics classification are tracked in the
[notebook roadmap](notebooks/BENCHMARK_ROADMAP.md#optional-qml-studies-on-simulation-results).
They consume simulation results and do not require a core ML dependency or
changes to the solver engines.

Open items are ordered by their impact on installation reliability,
reproducibility, and release confidence.

## Priority 0: Installation And Supported Environments

### Migrate supported Python and NumPy versions

Planned work:

- migrate the supported PennyLane/NumPy range together
- validate Python 3.13 before restoring it to package metadata and CI
- make the NumPy 2 job required once that range is supported

## Priority 1: CI And Merge Protection

### Require fast test jobs before merge

Planned work:

- configure branch protection/rulesets to require the fast test jobs before merge

This is a GitHub repository setting; workflow changes alone do not enforce it.

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

Implement shared problem inputs and prepared-state QPE first, then convergence
reporting, warm starts, and reusable trajectory analysis. Research validation is
tracked in the [notebook roadmap](notebooks/BENCHMARK_ROADMAP.md#validation-of-existing-solver-workflows).

### Share problem inputs across ADAPT and excited-state methods

Planned work:

- Extend ADAPT-VQE and the existing excited-state entrypoints through the shared
  problem-resolution layer to accept explicit geometries, active spaces,
  multiplicity, and expert Hamiltonians where their ansatz or operator pool
  supports them. Add explicit multiplicity input to QPE.
- Ensure post-VQE methods use the same resolved problem for their reference
  state and projected eigenproblem. Keep unsupported chemistry-dependent pools
  explicit rather than silently substituting a different problem or method.
- Include resolved inputs in cache signatures and result metadata, document
  method-specific restrictions, and forward supported chemistry options through
  the existing CLIs. Keep expert Hamiltonian inputs in the Python API.

Completion criteria:

- small active-space, explicit spin-reference, and expert-model cases verify
  shared Hamiltonian/reference conventions across supported methods
- cache entries distinguish physical input changes, unsupported combinations
  fail clearly, and existing molecule-name calls retain their behavior

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

### Report convergence and termination explicitly

Planned work:

- Start with VQE and VarQITE: add optional stopping criteria, actual update
  counts, non-finite-value detection, and explicit termination reasons.
  Preserve fixed-step execution when stopping controls are omitted.
- Separate budget exhaustion, tolerance satisfaction, and numerical failure.
  Record the criterion, threshold, and measured diagnostic; keep eigenstate
  quality diagnostics distinct from optimizer convergence.
- Carry status into benchmark records and comparisons so failed or non-finite
  runs cannot appear as successful evidence. Include stopping settings in cache
  identity and preserve status on cache hits.

Completion criteria:

- controlled cases cover tolerance satisfaction, budget exhaustion, and
  numerical failure, with histories aligned to the actual number of updates
- real-time evolution retains its requested time horizon; stationary energy
  alone is not used as a stopping criterion or an accuracy certificate for QRTE

### Support supplied parameters and geometry-scan warm starts

Planned work:

- Add `initial_params` to VQE and VarQITE, following the existing VarQRTE
  convention. Record parameter shape and initialization provenance, validate
  finite values and compatibility, and include supplied parameters in cache keys.
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

## Priority 2: Source Distribution Completeness

### Keep packaged documentation links valid

Audit repository-relative documentation links in the source distribution.

Planned work:

- remove/replace any remaining links that are intended to be repository-only
- add a package-content check for documentation targets referenced from included
  Markdown files

Completion criteria:

- documentation shipped in the sdist contains no repository-relative links to
  omitted files
- wheel and sdist metadata continue to pass `twine check`

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
