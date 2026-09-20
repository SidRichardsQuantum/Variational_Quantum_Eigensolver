# Notebook Benchmark Roadmap

This roadmap is intentionally narrow: add notebooks that answer method-selection or research-validation questions, not just more usage demos.

Clarity improvements to existing benchmark figures are tracked in the
[package engineering roadmap](../ROADMAP.md#priority-2-plot-and-diagram-clarity).

## Validation Of Existing Solver Workflows

These studies validate the planned
[scientific reference helpers](../ROADMAP.md#priority-1-scientific-references-and-state-diagnostics)
and [completion of existing solver workflows](../ROADMAP.md#priority-2-complete-existing-solver-workflows).
Implement API contracts and small regression checks in the package first. Extend
existing benchmarks where practical; these questions do not require six new
notebooks or any new solver family.

### Physical sectors and comparable method inputs

Question: do method rankings or excited-state assignments change when references
and trial states are compared within the intended physical sector?

- Extend an existing chemistry or excited-state comparison with full-space and
  electron-number/`M_s`-restricted exact references for the same resolved problem.
  Include a small case where the spectra differ and report particle number,
  `S²`, sector leakage, and energy variance where applicable.
- Exercise shared active-space and explicit-geometry inputs across the methods
  that support them, and extend an existing model benchmark to supported
  excited-state workflows. Identify ansatz/pool restrictions and match reference
  sectors before interpreting energy errors or state ordering.

### Prepared-state QPE

Question: does VQE or VarQITE preparation increase the probability of obtaining
the target spectral peak enough to justify its preparation cost?

- Extend an existing QPE calibration or cross-method benchmark with HF, VQE,
  and VarQITE preparations of the same small problem. Hold evolution time,
  ancillas, Trotter settings, and decoding policy fixed for the comparison.
- Report target-state or degenerate-subspace overlap, target-bin/window success
  probability, decoded energy error, preparation cost, and QPE execution cost.
  Separate preparation error, Trotter error, finite phase resolution, aliasing,
  and shot uncertainty; include a case where better preparation does not improve
  the decoded energy because another error source dominates.

### Convergence and warm-start efficiency

Question: can stopping criteria or transferred parameters reduce work at a
declared accuracy without hiding failures or scan-direction dependence?

- Extend existing optimizer/reproducibility benchmarks with fixed budgets and
  optional stopping criteria. Report termination reasons, actual updates,
  evaluation counts where available, runtime, reference error, and variance.
  Count failed and unconverged runs in aggregate outcomes.
- Extend the existing bond-scan example with independent starts and forward and
  reverse continuation, using matched geometries, ansatzes, and accuracy targets.
  Compare seed spread, total scan cost, and direction dependence. Validate
  orbital/parameter conventions before interpreting transferred parameters.
- Compare a small VQE → VarQITE refinement with independent VarQITE initialization,
  including the VQE preparation in the total cost.

### Observable trajectories

Question: do the shared analysis helpers preserve the existing QRTE evidence,
and which diagnostics reveal trajectory error when energy drift is small?

- Refactor the existing exact QRTE benchmark to use the shared helpers and
  verify its current observable and fidelity results.
- Extend that benchmark, only as needed, with a time-step or ansatz comparison
  against exact evolution from the same initial state. Report observable error,
  fidelity, energy drift, and reconstruction/runtime costs separately; energy
  conservation alone does not establish trajectory accuracy.

Follow `RESEARCH.md` and the benchmark row/artifact contract for all studies,
including resolved problem and sector labels, seeds, failures, environment
versions, and separate cache-hit and compute timings. Export refreshed evidence
through the existing curated-artifact workflow.

## Future QPE Calibration Scope

Keep future QPE calibration additions focused on new molecules or materially
different failure modes rather than duplicating the decision-map workflow.

## Cross-SDK Validation And Measurement Cost

Build on the optional integration in the
[package engineering roadmap](../ROADMAP.md#priority-3-optional-cross-sdk-validation-and-benchmarking).
Prioritize two questions using this package's existing small-system workloads:

- Does a frozen VQE state give matching energies across local SDKs, and how does
  execution cost vary for the same circuit and measurement task? Start with a
  small explicit Hamiltonian and H2. Report energy disagreement separately from
  ground-state error, repeated execution timings, and decomposed gate counts.
  Require actual SDK execution in addition to neutral translation verification.
- How much can qubit-wise commuting Pauli grouping reduce measurement settings
  for a useful VQE energy estimate? Reuse quantum-backend-bench's grouping helper
  on the same Hamiltonians and frozen states. Compare grouped and ungrouped
  estimates at equal total shot budgets, reporting group counts, energy bias,
  and uncertainty across repeats. Keep this a finite-shot evaluation study;
  grouping alone does not demonstrate faster analytic VQE optimization.

Extend an existing reproducibility or ansatz benchmark where practical. Add a
dedicated notebook only if these questions cannot be answered clearly there.

## Scope Guardrail

New notebooks should close a decision gap, not create another near-duplicate example.

When a new notebook overlaps strongly with an existing one, prefer replacing,
merging, or repurposing the older notebook instead of expanding the inventory.
