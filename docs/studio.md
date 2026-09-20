# VQE Experiment Studio

An optional, local web interface for composing, running, inspecting, and reusing
ground-state VQE, ADAPT-VQE, and VarQITE experiments. The scientific Python package remains the
only computational implementation. No generative AI, image generation, frontend
framework, Node runtime, or web dependency is added to `pip install vqe-pennylane`.
The studio runs from a source checkout; it is not installed in the Python wheel.

## Run in a fresh Codespace

From the repository root, use **Python 3.12** (the package supports 3.10–3.12).
The current default Codespaces `python` may be newer. These commands use the
system Python 3.12 and bootstrap with the existing Codespaces pip, so they also
work when the system `python3.12-venv` / `ensurepip` package is absent:

```bash
/usr/bin/python3.12 -m venv --without-pip .venv-studio
python -m pip --python .venv-studio/bin/python install -e .
export VQE_PENNYLANE_DATA_DIR="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-studio/bin/python -m studio --host 0.0.0.0 --port 8000
```

Open port **8000** in the Codespaces **Ports** panel and choose **Open in Browser**.
Keep port visibility **Private**. Locally, `python -m studio` defaults to
`http://127.0.0.1:8000`. Only one Studio server may own a data directory at a time.
A second instance is rejected before it can modify jobs or worker files. Stop the
first server or choose another `VQE_PENNYLANE_DATA_DIR` to start a separate instance.
The standard-library server is for a trusted local user, not a public deployment.
Host validation, a per-server submission token, a strict JSON body limit, and
same-origin static assets protect the local submission endpoint. These are not
user authentication or a multi-user job service.

Choose Ground-state VQE, ADAPT-VQE, or VarQITE in the method selector. Start with H2 / UCCSD / Adam for VQE. VQE automatically displays the selected optimizer's calibrated step size.
Changing optimizer updates that value while it is automatic; editing it creates
a custom override, and clearing it restores automatic selection. Method changes
load the new solver's defaults while retaining molecule and mapping. Molecule
changes refresh the registry basis and preserve your solver settings. These are
package defaults, not molecule-specific convergence guarantees. The molecule selector is populated directly from
`common.molecules.MOLECULES`. The registry controls basis, geometry, charge,
multiplicity, and active-space defaults; basis is therefore read-only. Other
bases require explicit geometry in the Python API. Some molecule/ansatz
combinations may fail, and larger systems may take substantial time; the actual
solver exception appears on the run card.

**Re-run** restores the configuration for editing. Press **Run experiment** to
submit it. Matching experiments reuse the core cache. There is no force-recompute
control in this MVP. A browser reload does not stop a job. **Cancel** on a card or in its detail view
cancels queued or running work. Queued cancellations never start a solver; running
cancellations kill the isolated worker, discard its provisional output, and let
the next queued run start after cleanup. Repeated cancellation is harmless. If
completion has already been recorded, cancellation returns that completed status.
Cancellation records `cancelled_at` and `finished_at`; it is an operational action,
not a numerical convergence or termination verdict.
Cancelled runs show their terminal status without an active progress indicator.

Ctrl-C cancels queued and running work and waits for worker cleanup. An abrupt
server exit closes the worker's parent-liveness pipe, causing it to exit; on the
next startup unfinished manifests are marked failed with a finish timestamp and
private staging directories are removed. Work is never automatically replayed.
Programmatic `Jobs.close(cancel=False)` explicitly drains the queue instead.
An OS-backed lock holds directory ownership until worker cleanup completes and
is automatically released if the server process exits. The persistent
`results/studio/.owner.lock` file is not itself evidence of a running server;
leave it in place, including after a crash.

## Refine a VQE run with VarQITE

Open a completed noiseless Studio VQE run and choose **Refine with VarQITE**.
The composer displays the source artifact and retains its molecule, mapping, and
ansatz. Those controls are locked while refinement is selected. Set the
imaginary-time step, update budget, linear solver, regularization, and optional
stopping tolerance, then press **Run experiment**. **Start independently** removes
the source while keeping your VarQITE settings. Changing method also resets the
initialization. **Re-run** on a refinement restores its source and settings.

The backend verifies the source digest, resolved Hamiltonian/geometry/active space,
reference, mapping, parameter shape, and prepared state (up to global phase)
before queueing and again in the worker's private snapshot. Missing, changed,
noisy, or incompatible sources fail explicitly. This transfers parameters; it
does not resume the VQE optimizer or claim orbital compatibility across geometries.
Imported artifacts without a recoverable Studio configuration remain view/export
only. Older VQE artifacts without recorded parameter shape cannot be refined.

Open a finished refinement and choose **Compare with source**, or select it and an
independent VarQITE run through the usual comparison controls. Details show the
source energy change and **Combined VQE + VarQITE compute runtime**, which adds
original source preparation and refinement compute times even on cache hits.
Invocation runtime remains separate. Missing source timings are not estimated.
A lower energy is not an exact-reference error or proof that the extra work was
worthwhile. See the reproducible [H2 refinement case study](../more_docs/qite/varqite.md#refinement-case-study)
for a comparison with independent initialization.

## Scientific outputs

Cards and detail views display real energy, energy history, completed iteration
count, qubits, active electrons (when returned), compute runtime, invocation
runtime, and cache-hit status. The last energy change is explicitly derived from
the last two history samples. VQE and VarQITE report `budget_exhausted`, `tolerance_satisfied`, or
`numerical_failure`, together with actual updates and the stopping diagnostic.
Leave the energy-change tolerance empty for fixed-step execution; otherwise
patience sets the required consecutive small changes. Budget exhaustion is a
completed execution, not a convergence claim. Numerical failures are failed runs;
the last finite iterate is retained where available. ADAPT reports its operator
budget, pool exhaustion, or pool-gradient tolerance separately; non-finite ADAPT
calculations raise an error.
Studio does **not** infer a convergence verdict from completion or energy change.

None of these runners returns exact/reference energies or reference errors, so none are invented. The browser plots the returned
`energies` array, including the initial state at iteration zero. `plot=False`
suppresses interactive Matplotlib output. Existing PNG names do not contain the
full deterministic run signature, so the studio does not associate potentially
overwritable PNG files with a specific run. Its SVG curve always uses that run's
actual energy samples. JSON export contains all results returned by the selected runner, requested and
resolved configuration, available runtime/environment metadata, artifact path, and
IDs. VQE and VarQITE include statevectors and parameter history; ADAPT includes selected
operators, inner energy histories, and scored pool gradients. Legacy ADAPT caches
may lack compute runtime; Studio leaves that measurement absent.

Energy charts label both axes: energy in hartree (Ha) and optimizer iteration, ADAPT outer
iteration, or imaginary-time update (a count). Live ADAPT inner curves identify their outer iteration.
Comparison legends use colors and line styles; iteration zero marks the start
of each plotted sequence.

Select two to four completed runs with **Compare**, then press **Compare selected**
to overlay energy curves and inspect metrics and configuration differences. Curves
are grouped by method because VQE optimizer steps, VarQITE updates, and ADAPT outer iterations have
different meanings. A warning identifies different or incomplete resolved problems;
absolute energies across such problems do not establish algorithm performance.

While a run executes, cards and open detail views show computed energy observations.
Progress bars show the percentage of VQE optimizer steps or VarQITE updates completed, or the current
ADAPT inner optimization's steps. ADAPT percentages restart for each outer iteration
and do not estimate overall completion. Preparation, pool scoring, and cache loading
use an indeterminate bar. Percentages measure iteration budgets, not elapsed time;
finishing optimization may still be followed by result serialization.
VQE and VarQITE report initial energy and computed updates. ADAPT reports inner optimization,
completed outer iterations, selected-operator count, and pool scoring. Observations
are provisional, held in memory, and replaced by the authoritative artifact when
execution completes. Polling may skip intermediate displays on short runs. Cache
hits report reuse rather than replaying optimization.

## Architecture and API

```text
Python registries + run_vqe signature
                 ↓
         /api/catalogue → browser composer
                               ↓
                 POST /api/runs (experiment JSON)
                               ↓
          strict adapter → isolated Python worker → selected Python runner
                               ↓                     ↓
                  submission manifest      existing VQE JSON/cache
                               ↓                     ↓
                  GET /api/runs ← artifact-backed history
                               ↓
                    cards → detail → restore composer
```

- `studio/adapter.py`: catalogue and validation; dispatch to `run_vqe()` or `run_adapt_vqe()`.
  Defaults come from its signature, options from the ansatz/optimizer/molecule
  registries, and mapping choices follow `common.encoding`. Unknown fields,
  unsupported methods, wrong types, invalid seeds and non-finite numbers fail
  before submission. The UI schema uses `ansatz`/`optimizer`, mapped centrally to
  `ansatz_name`/`optimizer_name`.
- `studio/jobs.py`: `submitted → running → completed/failed`, with cancellation
  from submitted or running. A single dispatcher starts one isolated worker process
  at a time, preserving deterministic seeds and serial execution. At most eight
  runs may be running/queued. UI changes cannot mutate a submitted run.
- `studio/worker.py`: invokes the existing runner in a private working/data
  directory with copies of existing VQE cache JSON files. Progress streams back
  to the server. Only successful, validated scientific artifacts are atomically
  published under the same lock as the completed lifecycle transition. Cancelling
  first prevents publication; completing first leaves the result intact. Existing
  cache files are never writable by the worker. Temporary files and chemistry
  scratch output are removed after cleanup. Copying the cache and starting Python
  add overhead, especially for large histories; history indexing remains planned.
- `studio/history.py`: reads authoritative `results/vqe/*.json`. It checks the
  VQE/ADAPT filenames and current package signature to exclude other methods
  sharing that directory and obsolete cache schemas. It links returned results
  back to the existing artifact without duplicating problem resolution or
  constructing another scientific cache key. A content digest detects an artifact
  replaced externally after a submission.
- `studio/server.py`: same-origin static assets and JSON endpoints. The browser
  polls every two seconds; numerical optimization is not exposed to JavaScript.
- `studio/static/model.mjs`: UI configuration, restoration, and presentation of
  result metrics. `comparison.mjs` compares resolved configurations and `charts.mjs` plots actual energy samples. `app.js` renders controls from catalogue fields using safe DOM
  text nodes; `style.css` supplies the responsive dark interface.

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/catalogue` | Versioned catalogue, defaults, submission token |
| `POST /api/runs` | Validate experiment; return submission with HTTP 202 |
| `POST /api/runs/{id}/cancel` | Cancel a submission; return current manifest with HTTP 200 (404 if unknown) |
| `GET /api/runs` | Status/history with compact results |
| `GET /api/runs/{id}` | Full run detail and JSON export source |

Both POST endpoints require the current `X-Studio-Token` and an allowed Host.
Cancellation accepts no configuration and does not change scientific cache identity.

Experiment schema version 1 (VQE):

```json
{
  "method": "vqe",
  "problem": {"molecule": "H2", "basis": "sto-3g", "mapping": "jordan_wigner"},
  "settings": {"ansatz": "UCCSD", "optimizer": "Adam", "steps": 75, "stepsize": null, "seed": 0}
}
```

ADAPT uses `"method": "adapt_vqe"` and settings `pool`, `max_ops`, `grad_tol`,
`optimizer`, `inner_steps`, `inner_stepsize`, and `seed`. Its pool choices are
`uccsd`, `uccs`, and `uccd`. Basis remains registry-controlled; it is descriptive
metadata rather than an argument to the ADAPT API. Zero `max_ops` evaluates the
Hartree–Fock reference. `inner_stepsize` must be positive and cannot be automatic.

Omitted fields use Python-derived defaults. `basis` must equal the selected
registry molecule's basis. Zero steps is allowed for an initial-energy run.
Step size must be positive or null; seed is an integer in NumPy's seed range.

## Persistence and reproducibility

The normal `VQE_PENNYLANE_DATA_DIR` / user-data-root rules apply. The commands
above choose repository-local output (ignored by Git):

- `results/vqe/*.json`: authoritative scientific configuration and results;
  unchanged package cache semantics and schema-versioned deterministic hashes.
- `results/studio/<submission-id>.json`: small operational manifests containing
  the submitted configuration, timestamps/status/errors, artifact reference and
  digest, and invocation runtime/cache-hit metadata. No scientific results are
  duplicated here. Removing manifests loses submission provenance, but existing
  scientific artifacts remain discoverable.

The random submission ID identifies one invocation; `config_id` is the shared
hash helper applied to the requested studio configuration; `signature` is the
package's scientific cache identifier, based on the resolved problem and settings.
These IDs have different meanings. Automatic and explicit step sizes can have
different requested IDs but the same scientific cache signature.

Current-schema Python/CLI artifacts are also visible, with modification time
explicitly labeled rather than a fabricated execution timestamp. They support
view/export; recreation is disabled because an original studio request was not
recorded and the artifact may use inputs outside the MVP. Missing, changed, or
corrupt referenced artifacts are reported instead of serving stale copied results.
The studio does not lock out independent CLI processes; avoid force-overwriting
artifacts while inspecting a studio session.

## Reference and extension points

The architecture was informed by direct inspection of
[OpenHiggsfield](https://github.com/wide-trace/open-higgsfield/tree/b16a0efe4d7e2707b56f8ccb02387fd2a9d2eddf):
`generation/catalog/{types,index,parse-settings}.ts`, `to-platform.ts`, `poll.ts`,
`openhiggsfield/history.ts`, and composer/gallery/viewer integration. The adopted
ideas are declared controls, a separate execution adapter, explicit lifecycle,
and configuration restoration. Its generation providers, branding, browser-only
history storage, and imagery are not used.

To add an algorithm, declare its method catalogue and validator, implement a
small adapter calling its existing Python entrypoint, and add dispatch by method
ID at the adapter boundary. Extend artifact discovery for its actual schema and
project returned metrics explicitly. Keep the generic composer and lifecycle;
add method-specific result views only where needed. Do not label dynamics methods
such as VarQRTE as energy minimizers.

## Release scope and next steps

Introduced in v0.3.28, the Studio is available only from a source checkout;
`pip install vqe-pennylane` installs the scientific package without the Studio.
Queued/running cancellation, isolated worker processes, solver termination
reporting, VarQITE, and compatible VQE → VarQITE refinement are included.
Sector-aware references, state diagnostics, explicit geometry controls, and further
algorithm adapters remain planned. The [engineering roadmap](../ROADMAP.md)
tracks completion criteria and the remaining work.
Python/NumPy compatibility is tracked as a separate package milestone.

## Python progress observers

All three runners accept `progress_callback=observer`. The synchronous observer receives
fresh dictionaries containing a `phase` and computed `energy`. VQE and VarQITE update
events include `iteration` and `total_iterations`; ADAPT inner events also identify
`outer_iteration` and `selected_operators`. ADAPT additionally emits
`outer_completed` and `pool_scored` events. A cache hit emits only `cache_hit`.
Callbacks are excluded from scientific cache identity, and callback exceptions
propagate to the caller. Their execution time is included in runtime measurements.

## Checks

No frontend build step or TypeScript compiler is required. For development:

```bash
python -m pip --python .venv-studio/bin/python install -e '.[dev]'
.venv-studio/bin/python -m pytest -q
.venv-studio/bin/ruff check studio tests/test_studio.py tests/test_progress_callbacks.py
.venv-studio/bin/black --check studio tests/test_studio.py tests/test_progress_callbacks.py
node --check studio/static/app.js
node studio/tests/model.test.mjs
```

Backend tests cover mapping, defaults, invalid requests, JSON determinism, real H2
cache reuse, lifecycle, cancellation/completion races, shutdown, abrupt parent exit,
restart, exclusive directory ownership, startup failures, missing artifacts, and HTTP boundaries. Node's
built-in test runner checks catalogue-derived defaults, restoration, input parsing,
ADAPT metrics, and comparison semantics without adding frontend dependencies. HTTP tests need
permission to bind a loopback port. Slow scientific integration tests remain
available through the repository's existing `pytest -q -o addopts=''` command.

An optional real-browser smoke check covers a real H2 calculation, catalogue
controls, detail/export, re-run restoration, cache reuse, ADAPT execution/restoration, comparisons, live observations, queued/running cancellation, refresh,
and mobile layout. The Studio CI job runs this check alongside the Node tests. Its browser dependencies are development-only:

```bash
python -m pip --python .venv-studio/bin/python install playwright
.venv-studio/bin/python -m playwright install --with-deps chromium
.venv-studio/bin/python -m studio.tests.browser_smoke
```

The check writes screenshots to `/tmp/vqe-studio-desktop.png` and
`/tmp/vqe-studio-mobile.png` and uses a temporary scientific data directory.
