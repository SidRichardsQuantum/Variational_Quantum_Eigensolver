# Benchmark Schema

This page defines the preferred fields for benchmark tables and curated
artifact metadata. It is a data contract, not a benchmark inventory. For the
current notebook list, use `SUMMARY.md`; for published tables and figures, use
`RESULTS.md`.

## Benchmark Row Fields

Use these columns when a benchmark produces one row per method, setting, seed,
or configuration.

| Field | Required | Meaning |
| --- | --- | --- |
| `benchmark_id` | recommended | Stable identifier for the benchmark study or notebook family. |
| `question` | recommended | Short research question answered by the row group. |
| `system` | required | Molecule, atom, ion, or model-Hamiltonian label. |
| `system_type` | recommended | `molecule`, `atom`, `ion`, or `model`. |
| `method` | required | Solver or workflow name, such as `VQE`, `QPE`, `VarQITE`, or `VarQRTE`. |
| `mapping` | when applicable | Fermion-to-qubit mapping. |
| `basis` | when applicable | Chemistry basis set. |
| `charge` | when applicable | Molecular or atomic charge. |
| `multiplicity` | when applicable | Spin multiplicity. |
| `active_electrons` | when applicable | Active-space electron count. |
| `active_orbitals` | when applicable | Active-space spatial-orbital count. |
| `num_qubits` | required | Number of qubits used by the resolved problem. |
| `hamiltonian_terms` | recommended | Number of Hamiltonian terms when available. |
| `ansatz` | when applicable | Ansatz or circuit family. |
| `optimizer` | when applicable | Classical optimizer. |
| `steps` | when applicable | Optimization or evolution step count. |
| `stepsize` | when applicable | Optimizer stepsize. |
| `seed` | when applicable | Random seed for the row. |
| `shots` | when applicable | Shot count; blank or `analytic` for analytic mode. |
| `noise_model` | when applicable | Noise model or channel name. |
| `noise_level` | when applicable | Scalar noise parameter when the study uses one. |
| `energy` | recommended | Reported energy or final energy. |
| `exact_energy` | when available | Exact diagonalization or analytical reference energy. |
| `abs_error` | when available | Absolute error against `exact_energy`. |
| `runtime_s` | recommended | Wall time for the call that produced or loaded the result. |
| `compute_runtime_s` | recommended | Compute time from the original fresh run. |
| `cache_hit` | recommended | Whether the row came from cache. |
| `status` | recommended | `ok`, `failed`, `skipped`, or a similarly explicit state. |
| `failure_reason` | when applicable | Short explanation for failed or skipped configurations. |

## Aggregate Fields

Use these fields for multi-seed, multi-shot, or calibration summaries.

| Field | Meaning |
| --- | --- |
| `n_runs` | Number of rows included in the aggregate. |
| `seed_count` | Number of distinct seeds. |
| `mean_abs_error` | Mean absolute error. |
| `std_abs_error` | Standard deviation of absolute error. |
| `max_abs_error` | Worst absolute error. |
| `mean_runtime_s` | Mean wall time. |
| `mean_compute_runtime_s` | Mean fresh-run compute time. |
| `failure_rate` | Fraction of configurations marked failed. |
| `branch_failure_rate` | QPE branch-selection failure rate. |
| `dominant_bin_failure_rate` | QPE dominant-bin failure rate. |
| `score` | Notebook-defined ranking score; document the formula nearby. |

## Artifact Manifest

`scripts/export_benchmark_artifacts.py` writes
`_artifacts/benchmark_manifest.json` with:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest format version. |
| `description` | Human-readable scope note. |
| `figures` | Curated figure entries with title, notebook, and artifact path. |
| `tables` | Curated table entries with title, notebook, CSV path, and Markdown path. |

The manifest describes the published evidence set. It does not include raw
local cache records from `results/` or raw generated plots from `images/`.

## Registered Suite Outputs

The command-line runner writes one directory per suite:

```bash
python -m common.benchmarks run --suite h2-cross-method --out benchmark_runs
```

Each suite directory contains:

| File | Meaning |
| --- | --- |
| `results.json` | Full suite payload with rows and environment metadata. |
| `results.csv` | Flat benchmark rows using the field names above where applicable. |
| `report.md` | Human-readable summary for quick review. |
| `manifest.json` | Artifact pointers, suite metadata, row count, and environment metadata. |

List available suites with:

```bash
python -m common.benchmarks list
```

Compare two suite outputs with:

```bash
python -m common.benchmarks compare \
  --base benchmark_runs_old/h2-cross-method \
  --head benchmark_runs_new/h2-cross-method
```

The comparison command checks row additions/removals, energy drift, absolute
error drift, and compute-runtime regressions. It exits with status 1 when a
threshold is exceeded.
