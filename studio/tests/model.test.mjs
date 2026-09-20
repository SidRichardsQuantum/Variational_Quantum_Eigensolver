import test from "node:test";
import assert from "node:assert/strict";
import {
  defaults,
  restore,
  inputValue,
  metrics,
  syncRegistry,
  comparisonData,
  runProgress,
  fieldValue,
} from "../static/model.mjs";
const method = {
  id: "vqe",
  fields: [
    { group: "problem", name: "molecule", default: "H2" },
    {
      group: "problem",
      name: "basis",
      type: "registry",
      values_by_molecule: { H2: "sto-3g", X: "other" },
    },
    { group: "settings", name: "steps", default: 75 },
    { group: "settings", name: "stepsize", default: null },
  ],
};
test("progress measures iteration budgets and distinguishes preparation and completion", () => {
  const row = { status: "running", method: "vqe" };
  assert.equal(runProgress(row).percent, null);
  row.progress = { phase: "optimization", iteration: 1, total_iterations: 2 };
  assert.equal(runProgress(row).percent, 50);
  row.progress.iteration = 0;
  assert.equal(runProgress(row).percent, 0);
  row.progress.total_iterations = 0;
  assert.equal(runProgress(row).percent, null);
  row.status = "completed";
  assert.equal(runProgress(row).percent, 100);
  row.status = "failed";
  assert.equal(runProgress(row), null);
  row.status = "cancelled";
  assert.equal(runProgress(row), null);
  delete row.progress;
  assert.equal(runProgress(row), null);
});
test("ADAPT progress is scoped to inner optimization, never overall completion", () => {
  const row = { status: "running", method: "adapt_vqe", progress: {
    phase: "inner_optimization", outer_iteration: 2, iteration: 3, total_iterations: 4,
  } };
  assert.equal(runProgress(row).percent, 75);
  assert.match(runProgress(row).label, /Inner optimization \(outer 2\)/);
  row.progress.phase = "pool_scored";
  assert.equal(runProgress(row).percent, null);
  row.progress.phase = "cache_hit";
  assert.equal(runProgress(row).percent, null);
  row.status = "submitted";
  assert.equal(runProgress(row).label, "Queued");
});
test("catalogue drives fields and molecule-dependent basis", () => {
  const config = defaults(method);
  assert.deepEqual(config, {
    method: "vqe",
    problem: { molecule: "H2", basis: "sto-3g" },
    settings: { steps: 75, stepsize: null },
  });
  config.problem.molecule = "X";
  syncRegistry(method, config);
  assert.equal(config.problem.basis, "other");
});
test("re-run restoration is independent and preserves automatic stepsize", () => {
  const original = defaults(method),
    copy = restore(original);
  copy.settings.steps = 3;
  assert.equal(original.settings.steps, 75);
  assert.equal(copy.settings.stepsize, null);
});
test("automatic step size follows optimizer while explicit edits are preserved", () => {
  const field = { group: "settings", name: "stepsize" };
  const catalogue = { optimizer_stepsizes: { Adam: 0.15, RMSProp: 0.01 } };
  const config = { settings: { optimizer: "Adam", stepsize: null } };
  assert.equal(fieldValue(field, config, catalogue), 0.15);
  config.settings.optimizer = "RMSProp";
  assert.equal(fieldValue(field, config, catalogue), 0.01);
  config.settings.stepsize = 0.3;
  config.settings.optimizer = "Adam";
  assert.equal(fieldValue(field, config, catalogue), 0.3);
  assert.equal(config.settings.stepsize, 0.3);
});
test("method change preserves supported problem choices and resets solver settings", () => {
  const next = { id: "adapt_vqe", fields: [
    { group: "problem", name: "molecule", type: "enum", values: ["H2", "X"], default: "H2" },
    method.fields[1],
    { group: "settings", name: "max_ops", default: 20 },
  ] };
  const config = defaults(next, { molecule: "X", basis: "stale" });
  assert.deepEqual(config, { method: "adapt_vqe", problem: { molecule: "X", basis: "other" }, settings: { max_ops: 20 } });
});
test("input parsing retains automatic and numeric zero", () => {
  assert.equal(inputValue({ type: "number", nullable: true }, ""), null);
  assert.equal(inputValue({ type: "integer" }, "0"), 0);
});
test("results hide unavailable metrics and distinguish invocation runtime", () => {
  const row = {
    result: {
      energy: -1.1,
      energies: [-1, -1.1],
      runtime_s: 5,
      compute_runtime_s: 5,
    },
    invocation: { runtime_s: 0.1, compute_runtime_s: 5, cache_hit: true },
  };
  const values = Object.fromEntries(metrics(row));
  assert.equal(values["Cache hit"], true);
  assert.equal(values["Invocation runtime (s)"], 0.1);
  assert.equal(values["Completed iterations"], 1);
  assert.ok(!("Qubits" in values));
  assert.deepEqual(metrics({}), []);
});

test("ADAPT metrics use outer iterations and real inner step counts", () => {
  const row = {
    method: "adapt_vqe",
    result: {
      energy: -1.1,
      energies: [-1, -1.1],
      inner_energies: [[-1], [-1, -1.05, -1.1]],
      selected_operators: [{ kind: "double", wires: [0, 1, 2, 3] }],
      max_gradients: [0.2, -1],
    },
  };
  const values = Object.fromEntries(metrics(row));
  assert.equal(values["Completed outer iterations"], 1);
  assert.equal(values["Selected operators"], 1);
  assert.equal(values["Total inner optimizer steps"], 2);
  assert.ok(!("Completed iterations" in values));
  assert.ok(!("Last scored pool gradient" in values));
});

test("comparison reports configuration differences without treating key order as a difference", () => {
  const a = {
    method: "vqe",
    config: { settings: { steps: 2 } },
    resolved_config: {
      molecule: "H2",
      geometry: [[0, 0, 0]],
      basis: "sto-3g",
      optimizer: { name: "Adam", stepsize: 0.15 },
    },
  };
  const b = structuredClone(a);
  b.resolved_config.optimizer = { stepsize: 0.15, name: "Adam" };
  assert.deepEqual(comparisonData([a, b]), {
    differences: [],
    problemMismatch: false,
    methodsDiffer: false,
  });
  b.resolved_config.optimizer.name = "GradientDescent";
  assert.equal(
    comparisonData([a, b]).differences[0].path,
    "resolved.optimizer.name",
  );
  assert.equal(comparisonData([a, b]).problemMismatch, false);
  b.resolved_config.geometry = [[0, 0, 1]];
  assert.equal(comparisonData([a, b]).problemMismatch, true);
  b.method = "adapt_vqe";
  assert.equal(comparisonData([a, b]).methodsDiffer, true);
});

test("comparison flags missing problem metadata and distinguishes absent/null", () => {
  const a = { method: "vqe", config: { settings: { stepsize: null } } },
    b = { method: "vqe", config: { settings: { stepsize: 0.15 } } };
  const diff = comparisonData([a, b]);
  assert.equal(diff.problemMismatch, true);
  assert.deepEqual(diff.differences[0].values, ["null", "0.15"]);
});


test("refinement metrics include preparation cost on cache hits and omit missing costs", () => {
  const row = {method: "varqite", result: {energy: -1.1,
    initialization: {source: "supplied", provenance: {artifact: "source.json", energy: -1, compute_runtime_s: 4}}},
    invocation: {compute_runtime_s: 3, runtime_s: 0.1, cache_hit: true}};
  let values = Object.fromEntries(metrics(row));
  assert.equal(values["Combined VQE + VarQITE compute runtime (s)"], 7);
  assert.ok(Math.abs(values["Energy change from VQE source (Ha)"] + 0.1) < 1e-12);
  delete row.result.initialization.provenance.compute_runtime_s;
  values = Object.fromEntries(metrics(row));
  assert.ok(!("Combined VQE + VarQITE compute runtime (s)" in values));
});
