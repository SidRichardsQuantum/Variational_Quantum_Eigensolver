// UI-only configuration state; the backend validates and executes all experiments.
export function defaults(method, problem = {}) {
  const config = { method: method.id, problem: {}, settings: {} };
  for (const f of method.fields) config[f.group][f.name] = f.default;
  for (const f of method.fields) {
    if (f.group === "problem" && f.type === "enum" && f.values?.includes(problem[f.name]))
      config.problem[f.name] = problem[f.name];
  }
  return syncRegistry(method, config);
}
export function fieldValue(field, config, catalogue) {
  const value = config[field.group][field.name];
  if (field.name === "stepsize" && value === null)
    return catalogue.optimizer_stepsizes[config.settings.optimizer];
  return value ?? "";
}
export function syncRegistry(method, config) {
  for (const f of method.fields) {
    if (f.type === "registry")
      config[f.group][f.name] = f.values_by_molecule[config.problem.molecule];
  }
  return config;
}
export function restore(config) {
  return structuredClone(config);
}
export function inputValue(field, text) {
  if (field.nullable && text === "") return null;
  return ["number", "integer"].includes(field.type) ? Number(text) : text;
}
export function runProgress(row) {
  if (row.status === "completed") return { percent: 100, label: "Completed" };
  if (row.status === "failed") return null;
  if (row.status === "submitted") return { percent: null, label: "Queued" };
  const p = row.progress;
  if (!p) return { percent: null, label: "Preparing experiment" };
  if (p.phase === "cache_hit")
    return { percent: null, label: "Loading cached result" };
  if (
    ["optimization", "inner_optimization"].includes(p.phase) &&
    Number.isInteger(p.iteration) &&
    Number.isInteger(p.total_iterations) &&
    p.total_iterations > 0
  ) {
    const percent = Math.min(
      100,
      Math.max(0, Math.floor((100 * p.iteration) / p.total_iterations)),
    );
    const scope = row.method === "adapt_vqe"
      ? `Inner optimization (outer ${p.outer_iteration})`
      : "Optimization";
    return { percent, label: `${scope} · ${p.iteration} / ${p.total_iterations} steps` };
  }
  return { percent: null, label: row.method === "adapt_vqe"
    ? "Evaluating operator pool / preparing next iteration"
    : "Finalizing result" };
}
export function metrics(row) {
  const result = row.result ?? {};
  const runtime = row.invocation ?? result;
  const values = [
    ["Final energy (Ha)", result.energy],
    ["Qubits", result.num_qubits],
    ["Active electrons", result.active_electrons],
    ["Compute runtime (s)", runtime.compute_runtime_s],
    ["Invocation runtime (s)", runtime.runtime_s],
    ["Cache hit", runtime.cache_hit],
  ];
  if (Array.isArray(result.energies) && result.energies.length) {
    values.push([
      row.method === "adapt_vqe"
        ? "Completed outer iterations"
        : "Completed iterations",
      result.energies.length - 1,
    ]);
    if (result.energies.length > 1)
      values.push([
        "Last energy change (Ha)",
        result.energies.at(-1) - result.energies.at(-2),
      ]);
  }
  if (row.method === "adapt_vqe" && Array.isArray(result.selected_operators)) {
    values.push(["Selected operators", result.selected_operators.length]);
    if (Array.isArray(result.inner_energies))
      values.push([
        "Total inner optimizer steps",
        result.inner_energies.reduce(
          (n, values) => n + Math.max(0, values.length - 1),
          0,
        ),
      ]);
    const gradient = result.max_gradients?.at(-1);
    if (Number.isFinite(gradient) && gradient >= 0)
      values.push(["Last scored pool gradient", gradient]);
  }
  return values.filter(([, value]) => value !== null && value !== undefined);
}

export function methodLabel(row) {
  return row.method === "adapt_vqe" ? "ADAPT-VQE" : "VQE";
}
export function curveAxis(row) {
  return row.method === "adapt_vqe"
    ? "ADAPT outer iteration"
    : "optimizer iteration";
}
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((k) => [k, canonical(value[k])]),
    );
  return value;
}
function flatten(value, prefix = "", out = {}) {
  for (const [key, item] of Object.entries(value ?? {})) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (
      item &&
      typeof item === "object" &&
      !Array.isArray(item) &&
      Object.keys(item).length
    )
      flatten(item, path, out);
    else out[path] = JSON.stringify(canonical(item));
  }
  return out;
}
export function comparisonData(rows) {
  const configs = rows.map((row) =>
    flatten({
      method: row.method,
      requested: row.config,
      resolved: row.resolved_config,
    }),
  );
  const keys = [...new Set(configs.flatMap(Object.keys))].sort();
  const differences = keys
    .map((path) => ({
      path,
      values: configs.map((config) => config[path] ?? "— (not recorded)"),
    }))
    .filter(({ values }) => new Set(values).size > 1);
  const problemKeys = [
    "molecule",
    "symbols",
    "geometry",
    "basis",
    "charge",
    "multiplicity",
    "unit",
    "mapping",
    "active_electrons",
    "active_orbitals",
    "hamiltonian",
    "reference_state",
    "num_qubits",
  ];
  const problems = rows.map((row) =>
    JSON.stringify(
      canonical(
        Object.fromEntries(
          problemKeys
            .filter((k) => k in (row.resolved_config ?? {}))
            .map((k) => [k, row.resolved_config[k]]),
        ),
      ),
    ),
  );
  return {
    differences,
    problemMismatch:
      rows.some((row) => !row.resolved_config) || new Set(problems).size > 1,
    methodsDiffer: new Set(rows.map((row) => row.method)).size > 1,
  };
}
