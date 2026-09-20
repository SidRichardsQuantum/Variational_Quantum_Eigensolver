import { comparisonData, metrics, methodLabel, curveAxis } from "./model.mjs";
import { energyChart } from "./charts.mjs";
import { el, number } from "./ui.mjs";

function table(headers, rows) {
  const wrapper = el("div", undefined, "table-scroll"),
    table = el("table");
  const head = el("thead"),
    titles = el("tr");
  headers.forEach((text) => {
    const cell = el("th", text);
    cell.scope = "col";
    titles.append(cell);
  });
  head.append(titles);
  const body = el("tbody");
  for (const [label, ...values] of rows) {
    const tr = el("tr"),
      th = el("th", label);
    th.scope = "row";
    tr.append(th);
    for (const value of values) tr.append(el("td", value));
    body.append(tr);
  }
  table.append(head, body);
  wrapper.append(table);
  return wrapper;
}
export function renderComparison(rows) {
  const content = el("div"),
    comparison = comparisonData(rows);
  const labels = rows.map(
    (r) =>
      `${r.config?.problem.molecule ?? r.resolved_config?.molecule} / ${methodLabel(r)} / ${r.id.slice(0, 8)}`,
  );
  content.append(el("p", `${rows.length} completed experiments`, "eyebrow"));
  if (comparison.problemMismatch)
    content.append(
      el(
        "p",
        "Different or incomplete resolved problems: absolute energies must not be ranked as algorithm performance. Inspect molecule, geometry, basis, encoding, and active-space differences below.",
        "comparison-warning",
      ),
    );
  if (comparison.methodsDiffer)
    content.append(
      el(
        "p",
        "Iteration counts have different meanings across algorithms. Curves are grouped by method; an ADAPT outer iteration includes inner optimization and pool scoring.",
        "comparison-warning",
      ),
    );
  for (const method of new Set(rows.map((r) => r.method))) {
    const group = rows
      .map((r, i) => ({ row: r, label: labels[i] }))
      .filter(({ row }) => row.method === method);
    content.append(
      el("h3", methodLabel(group[0].row)),
      energyChart(
        group.map(({ row, label }) => ({
          energies: row.result.energies,
          label,
        })),
        curveAxis(group[0].row),
      ),
    );
  }
  const values = rows.map((row) => Object.fromEntries(metrics(row)));
  const metricNames = [...new Set(values.flatMap(Object.keys))];
  content.append(
    el("h3", "Computed outputs & runtime"),
    table(
      ["Metric", ...labels],
      metricNames.map((name) => [
        name,
        ...values.map((value) =>
          value[name] === undefined ? "— (not returned)" : number(value[name]),
        ),
      ]),
    ),
  );
  content.append(el("h3", "Configuration differences"));
  if (!comparison.differences.length)
    content.append(
      el("p", "Requested and resolved configurations are identical.", "muted"),
    );
  else
    content.append(
      table(
        ["Field", ...labels],
        comparison.differences.map(({ path, values }) => [path, ...values]),
      ),
    );
  content.append(
    el(
      "p",
      "Runtime includes solver work and observer overhead, and depends on the execution environment. Combined VQE + VarQITE compute runtime includes the source preparation and refinement, even on cache hits; it is not invocation latency. Energy change is not a reference-energy error or proof that refinement is cost-effective.",
      "muted",
    ),
  );
  return content;
}
