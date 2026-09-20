import {
  defaults,
  syncRegistry,
  restore,
  inputValue,
  metrics,
  methodLabel,
  curveAxis,
  runProgress,
  fieldValue,
} from "./model.mjs";
import { el, number, metricList } from "./ui.mjs";
import { energyChart } from "./charts.mjs";
import { renderComparison } from "./comparison.mjs";
const $ = (id) => document.getElementById(id);
const selected = new Set();
let latestRows = [],
  detailId = null,
  detailStatus = null;
function curve(energies, axis = "optimizer iteration") {
  return energyChart([{ energies }], axis);
}
let catalogue,
  method,
  config,
  lastHistory = "",
  submitting = false;
const controls = new Map();
async function api(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error ?? `HTTP ${response.status}`);
  return value;
}
function renderComposer() {
  controls.clear();
  const initialization = $("initialization");
  initialization.replaceChildren();
  if (config.refinement) {
    initialization.append(
      el("p", `Refining VQE source: ${config.refinement.artifact}`, "muted"),
      button("Start independently", () => {
        delete config.refinement;
        renderComposer();
        $("notice").textContent = "VarQITE will start from its seeded parameters.";
      }),
    );
  }
  $("fields").replaceChildren();
  for (const f of method.fields) {
    const label = el("label", f.label);
    const control = el(f.type === "enum" ? "select" : "input");
    control.name = `${f.group}.${f.name}`;
    if (f.type === "enum") {
      for (const value of f.values) {
        const option = el("option", value);
        option.value = value;
        control.append(option);
      }
    } else {
      control.type = ["integer", "number"].includes(f.type) ? "number" : "text";
      if (f.minimum !== undefined) control.min = f.minimum;
      if (f.maximum !== undefined) control.max = f.maximum;
      control.step = f.type === "integer" ? "1" : "any";
      control.readOnly = f.type === "registry";
      control.required = !f.nullable;
      if (f.nullable) control.placeholder = "Automatic";
    }
    if (config.refinement && (f.group === "problem" || f.name === "ansatz"))
      control.disabled = true;
    control.value = fieldValue(f, config, catalogue);
    const help = el("small", f.help);
    help.id = `help-${f.name}`;
    control.setAttribute("aria-describedby", help.id);
    control.addEventListener("input", () => {
      config[f.group][f.name] = inputValue(f, control.value);
    });
    control.addEventListener("change", () => {
      config[f.group][f.name] = inputValue(f, control.value);
      syncRegistry(method, config);
      refreshDefaults();
    });
    controls.set(f.name, control);
    label.append(control, help);
    $("fields").append(label);
  }
  refreshDefaults();
}
function refreshDefaults() {
  for (const field of method.fields) {
    const control = controls.get(field.name);
    control.value = fieldValue(field, config, catalogue);
    if (field.name === "stepsize") {
      const automatic = config.settings.stepsize === null;
      $(`help-${field.name}`).textContent = automatic
        ? `Automatic ${config.settings.optimizer} default. Edit to override; clear to restore.`
        : "Custom step size. Clear to restore the optimizer default.";
    }
  }
}
function reuse(row) {
  if (!row.config) return;
  config = restore(row.config);
  method = catalogue.methods.find((m) => m.id === config.method);
  $("method").value = method.id;
  renderComposer();
  $("detail").close();
  $("notice").textContent =
    "Configuration restored. Press Run experiment to submit; matching runs may use cache.";
  $("composer").scrollIntoView({ behavior: "smooth", block: "start" });
  controls.get("molecule").focus();
}
function button(label, action) {
  const b = el("button", label);
  b.type = "button";
  b.addEventListener("click", action);
  return b;
}
async function cancelRun(row) {
  try {
    await api(`/api/runs/${encodeURIComponent(row.id)}/cancel`, {
      method: "POST",
      headers: { "X-Studio-Token": catalogue.submission_token },
    });
    renderHistory(await api("/api/runs"));
  } catch (error) {
    showError(error);
  }
}
async function view(row, refresh = false) {
  try {
    const full = await api(`/api/runs/${encodeURIComponent(row.id)}`);
    if (refresh && (!$("detail").open || detailId !== row.id)) return;
    detailId = row.id;
    detailStatus = full.status;
    $("detail-title").textContent = "Experiment detail";
    const content = $("detail-content");
    content.replaceChildren();
    content.append(el("p", `${methodLabel(full)} · ${full.status}`, "eyebrow"));
    if (full.error) content.append(el("p", full.error, "failure"));
    if (full.result) {
      content.append(
        metricList(metrics(full)),
        curve(full.result.energies, curveAxis(full)),
      );
      content.append(
        el(
          "p",
          "A satisfied stopping tolerance does not certify ground-state accuracy. Exact/reference energy is not returned by this runner.",
          "muted",
        ),
      );
    }
    const live = el("div");
    live.id = "detail-progress";
    content.append(live);
    if (!full.result) renderProgress(live, full);
    if (full.result?.selected_operators) {
      content.append(
        el("h3", "Selected operators"),
        el("pre", JSON.stringify(full.result.selected_operators, null, 2)),
      );
    }
    const actions = el("div", undefined, "actions");
    if (["submitted", "running"].includes(full.status))
      actions.append(button("Cancel", () => cancelRun(full)));
    if (full.config) actions.append(button("Re-run", () => reuse(full)));
    if (full.config && full.method === "vqe" && full.status === "completed" && full.result?.final_params_shape && full.artifact && full.artifact_digest && !Object.keys(full.resolved_config?.noise ?? {}).length)
      actions.append(button("Refine with VarQITE", () => {
        method = catalogue.methods.find((m) => m.id === "varqite");
        config = defaults(method, full.config.problem);
        config.settings.ansatz = full.config.settings.ansatz;
        config.refinement = {artifact: full.artifact.split("/").at(-1), digest: full.artifact_digest};
        $("method").value = method.id;
        renderComposer();
        $("detail").close();
        $("notice").textContent = "VQE parameters selected for refinement. Matching problem and circuit preparation are checked before submission. Change method to start independently.";
        $("composer").scrollIntoView({behavior: "smooth"});
      }));
    const sourceRef = full.result?.initialization?.provenance;
    const source = sourceRef && latestRows.find((r) =>
      r.artifact?.split("/").at(-1) === sourceRef.artifact &&
      r.artifact_digest === sourceRef.digest && r.status === "completed");
    if (source && full.status === "completed")
      actions.append(button("Compare with source", () => {
        detailId = null;
        $("detail-title").textContent = "Experiment comparison";
        content.replaceChildren(renderComparison([source, full]));
      }));
    actions.append(
      button("Export JSON", () => {
        const blob = new Blob([JSON.stringify(full, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob),
          link = el("a");
        link.href = url;
        link.download = `vqe-${full.id}.json`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }),
    );
    content.append(actions);
    content.append(
      metricList([
        ["Run ID", full.id],
        ["Timestamp", full.timestamp],
        ["Timestamp source", full.timestamp_source],
        ...["started_at", "cancelled_at", "finished_at", "signature", "config_id", "artifact"]
          .filter((k) => full[k])
          .map((k) => [k, full[k]]),
      ]),
    );
    for (const [title, value] of [
      ["Requested configuration", full.config],
      ["Resolved scientific configuration", full.resolved_config],
      ["Environment", full.result?.environment],
    ]) {
      if (value)
        content.append(
          el("h3", title),
          el("pre", JSON.stringify(value, null, 2)),
        );
    }
    if (!full.config)
      content.append(
        el(
          "p",
          "Imported Python artifact: view/export supported. No original studio configuration was recorded, so recreation is disabled.",
          "muted",
        ),
      );
    if (!$("detail").open) $("detail").showModal();
  } catch (error) {
    showError(error);
  }
}
function updateCompareButton() {
  $("compare").disabled = selected.size < 2;
  $("compare").textContent = `Compare selected (${selected.size}/4)`;
  $("clear-comparison").disabled = selected.size === 0;
}
function renderProgress(container, row) {
  container.replaceChildren();
  const status = runProgress(row);
  if (status) {
    const label = `${status.label}${status.percent === null ? "…" : ` · ${status.percent}%`}`;
    const wrapper = el("div", undefined, "run-progress");
    const bar = el("progress");
    bar.max = 100;
    if (status.percent !== null) bar.value = status.percent;
    bar.setAttribute("aria-label", label);
    wrapper.append(el("p", label, "muted"), bar);
    container.append(wrapper);
  }
  if (row.result) {
    container.append(
      el(
        "p",
        "Completed · current result available. Reopen View for full detail.",
        "muted",
      ),
    );
    return;
  }
  if (row.error) {
    container.append(el("p", row.error, "failure"));
    return;
  }
  const p = row.progress;
  if (!p) return;
  if (p.phase === "cache_hit") {
    container.append(
      el("p", "Cache hit · loading the existing scientific artifact.", "muted"),
    );
    return;
  }
  container.append(
    el(
      "p",
      "Live computed observations · provisional until completion",
      "eyebrow",
    ),
  );
  const values = [["Current energy (Ha)", p.energy]];
  if (p.outer_iteration !== undefined)
    values.push(
      ["Outer iteration", p.outer_iteration],
      ["Selected operators", p.selected_operators],
    );
  if (p.iteration !== undefined)
    values.push([
      row.method === "adapt_vqe" ? "Inner iteration" : "Iteration",
      `${p.iteration} / ${p.total_iterations}`,
    ]);
  if (p.phase === "pool_scored" && p.max_gradient !== null)
    values.push(["Largest remaining pool gradient", p.max_gradient]);
  container.append(metricList(values));
  if (p.energies?.length)
    container.append(
      curve(
        p.energies,
        row.method === "adapt_vqe"
          ? `inner optimizer iteration (outer ${p.outer_iteration})`
          : curveAxis(row),
      ),
    );
  if (p.outer_energies?.length)
    container.append(curve(p.outer_energies, "ADAPT outer iteration"));
}
function showError(error) {
  $("error").hidden = false;
  $("error").textContent = error.message;
}
function renderHistory(rows) {
  latestRows = rows;
  for (const id of selected)
    if (
      !rows.some(
        (row) => row.id === id && row.status === "completed" && row.result,
      )
    )
      selected.delete(id);
  updateCompareButton();
  if ($("detail").open && detailId) {
    const current = rows.find((row) => row.id === detailId);
    const live = $("detail-progress");
    if (current && live && ["submitted", "running"].includes(detailStatus)) {
      if (["completed", "failed", "cancelled"].includes(current.status)) {
        detailStatus = current.status;
        view(current, true);
      } else renderProgress(live, current);
    }
  }
  const serial = JSON.stringify(rows);
  if (serial === lastHistory) return;
  lastHistory = serial;
  $("history").replaceChildren();
  $("count").textContent = `${rows.length} runs`;
  if (!rows.length)
    $("history").append(
      el(
        "div",
        "No experiments yet. Configure a molecule and run your first VQE experiment.",
        "empty",
      ),
    );
  for (const row of rows) {
    const card = el("article", undefined, "card");
    const c = row.config,
      resolved = row.resolved_config;
    const head = el("div", undefined, "card-head");
    head.append(
      el(
        "h3",
        `${c?.problem.molecule ?? resolved?.molecule ?? "Experiment"} / ${methodLabel(row)}`,
      ),
      el("span", row.status, `pill ${row.status}`),
    );
    card.append(head);
    card.append(
      el(
        "p",
        `${c?.settings.ansatz ?? c?.settings.pool ?? resolved?.ansatz ?? ""} · ${c?.settings.optimizer ?? resolved?.optimizer?.name ?? ""}`,
        "muted",
      ),
    );
    if (row.result) {
      card.append(
        el("div", `${number(row.result.energy)} Ha`, "energy"),
        curve(row.result.energies, curveAxis(row)),
      );
      card.append(
        metricList(
          metrics(row).filter(([label]) =>
            [
              "Termination reason",
              "Actual updates",
              "Compute runtime (s)",
              "Cache hit",
              "Completed iterations",
              "Completed outer iterations",
              "Selected operators",
              "Last energy change (Ha)",
            ].includes(label),
          ),
        ),
      );
    } else
      card.append(
        el(
          "p",
          row.error ??
            (row.status === "cancelled"
              ? "Cancelled. No result was published for this submission."
              : row.status === "submitted"
              ? "Queued for the Python worker."
              : "Python is preparing the problem or computing the next observation."),
          row.error ? "failure" : "muted",
        ),
      );
    if (!row.result && ["submitted", "running"].includes(row.status)) {
      const live = el("div");
      renderProgress(live, row);
      card.append(live);
    }
    card.append(el("p", new Date(row.timestamp).toLocaleString(), "timestamp"));
    const actions = el("div", undefined, "actions");
    actions.append(button("View", () => view(row)));
    if (["submitted", "running"].includes(row.status))
      actions.append(button("Cancel", () => cancelRun(row)));
    if (row.config) actions.append(button("Re-run", () => reuse(row)));
    if (row.status === "completed" && row.result) {
      const label = el("label", undefined, "compare-choice"),
        check = el("input");
      check.type = "checkbox";
      check.checked = selected.has(row.id);
      check.setAttribute(
        "aria-label",
        `Compare ${methodLabel(row)} run ${row.id.slice(0, 8)}`,
      );
      check.addEventListener("change", () => {
        if (check.checked && selected.size >= 4) {
          check.checked = false;
          showError(new Error("Select up to four runs for comparison"));
          return;
        }
        if (check.checked) selected.add(row.id);
        else selected.delete(row.id);
        updateCompareButton();
      });
      label.append(check, el("span", "Compare"));
      actions.append(label);
    }
    card.append(actions);
    $("history").append(card);
  }
}
async function poll() {
  try {
    renderHistory(await api("/api/runs"));
    $("connection").textContent = "Python connected";
  } catch (error) {
    $("connection").textContent = "Connection interrupted · retrying";
    showError(error);
  } finally {
    setTimeout(poll, 2000);
  }
}
$("close").addEventListener("click", () => $("detail").close());
$("detail").addEventListener("close", () => {
  detailId = null;
});
$("compare").addEventListener("click", async () => {
  try {
    const ids = [...selected];
    const rows = await Promise.all(
      ids.map((id) => api(`/api/runs/${encodeURIComponent(id)}`)),
    );
    if (
      rows.length < 2 ||
      rows.some((row) => row.status !== "completed" || !row.result)
    )
      throw new Error(
        "Choose at least two completed runs with available results",
      );
    detailId = null;
    $("detail-title").textContent = "Experiment comparison";
    $("detail-content").replaceChildren(renderComparison(rows));
    $("detail").showModal();
  } catch (error) {
    showError(error);
  }
});
$("clear-comparison").addEventListener("click", () => {
  selected.clear();
  lastHistory = "";
  renderHistory(latestRows);
});
$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitting) return;
  submitting = true;
  $("run").disabled = true;
  $("error").hidden = true;
  try {
    const row = await api("/api/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Studio-Token": catalogue.submission_token,
      },
      body: JSON.stringify(config),
    });
    $("notice").textContent =
      `Submitted ${row.id.slice(0, 8)}. You can configure another experiment.`;
    renderHistory(await api("/api/runs"));
  } catch (error) {
    showError(error);
  } finally {
    submitting = false;
    $("run").disabled = false;
  }
});
try {
  catalogue = await api("/api/catalogue");
  method = catalogue.methods[0];
  config = defaults(method);
  for (const m of catalogue.methods) {
    const o = el("option", m.label);
    o.value = m.id;
    $("method").append(o);
  }
  $("method").addEventListener("change", () => {
    method = catalogue.methods.find((m) => m.id === $("method").value);
    config = defaults(method, config.problem);
    renderComposer();
  });
  renderComposer();
  $("run").disabled = false;
  $("notice").textContent = "Idle · ready to configure";
  poll();
} catch (error) {
  showError(error);
  $("connection").textContent = "Unable to load catalogue · reload to retry";
}
