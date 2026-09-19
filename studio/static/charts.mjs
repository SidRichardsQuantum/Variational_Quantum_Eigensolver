import { el } from "./ui.mjs";
const COLORS = ["#7bd9c4", "#e9c88a", "#bba3ff", "#ff9da9"];
const NS = "http://www.w3.org/2000/svg";
function svgNode(tag, attrs, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs))
    node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
export function energyChart(series, axis = "optimizer iteration") {
  const fig = el("figure");
  const valid = series.filter(
    (s) =>
      Array.isArray(s.energies) &&
      s.energies.length &&
      s.energies.every(Number.isFinite),
  );
  if (!valid.length) return fig;
  let low = Infinity,
    high = -Infinity,
    last = 0;
  for (const s of valid) {
    for (const energy of s.energies) {
      low = Math.min(low, energy);
      high = Math.max(high, energy);
    }
    last = Math.max(last, s.energies.length - 1);
  }
  if (high === low) {
    const padding = Math.max(Math.abs(low) * 0.001, 0.001);
    low -= padding;
    high += padding;
  }
  const span = high - low;
  const tick = (value) => {
    const decimals = Math.min(12, Math.max(6, Math.ceil(-Math.log10(span)) + 1));
    return Math.abs(value) >= 1e6 || span < 1e-10
      ? value.toExponential(12)
      : value.toFixed(decimals);
  };
  const left = Math.max(120, Math.max(tick(high).length, tick(low).length) * 8 + 20);
  const svg = svgNode("svg", {
    viewBox: "0 0 600 270",
    role: "img",
    "aria-label": `Energy in hartree (Ha) by ${axis}; zero is the start of the plotted sequence`,
  });
  svg.append(
    svgNode("text", { x: 12, y: 22, class: "axis-label" }, "Energy (Ha)"),
    svgNode("text", { x: (left + 580) / 2, y: 253, "text-anchor": "middle", class: "axis-label" }, axis),
    svgNode("path", { d: `M${left} 45 V205 H580`, fill: "none", stroke: "#526273" }),
  );
  valid.forEach((s, index) => {
    const color = COLORS[index % COLORS.length];
    const points = s.energies.map((energy, i) => [
      left + ((580 - left) * i) / Math.max(1, last),
      205 - (160 * (energy - low)) / span,
    ]);
    svg.append(
      svgNode("polyline", {
        points: points.map((p) => p.join(",")).join(" "),
        stroke: color,
        "stroke-dasharray": ["none", "8 4", "3 3", "10 3 2 3"][index % 4],
      }),
    );
    if (points.length === 1)
      svg.append(
        svgNode("circle", {
          cx: points[0][0],
          cy: points[0][1],
          r: 4,
          fill: color,
        }),
      );
  });
  for (const [x, y, text] of [
    [8, 49, tick(high)],
    [8, 205, tick(low)],
    [left, 225, "0"],
    ...(last ? [[570, 225, String(last)]] : []),
  ])
    svg.append(svgNode("text", { x, y }, text));
  fig.append(
    svg,
    el(
      "figcaption",
      "Ha = hartree · 0 = start of plotted sequence",
    ),
  );
  if (valid.length > 1) {
    const legend = el("ul", undefined, "legend");
    valid.forEach((s, index) => {
      const item = el("li");
      const marker = svgNode("svg", {
        viewBox: "0 0 20 10",
        "aria-hidden": "true",
      });
      marker.append(
        svgNode("line", {
          x1: 0,
          y1: 5,
          x2: 20,
          y2: 5,
          stroke: COLORS[index % COLORS.length],
          "stroke-width": 3,
          "stroke-dasharray": ["none", "8 4", "3 3", "10 3 2 3"][index % 4],
        }),
      );
      item.append(marker, el("span", s.label));
      legend.append(item);
    });
    fig.append(legend);
  }
  return fig;
}
