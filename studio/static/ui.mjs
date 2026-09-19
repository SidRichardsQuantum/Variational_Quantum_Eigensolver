export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
export function number(value) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value !== "number") return String(value);
  return Number.isInteger(value) ? String(value) : value.toPrecision(10);
}
export function metricList(values) {
  const dl = el("dl", undefined, "metrics");
  for (const [label, value] of values)
    dl.append(el("dt", label), el("dd", number(value)));
  return dl;
}
