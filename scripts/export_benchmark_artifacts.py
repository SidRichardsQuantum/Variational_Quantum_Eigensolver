from __future__ import annotations

import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "notebooks" / "benchmarks"
ARTIFACTS = BENCHMARKS / "_artifacts"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"
MANIFEST = ARTIFACTS / "benchmark_manifest.json"
RESULTS_MD = BENCHMARKS / "RESULTS.md"

START = "<!-- benchmark-artifacts:start -->"
END = "<!-- benchmark-artifacts:end -->"

FIGURE_SPECS = [
    {
        "title": "H2 Ansatz Comparison",
        "source": ROOT / "images/vqe/H2/ansatz_conv_Adam_s0.png",
        "target": FIGURES / "h2_ansatz_comparison.png",
        "notebook": "vqe/H2/Ansatz_Comparison.ipynb",
    },
    {
        "title": "H2 Mapping Comparison",
        "source": ROOT / "images/vqe/H2/mapping_comparison_UCCSD_Adam.png",
        "target": FIGURES / "h2_mapping_comparison_uccsd.png",
        "notebook": "vqe/H2/Mapping_Comparison.ipynb",
    },
    {
        "title": "Low-Qubit VQE Benchmark",
        "source": ROOT
        / "images/vqe/multi_molecule/low_qubit_benchmark_UCCSD_Adam_jordan_wigner_max10q.png",
        "target": FIGURES / "low_qubit_vqe.png",
        "notebook": "comparisons/multi_molecule/Low_Qubit_VQE_Benchmark.ipynb",
    },
]

TABLE_SPECS = [
    {
        "title": "H2 Cross-Method Runtime",
        "notebook": "comparisons/H2/Cross_Method_Comparison.ipynb",
        "cell": 4,
        "target": "h2_cross_method_runtime",
    },
    {
        "title": "LiH Problem Summary",
        "notebook": "comparisons/LiH/Cross_Method_Comparison.ipynb",
        "cell": 3,
        "target": "lih_problem_summary",
    },
    {
        "title": "LiH Cross-Method Results",
        "notebook": "comparisons/LiH/Cross_Method_Comparison.ipynb",
        "cell": 7,
        "target": "lih_cross_method_results",
    },
    {
        "title": "QPE H2 Best Configurations",
        "notebook": "qpe/H2/Calibration_Decision_Map.ipynb",
        "cell": 5,
        "target": "qpe_h2_best_configurations",
        "rows": 10,
    },
    {
        "title": "QPE H2 Ranked Summary",
        "notebook": "qpe/H2/Calibration_Decision_Map.ipynb",
        "cell": 7,
        "target": "qpe_h2_ranked_summary",
        "rows": 10,
    },
    {
        "title": "Low-Qubit VQE Summary",
        "notebook": "comparisons/multi_molecule/Low_Qubit_VQE_Benchmark.ipynb",
        "cell": 2,
        "target": "low_qubit_vqe_summary",
    },
    {
        "title": "H2 Noise Robustness Reference",
        "notebook": "vqe/H2/Noise_Robustness_Benchmark.ipynb",
        "cell": 7,
        "target": "h2_noise_reference",
    },
]


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and not self.in_table:
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_table and self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and self.in_cell and tag in {"th", "td"}:
            text = " ".join("".join(self.current_cell).split())
            self.current_row.append(text)
            self.current_cell = []
            self.in_cell = False
        elif self.in_table and self.in_row and tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
            self.in_row = False
        elif self.in_table and tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def _ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def _dataframe_from_html_table(html_text: str) -> pd.DataFrame:
    parser = _HTMLTableParser()
    parser.feed(html_text)
    if len(parser.rows) < 2:
        raise RuntimeError("No complete HTML table found")

    headers = parser.rows[0]
    rows = parser.rows[1:]
    width = len(headers)
    rows = [row[:width] + [""] * max(0, width - len(row)) for row in rows]

    if headers and headers[0] == "":
        headers = headers[1:]
        rows = [row[1:] for row in rows]

    return pd.DataFrame(rows, columns=headers)


def _first_html_table(notebook_path: Path, cell_index: int) -> pd.DataFrame:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = notebook["cells"][cell_index]
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if not html:
            continue
        html_text = "".join(html) if isinstance(html, list) else str(html)
        return _dataframe_from_html_table(html_text)
    raise RuntimeError(f"No HTML table found in {notebook_path}:{cell_index}")


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.3e}"
        return f"{value:.6g}"
    text = str(value)
    if text.lower() in {"nan", "none"}:
        return ""
    return text.replace("\n", " ").strip()


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(_format_value(row[col]) for col in df.columns) + " |"
        )
    return "\n".join(lines) + "\n"


def export_figures() -> list[dict[str, str]]:
    exported: list[dict[str, str]] = []
    for spec in FIGURE_SPECS:
        source = spec["source"]
        target = spec["target"]
        if source.exists():
            shutil.copy2(source, target)
        elif not target.exists():
            continue
        exported.append(
            {
                "title": str(spec["title"]),
                "path": target.relative_to(BENCHMARKS).as_posix(),
                "notebook": str(spec["notebook"]),
            }
        )
    return exported


def export_tables() -> list[dict[str, str]]:
    exported: list[dict[str, str]] = []
    for spec in TABLE_SPECS:
        notebook = BENCHMARKS / str(spec["notebook"])
        df = _first_html_table(notebook, int(spec["cell"]))
        rows = spec.get("rows")
        if isinstance(rows, int):
            df = df.head(rows)

        target_stem = str(spec["target"])
        csv_path = TABLES / f"{target_stem}.csv"
        md_path = TABLES / f"{target_stem}.md"
        df.to_csv(csv_path, index=False)
        md_path.write_text(_markdown_table(df), encoding="utf-8")
        exported.append(
            {
                "title": str(spec["title"]),
                "csv": csv_path.relative_to(BENCHMARKS).as_posix(),
                "md": md_path.relative_to(BENCHMARKS).as_posix(),
                "notebook": str(spec["notebook"]),
            }
        )
    return exported


def write_manifest(figures: list[dict[str, str]], tables: list[dict[str, str]]) -> None:
    payload = {
        "schema_version": 1,
        "description": (
            "Curated benchmark artifacts exported from saved benchmark notebook "
            "outputs. Raw results/ and images/ files are local caches, not this "
            "published evidence set."
        ),
        "figures": figures,
        "tables": tables,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generated_section(
    figures: list[dict[str, str]], tables: list[dict[str, str]]
) -> str:
    lines = [
        START,
        "",
        "## Curated Figures",
        "",
    ]
    if figures:
        for figure in figures:
            lines.extend(
                [
                    f"### {figure['title']}",
                    "",
                    f"Source notebook: `{figure['notebook']}`",
                    "",
                    f"![{figure['title']}]({figure['path']})",
                    "",
                ]
            )
    else:
        lines.extend(["No curated figures were exported.", ""])

    lines.extend(["## Curated Tables", ""])
    if tables:
        for table in tables:
            table_text = (BENCHMARKS / table["md"]).read_text(encoding="utf-8").strip()
            lines.extend(
                [
                    f"### {table['title']}",
                    "",
                    f"Source notebook: `{table['notebook']}`",
                    "",
                    f"CSV artifact: `{table['csv']}`",
                    "",
                    table_text,
                    "",
                ]
            )
    else:
        lines.extend(["No curated tables were exported.", ""])

    lines.extend([END, ""])
    return "\n".join(lines)


def update_results_md(
    figures: list[dict[str, str]], tables: list[dict[str, str]]
) -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")
    generated = _generated_section(figures, tables)
    if START not in text or END not in text:
        text = text.rstrip() + "\n\n" + generated
    else:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        text = before + "\n\n" + generated + after
    RESULTS_MD.write_text(text, encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    figures = export_figures()
    tables = export_tables()
    write_manifest(figures, tables)
    update_results_md(figures, tables)
    print(f"Exported {len(figures)} figures and {len(tables)} tables.")


if __name__ == "__main__":
    main()
