from __future__ import annotations

import sys
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/check_docs_html.py docs/_build/html")
        return 2

    build_dir = Path(sys.argv[1])
    if not build_dir.is_dir():
        print(f"docs build directory does not exist: {build_dir}")
        return 2

    failures: list[str] = []

    html_files = sorted(build_dir.rglob("*.html"))
    if not html_files:
        failures.append("no generated HTML files found")

    for path in html_files:
        text = _read(path)
        if "$$" in text:
            failures.append(f"raw dollar math marker found in {path}")
        if "Table of Contents" in text:
            failures.append(f"manual table of contents heading found in {path}")
        if "research.html#id1" in text:
            failures.append(f"duplicate Research Use anchor found in {path}")

    theory_page = build_dir / "user" / "theory.html"
    if not theory_page.exists():
        failures.append("missing generated theory overview page")
    else:
        theory_html = _read(theory_page)
        if "math-wrapper" not in theory_html:
            failures.append("theory overview does not contain rendered math blocks")
        if "MathJax" not in theory_html:
            failures.append("theory overview does not load MathJax")

    landing_page = build_dir / "index.html"
    if landing_page.exists():
        landing_html = _read(landing_page)
        if 'class="portfolio-page"' not in landing_html:
            failures.append("landing page is missing portfolio-page markup")

    if failures:
        for failure in failures:
            print(f"docs check failed: {failure}")
        return 1

    print(f"docs HTML checks passed for {build_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
