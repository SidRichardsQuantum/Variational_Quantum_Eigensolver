from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Quantum Simulation Suite"
author = "Sid Richards"
copyright = "2026, Sid Richards"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "benchmarks/_artifacts", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["portfolio.css"]

myst_enable_extensions = [
    "colon_fence",
]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

# The included repository Markdown contains GitHub-oriented relative links
# to notebooks and root files that are valid in GitHub/PyPI but not all
# resolvable as Sphinx document references.
suppress_warnings = [
    "docutils",
    "myst.header",
    "myst.xref_missing",
]
