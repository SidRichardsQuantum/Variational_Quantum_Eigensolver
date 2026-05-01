# Quantum Simulation Suite

```{raw} html
<div class="portfolio-page" id="top">
  <header class="site-header">
    <a class="brand" href="#top" aria-label="Quantum Simulation Suite home">
      <span class="brand-mark">QSS</span>
      <span>Quantum Simulation Suite</span>
    </a>
    <nav class="nav-links" aria-label="Primary navigation">
      <a href="#methods">Methods</a>
      <a href="#package">Package</a>
      <a href="#notebooks">Notebooks</a>
      <a href="#docs">Docs</a>
    </nav>
  </header>

  <main>
    <section class="hero section">
      <div class="hero-copy">
        <p class="eyebrow">PennyLane quantum chemistry and simulation</p>
        <h1>Quantum Simulation Suite</h1>
        <p class="hero-text">
          A modular research toolkit for small-molecule variational algorithms,
          phase estimation, imaginary-time evolution, real-time dynamics, and
          reproducible benchmark workflows.
        </p>
        <div class="badges" aria-label="Project badges">
          <a href="https://pypi.org/project/vqe-pennylane/">
            <img src="https://img.shields.io/pypi/v/vqe-pennylane?style=flat-square" alt="PyPI version">
          </a>
          <a href="https://pypi.org/project/vqe-pennylane/">
            <img src="https://img.shields.io/pypi/pyversions/vqe-pennylane?style=flat-square" alt="Python versions">
          </a>
          <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/actions/workflows/tests.yml">
            <img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Variational_Quantum_Eigensolver/tests.yml?label=tests&style=flat-square" alt="Tests">
          </a>
          <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/LICENSE">
            <img src="https://img.shields.io/github/license/SidRichardsQuantum/Variational_Quantum_Eigensolver?style=flat-square" alt="License">
          </a>
        </div>
        <div class="hero-actions" aria-label="Project links">
          <a class="button primary" href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver">GitHub</a>
          <a class="button" href="https://pypi.org/project/vqe-pennylane/">PyPI</a>
          <a class="button" href="#docs">Documentation</a>
        </div>
      </div>

      <aside class="focus-panel" aria-label="Project focus">
        <h2>Project Scope</h2>
        <ul>
          <li>Ground-state VQE, ADAPT-VQE, and ansatz comparisons</li>
          <li>Excited-state solvers including QSE, EOM, LR-VQE, SSVQE, and VQD</li>
          <li>QPE, VarQITE, VarQRTE, noise studies, and calibration notebooks</li>
          <li>Shared chemistry, Hamiltonian, caching, plotting, and benchmark tooling</li>
        </ul>
      </aside>
    </section>

    <section id="methods" class="section">
      <div class="section-heading">
        <p class="eyebrow">Research workflows</p>
        <h2>One shared pipeline for quantum simulation methods</h2>
        <p>
          The repository keeps VQE, QPE, and QITE workflows on a common problem
          resolution layer so small-system studies can be compared with consistent
          Hamiltonians, run signatures, outputs, and exact references.
        </p>
      </div>

      <div class="project-grid">
        <article class="project-card">
          <div>
            <h3>Ground-State VQE</h3>
            <p>
              Run molecular VQE studies with calibrated defaults, multiple ansatzes,
              optimizer choices, geometry scans, and low-qubit benchmark helpers.
            </p>
          </div>
          <div class="tags">
            <span>VQE</span>
            <span>ADAPT-VQE</span>
            <span>UCCSD</span>
            <span>PennyLane</span>
          </div>
          <div class="card-links">
            <a href="vqe/defaults.html">Defaults</a>
            <a href="vqe/ansatzes.html">Ansatzes</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>Excited-State Methods</h3>
            <p>
              Explore post-VQE and direct variational excited-state workflows for
              compact molecule studies and reference-state diagnostics.
            </p>
          </div>
          <div class="tags">
            <span>QSE</span>
            <span>EOM-QSE</span>
            <span>LR-VQE</span>
            <span>SSVQE</span>
            <span>VQD</span>
          </div>
          <div class="card-links">
            <a href="vqe/excited_states.html">Guide</a>
            <a href="user/notebooks.html">Notebooks</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>Quantum Phase Estimation</h3>
            <p>
              Compare spectral and phase-estimation workflows with controlled
              time-evolution settings, ancilla studies, shots, and noise options.
            </p>
          </div>
          <div class="tags">
            <span>QPE</span>
            <span>Phase Estimation</span>
            <span>Time Evolution</span>
            <span>Noise</span>
          </div>
          <div class="card-links">
            <a href="qpe/phase_estimation.html">Phase Estimation</a>
            <a href="qpe/time_evolution.html">Evolution</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>QITE and QRTE</h3>
            <p>
              Use projected variational imaginary-time relaxation and real-time
              dynamics with shared Hamiltonian inputs and comparable output records.
            </p>
          </div>
          <div class="tags">
            <span>VarQITE</span>
            <span>VarQRTE</span>
            <span>Dynamics</span>
            <span>State Preparation</span>
          </div>
          <div class="card-links">
            <a href="qite/varqite.html">VarQITE</a>
            <a href="qite/varqrte.html">VarQRTE</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>Shared Chemistry Layer</h3>
            <p>
              Resolve registry molecules, generated geometry tags, explicit
              geometries, and expert-mode qubit Hamiltonians through common helpers.
            </p>
          </div>
          <div class="tags">
            <span>Hamiltonians</span>
            <span>Molecules</span>
            <span>Geometry</span>
            <span>Caching</span>
          </div>
          <div class="card-links">
            <a href="common/molecule_registry.html">Registry</a>
            <a href="common/problem_resolution.html">Resolution</a>
          </div>
        </article>

        <article class="project-card">
          <div>
            <h3>Benchmark Evidence</h3>
            <p>
              Maintain notebooks for cross-method comparisons, default calibration,
              noise studies, non-molecule Hamiltonians, and reproducibility checks.
            </p>
          </div>
          <div class="tags">
            <span>Benchmarks</span>
            <span>Notebooks</span>
            <span>Reproducibility</span>
            <span>Exact References</span>
          </div>
          <div class="card-links">
            <a href="benchmarks/summary.html">Summary</a>
            <a href="benchmarks/results.html">Results</a>
          </div>
        </article>
      </div>
    </section>

    <section id="package" class="section">
      <div class="section-heading">
        <p class="eyebrow">Published package</p>
        <h2>Installable Python tooling</h2>
        <p>
          The PyPI package exposes four importable stacks:
          <code>vqe</code>, <code>qpe</code>, <code>qite</code>, and <code>common</code>.
        </p>
      </div>

      <div class="package-list">
        <article class="package-row">
          <div>
            <h3>vqe-pennylane</h3>
            <p>Quantum Simulation Suite with VQE, QPE, and QITE modules.</p>
          </div>
          <code>pip install vqe-pennylane</code>
          <a href="https://pypi.org/project/vqe-pennylane/">PyPI</a>
        </article>

        <article class="package-row">
          <div>
            <h3>CLI entrypoints</h3>
            <p>Run solver workflows directly from the terminal.</p>
          </div>
          <code>python -m vqe -m H2</code>
          <a href="user/usage.html">Usage</a>
        </article>

        <article class="package-row">
          <div>
            <h3>Python APIs</h3>
            <p>Import high-level runners for notebooks, scripts, and tests.</p>
          </div>
          <code>from vqe import run_vqe</code>
          <a href="user/readme.html#quickstart">Quickstart</a>
        </article>
      </div>
    </section>

    <section id="notebooks" class="section split-section">
      <div class="section-heading">
        <p class="eyebrow">Notebook library</p>
        <h2>Examples and benchmark studies</h2>
      </div>
      <div class="about-copy">
        <p>
          The notebook tree separates getting-started examples from benchmark
          notebooks. Start with the H2 comparison, QITE, and QRTE examples before
          moving into calibration or cross-method studies.
        </p>
        <div class="link-stack">
          <a href="user/notebooks.html">Notebook guide</a>
          <a href="benchmarks/summary.html">Benchmark summary</a>
          <a href="benchmarks/results.html">Benchmark results</a>
          <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/tree/main/notebooks">Notebook source tree</a>
        </div>
      </div>
    </section>

    <section id="docs" class="section contact-section">
      <div>
        <p class="eyebrow">Documentation and source</p>
        <h2>Read the full project materials</h2>
        <p>
          This page is generated by the repository's Sphinx Pages workflow.
          The deeper project documentation remains available as generated HTML,
          Markdown source, and notebooks.
        </p>
      </div>
      <div class="contact-actions">
        <a class="button primary" href="user/readme.html">Overview</a>
        <a class="button" href="user/usage.html">Usage</a>
        <a class="button" href="user/theory.html">Theory</a>
        <a class="button" href="research.html">Research</a>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <span>&copy; 2026 Sid Richards</span>
    <a href="https://sidrichardsquantum.github.io/">Main portfolio</a>
    <a href="#top">Back to top</a>
  </footer>
</div>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: User Guide

Overview <user/readme>
Usage <user/usage>
Theory <user/theory>
Notebooks <user/notebooks>
Research Use <research>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Architecture

architecture
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Benchmarks

Benchmark Summary <benchmarks/summary>
Benchmark Results <benchmarks/results>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Common

Molecule Registry <common/molecule_registry>
Expert Mode <common/expert_mode>
Problem Resolution <common/problem_resolution>
Caching And Artifacts <common/caching_and_artifacts>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: VQE

Defaults <vqe/defaults>
Ansatzes <vqe/ansatzes>
Optimizers <vqe/optimizers>
Mappings <vqe/mappings>
Noise <vqe/noise>
Excited States <vqe/excited_states>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: QPE

Phase Estimation <qpe/phase_estimation>
Time Evolution <qpe/time_evolution>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: QITE

VarQITE <qite/varqite>
VarQRTE <qite/varqrte>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Reference

API Reference <api>
```
