# Release checks

The scientific package is published by `.github/workflows/publish.yml` when a
`v*` tag is pushed. The Studio is distributed through the source checkout only.

Before creating a release tag:

1. Update `pyproject.toml`, `CHANGELOG.md`, and the relevant usage/roadmap notes.
2. Run the fast tests, formatter, lint, type checks, frontend tests, browser smoke,
   and documentation build. Review the full integration job before publishing.
3. Build the wheel and sdist, run `twine check`, and run
   `python scripts/check_sdist_links.py dist/*.tar.gz`. Package CI also installs
   the built wheel outside the checkout and checks cache writes and CLI imports.
4. Merge only after the Python 3.10/3.11/3.12 fast tests, `lint`, `package`, and
   `studio` checks pass for the release change. Tag the validated commit and
   verify the publish workflow and resulting PyPI version.

## Automated release gate

There are no scheduled workflows. Pushes and pull requests run the normal
checks; manual runs remain available for tests, Studio, Pages, and publishing.
The advisory NumPy 2 probe runs only when the tests workflow is started manually.

Pushing a tag runs the same lint, Python 3.10/3.11/3.12 tests, full integration
tests, package validation, Studio browser checks, and docs build used by CI.
Publishing waits for all of them. The tag must exactly match
`v<project.version>` and its commit must be reachable from `main`.
Manual publishing must also select a matching tag, not a branch.

The package job uploads the wheel and sdist only after metadata, contents,
documentation links, and installed-wheel smoke checks pass. PyPI receives
these exact artifacts without rebuilding. Only the publishing job receives
the OIDC permission used by the existing PyPI trusted publisher.

## Terminal release sequence

Run this from the repository root with Python 3.11 or 3.12. The subshell stops
on any failed command without closing your interactive terminal. Review your
changes first: `git add .` includes all non-ignored changes.

```bash
(
  set -euo pipefail
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

  test "$(git branch --show-current)" = main
  python -m pip install -e ".[dev]" build twine

  # Ruff may change imports; format its result before validating.
  python -m ruff check . --fix
  python -m black .
  python -m ruff check .
  python -m mypy
  python -m pytest -q

  VERSION=$(python -c 'import tomllib, pathlib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')
  if git rev-parse --verify --quiet "refs/tags/v$VERSION" >/dev/null; then
    echo "Tag v$VERSION already exists. Aborting."
    exit 1
  fi

  rm -rf dist build ./*.egg-info
  python -m build
  python -m twine check dist/*
  python scripts/check_sdist_links.py dist/*.tar.gz

  git add .
  git commit -m "Release v$VERSION"
  git tag -a "v$VERSION" -m "Release v$VERSION"
  git push --atomic origin main "refs/tags/v$VERSION"
)
```

The atomic push updates the branch and tag together or neither. If branch
protection requires a pull request, merge the release changes through that
process first, then tag and push the validated merge commit instead.
A failed push leaves the local commit and tag available for inspection.

Tool versions come from `.[dev]`, matching CI. Installing `black[jupyter]` is
unnecessary with the current configuration: Black explicitly excludes notebooks.
Local fast tests exclude `slow` tests. CI runs these fast tests on Python
3.10/3.11/3.12 and the additional `slow` tests on Python 3.12, covering the full
suite without repeating the Python 3.12 fast tests. Both jobs report their
15 slowest tests to make performance regressions visible. The terminal sequence
uses the same single-threaded OpenMP/BLAS settings as CI to avoid thread
oversubscription in these small scientific workloads.

## Optional enforcement of GitHub checks

The ready-to-apply ruleset is
[main.json](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/.github/rulesets/main.json).
It requires the six checks above on `main`, with an up-to-date branch and no
bypass actors. It does not require the intentionally advisory NumPy 2 probe.

This ruleset is optional branch protection. Until activated, verify the checks
manually before merging. The tag-triggered publishing gate runs independently
of whether branch protection is enabled.

Apply it once using a GitHub token with repository Administration write access:

```bash
gh api --method POST \
  repos/SidRichardsQuantum/Variational_Quantum_Eigensolver/rulesets \
  --input .github/rulesets/main.json
```

Inspect existing rulesets before creating another copy. Workflow configuration
alone does not make checks mandatory. During v0.3.28 preparation, GitHub rejected
this update with HTTP 403 (`Resource not accessible by integration`); activation
remains an administrator action until the integration has the required scope.
