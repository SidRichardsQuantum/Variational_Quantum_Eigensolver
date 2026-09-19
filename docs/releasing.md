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

## Optional enforcement of GitHub checks

The ready-to-apply ruleset is
[main.json](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/.github/rulesets/main.json).
It requires the six checks above on `main`, with an up-to-date branch and no
bypass actors. It does not require the intentionally advisory NumPy 2 probe.

This ruleset is optional repository protection, not a runtime or publishing
requirement. Until activated, verify the checks manually before merging.

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
