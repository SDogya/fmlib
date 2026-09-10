# Repository Guidelines

## Project Structure & Module Organization

`fmlib/` is the distributable Python package. Keep domain code close to its
area: neural-network components are in `fmlib/nn/`, training and callbacks in
`fmlib/training/`, pipelines in `fmlib/pipeline/`, and shared helpers in
`fmlib/utils/` and `fmlib/constants/`. Feature-selection code lives in
`fmlib/feature_selection/`. Place unit tests in the nearest `tests/` directory
(for example, `fmlib/nn/transforms/tests/`). Put runnable configurations and
examples in `examples/`; user-facing guides belong in `docs/`.

## Build, Test, and Development Commands

Use Python 3.9 or later and install the editable development environment with
`poetry install` (access to the configured SberOSC package source is required).

- `poetry run pytest .` runs the full unit suite.
- `scripts/run_tests.sh path/to/test.py` runs pytest with CPU-thread settings
  used for faster local tests.
- `poetry run ruff check .` checks imports, style, naming, and common errors.
- `poetry run ruff format .` formats Python files before review.
- `python3 scripts/clean_up_notebooks.py examples/foo.ipynb` removes notebook
  outputs; it first saves an `.unclean.ipynb` backup.

## Coding Style & Naming Conventions

Use four-space indentation, double-quoted strings, and a 128-character line
limit, as configured in `pyproject.toml`. Ruff is the formatter and linter;
run it instead of hand-formatting imports. Follow PEP 8 naming: `snake_case`
for modules, functions, variables, and test files; `PascalCase` for classes;
and `UPPER_CASE` for constants. Add concise docstrings, including arguments and
return values, to public classes and functions. Prefer typed interfaces and
keep reusable model blocks small and composable.

## Testing Guidelines

Pytest discovers tests named `test_*.py`; name test functions `test_<behavior>`
and keep them beside the implementation. Add regression coverage for every bug
fix and focused unit coverage for new behavior. Run the relevant test file
while developing, then `poetry run pytest .` and `poetry run ruff check .`
before opening a PR. Tests requiring optional ML backends should fail clearly
when their required extras are unavailable rather than silently masking issues.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects such as `Add CatBoost RFE mode` or `Fix
configuration validation`; recent history favors that style without ticket
prefixes. Work on `feature/<name>`, `fix/<name>`, or `hotfix/<name>` branches.
PRs target `master`, explain the change and validation performed, include tests,
and add an `examples/` usage example for substantial features. Ensure CI passes
and document public API changes; include screenshots only when notebooks or
visual output materially changes.

## Feature-Selection Change Notes

For every substantial change under `fmlib/feature_selection/`, add a concise
Markdown note in `fmlib/feature_selection/updates/` named
`NNN-short-topic.md` (for example, `001-categorical-handling.md`). State the
user-visible YAML/API change, behavioral defaults, supported modes and known
limitations, affected model paths, and validation performed. Keep it factual;
do not copy raw datasets, credentials, or large logs. Tiny formatting-only
changes do not need a note. Read the latest existing note before adding a new
one, so numbering and terminology remain consistent.
