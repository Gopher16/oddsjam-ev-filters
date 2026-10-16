# OddsJam EV Filters — Repository Instructions

## Scope and purpose

This repository is an offline, reproducible research platform for historical OddsJam
Bet Tracker data and saved positive-EV betting filters. It does not operate the live
OddsJam scanner, place wagers, or automatically deploy filter changes.

## Working style

- Prefer small, reviewable, independently testable changes; preserve existing behavior
  unless the task explicitly calls for a methodological or contract change.
- Inspect relevant existing modules and tests before introducing abstractions.
- Keep configuration and shared logic in reusable modules under `src/oddsjam_ev/`;
  notebooks should orchestrate analyses and present interpretable outputs.
- Do not introduce arbitrary commit numbers or temporary development milestones into
  enduring docstrings, notebook explanations, or comments.
- Clearly distinguish verified observations, assumptions, and proposed next steps.

## Python conventions

- Target Python 3.11 and follow the Ruff configuration in `pyproject.toml`.
- Use four-space indentation, type hints where helpful, snake_case for functions and
  variables, and PascalCase for classes.
- Match the established module-level documentation style in
  `src/oddsjam_ev/analysis/scorecards.py`: module name, Purpose, What this module
  provides, and Design Notes, with consistent section separators.
- Use NumPy-style docstrings for public functions and configuration classes,
  including Parameters, Returns, Notes, and Attributes when relevant.
- Prefer focused functions and straightforward pandas/NumPy operations over
  unnecessary abstractions.
- Validate required columns and configuration values early; fail with informative
  errors rather than silently producing misleading results.

## Research and data contracts

- Distinguish execution-level records from opportunity-group records; never
  silently treat repeated executions or overlapping filter memberships as
  independent betting opportunities.
- Preserve units, denominator definitions, time windows, and missingness semantics.
- Treat reported expected value and realized ROI as distinct measures. Neither a
  positive historical ROI nor an EV realization gap alone proves a deployable edge.
- Make data snapshot selection and provenance explicit. Do not hardcode duplicate
  snapshot paths across notebooks when a shared loader exists.
- Use chronological, group-aware validation when evaluating proposed filters.
  Prevent leakage across splits and disclose small samples, uncertainty, multiple
  comparisons, liquidity limitations, and overlapping saved-filter exposure.
- Historical saved-filter labels and current live UI settings are not necessarily
  equivalent; do not assume current settings applied retroactively.
- Do not silently alter canonical output schemas, evaluation grains, or metric
  definitions.

## Testing and notebooks

- Add or update pytest coverage for behavior changes, including invalid inputs,
  missing values, duplicates, boundary conditions, and empty datasets when relevant.
- Keep unit tests runnable on small synthetic fixtures without private/local
  historical Parquet files or network access.
- Preserve useful notebook explanations, diagnostics, and plots unless removal is
  explicitly justified. Re-run affected notebooks locally and inspect outputs when
  data is available; CI does not execute full historical-data notebooks.
- Before committing, run:

  ```bash
  poetry check --lock
  poetry run ruff check . --fix
  poetry run ruff format .
  poetry run ruff check .
  poetry run ruff format --check .
  poetry run pytest -q
  git diff --check
  ```

- Inspect formatting changes before staging; do not include unrelated files.

## Security and repository boundaries

- Never commit API keys, credentials, private account information, raw exports,
  or local processed betting datasets.
- Do not copy proprietary employer code, documentation, infrastructure settings,
  or internal data into this personal repository.
- Do not represent exploratory filter recommendations as production-ready without
  independent validation and an explicit deployment decision.

## Pull requests

- Use `.github/pull_request_template.md` and document testing evidence, snapshot
  provenance, research limitations, and remaining follow-up work.
- Required quality checks should fail on lint, format, or test errors. No approving
  reviewer is required for this single-maintainer repository.
