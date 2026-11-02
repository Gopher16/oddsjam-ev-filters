# Contributor guide

Use Python 3.11 and the locked Poetry environment. Read the
[repository instructions](../../.github/copilot-instructions.md),
[architecture](../ARCHITECTURE.md) and [implementation ledger](../plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md)
before extending a contract. Historical plans describe intent; module contracts and
versioned policies establish current behavior.

## Find the implementation

| Location | Responsibility |
| --- | --- |
| [analysis](../../src/oddsjam_ev/analysis/) | Eligibility, grains, predicates, economics, statistical policies, decisions and lifecycle validation |
| [io](../../src/oddsjam_ev/io/) | Explicit source loading, registers, immutable writers/loaders, manifests and integrity checks |
| [pipeline](../../src/oddsjam_ev/pipeline/) | Core frozen plans, stage reuse, attempts, bounded retries and final publication |
| [dashboard package](../../src/oddsjam_ev/dashboard/) | Verified attachments, display queries, chart builders and pages |
| [dashboard entry point](../../dashboard/app.py) | Streamlit setup, selected report and read-only component attachment |
| [scripts](../../scripts/) | Explicit CLI orchestration; shared analytical logic belongs in the package |
| [configs](../../configs/) | Versioned policies, declared search views, templates and local publication locators |
| [tests](../../tests/) | Synthetic contracts, publications, failure paths, AppTest and full-cycle acceptance |
| [notebooks](../../notebooks/) / [plotting](../../src/oddsjam_ev/plotting/) | Retained legacy exploration and reusable plotting concepts |

## Run the checks

```bash
poetry install --with dashboard
poetry run pre-commit install
poetry check --lock
poetry run ruff check . --fix
poetry run ruff format .
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest -q
git diff --check
```

Inspect formatting changes before staging. Unit tests use synthetic fixtures without
private historical data or network access. The dashboard group is required for its
imports/AppTest. [Release acceptance](../RELEASE_ACCEPTANCE.md) adds an explicit private
snapshot smoke and real-browser checks when changes affect that surface. AppTest alone
is not visual acceptance. Do not execute notebooks unless the change requires it;
retain their inputs and document any actual notebook validation performed.

## Extend without changing historical meaning

- Preserve original sources, published versions, units, grains and exclusions.
- Add versioned policy/schema changes deliberately; keep incompatible readers explicit.
- Test boundaries, missingness, duplicates, reconciliation, corrupt artifacts and
  budget failures where behavior changes.
- Keep complete inference families separate from display filters and overlap suppression.
- Preserve unknown evidence; never backfill history from current settings or names.
- Keep exploratory research, human approval and confirmed application separate.
- Run fitting/builds explicitly outside ordinary dashboard interactions.
- Keep private source/configuration evidence under ignored paths, with synthetic fixtures
  and safe public templates. Never stage raw history, credentials or account screenshots.

Use Conventional Commits and the [PR template](../../.github/pull_request_template.md).
State validation and limitations; user review/merge remains separate from creating a PR.
The [documentation sequence](../plans/DOCUMENTATION_OVERHAUL.md) keeps navigation,
walkthroughs, research reference and maintenance in independently reviewable changes.

[Documentation home](../README.md) · [Glossary](../GLOSSARY.md)
