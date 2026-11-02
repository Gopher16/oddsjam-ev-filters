# OddsJam EV Filters

An offline research workspace for evaluating historical OddsJam Bet Tracker data,
reviewing potential positive-EV filter hypotheses, and following prospective paper
studies through human review and versioned filter changes.

The dashboard brings together published opportunities, production concerns,
notebook-inspired diagnostics and frozen model explanations. Historical associations
remain exploratory; missing scanner settings and source evidence remain explicit.
The project does not operate the live scanner, place wagers or automatically deploy
changes.

## Start here

| Your task | Entry point |
| --- | --- |
| Review candidates, inspect settings and prepare a paper study | [Operator guide](docs/guides/OPERATOR.md) |
| Build publications and interpret research evidence | [Analyst guide](docs/guides/ANALYST.md) |
| Understand modules, run checks and extend contracts | [Contributor guide](docs/guides/CONTRIBUTOR.md) |
| Find a reference or understand a term | [Documentation home](docs/README.md) · [Contract reference](docs/reference/README.md) · [Glossary](docs/GLOSSARY.md) |
| Understand what is implemented and what remains | [Discovery implementation status](docs/plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md) |

## Open the dashboard

Use Python 3.11 and Poetry 2. Run from the repository root:

```bash
poetry install --with dashboard
poetry run streamlit run dashboard/app.py
```

The default configuration is [configs/dashboard.yaml](configs/dashboard.yaml).
An installation without the private publications will display an unavailable or
empty state; it does not include a public historical dataset. See the
[analyst build instructions](docs/guides/ANALYST.md#build-a-research-report) to publish
from your own explicitly selected inputs. Restart Streamlit after updates because
file watching is disabled in the repository configuration.

## Implemented workflow

1. Import a manually exported CSV and preserve its original bytes and metadata.
2. Build canonical wager facts and research from an explicit snapshot and policies.
3. Publish optional within-filter comparisons, prioritization, complete setup
   sheets, visuals, review queues and explanatory models using explicit builders.
4. Attach exact publications to the dashboard and review candidate evidence.
5. Freeze a prospective definition, review/approve it, then separately confirm launch.
6. Capture observations, original-quote checks and outcomes; publish a scheduled review.
7. Record a reviewed decision, approve any settings change and separately confirm
   its external application before version-specific monitoring.

[Architecture diagrams](docs/ARCHITECTURE.md) show these boundaries. The existing
weekly runner orchestrates its core stages; it does not automatically build every
optional extension. Ordinary dashboard interaction reads verified outputs and never
fits models or launches studies.

## Interpretation and release status

- Keep real-money, paper and unknown funding populations and product economics separate.
- Exact EV predicates use original numeric values; display bands do not replace them.
- Weighted and equal-stake returns answer different questions. Neither proves an edge.
- A `PROD` name is not proof of an applied configuration. Unknown historical settings
  cannot be filled from today's scanner settings.
- Liquidity, timing and closing-price series require supported source evidence.
- A qualifying prospective assessment permits explicit bounded-pilot human review;
  approval and actual application remain separate.

See [release acceptance and first-study workflow](docs/RELEASE_ACCEPTANCE.md) and
[dated release evidence](docs/releases/2026-11-02-acceptance.md). Acceptance covers
software behavior and supported source states, not profitable filters or verified
fills. Private data, publications and account evidence belong in ignored local storage.

## Legacy notebooks and development

The existing [preprocessing script](scripts/preprocess_bet_tracker.py),
[preprocessing configuration](configs/preprocess.yaml), [notebooks](notebooks/) and
[plotting helpers](src/oddsjam_ev/plotting/) remain available. Their historical
assumptions and input defaults are independent of the canonical dashboard workflow;
legacy timestamp or CLV transforms do not establish verified source semantics.

For code checks, synthetic tests and contribution boundaries, use the
[contributor guide](docs/guides/CONTRIBUTOR.md). For historical plans and the next
four documentation PRs, see [planning and status](docs/README.md#planning-and-status).
