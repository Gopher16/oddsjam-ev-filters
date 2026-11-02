# Analyst guide

Build from explicit verified inputs and interpret each publication at its declared
grain. Start with [the architecture](../ARCHITECTURE.md) and [glossary](../GLOSSARY.md).
All commands below run from the repository root. Paths and IDs in angle brackets
are placeholders; choose the intended local artifacts before running a build.

## Build a research report

Install the locked environment, including the optional dashboard when needed:

```bash
poetry install --with dashboard
```

For manual acquisition, download your own export outside this application. Declare
supported source metadata in [data_sources.yaml](../../configs/data_sources.yaml);
leave unverified export times, timezone and scope unknown. Follow
[ingestion](../INGESTION.md) for immutable import/snapshot commands and reconciliation.

The existing core runner supports a manual CSV or an explicit historical snapshot.
For historical analysis:

```bash
poetry run python scripts/run_weekly.py \
  --config configs/weekly_research.yaml \
  --snapshot data/raw/ingestion/snapshots/<snapshot-id> \
  --as-of <timezone-aware-knowledge-cutoff>
```

Replace both placeholders, keeping the cutoff within supported snapshot knowledge
and system time. See [pipeline inputs, notices and resume](../WEEKLY_RESEARCH.md) for
CSV mode and failure handling. Unknown freshness may require historical mode; do not
invent metadata to make a current-data gate pass. The runner publishes canonical,
atlas, core discovery, recommendations and optional experiment follow-up. It does
not orchestrate every newer optional builder.

## Publish optional evidence explicitly

| Build | Inputs and documentation |
| --- | --- |
| Continuous EV and sizing | [EV units/provenance](../CONTINUOUS_EV_CONTRACT.md), [sizing history](../SIZING_HISTORY.md), [diagnostics](../SIZING_DIAGNOSTICS.md) |
| Within-filter candidates | [Bounded declared views and exact thresholds](../WITHIN_FILTER_DISCOVERY.md) from canonical, atlas and diagnostics |
| Exploratory inference | [Prioritization](../STATISTICAL_PRIORITIZATION.md) from the frozen family and exact diagnostic publication |
| Setup sheets and queues | [Decision records](../DECISION_CONFIGURATION_RECORDS.md) and [review queues](../REVIEW_QUEUES.md) with explicit configuration/lifecycle evidence |
| Visuals and models | [Visual diagnostics](../VISUAL_DIAGNOSTICS.md), then [offline explanatory models](../EXPLANATORY_MODELS.md) on exact source publications |
| Prospective and applied outcomes | [Prospective reports](../PROSPECTIVE_EXPERIMENT_INTEGRATION.md) and [lifecycle reports](../REVIEWED_FILTER_LIFECYCLE.md) from explicit versioned records |

Each guide contains its builder's exact CLI and configuration. Use `--help` to confirm
the interface; do not concatenate publication paths from different snapshots merely
because directory names look compatible. Changed policies/evidence create new identities.
Keep old publications and manifests intact.

## Attach and verify

Set the artifact/evidence roots and exact run-keyed publication paths in
[dashboard.yaml](../../configs/dashboard.yaml), or a separate local configuration
selected with `ODDSJAM_DASHBOARD_CONFIG`. The viewers verify manifests, confinement
and matching lineage. Missing optional attachments remain unavailable; they are
not found by guessing the latest directory.

```bash
poetry run python scripts/verify_release.py \
  --config configs/dashboard.yaml \
  --run-id <exact-published-run-id> --repeats 3
```

This read-only check reports attachment availability, economics reconciliation,
chart readiness and loader performance. It does not run discovery, fit a model or
approve a strategy. See [acceptance procedure](../RELEASE_ACCEPTANCE.md) for browser
checks and optional engineering budgets.

## Interpret the result

The older core discovery contract, newer within-filter ledger, separate corrected
prioritization and untested global descriptive catalogue are different publications.
Do not transfer a rejection between them. Review denominators, exclusions, matched
EV/return cohorts, event dependence, chronology, concentration and sizing regimes.
Unverified timing blocks forward-validation claims. Liquidity and closing-price
series require their own evidence. See the [implementation ledger](../plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md)
for supported, blocked and deferred requirements.

Legacy notebooks and [preprocessing](../../scripts/preprocess_bet_tracker.py) retain
their original inputs/assumptions; they are not executed by the runner or dashboard.
The [synthetic walkthroughs](../walkthroughs/README.md) exercise the build/refresh
chain. Use the [configuration/publication map](../CONFIGURATION_PUBLICATION_MAP.md)
and [troubleshooting guide](../TROUBLESHOOTING.md) to locate inputs, outputs and remedies.

[Documentation home](../README.md) · [Operator route](OPERATOR.md)
