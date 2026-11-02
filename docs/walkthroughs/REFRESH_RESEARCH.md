# Refresh research from a new export

Continue after [monitoring the synthetic TEST change](MODIFY_FILTER.md). The refresh
uses the same fake export plus one new, distinct fictional wager.

```bash
poetry run python docs/examples/operator_demo.py --workspace "$WALKTHROUGH_ROOT" refresh
```

## What changes

1. Creates `source-refresh.csv`; the original `source.csv` remains intact.
2. Imports the new bytes explicitly into the same synthetic ingestion archive.
3. Creates an explicit new snapshot at the supported synthetic knowledge cutoff.
4. Runs the core historical pipeline and the newer explicit builders into `refresh/`.
5. Rebuilds candidate IDs, source-bound configuration evidence, queues, visuals and
   a model publication appropriate to the new lineage.
6. Runs read-only release reconciliation for the new report.
7. Writes `dashboard-refresh.yaml` with exact new publication links.

The canonical candidate count moves from **240 to 241**, not 481. Repeated export
rows are source observations, not additional independent wagers. The original
`first/` publications and earlier definitions/outcomes remain byte-for-byte intact.

```bash
ODDSJAM_DASHBOARD_CONFIG="$WALKTHROUGH_ROOT/dashboard-refresh.yaml" \
  poetry run streamlit run dashboard/app.py
```

The new configuration does not attach old definition/outcome/lifecycle publications
to incompatible new research lineage. The earlier `dashboard-monitored.yaml` remains
available to inspect the first study. A real source refresh does not retrospectively
enroll old observations in a prospective test or silently rewrite its candidate.

## Apply this pattern to a real export

Use [ingestion](../INGESTION.md) to import a manually acquired file with truthful
namespace and acquisition metadata. Use [the core runner](../WEEKLY_RESEARCH.md) for
an explicit CSV or pinned historical snapshot; unknown freshness is not fixed by
inventing export timestamps. Build needed optional evidence with explicit paths
according to [the configuration/publication map](../CONFIGURATION_PUBLICATION_MAP.md).

Reconcile unique wager counts, original stakes/profits, population eligibility and
attribution coverage. Do not add overlapping saved-filter totals. Preserve old
snapshots/configurations, then update dashboard links deliberately. Revisions to
research policy or scanner settings require their own provenance; a new export alone
does not establish them.

The tutorial uses a simulated calendar; real imports/builds use actual knowledge
times and supported source timestamps. See [troubleshooting](../TROUBLESHOOTING.md)
for notices, failed integrity checks and incompatible attachments.

[Walkthrough home](README.md) · [Analyst guide](../guides/ANALYST.md)
