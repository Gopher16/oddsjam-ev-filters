# Publish reviewed historical attribution

The dated `configs/filter_history/oddsjam_filter_configurations_2026-10-30.yaml`
revision applies confirmed identity decisions from the review conversation. It
embeds the proposal ID, private decision-file hashes, confirmed names, unresolved
cases and previous registry hash. Canonical specifications embed this entire
registry; pipeline configuration hashes and downstream manifests pin its lineage.
Private evidence files remain under ignored data paths.

The old registry stays unchanged. Existing IDs are retained, including the legacy
PROD DFS ID. Its explicit `attribution_target_id` redirects membership to the
existing TEST DFS identity. This represents a confirmed same-identity relationship,
not a clone; unknown targets, cycles and conflicting aliases are rejected. Registry
entries with no redirect retain their previous behavior. New historical identities
reuse provisional IDs from the reviewed proposal. Their parameters remain unknown.

Only confirmed name relationships are applied. Bare `DFS Platforms`, bare Aggressive
Line Chaser, `Offshore Sportsbooks` and `Rebet` remain unresolved. The first two
were described as likely matches, not confirmed aliases. Existing exact mappings
are retained except for the explicit reviewed legacy DFS membership redirect.
No funding policy, source economics or historical configuration effectivity changes.
The revision date records knowledge of identities, not dates when parameters changed.

## Explicit publication

From the repository root, select the original snapshot and original analysis cutoff:

```sh
poetry run python scripts/run_weekly.py \
  --config configs/weekly_research_attribution_2026-10-30.yaml \
  --snapshot data/raw/ingestion/snapshots/<snapshot-id> \
  --as-of <original-aware-cutoff>
```

This separate configuration leaves the default pipeline configuration unchanged.
It builds new immutable canonical, atlas, discovery, recommendation and report
artifacts under the existing private research root. No acquisition or scheduling
is invoked. Old reports remain selectable. Generation time does not change source
freshness; the original historical cutoff and unknown chronology remain visible.

Then compare explicit before/after canonical bundles:

```sh
poetry run python scripts/audit_filter_attribution.py \
  --before <original-canonical-directory> \
  --after <new-canonical-directory>
```

The audit requires identical ingestion manifests and economic policy. It verifies
both bundles and rejects any change in the eight non-attribution tables, covering
wager identities, source/leg records, eligibility, product, source amounts,
opportunity links and economic summaries. It publishes immutable coverage counts,
added/removed membership pairs and per-filter candidate counts, with both manifest
hashes and registry fingerprints. Status/membership totals are non-additive.

The dashboard consumes completed publications from the existing research root.
Reload it and select the newly generated report in the report selector. Its filter
names and selectable cohorts come from that publication's pinned registry/atlas;
no dashboard-side remapping of old results is performed. The default 100-wager
threshold can still hide small cohorts. Old reports remain usable with old identity
coverage. A successful publication and reconciliation are required before treating
the revised report as ready for review; neither implies strategy validation.
