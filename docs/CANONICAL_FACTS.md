# Canonical bets, opportunities, and economics

Canonical bundles are opt-in research artifacts built from a pinned PR 1 ingestion
snapshot. They do not acquire data, place bets, change filters, update the legacy
snapshot resolver, or execute notebooks. The acquisition draft is not a dependency.

## Build and inspect

Run from the repository root with the existing Poetry environment:

```bash
poetry run python scripts/build_canonical_facts.py \
  --snapshot data/raw/ingestion/snapshots/861924428766080b4468adad998805037901532c68f8bdd041e1aa722994c7d1
```

The snapshot must already exist locally. No automatic latest selection or ingestion
reconciliation takes place. Optional `--config`, `--registry`, and `--output-root`
arguments select explicit alternate inputs/destinations. The default outputs are
private, Git-ignored `data/processed/canonical/<bundle-id>/` directories.

The command prints the bundle path and reconciliation counts. Read its manifest
for exact inputs, policies, code hashes, versions, exclusions, and table counts.
The Python reader verifies output hashes before returning tables:

```python
from pathlib import Path
from oddsjam_ev.io.canonical import load_canonical_bundle

tables = load_canonical_bundle(Path("data/processed/canonical/BUNDLE_ID"))
bets = tables["bet_facts"]
reported_roi_population = bets.loc[bets["reported_roi_eligible"]]
```

Use `economic_summary` for precomputed reported-return cohorts. Do not sum the
source features, source audit, ticket legs, opportunities, or attribution bridge
as a wager portfolio. Actual source records remain recoverable from ingestion.

## Contract version 1

| Table | Grain and contract |
| --- | --- |
| `bet_facts` | One current non-ticket source candidate, or one namespace/parlay-ID ticket candidate. These are not all economically eligible wagers. |
| `bet_legs` | Current and unresolved explicit ticket leg observations. No ticket stake, profit, or payout columns are carried here. |
| `source_audit` | Every ingestion observation, including unresolved and historical rows, with canonical disposition and linkage where known. |
| `source_features` | Every current ingestion source record, plus its canonical bet ID. Original source field values remain strings; monetary fields here are source-grain audit values, not additive wager facts. |
| `opportunity_facts` | Distinct labelled selection/quote proxies derived from current logged source records. No stake, profit, ROI, or selected representative execution. |
| `bet_opportunity_links` | One link per current source observation. A ticket can link to several leg opportunities. |
| `filter_attribution` | Unique resolved bet/filter memberships, plus explicit missing/unresolved label diagnostics. No economic fields. |
| `feature_catalog` | Actual current-source column availability, denominator, meaning, and temporal limitations. |
| `economic_summary` | Matching eligible profit/stake aggregates by product and assumed execution mode. This is reported ROI, not fee-verified net ROI. |

Every table carries ingestion snapshot ID and canonical contract version. Bet IDs
for singles reuse ingestion record identities. Ticket IDs hash namespace plus
source parlay ID. Source observation IDs link back to exact import/row provenance.
Namespaced ticket IDs do not assert the source identifier is globally unique or
stable across providers/accounts. Conflicting tickets remain ineligible.

All source observations reconcile to `current_single`, `current_ticket_leg`,
`unresolved_source`, or `historical_observation`. Revisions do not create extra
current wagers. Unresolved singles remain in the audit, rather than being promoted
into accepted economic facts. Unresolved explicit legs appear in `bet_legs` and
flag their ticket. Ticket grouping never silently removes an ambiguous leg.

## Wagers, explicit tickets, and candidate legs

See the authoritative [grain and ticket contract](reference/DATA_GRAINS.md).
The table inventory above describes this bundle's API; candidate identity never
implies ticket completeness or economic eligibility.

## USD, reported profit, and fee uncertainty

Use the authoritative [economics and eligibility reference](reference/ECONOMICS.md)
for canonical exclusions, settlement consistency, fee uncertainty and matching
profit/stake denominators. The dated validation counts below are historical evidence,
not current account results.

## Opportunities and filter attribution

The [opportunity and attribution reference](reference/DATA_GRAINS.md#opportunities-and-filter-attribution)
defines grouping fallbacks, non-additivity and historical-label limitations.

## Time, leakage, and feature availability

The [chronology reference](reference/CHRONOLOGY.md) defines source timestamps,
knowledge cutoffs, decision-time availability and blocked temporal claims. Unknown
timezone remains unknown; a successful build does not verify placement semantics.

## Publication and research compatibility

Bundle identity hashes its complete source manifest, economic policy, registry,
transformation code hashes, pandas version, and PyArrow version. Tables and counts
have output hashes. Rebuilding the same specification checks logical table equality
and reuses the bundle. Library/code/policy changes produce a different bundle ID.
Use the pinned code/lock version for reproducible historical replay.

Writes use a single-writer guard, staging directory, and atomic publication.
Failed builds leave no published partial bundle. Historical bundles are never
overwritten, and no latest pointer or research default is updated.

Notebook 01 remains legacy QA/readiness; Notebook 02 remains legacy descriptive
filter evaluation. No cells, IDs, metadata, outputs, or historical metric functions
were modified or executed. Future notebook integration must explicitly opt into
canonical grains and explain exclusions, rather than replace its input silently.
The existing row-bootstrap confidence intervals are not made group-aware by this
change. Opportunity discovery, statistical promotion, fee/payout inference,
acquisition, scheduling, and dashboard changes remain outside this PR.

## Validation

Synthetic tests cover source accounting, repeated ticket amounts, contradictory
leg statuses, unresolved members, one-leg incompleteness, missing money, explicit
settlement formulas, exchange fee uncertainty, Pick'em exclusions, literal commas,
registry errors, proxy fallbacks, matching ROI populations, empty inputs,
as-of settlement separation, output corruption, atomic failure, and replay.
Tests require no private data, network, acquisition branch, or notebook execution.

The October 22 ingestion snapshot is the private-data validation input. Record
counts and known limitations may be shared; raw records and monetary reports
remain under ignored data paths. `.local/` is independently ignored on this branch;
no credential files are read or needed by canonical processing.

### October snapshot validation evidence

The final implementation was built and replayed against ingestion snapshot
`861924428766080b4468adad998805037901532c68f8bdd041e1aa722994c7d1`.

| Result | Count |
| --- | ---: |
| Source audit observations | 106,246 |
| Current source records / opportunity links | 106,168 |
| Non-ticket wager candidates | 61,598 |
| Explicit ticket candidates | 9,037 |
| Total canonical bet candidates | 70,635 |
| Current plus unresolved ticket legs | 44,615 |
| Tickets affected by unresolved observations | 42 |
| Reported-ROI eligible single candidates | 53,853 |
| Fee-verified net-ROI eligible candidates | 0 |
| Opportunity proxies (not verified unique quotes) | 106,111 |

All nine tables reproduced with logical equality. Original raw files, ingestion
history, root-level processed Parquet datasets, and notebook hashes remained
unchanged. The 138-test suite passed with pandas FutureWarnings treated as errors;
Poetry lock validation, Ruff lint/format checks, and `git diff --check` passed.

All 70,635 bet candidates have unknown source timezone and null canonical UTC
creation time in this input. No complete temporal validation or decision-time
feature availability is claimed. Eligibility is reported-return arithmetic and
product policy, not proof of funded execution, net fees, or profitability.
