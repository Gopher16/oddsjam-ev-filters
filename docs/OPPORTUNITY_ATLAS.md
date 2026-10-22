# Cross-filter opportunity atlas

The atlas describes logged records from an explicit canonical bundle. It does not
estimate scanner coverage, establish executable opportunities, promote strategies,
change filters, or schedule jobs. The draft acquisition adapter is not required.

## Build and read

From the repository root:

```bash
poetry run python scripts/build_opportunity_atlas.py \
  --bundle data/processed/canonical/df99ad86af5f1c7b09b7fb93fdab8c039306217239fe528242cf39b30f886dc7
```

The command prints an immutable `data/processed/atlas/<atlas-id>` directory.
Optional `--config` and `--output-root` arguments select explicit alternatives.
Raw files, canonical bundles, ingestion history, the March preprocessing input,
and Notebook 01/02 defaults are never changed. No latest pointer is used.

```python
from pathlib import Path
from oddsjam_ev.io.atlas import load_atlas

tables = load_atlas(Path("data/processed/atlas/ATLAS_ID"))
```

The reader verifies specification identity, file hashes, and row counts. Publication
uses staging and a writer lock; failures publish no partial bundle. Rebuilding the
same specification verifies logical table equality. The specification pins the
canonical manifest hash, canonical/ingestion IDs, mapping and bands, transformation
code hashes, pandas/PyArrow versions, and interpretation limits. Outputs carry
canonical and ingestion IDs plus the atlas contract version. The atlas directory
ID identifies the full policy and code specification.

## Contracts and denominators

| Table | Grain / interpretation |
| --- | --- |
| `wager_dimensions` | One canonical candidate per bet ID. Canonical eligibility and economics are preserved. |
| `opportunity_dimensions` | One selection/quote proxy, with consensus source metadata and linked source-record count. No monetary fields. |
| `wager_segments` | One view / execution mode / product / dimension combination. Each wager enters once per slice. |
| `opportunity_segments` | One view / execution mode / dimension combination; distinct proxy/selection counts and source-link counts. No wager economics. |
| `filter_segment_diagnostics` | Deduplicated memberships, with overall, sportsbook, and market-family slices per filter. Non-additive across filters. |
| `atlas_quality` | Dimension-category coverage and source/exclusion accounting, with named populations and denominators. |

`view=overall` is the reconciliation baseline. Do not sum multiple views. Distinct
selection/proxy counts are also non-additive across slices and execution modes:
one proxy may link to both paper and real wagers. Counts are grouping diagnostics,
not independent sample sizes. Source-link counts include logged legs; candidate
counts include tickets. These are deliberately different populations.

Economic rows always separate `real_money`, `paper`, and `unknown`, as well as
canonical product. Real-money status is the historical user assumption, not
independently verified fills. Unknown stays unknown. Reported ROI is the eligible
reported profit sum divided by stake for exactly the same eligible candidates.
Empty cohorts have null monetary totals and ROI. All ticket candidates, unlinked
Pick'em legs, refunds/cancellations, unsupported products, and other canonical
exclusions remain excluded; their counts and exclusion combinations remain visible.
Ticket economics never flow through leg or opportunity joins. Fees remain
unverified; no estimated fee is deducted. Never call reported ROI verified net ROI.

The atlas validates link identities and source coverage before aggregation. It
uses current source records only for metadata consensus, not to re-sum source
money. Multiple values, including missing versus present values, yield explicit
`<conflict>` metadata rather than an arbitrarily selected observation.

## Dimensions

Source sportsbook, sport, league, and market labels are preserved (surrounding
whitespace is normalized). Market families use **exact** label mappings in
`configs/opportunity_atlas.yaml`; the initial mapping is conservative and incomplete.
Unmapped labels remain `<unmapped>` and raw labels are retained for review. Saved
filter names never determine discovery dimensions or market families.

American odds bands use boundaries -200, -150, -110, +100, +150, +200. Nonfinite
values and absolute odds below 100 are invalid. Fractional American odds above
that magnitude are retained. Advertised EV bands use percentage boundaries 0, 1,
2.5, 5, 10, 20. Intervals are left closed, right open, with open-ended tails.
Missing, invalid, and conflicting values have distinct categories. Negative and
extreme advertised EV remain visible; values are never clipped or treated as
verified expected returns. Display bands are not validated betting thresholds.

Default views include each dimension and sportsbook × market family, league ×
market family, and odds × advertised EV. For an explicit drill-down, call
`summarize_wagers(wager_dimensions, ["sportsbook", "market_family", "league"])`.
Execution mode and product are always added to grouping keys. Arbitrary outcome
columns are not accepted as atlas dimensions.

## Uncertainty, concentration, and unsupported analysis

`largest_stake_share` is the largest eligible stake divided by eligible stake.
`largest_absolute_profit_share` is the largest absolute eligible profit divided by
total absolute eligible profit. A zero denominator yields null. Event-ID counts
and selection-proxy counts do not establish effective independent sample sizes.

Uncertainty is explicitly `not_estimated_dependence_unverified`. Confidence
intervals, cluster-bootstrap policy, significance, multiple-testing correction,
and formal validation are deferred to robustness work. This implementation uses
the assessment's recommended concentration diagnostics; its optional cluster
intervals are not enabled. Existing row-bootstrap metrics are not reused.

Timezone, chronology, export scope, and decision-time feature availability remain
unverified. No temporal rankings, time-to-event bands, drawdown histories, holdouts,
or freshness claims are produced. Liquidity is unavailable in the October source;
tags and market width do not establish executable capacity. CLV, settlement,
and profit are never pre-bet discovery dimensions. No full-market opportunity
universe or causal interpretation follows from these observational slices.

## Notebook 03

`notebooks/03_opportunity_discovery.ipynb` loads only an explicitly selected atlas
through the shared verified reader. Build first, then launch Jupyter from the
repository root, replacing `ATLAS_ID` with the printed directory name:

```bash
ODDSJAM_ATLAS_DIR="$PWD/data/processed/atlas/ATLAS_ID" poetry run jupyter lab
```

Select the project Python kernel. An already-running VS Code kernel will not
inherit a newly exported terminal variable; restart Jupyter with the environment
above or set the input explicitly in your local session. No default private
bundle path is embedded in the checked-in notebook.

Notebook sections cover provenance, population reconciliation, coverage/mapping,
broad segments, numeric bands, explicit drill-down, within-filter comparisons,
and concentration. Parent versus subsegment ROI is descriptive only. No automatic
best-segment ranking or action recommendation is generated. Keep executed notebooks
and monetary reports in ignored data paths. The checked-in notebook has no outputs.

## Validation and follow-up

Synthetic tests cover boundaries, invalid policies, absent optional source columns,
empty data, metadata conflicts, source/link integrity, input order, repeated filter
memberships, shared proxies across modes, ticket exclusions, null economics,
actual canonical-to-atlas integration, replay, policy versioning, corruption,
manifest tampering, CLI invocation, and atomic failure. They require no credentials,
private history, or network.

Follow-ups: expand reviewed exact market mappings, establish chronology and source
coverage, verify economics and fills, and define dependence/uncertainty methods
before interpreting segment differences as evidence for a strategy. Manual CSV
imports remain supported; weekly scheduling remains disabled.

### October validation evidence

Validated canonical bundle:
`df99ad86af5f1c7b09b7fb93fdab8c039306217239fe528242cf39b30f886dc7`.

Published atlas:
`198d2851aaf045f115124015c76f410ad6808ccf5f83cf674d3826b660ebb5e6`.

| Table | Rows |
| --- | ---: |
| Wager dimensions | 70,635 |
| Opportunity dimensions | 106,111 |
| Wager segments | 1,965 |
| Opportunity segments | 1,366 |
| Filter diagnostics | 180 |
| Quality audit | 827 |

Overall candidate counts, eligible counts, eligible stake/profit, and reported ROI
matched the canonical economic summary by product and execution mode. Every
configured wager view reconciled to those parent counts and monetary totals.
The atlas rebuilt with logical equality across all six tables. The exact market
mapping leaves 21,400 wager candidates unmapped; they remain visible rather than
being inferred from filter names. The eligible population remains 53,853 candidates.

Notebook 03 executed successfully against this explicit atlas. Its executed copy,
including the volume chart and monetary tables, is stored only under ignored
`data/processed/atlas-notebook-validation.ipynb`. The checked-in notebook retains
empty outputs. Existing notebooks, raw data, and historical defaults were not
modified or executed.

Final checks passed: `poetry check --lock`, Ruff lint and format checks,
`git diff --check`, and 152 tests with pandas FutureWarnings treated as errors.
The Notebook 03 volume chart was also inspected visually. No credentials or
network access were required for atlas construction or synthetic tests.
