# Robust segment research

This framework evaluates **predeclared descriptive hypotheses** from a pinned
atlas. It never promotes strategies, recommends filter edits, places wagers,
changes notebook defaults, or enables scheduling. The acquisition draft is not
required. Historical results remain exploratory, even when an approximate test
passes a correction.

## Run

From the repository root:

```bash
poetry run python scripts/build_discovery_report.py \
  --atlas data/processed/atlas/198d2851aaf045f115124015c76f410ad6808ccf5f83cf674d3826b660ebb5e6
```

Optional `--config`, `--output-root`, and `--temporal-evidence` select explicit
policy, private output root, and JSON temporal evidence. No input is selected by
latest date. Outputs live under ignored `data/processed/discovery/<report-id>`.
The CLI prints counts and gate reasons, not monetary reports or source records.

```python
from pathlib import Path
from oddsjam_ev.io.discovery import load_discovery

tables = load_discovery(Path("data/processed/discovery/REPORT_ID"))
```

The manifest pins the complete policy, atlas manifest hash and IDs, external
attestations, transformation code hashes, and NumPy/pandas/SciPy/PyArrow versions.
Output hashes and row counts are verified on read. Writes use staging, exclusive
locking, and atomic publication. Repeating a specification checks logical equality
of every table. Policy, source, evidence, seed, or code changes create a different
report. Existing raw, ingestion, canonical, atlas, and notebook files are untouched.

## Tables

| Table | Grain and contract |
| --- | --- |
| `candidate_ledger` | One declared predicate / execution status / product. Includes candidates with no eligible economics. |
| `candidate_memberships` | Unique candidate/bet membership. Non-additive across candidates and views. No economics. |
| `candidate_results` | Full eligible economics, event-covered economics, missing-event coverage, conditional intervals, sensitivities, raw/adjusted tests, and limitations. |
| `temporal_gates` | Overall chronology availability and explicit blocking reasons. |
| `temporal_partitions` | One eligible bet/fold with assigned phase or exclusion reason. |
| `frozen_candidates` | Development-only predicate ledger per fold. No holdout-created categories. |
| `temporal_metrics` | Frozen candidate/phase reported economics, including empty later cohorts. Descriptive, not a validation certificate. |
| `temporal_stability` | Included phase/month (UTC)/execution/product reported-return summaries where time evidence exists. Non-additive across repeated training windows. |
| `temporal_fold_audit` | Input, included and excluded counts, reconciliation, and incomplete/empty-phase flags per fold. |

Every output carries atlas ID and discovery contract version. Report identity
links all tables to the full manifest. No result is a scanner opportunity count or
verified independent sample size.

## Search policy

`configs/discovery_policy.yaml` declares operational settings:

- Single atlas dimensions and its three paired views, plus an overall baseline.
- Separate `real_money`, `paper`, and `unknown` cohorts and canonical products.
- Assumed real-money results are primary; paper/unknown are secondary analyses.
- Minimum 100 event-proxy groups for conditional intervals and tests.
- 95% percentile intervals, 5,000 cluster resamples, fixed recorded seed.
- One-sided zero-reported-return null and Holm familywise alpha 0.05.
- `min_eligible_stake_usd: null`: **formal testing is blocked** until the user
  explicitly sets a stake floor. The floor applies to the event-covered inference
  cohort, not to money excluded because event IDs are missing.
- No temporal windows or embargo are guessed.

These thresholds are guardrails, not power guarantees. Changing a threshold after
viewing results is additional researcher selection and must be disclosed.

Categories are enumerated from observed dimension values without consulting
profit or eligibility. Predicates, including missing/unmapped categories, receive
stable IDs. All observed candidates remain in the ledger, including entirely
ineligible ones. Unobserved Cartesian combinations are not fabricated; an absent
category is an unobserved coverage gap, not an evaluated zero-return strategy.
Saved-filter labels never define search keys. Candidate dollar totals cannot be
summed across overlapping predicates. Canonical parlay/Pick'em, refund/cancellation,
promotion, unsupported-product, and unsettled exclusions are preserved.

## Economics, dependence and missing groups

Full eligible ROI is sum of eligible reported profit divided by corresponding
eligible stake. Empty eligible cohorts have null totals/ROI. USD is user-confirmed;
fees remain unverified and no estimated fee is deducted.

Event groups use namespace plus nonempty source game ID. They are dependence
**proxies**, not proven independent units. Wagers without namespace/event evidence
remain in descriptive counts and money but are excluded from group inference.
Their counts, stake and profit are disclosed separately. Bootstrap intervals,
tests, and leave-one-event-out diagnostics describe only the event-covered subset;
do not present them as intervals for the entire eligible population.

Whole events are resampled with replacement, keeping all event stake/profit
paired. Each resample computes sum profit / sum stake. Per-bet ROIs are not averaged
for this estimate. Degenerate resampling yields unavailable bounds. Cross-event
team/player/regime dependence is unresolved; neither grouping nor a large nominal
count establishes complete independence.

Sensitivity outputs include removing the largest-stake event, removing the largest
positive-profit event, leave-one-event-out ROI extrema, largest event stake share,
and equal-wager mean ROI. The latter is a different weighting estimand, not a
corrected bankroll return. Original economics are never overwritten or wins/losses
silently removed. Empirical-Bayes shrinkage is deferred until a defensible model,
prior and fitting population are specified.

## Approximate tests and multiple comparisons

With an explicit stake threshold and adequate groups, the optional test uses a
one-sided Student t statistic on **event profit totals**, under an assumed
independent finite-variance event model. Positive expected group stake makes the
zero expected-profit null correspond to the sign of the population aggregate
reported-return ratio. This is not an exact test for heavy-tailed or dependent
returns and is not a test of an average individual-bet ROI.

Degenerate event profit variance produces no p-value. No p-values are obtained by
counting signs in an uncentered bootstrap distribution. Conditional bootstrap
intervals are not simultaneous intervals or post-selection-valid confidence bounds.

The Holm family includes **every candidate in the declared run**, across all
cohorts, products and views, including overall baselines. Unavailable tests occupy
p=1 placeholders for correction but retain null published raw/adjusted p-values.
They can never be flagged as rejected. Only explicitly available approximate tests
can receive `exploratory_holm_reject`; `promotion_allowed` is always false.

Holm does not repair biased, invalid, or selectively reported marginal tests.
Repeated searches under changed configurations are new researcher choices; the
software records immutable runs but cannot retroactively correct all human
exploration across runs. Reusing the already-inspected October history does not
create an untouched confirmatory holdout.

Method references: [SciPy Student t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html),
[SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
[Statsmodels multiple-testing reference implementation](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/multitest.py).
No new dependencies are needed.

## Chronological evidence gates

The current October atlas cannot support chronological validation. A future input
requires these explicit UTC columns on eligible wager facts:

- `created_at_utc`: verified placement time.
- `features_available_at_utc`: candidate features known by placement.
- `outcome_available_at_utc`: settlement information actually available.
- `information_end_utc`: end of the record's information/dependence window, no
  earlier than outcome availability.

Canonical contracts currently do not supply the three additional availability
columns. An evidence JSON alone cannot enable temporal analysis or synthesize
these fields. Supporting future contracts must retain their real provenance.

Optional evidence JSON must include `placement_time_verified`,
`decision_features_verified`, `outcome_availability_verified`, and
`outcome_values_asof_verified` as true, plus a meaningful `source_reference` and
timezone-aware `observed_as_of_utc`. These are external attestations requiring
supporting evidence, not independent verification performed by this program.
Never assert them merely to make a gate pass.

Each configured temporal window requires `fold_id`, `development_start`, `selection_start`,
`test_start`, and `test_end`. All time values must be explicitly timezone-aware and
strictly increasing. Use justified calendar boundaries, not guessed October dates.
Multiple expanding origins require the same development start and non-overlapping
test windows. `embargo_days` must be explicitly justified and set; zero is an
explicit research choice, not an implicit fallback.

Intervals are left closed/right open. Event groups spanning phases are excluded
in full. Missing times, features known only after placement, unavailable outcomes,
and missing event identities are audited; a known event with an invalid member is
quarantined as a whole. Development/selection information must end before the next
phase minus embargo. If the observation cutoff precedes the test-window end, the
fold is incomplete and publishes no phase metrics. Each fold reconciles all
eligible input rows, including outside-window exclusions.

Development data alone determines each fold's predicate ledger. Selection/test
outcomes and new later categories cannot alter it. All frozen candidates are
reported; there is no automated winner-selection or production decision. Monthly
stability and fold metrics remain descriptive. Full-period exploratory results and
fold results have separate tables and must never be substituted for each other.
A clean partition is not proof of executable, fee-adjusted or prospective edge.

## Validation and deferred work

Synthetic tests cover invalid policies, empty and excluded cohorts, stable search
IDs, missing event coverage, whole-event economics, degenerate samples, weighted
ratios, Holm family accounting, planted signals, null families, overlapping searches,
unequal stakes, conditional interval coverage, delayed outcomes, group purging,
embargoes, unknown time, incomplete folds, multiple expanding origins and holdout
outcome/category isolation. Distribution-specific simulation results do not verify
independence or calibration for the private historical data.

Integration tests use synthetic canonical and atlas bundles for CLI invocation,
immutable replay, configuration versioning, corruption, manifest tampering and
atomic publication failure. They need no credentials, private data or network.

Deferred: economic stake floor approval, externally supported chronology/features,
fee/fill verification, cross-event dependence models, justified temporal windows
and embargo, empirical-Bayes modeling, and prospective validation. Existing
notebooks and their input selection are unchanged.


### October validation evidence

Validated atlas: `198d2851aaf045f115124015c76f410ad6808ccf5f83cf674d3826b660ebb5e6`.

Published report: `102bb78305423fe24add23bd5d75dc2283e8980cfb83c643e458e72bd82ba51b`.

- 1,965 candidate ledger/results rows and 706,350 candidate memberships.
- 53,853 eligible wagers reconcile to the atlas overall economics by execution
  mode and product; reported stake, profit and ROI match.
- 7,408 eligible wagers lack event evidence and remain outside event inference.
- 341 conditional exploratory intervals; the other 1,624 candidates have
  insufficient event-proxy groups under the configured floor.
- All 1,965 formal tests are blocked by the unset stake threshold. No raw or
  adjusted p-values, rejection flags, or promotion flags are published as positive.
- Chronological gates are blocked: placement/feature/outcome evidence, availability
  columns, temporal windows, and embargo policy are absent. All five temporal
  partition/candidate/metric/stability/fold tables are empty.
- All nine tables reproduced with logical equality on replay.
- 179 tests passed with FutureWarnings treated as errors. Poetry lock validation,
  Ruff lint/format checks, and `git diff --check` passed.

No notebook was executed or modified. Raw data, ingestion history, canonical/atlas
bundles and the historical preprocessing defaults were preserved. Private report
outputs remain Git-ignored.
