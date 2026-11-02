# Sizing-aware diagnostic economics

This optional stage compares source portfolio ROI with hypothetical equal-stake
returns on the **same diagnostic-eligible wager cohort**. Canonical reported ROI,
funding, eligibility and historical outputs are not redefined. A larger wager has
more influence on portfolio ROI; equal-stake arithmetic answers a different question.
Neither result alone measures intrinsic filter quality or establishes causality.

## Metrics and denominators

The [economics reference](reference/ECONOMICS.md#metrics-and-denominators) defines
weighted/equal-stake returns, fixed-unit profit, concentration and units. This guide
retains stage-specific cohort, regime, uncertainty and publication requirements.

## Cohorts, exclusions and regime scope

The candidate table retains one row per canonical wager candidate, source amounts,
canonical eligibility/exclusion reasons and separate diagnostic exclusions. Both
new estimands exclude identical rows: canonical-ineligible candidates, unsupported
products (this version supports fixed-odds/exchange), nonpositive/nonfinite stakes,
nonfinite profits or nonfinite unit returns. Half settlements follow source profit
arithmetic without an invented settlement conversion. Tickets are counted once as
candidates; source legs are never expanded into additional economic wagers.

Counts distinguish canonical eligibility from additional diagnostic exclusions.
If no extra exclusions exist, weighted ROI reconciles to the canonical baseline.
When extra exclusions exist, the baseline and new denominator remain separate and
visible. Undefined empty/zero denominators are unavailable, not zero returns.

Published views are overall, applied regime, resolved saved filter and
filter-by-regime, always separated by product and funding. Overall and regime views
partition the candidate population. Resolved-filter views overlap and must not be
summed; missing/unresolved labels remain in overall/regime coverage but do not
receive invented per-filter returns. Filter membership creates no extra money.

Regimes use **verified `applied_policy_id` only** from the matching sizing publication.
Settings-interval matches, contradictory evidence and unavailable histories stay in
`<unknown>`. Partial settings can support a verified regime while bankroll remains
unknown. Bankroll-relative fields have their own availability count/status and are
not computed from current account settings or accumulated profit. Current source
stake remains reported stake, not proof of an accepted fill.

## Conditional event-proxy uncertainty

Both percentile intervals resample the same whole namespace/event clusters using
identical seeded draws. Weighted draws are summed profit / summed stake; equal-stake
draws are summed unit returns / summed candidate counts. The existing cluster
bootstrap implementation is reused. Draw order is deterministic for the same
eligible candidate IDs, policy and source inputs.

Unknown/missing event identities are never assigned independent synthetic clusters.
Intervals describe only the **common event-covered subset**, not the full point
estimate cohort. Publish covered counts/stake, missing counts, paired covered point
estimates and separate cohort fingerprints. Minimum event support and degenerate
bootstrap distributions produce explicit unavailable states. Finite intervals remain
conditional on event proxies that are not proven independent; timing and selection
limitations remain. No p-values, multiplicity-adjusted rankings, filter optimization
or validated edge claims are produced.

The support setting is not the dashboard's minimum-wager display control. This
stage publishes private diagnostic tables; it adds no dashboard rendering or sliders.
Current [visual datasets](VISUAL_DIAGNOSTICS.md) and the
[decision workspace](DASHBOARD_DECISION_WORKSPACE.md) consume these diagnostics;
the display policy remains independent of research eligibility.

## Explicit reproducible publication

```bash
poetry run python scripts/build_sizing_diagnostics.py \
  --canonical <verified-canonical-directory> \
  --sizing <matching-sizing-publication-directory> \
  --config configs/sizing_diagnostics.yaml
```

Both inputs must pin the **same exact canonical bundle and manifest hash**. Wager
identities, product/funding scope and source stake are reconciled before calculation.
The declared policy fixes notional unit, support, confidence level, bootstrap budget
and seed; it does not learn cutpoints or ranking rules from outcomes.

Private outputs live under `data/processed/sizing_diagnostics/<id>/`:
`candidate_diagnostics.parquet`, `cohort_metrics.parquet`, `audit.json` and manifest.
The manifest pins canonical and sizing evidence, diagnostic policy, implementation
(including the reused bootstrap) and runtime versions. The audit reconciles
partitions, eligibility and exclusions. Verified loading and deterministic replay
preserve previous publications. Source bundles and default pipeline/dashboard
behavior remain unchanged. No acquisition or scheduling is introduced.

Tests cover equal/unequal stakes, sign reversals, capital scaling, partial and
unknown regimes, missing events, event dependence, unsupported/invalid economics,
filter overlap, exact provenance, immutable sources and corrupt/replayed artifacts.
Descriptive plots can consume this contract in a later publication/rendering change.
