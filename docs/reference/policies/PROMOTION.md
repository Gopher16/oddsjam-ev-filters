# Approved prospective promotion and pilot policy

This is the authoritative prospective-policy reference, relocated from the
lifecycle guide with the approved gates, claim family and caps unchanged.
It does not inherit approval from [exploratory policies](EXPLORATORY.md).
Operational events and payloads remain in the
[reviewed lifecycle guide](../../REVIEWED_FILTER_LIFECYCLE.md).

## Review policies

Two explicit policy scopes are supported. Neither is inferred from positive ROI,
a PROD name, an elapsed date or a historical label.

- `descriptive_paper_review_only`: declared human criteria and monitoring rules;
  cannot approve production activation or modification. Use the checked-in
  [configs/filter_review_policy.example.json](../../../configs/filter_review_policy.example.json) as a template.
- `bounded_prod_pilot_review`: the prospective policy approved by the user on
  **November 2, 2026**. Use [configs/promotion_policy.example.json](../../../configs/promotion_policy.example.json), replacing its
  placeholder publication/version references with exact **future** definitions.
  The policy and complete claim family must be recorded before launch. The same
  definition version cannot be redeclared under another policy to retry a result.

The approved prospective policy requires all of the following:

1. Freeze every candidate definition and its trial arm before actual launch. Each
   definition belongs to one promotion family; record the full family, not only its
   eventual winners. At most 16 definitions and 50 million bootstrap cells are
   supported in one explicit build; budget failure publishes no partial evaluation.
2. Evaluate **once**, after all planned enrollment ends and settlement follow-up
   periods. The source reports must include actual end and scheduled-review evidence.
   Pending outcomes, unresolved quote availability, deviations and insufficient
   support remain blockers. A later result correction invalidates the old assessment
   for approvals; it does not permit repeated looks at the same study. Outcomes or
   settlement corrections recorded after the fixed terminal knowledge cutoff are
   blockers, even if the human review happens later.
3. Each subgroup **and** complement needs at least 100 settled eligible wagers and
   100 namespace/event groups, complete event IDs, positive finite paper stakes,
   no wager above 5% and no event above 10% of arm stake. Voids and pending outcomes
   do not establish settled support. Timely original-quote checks must cover the
   evaluated stake; an available quote is not a fill.
4. Use 5,000 deterministic paired namespace/event bootstrap draws for **marginal
   percentile intervals**. Directional p-values use event-cluster influence t
   approximations, as in the existing research engine. Apply Holm at 0.05 across
   **eight claims per definition**: both return measures × both directions ×
   absolute trial return and trial-minus-other-arm return. Failed/unavailable tests
   retain their full correction slots; they never become p=0. Intervals themselves
   are not simultaneous or Holm-adjusted.
5. Both measures must show corrected positive absolute trial return and the
   predeclared favorable comparison, with positive estimates and marginal interval
   lower bounds. Fixed-stake measures coincide; this does not make them independent
   confirmations. Event clustering and Holm do not prove event independence,
   causal effects, complete scanner capture, or the absence of selection bias.
6. Before approving settings, verify the exact complete configuration publication
   pinned by the tested definition: the after-settings must equal the tested trial
   sheet, and every changed scanner control needs exact state/value/UI-version
   evidence. A parameter tweak that was not tested needs a new paper study.
7. Passing permits **human review of a bounded PROD pilot only**. It does not
   approve a strategy, establish executable net profitability, or prove fills.
   Source links and outcome/quote evidence are operator attestations; hypothetical
   quote returns do not estimate unknown fees or execution slippage.

The initial pilot caps, selected at the user's request, are:

| Limit | Initial cap |
| --- | --- |
| Stake per wager | $1 |
| Cumulative deployed stake across recorded pilots in the journal | $100 |
| Duration of an individual pilot | 14 days from confirmed application |
| Concurrent confirmed pilots | 1 |

These are conservative engineering defaults, **not** bankroll-based affordability
advice, an allocation recommendation or a statistical power guarantee. Approval
requires an explicit human budget/duration attestation; choosing not to launch is
always possible. Gross deployed stake is counted without subtracting wins or void
refunds. Policy versions, filter names and rollbacks do not reset the recorded
program budget. The same tested definition cannot authorize repeated pilots.
An expired pilot stays externally unconfirmed until the operator actually retires
it; elapsed time does not fabricate a stop or free the concurrent-pilot slot.

The general principle of committing to a budget and time limit beforehand is
consistent with [NCPG's responsible-play guidance](https://www.ncpgambling.org/pa-resources/responsible-play-toolkit/).
That source does not establish the specific dollar/day values above. Local records
cannot enforce limits inside OddsJam or detect omitted trades. Observations that
show an external cap breach are retained and flagged, not hidden or interpreted as
permission. Platform limits and actual external actions remain the operator's task.

## Implementation and verification

[Promotion validator/evaluator](../../../src/oddsjam_ev/analysis/promotion.py)
validates the exact approved gates, caps, family and evaluation rule.
[Lifecycle writer](../../../src/oddsjam_ev/io/filter_lifecycle.py) checks source
freshness, tested settings and authorization again at approval and application.
See [promotion tests](../../../tests/test_promotion.py) and
[production lifecycle tests](../../../tests/test_production_lifecycle.py) for
complete-family, late-outcome, stale-evidence and cap enforcement cases.

[Contract reference](../README.md)
