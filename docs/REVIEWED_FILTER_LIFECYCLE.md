# Reviewed outcomes, applied configuration versions and monitoring

The workflow now records the entire local chain: frozen paper definition → explicit
scheduled review → human continue/modify/stop/inconclusive decision → exact proposed
settings → human approval → confirmed external application → version-specific
monitoring → rollback/retest/stop review. No command operates OddsJam, places a bet,
starts a scheduler or automatically deploys/rolls back a filter.

## Review policies

Two explicit policy scopes are supported. Neither is inferred from positive ROI,
a PROD name, an elapsed date or a historical label.

- `descriptive_paper_review_only`: declared human criteria and monitoring rules;
  cannot approve production activation or modification. Use the checked-in
  `configs/filter_review_policy.example.json` as a template.
- `bounded_prod_pilot_review`: the prospective policy approved by the user on
  **November 2, 2026**. Use `configs/promotion_policy.example.json`, replacing its
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

## Explicit commands

All payloads, journals and publications belong under ignored `data/processed/`
paths. Example configurations contain no real approvals, registrations or trades.

```bash
poetry run python scripts/manage_filter_lifecycle.py \
  --register data/processed/filter_lifecycle inspect

poetry run python scripts/manage_filter_lifecycle.py \
  --register data/processed/filter_lifecycle record \
  --kind KIND --payload data/processed/payloads/event.json \
  --expected-event-id CURRENT_GLOBAL_JOURNAL_HEAD \
  --actor 'Your name' --rationale 'The evidence and reason for this action'
```

Omit `--expected-event-id` only for the first event. Recording time always comes
from the system; it cannot be backdated through the CLI. The hash chain and explicit
head guard detect accidental corruption and concurrent updates. Actor/evidence
references are attestations, not authenticated identities; detecting tail deletion
requires an external head anchor. One journal should cover the whole program.
Cross-journal duplicate/risk checks are not possible.

### Policy, terminal assessment and reviewed outcome

| Event | Required evidence and result |
| --- | --- |
| `POLICY` | Complete policy object, review/error-control text, all four action criteria, monitoring stake/support/reference/alert rules. Production scope additionally requires the exact reviewed gates/caps and future family references. New versions append; old policies remain pinned. |
| `EVALUATE_PROMOTION` | Production `policy_event_id` and complete `sources` list. Each source names `experiment_register`, `experiment_id`, `review_report`, `review_event_id`. Verified reports must include the completed scheduled review and current exact experiment head. All family members must be supplied. The journal records one deterministic terminal assessment, hypotheses and source anchors. |
| `DECISION` | `policy_event_id`, `experiment_register`, `experiment_id`, `review_report`, `review_event_id`, `action`, `evidence_reference`. Continue/modify also require `followup_publication` and `followup_version_id` naming a new future frozen definition. A replacement decision must name `supersedes_decision_id`. |

The original paper workflow remains documented in
[Prospective experiment integration](PROSPECTIVE_EXPERIMENT_INTEGRATION.md). Report
publication alone is not a review. A losing subgroup does not automatically become
an exclusion and insufficient support never becomes an automatic pass. Stop and
inconclusive decisions are still recordable without a statistically favorable result.

A new definition does not launch itself. Continue means continue under a new version;
modify means a separately frozen future rule set. Old observations remain with their
old definition, including corrections recorded later. Reviewed decisions are anchored
to the current report and exact experiment head; subsequent evidence requires a new
reviewed decision and cannot silently support the earlier approval. For a late
outcome correction, first amend the scheduled experiment `REVIEW` with
`supersedes_review_event_id` naming the latest review and `correction_reason`.
Its pinned report must include the current evidence. The amendment appends new
review provenance; it does not grant a second promotion evaluation.

To close an old experiment explicitly, use the existing experiment `TRANSITION` to
`CLOSED` from `READY_FOR_REVIEW`, supplying `review_decision_reference` (an immutable
lifecycle report), `review_decision_event_id`, and evidence reference. Closure records
the reviewed action. It does not retire an external scanner. A properly linked final
closure is accepted without treating it as changed outcome data.

### Before/after configuration sheets and actual application

- `BASELINE` imports one **preexisting operator-confirmed** configuration per filter:
  filter ID, TEST/PROD/retired status, effective date, complete-or-explicitly-unknown
  settings, declared policy, legacy context and evidence. A PROD baseline is labeled
  legacy/unknown approval; importing it never promotes a paper candidate. A name
  alone cannot confirm PROD or reconstruct historical settings.
- `CHANGE` pins a reviewed `decision_event_id`, target filter ID/status, change kind
  (`new_test`, `modify`, `stop`, `rollback`, `promote`), exact
  `previous_application_id` (null for a new filter), planned effective UTC date,
  complete `after_settings`, UI evidence and reason. The writer preserves the exact
  applied before-settings and a semantic state/value/platform-version diff.
- Every control uses the same setting contract as
  [Complete decision/configuration records](DECISION_CONFIGURATION_RECORDS.md).
  Omitted controls remain unknown; unset, unrestricted, not applicable and verified
  platform defaults are distinct. Unknown target settings block approval. For paper
  targets, fixed sizing must match the declared monitoring policy and future
  definition. No extra feature/setting names are silently translated.
- Each changed-control UI evidence row names `control`, `state`, `value`,
  `semantics_verified: true`, `platform_version`, `evidence_reference`, and aware
  `observed_at_utc`. Its state/value must match the proposed setting exactly.
- A production request also supplies `promotion_assessment_event_id`,
  `configuration_publication` (the original tested parameter sheet), and an explicit
  `pilot_budget_attestation`. The writer checks full-family freshness, qualification,
  tested settings, active pilots and remaining recorded budget. Blocked proposals
  remain visible but cannot be approved.
- `APPROVE_CHANGE` is an explicit human action referencing the exact change and
  evidence. Approval does not establish an application date or current settings.
  `REJECT_CHANGE` can withdraw a proposal/unused approval; it cannot undo an already
  applied configuration. Changes to source decisions, outcomes or applied baselines
  invalidate dependent approvals. Production conditions are checked again at approval
  and application, not merely when the proposal was written.
- `CONFIRM_APPLICATION` references the approved change and approval event, supplies
  actual `filter_id`, `status`, `settings`, `effective_at_utc`, and external evidence.
  Identity/status/settings must match the approval. Actual dates cannot be future
  dates, precede approval/planned date, or overlap a previous application start.
  The new application ID is the version identity for subsequent monitoring.

A rollback is a newly proposed/approved/confirmed application of an exact earlier
configuration of the **same** filter. It names `rollback_application_id`; it does
not rewrite the earlier interval or relabel intervening observations. Rolling a
filter back to PROD still requires eligible, tested production evidence. Retest
requests do not register or launch studies implicitly.

## Monitoring and late outcomes

`OBSERVE` requires a current non-retired application ID and exact settings hash,
a stable observation key, namespace/event identity, aware observation/event times,
selection, line, American odds, stake, execution mode, availability status and
source evidence. Recording must follow the recorded application confirmation and
precede event start. Do not backfill historical bets into this prospective record.
Duplicate observation keys across application versions are rejected.

Execution modes remain separate:

- `paper`: use the policy's predeclared fixed stake. Paper settlements derive unit
  returns from the captured original quote; they are hypothetical and do not infer
  execution or fees.
- `operator_reported_real_money`: requires an explicit PROD application and operator
  fill reference. Settlement requires independently supplied **reported net profit**
  and its evidence reference; the code never substitutes theoretical quote profit.
  This is operator-reported economics, not independent fill verification.

Record `not_checked`, `checked_unavailable`, `inconclusive` or `checked_available`.
Failed/inconclusive checks need a reason. Available observations need an
`availability_check` with contemporaneous UTC time, matching selection/line/odds,
sufficient `available_stake_usd`, and evidence reference. Changed quotes and missing
capacity cannot enter the original-quote-checked cohort. Capture optional finite
advertised-EV features in percentage points; odds and time-to-event minutes are
recorded/derived from decision-time evidence.

`SETTLE` names `observation_event_id`, explicit result (won/lost/push/void/pending),
aware `outcome_available_at_utc` and evidence. Pending profit stays unknown, pushes
are zero and voids are excluded from settled return denominators. Corrections append
`supersedes_event_id` plus `correction_reason`. A late outcome stays with the original
application version even after rollback or retirement. If a later application is
confirmed with an earlier effective date, older captures outside their version's
newly known half-open interval remain visible as ineligible; they are not reassigned
to the replacement version.

Monitoring tables separate application version, execution mode and availability
cohort. They report counts, pending/void results, event breadth, weighted ROI and
mean per-wager unit return. ROI is a fraction (`0.03 = 3%`). Fixed-stake measures
coincide by construction. Event groups are not automatically independent and no
cross-version or paper/funded pooling is performed.

Drift checks compare means with **explicitly declared** reference means and absolute
alert deltas for advertised EV, American odds and time-to-event minutes. Empty
references or no observations yield unknown, not “no drift.” Counts and denominators
show coverage. These are descriptive checks, not learned thresholds or new hypothesis
tests. Alerts never deploy, stop, roll back or authorize a retest.

## Publish, review and attach

```bash
poetry run python scripts/build_filter_lifecycle_report.py \
  --register data/processed/filter_lifecycle \
  --as-of 'ACTUAL_AWARE_KNOWLEDGE_CUTOFF' \
  --output-root data/processed/filter_lifecycle_reports
```

Publications pin policies, source/report hashes, all human events, exact settings,
application intervals, monitoring observations/settlements and code hashes.
Promotion assessments additionally pin the evaluation code hashes and NumPy/pandas/
SciPy versions at evaluation time; publishing a later report does not relabel that
statistical provenance. Future
cutoffs are rejected and old outputs are preserved. A `MONITOR_REVIEW` names the
exact current `monitoring_report`, `application_id`, one of
continue/pause/investigate/rollback/retest, and explicit human evidence/reason. It
records intent; the subsequent actual change still needs its own approval/application.

In the trusted dashboard YAML, set:

```yaml
lifecycle_report_publication: filter_lifecycle_reports/PUBLICATION_ID
```

The selected definition shows reviewed outcomes, approval blockers, complete before/
after settings, confirmed configuration periods, subsequent economics and drift
coverage. Pilot caps, expiry review flags and recorded breaches are visible. The
main decision workspace links the exact candidate to its reviewed follow-up. Files
are verified under the trusted evidence root on each load. A missing/corrupt attachment
clears results without reading a live journal or substituting another publication.
Definitions are joined by exact immutable identity, never similar names.

No real policy registration, study approval, scanner application or trade was created
while implementing this feature. The included positive, negative, ambiguous, rollback,
missingness and cap-breach examples are synthetic. End-to-end release/browser/design
acceptance remains the next planned workstream.
