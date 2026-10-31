# Complete decision and configuration records

This optional publication binds a historical candidate to a complete **prospective
setup sheet**. It does not rank new opportunities, recalculate historical returns,
confirm a fill, approve a test or apply a production change. Every known control is
shown even when the evidence is missing. A ready sheet means ready for operator
configuration review, not permission to deploy.

## States and provenance

`analysis/decision_config.py` defines the complete control catalog: Recommended
Filters, formula builder, devig books/weights/method/model version, platform,
sport/league/market types/exact markets/games, odds and EV endpoints, liquidity,
minimum recommended bet, market width, date range, pregame/live/time-to-event,
timezone, warning behavior, paper sizing and bankroll/Kelly context.

Each input setting has an explicit state:

| State | Interpretation |
| --- | --- |
| `observed` | A sourced value, not a guessed default. |
| `unrestricted` | A deliberate lack of an additional restriction within the prospective universe. |
| `unset` | Confirmed inactive control; value must be null, not zero. |
| `platform_default` | Verified default with platform/version evidence. |
| `not_applicable` | Explicit reason/evidence for inapplicability. |
| `unknown` | Missing evidence; blocks complete setup readiness. |

Output sheets add `inherited` and `override`. Inheritance preserves the input state,
value and baseline fingerprint in separate columns. A null inherited value may be
confirmed unset/unrestricted; it is not silently converted to zero or unknown.
Unknown is never promoted to unset, unrestricted or default.

Known input states require a reason, evidence reference, observation time and a basis:
`historical_observation`, `prospective_design`, `operational_constraint` or
`platform_requirement`. Observation time cannot follow recording time. The basis
explains why the value was chosen, not proof of its profitability. Unrestricted logged
history does not establish that an unrestricted scanner is profitable.

## Explicit evidence input

Copy `configs/decision_evidence.yaml` to ignored `data/processed`. Its empty defaults
intentionally assert no baseline, scanner translation or production lifecycle.
Populate only supported evidence. A partial baseline is valid, but omitted controls
remain unknown and block readiness. Never copy today's values into old snapshots.

Example input (illustrative only):

```yaml
contract_version: 1
baselines:
  - baseline_id: my-parent-settings-v1
    filter_id: '<exact-registry-filter-id>'
    kind: prospective_parent
    recorded_at: '<aware-recording-time>'
    effective_from: '<aware-effective-start>'
    effective_until: null
    evidence_reference: '<current-settings-capture-and-operator-context>'
    settings:
      min_ev:
        state: observed
        value: {value: 2.5, operator: '>=', units: percentage_points}
        reason: Captured parent setting
        basis: prospective_design
      minimum_liquidity:
        state: unset
        value: null
        reason: Operator confirmed this control is inactive
        basis: operational_constraint
      paper_sizing:
        state: observed
        value: {method: fixed_paper_stake, stake_usd: 1, applies_to: both_arms}
        reason: Explicit fixed-unit paper design
        basis: prospective_design
lifecycle: []
translations: []
```

Field-level `observed_at` and `evidence_reference` may override baseline metadata.
USD controls are numeric USD values, odds/EV bounds carry explicit units/operators,
and false/zero are not missing. Known string/list/boolean/map settings are type checked.
Defaults require `platform_version`. Full settings and evidence remain private.

Baselines are explicit prospective records (`prospective_parent`), not historical
registry snapshots. `new_prospective` is reserved for a deliberately new global
universe; a global analytical slice cannot inherit a convenient parent. The current
publication builder consumes the existing within-filter family only; the pure
resolver supports explicit new-universe scope for future global adapters.

Conflicting applicable baselines block resolution rather than choosing the latest.
Changing any input, effective time, knowledge cutoff or implementation produces a
new publication identity. Preserve old files.

## Effective-dated lifecycle

Lifecycle records require exact `filter_id`, `status` (`PROD`, `TEST`, `retired`,
`unknown`), `recorded_at`, `effective_from`, `effective_until`, `actor`,
`evidence_reference` and `confirmation: operator_confirmed`. Effectiveness is a
half-open interval. Only evidence known by the explicit cutoff is used. Conflicting
applicable statuses resolve to unknown/conflicting, not whichever sounds actionable.
A PROD/TEST name prefix, historical funding label or selected screenshot checkbox is
not an independent confirmation of deployed production use. Status is as of the
specified effective time, not a live account query.

## Exact candidate overrides and exclusions

The candidate predicate remains the historical analytical rule. Scanner translations
require exact `term`, target `control`, typed `value`, `recorded_at`,
`evidence_reference`, `platform_version` and `semantics_verified: true`. This is an
explicit supported operator/UI mapping, not independent verification by the program.
Only translations known by the cutoff apply. Multiple matches block rather than
silently choosing one. Unmapped dimensions remain blocked.

For example, a market family must be mapped to its exact scanner market set. Book
aliases require explicit mapping. A `>3%` analytical rule cannot become `>=3%` or
`>=3.01%`; a UI that cannot express the exact endpoint needs a separate reviewed
implementation/capture design. Odds and EV bands retain inclusive/exclusive bounds.
The resolver intersects supported overrides with baseline restrictions; it never
weakens a stricter parent or silently widens a known platform/market universe.

For `exclusion_comparison`, the logical change is **NOT(the entire conjunction)**
within the parent. Excluding Novig totals is not equivalent to excluding all Novig
bets or all totals. This initial contract deliberately leaves compound scanner
exclusion implementation blocked. The sheet shows the unchanged baseline controls,
exact logical exclusion, production confirmation and unresolved implementation.
Future verified split-filter/implementation adapters must preserve that logic.

## Publish and view

```bash
poetry run python scripts/build_decision_records.py \
  --discovery data/processed/within_filter_discovery/<id> \
  --prioritization data/processed/signal_prioritization/<id> \
  --evidence data/processed/my-decision-evidence.yaml \
  --as-of '<aware-knowledge-cutoff>' \
  --effective-at '<aware-effective-time-at-or-before-cutoff>'
```

The builder verifies exact source/diagnostic lineage, preserves the complete source
family and publishes two design sheets per candidate. These are available setup
variants, not two recommendations. Existing categories/statistics are copied with
source references, never reinterpreted as performance of the proposed configuration.
The immutable manifest pins raw evidence, code, input hashes and times. JSON records
include baseline fingerprints, each control's before/change/after values and states,
setup blockers, lifecycle evidence and an individual record hash. Replay verifies
all output bytes. Publications remain under ignored `data/processed/decision_records`.

Set `decision_records_publication: decision_records/<id>` in dashboard configuration
(relative to `evidence_root`). **Within-filter evidence → Complete prospective setup**
shows each candidate's sheet and design selector, plus download. Missing attachments
leave old reports usable. Root/symlink confinement, exact report/source linkage and
reverification prevent silent fallback or stale cached sheets.

Legacy Signals & leaks expanders are labeled historical subgroup definitions and
explain that omitted controls are not defaults. Legacy paper definitions remain
readable, with an untyped-settings notice; they do not acquire complete setup status.

## Configuration-bound paper definitions

A ready subgroup sheet can export a draft with `configuration_reference` (contract
version, publication ID, decision ID, record hash). Finish dates/review fields and
save via `manage_paper_definitions.py save --decision-publication <directory>` plus
the existing required arguments. The extra reference is a backwards-compatible
protocol extension; old event/publication bytes and readers remain usable.

Saving re-verifies the actual publication, source candidate/design, baseline values,
evidence reference, effective enrollment window and paper sizing. Copying a ready
flag or editing a draft reference cannot bypass setup blockers. Changed configuration
requires a new decision publication and paper definition version. Bound definitions
cannot use future-known configuration evidence. Legacy unbound definitions still
register under their existing contract but remain explicitly unverified as complete
scanner setups.

## Boundaries and follow-up

This implements plan A. It does not infer missing historical configuration, optimize
unknown knobs from outcomes, alter statistical thresholds, deploy filters, train
explanatory models or deliver the future redesigned home page. See
`plans/DASHBOARD_DECISION_WORKSPACE.md` for the updated A–I sequence, including
notebook visuals, all-history prioritization, SHAP model/exploration workstreams and
prospective experiment integration. No new ML dependency is introduced here.
