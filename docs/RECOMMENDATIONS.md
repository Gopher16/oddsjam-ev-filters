# Research proposals and decision register

Recommendations are **research proposals**, not filter execution instructions.
The engine does not contact OddsJam, edit filters, create experiments, fund paper
tests, place wagers, or enable a weekly schedule. It works with manual imports and
has no dependency on the draft acquisition adapter. Existing notebooks, research
snapshots and preprocessing defaults are unchanged.

## Build from explicit linked inputs

From the repository root:

```bash
poetry run python scripts/build_recommendations.py \
  --report data/processed/discovery/102bb78305423fe24add23bd5d75dc2283e8980cfb83c643e458e72bd82ba51b \
  --atlas data/processed/atlas/198d2851aaf045f115124015c76f410ad6808ccf5f83cf674d3826b660ebb5e6 \
  --canonical data/processed/canonical/df99ad86af5f1c7b09b7fb93fdab8c039306217239fe528242cf39b30f886dc7
```

These bundles must already exist. The command verifies all source tables and the
cross-bundle identities and manifest hashes. Mismatched inputs fail rather than
combining evidence from different snapshots. Optional `--policy`, `--registry`,
`--capabilities` and `--output-root` select explicit alternatives. No latest-input
search occurs. The command prints a bundle path and action counts, not private
monetary evidence.

```python
from pathlib import Path
from oddsjam_ev.io.recommendations import load_recommendations

tables = load_recommendations(Path("data/processed/recommendations/BUNDLE_ID"))
proposals = tables["proposals"]
# Stable ID ordering is not a profitability or action-priority rank.
print(proposals[["proposal_id", "candidate_id", "action", "approval_allowed"]].head(20))
```

Inspect `next_steps_json`, evidence/review/approval blockers, exact predicate,
translation, and protocol alongside the original candidate evidence before making
a decision. Source IDs and policy/registry/capability evidence are pinned in the
manifest. Output hashes and row counts are verified on read. Staging and locking
prevent partial publication. Replay checks logical table equality. Source, policy,
capability, registry, or code changes produce a different bundle.

## Contracts

| Table | Grain / interpretation |
| --- | --- |
| `proposals` | One source-report/candidate/action/target proposal. Includes human-readable follow-ups, protocol, translation, evidence blockers, scope and initial status. |
| `candidate_evidence` | Exact discovery candidate result rows, retaining cohort, full/event-covered economics, missingness, sensitivity, test and uncertainty fields. No invented holdout results. |
| `coverage_audit` | One proposal's observed target/cohort counts and stake shares. Repeated actions and overlapping predicates are non-additive. |

Every table carries recommendation bundle ID and contract version. Proposal
identity binds source report, exact predicate, action, explicit target, policy,
capability evidence, registry and candidate evidence. Decisions additionally bind
the full recommendation bundle. An old approval cannot transfer to a changed
bundle, even where a semantic candidate ID remains stable.

Generated initial state is always `PROPOSED`. Human decisions are separate events;
building or replaying a report never creates or modifies approvals.

## Action policy

`configs/research_policy.yaml` permits research actions only:

| Action | Meaning |
| --- | --- |
| `COLLECT_MORE_DATA` | Resolve named missing evidence, including policy decisions as well as additional observations. |
| `REVIEW_MANUALLY` | Review unsupported translations, diagnostic categories, incomplete attribution or an unconfirmed target. |
| `CREATE_TEST` | Draft an unproven, paper-only prospective hypothesis; no experiment or live filter is created. |
| `NO_CHANGE` | Policy abstention where no permitted research action is emitted; never means a filter is validated. |
| `TIGHTEN_EXISTING` | Recognized action vocabulary but unsupported and prohibited by this contract. |
| `RETIRE_REDUNDANT` | Recognized vocabulary but prohibited: logged history does not establish broader scanner coverage. |

Production actions cannot be enabled by a configuration flag. They require a
separately designed future contract and supporting evidence. A high historical ROI,
positive interval, Holm flag, or altered upstream promotion flag cannot override
this rule. No scalar confidence score or profit-driven recommendation rank is used.

The source's unset stake threshold, blocked chronology, incomplete event evidence,
unknown fees and fillability, and exploratory interpretation remain explicit.
Positive-return tests do not establish that a negative segment should be excluded.
No new harm-test family is silently introduced.

Action policy may disable categories. All candidate evidence remains available
even when a candidate produces no allowed proposal. If only `NO_CHANGE` is enabled,
it records abstention while retaining blockers and required follow-ups.

## UI translation and target selection

`configs/filter_capabilities.yaml` starts with **no verified exact mappings**.
Historical screenshots are not asserted to prove current UI capabilities. Each
reviewed dimension/source-value mapping requires all of:

```yaml
ui_control: reviewed_control_name
ui_value: reviewed_exact_value
semantics_verified: true
evidence_reference: reference_to_actual_reviewed_evidence
```

Record mappings under `exact_values.<dimension>.<source value>`. These are external
review attestations; the program does not authenticate them or inspect OddsJam.
Do not set verification merely to bypass a blocker. The source advertised-EV units,
American odds, market sets and exact inclusive/exclusive boundaries must match.
A right-open atlas interval is not automatically equivalent to an inclusive UI
maximum; no epsilon, rounding or inferred market IDs are invented. Missing,
invalid, conflicting and unmapped categories remain research diagnostics.

Explicit `candidate_targets` map candidate IDs to stable historical registry IDs.
Names and ID prefixes never imply funding or target status. Planned, legacy,
deferred or otherwise unconfirmed lifecycle states do not become confirmed existing
filters. Stable IDs are preserved, and unknown target IDs are rejected. Even a
registry-confirmed existing filter requires current settings review before any
external action. Registry dates never establish historical effective settings.

## Paper protocol requirements

A draft paper proposal requires an actual nonempty predicate, reviewed exact UI
translation, eligible research observations, and either no selected target or an
existing TEST target. There is no positive-ROI selection requirement. The draft
can investigate an unproven hypothesis while temporal or economic validation is
blocked; it is not a validated strategy recommendation.

Without a complete protocol, `approval_allowed` is false. Optional `paper_protocols`
are keyed by exact candidate ID and require:

- A nonempty `hypothesis`.
- Explicit positive integer `duration_days` and `min_event_groups`.
- `primary_metric: reported_roi`, retaining fee limitations.
- `funding: paper_only` and `max_funded_stake_usd: 0`.
- Explicit nonempty `review_criteria`.

These fields are user decisions, not generated sample/risk defaults. They are
pinned in the proposal bundle; adding them creates new proposal identities.
Approval scope remains `draft_paper_protocol_only`. Experiment registration,
launch dates, controls, stop monitoring and outcome tracking are later work. No
approved protocol is automatically launched or funded.

## Coverage and overlap

Coverage uses unique canonical bet IDs within the proposal's execution/product
cohort and resolved filter memberships. Duplicate bridge rows cannot multiply
money. Eligible stake comes only from canonical wager facts. Audits show matching
and nonmatching target wagers, eligible counts and stake shares, other-filter
overlap, and incomplete attribution. Missing bridge rows count as unresolved.
Without an explicit target, target impact remains unavailable rather than zero.
Empty eligible stake yields null shares.

Matching a candidate is not an instruction to retain or remove it. The two sides
are observed historical partitions, not projected future scanner coverage or
causal profit improvement. Never sum audits across actions, predicates or filters.
Unknown opportunity coverage cannot justify retirement.

## Human review workflow

The private default register is `data/processed/decisions/`. No decision is recorded
by report generation. Review the exact bundle/proposal first, then explicitly:

```bash
poetry run python scripts/record_research_decision.py record \
  --bundle data/processed/recommendations/BUNDLE_ID \
  --proposal-id PROPOSAL_ID \
  --to-status UNDER_REVIEW \
  --reviewer "YOUR_NAME" \
  --rationale "Explain the research review being undertaken" \
  --expected-status PROPOSED
```

To approve the stated research scope, use the returned event digest:

```bash
poetry run python scripts/record_research_decision.py record \
  --bundle data/processed/recommendations/BUNDLE_ID \
  --proposal-id PROPOSAL_ID \
  --to-status APPROVED \
  --reviewer "YOUR_NAME" \
  --rationale "Explain why this specific research scope is approved" \
  --expected-status UNDER_REVIEW \
  --expected-event-id PREVIOUS_EVENT_ID
```

Inspect local history with:

```bash
poetry run python scripts/record_research_decision.py inspect
```

`--register` can select a different private root and must precede the subcommand.
Reviewer names and rationales are private local audit data; do not include secrets
or commit decision records. Actor attribution is explicit, not inferred from OS
identity, and is not authentication.

Transitions:

- `PROPOSED` → `UNDER_REVIEW`, `DEFERRED`, `REJECTED`, or `SUPERSEDED`.
- `UNDER_REVIEW` → `APPROVED`, `REJECTED`, `DEFERRED`, or `SUPERSEDED`.
- `DEFERRED` → `UNDER_REVIEW`, `REJECTED`, or `SUPERSEDED`.
- `APPROVED` or `REJECTED` → `SUPERSEDED` only.
- `SUPERSEDED` is terminal.

Direct approval from `PROPOSED` is rejected. Incomplete paper protocols cannot be
approved. Approval of an evidence-collection/manual-review proposal approves only
that research follow-up; it does not resolve the evidence gaps or authorize a
filter change. Approval always records `external_execution_performed: false`.

Each event records exact proposal/bundle/report IDs, previous status, requested
status, reviewer, rationale, UTC recording time, scope, sequence and previous-event
hash. Stale expected status or proposal-event digest is rejected. A writer lock
serializes appends; atomic exclusive links prevent replacement of an existing
event. Never remove a lock without first investigating a stale writer.

Hashes detect accidental corruption, gaps and reordering. This local audit trail
is **not tamper-proof**: a party able to rewrite all files can recompute hashes,
and deleting the final suffix cannot be proven without an external anchor.
Historical approval remains attached only to its original bundle. No active/live
filter state is inferred from this register.

## Validation and deferred work

Synthetic tests cover production-action rejection, unsupported/boundary UI
translations, stable IDs, planned targets, duplicate memberships, null denominators,
positive-result escalation prevention, funding/protocol gates, reproducible proposal
identity, stale approvals, invalid transitions, missing reviewers/rationales,
corruption, locks, failed publication and CLI integration. They need no private
credentials, network access or private historical records.

Deferred: verified current UI mappings, complete paper protocols, prospective
experiments, settlement/feature chronology, stake-floor approval, fee/fill evidence,
broader opportunity coverage and production-change policy. Weekly scheduling and
live OddsJam integration remain disabled. No notebooks are modified or executed.

### October validation evidence

Validated discovery report:
`102bb78305423fe24add23bd5d75dc2283e8980cfb83c643e458e72bd82ba51b`.

Published recommendation bundle:
`71ad1f0bfce6c058b8b66293b73108f4d1c18992b8e79dd5f2e0979cc42d0138`.

- All 1,965 original candidate evidence rows are preserved exactly.
- 3,930 proposals: 1,965 `COLLECT_MORE_DATA` and 1,965 `REVIEW_MANUALLY`.
- 3,930 observed coverage audit rows; no target is implicitly selected by the
  default policy, so target-specific impact is unavailable.
- No `CREATE_TEST` proposal is generated because reviewed exact UI mappings are
  absent. No tightening or retirement proposals are generated.
- Every proposal remains `PROPOSED`; production and live-execution permissions
  are false. All retain temporal and stake-threshold blockers with concrete next
  steps. No real decision event was recorded during validation.
- All three tables reproduced with logical equality on replay.
- 200 tests passed with FutureWarnings treated as errors. Poetry lock validation,
  Ruff lint/format checks and whitespace checks passed.

Notebook behavior, raw data, ingestion history, canonical/atlas/discovery bundles
and historical preprocessing defaults were preserved. No notebook was executed.
Generated reports remain Git-ignored; no credentials or network were required.
