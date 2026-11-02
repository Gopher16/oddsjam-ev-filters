# Prospective versions, review and applied changes

## Distinct records and authorizations

| Record | Contract and boundary |
| --- | --- |
| Definition version | Immutable candidate, baseline, full configuration reference, future dates, sizing, availability and review rules. Draft download/save is not approval. |
| Paper approval | Human decision for the exact proposal/version. Registration consumes that approval; registration is not launch. |
| Actual launch / resume | Separately recorded confirmation within the approved capture window and exact baseline. Elapsed planned dates do not launch a study. |
| Enrollment | Timely parent observation under the launched version with original features/quote and membership evidence. Never retrospective export enrollment. |
| Availability | Original selection/line/odds, timely capacity and evidence; changed/unavailable/unknown checks remain visible. Availability is not a fill. |
| Settlement | Explicit original-quote paper outcome or separately evidenced operator-reported funded result. No inferred outcome from event completion. |
| Scheduled review | Exact due date, version, current experiment head and immutable report, including blockers. Publication alone is not a performed review. |
| Reviewed decision | Continue/modify/stop/inconclusive for the exact report/version. Continue/modify need a separately frozen future definition, not pooling old observations. |
| Change and approval | Full before/after settings, exact baseline application, changed-control UI evidence and explicit human approval. Approval is not application. |
| Confirmed application | Exact approved settings/status and actual evidenced effective date; creates the application identity monitored thereafter. |
| Monitoring review | Human continue/pause/investigate/rollback/retest intent against an exact report. An alert never performs an external action. |

[The approved prospective promotion policy](policies/PROMOTION.md) is authoritative
for production eligibility and pilot caps. Descriptive paper review cannot approve
production. [Exploratory policies](policies/EXPLORATORY.md) do not confer eligibility.
Passing prospective gates only permits explicit human review of a bounded pilot.

## Version and time invariants

Capture every sampled parent observation, including subgroup and complement, under
the predeclared sampling policy. Missing predicate values are rejected rather than
assigned to complement. Duplicate keys/quote identities are checked across versions
of the same test key within the selected register; cross-register checks cannot
prove uniqueness. One program register is the supported operational pattern.
Manual completeness and membership remain operator attestations.

Pauses/deviations block captures. Source links preserve identity evidence but do
not import old historical profits or create enrollment. Definition-bound paper
settlement uses the captured original quote and predeclared fixed stake; pending
stays null, push is zero and void is excluded. Hypothetical and original-quote-checked
cohorts stay separate; parent totals overlap the arms. No cross-version pooling.

Settlement corrections append `supersedes_event_id` and `correction_reason`; they
remain tied to the original definition/application, including after closure or
rollback. Old reports remain immutable. Stale source heads/outcome evidence invalidate
dependent review/approval. A review amendment supplies `supersedes_review_event_id`
and a reason; it cannot create a second promotion look. Terminal knowledge-cutoff
corrections remain promotion blockers under the fixed-look policy.

## Applied periods, monitoring and rollback

A baseline records a preexisting operator-confirmed configuration; a legacy PROD
baseline is not a promoted candidate. A change preserves its exact before-settings,
proposed after-settings and semantic diff. Changed controls need matching state,
value, platform version and UI evidence. Untested changes require a new paper study.
Confirmation must match the approval and cannot fabricate a future/earlier application.

Monitoring observations pin the applied configuration hash and application ID.
Late outcomes stay on their original version. Newly known half-open boundaries can
make old captures ineligible; they are retained rather than reassigned. Paper and
operator-reported real-money cohorts remain distinct; the latter require fill and
reported-profit evidence, not theoretical quote profit. Drift uses predeclared
reference means/deltas and visible coverage; absent reference is unknown, not no drift.

Rollback is a new proposed, approved and confirmed application restoring an exact
prior configuration of the same filter. It does not rewrite old periods, reset
program pilot budgets or bypass production evidence gates. Closure is not external
retirement; elapsed expiry is not a confirmed stop. The repository never controls
the scanner, verifies authenticated identities or detects omitted external trades.

Payload/CLI authorities: [paper definitions](../PROSPECTIVE_PAPER_DEFINITIONS.md),
[prospective integration](../PROSPECTIVE_EXPERIMENT_INTEGRATION.md),
[reviewed lifecycle](../REVIEWED_FILTER_LIFECYCLE.md).
Implementation: [prospective validator/tables](../../src/oddsjam_ev/analysis/prospective.py),
[experiment register](../../src/oddsjam_ev/io/experiment_register.py),
[lifecycle writer](../../src/oddsjam_ev/io/filter_lifecycle.py).
Verification: [prospective integration](../../tests/test_prospective_integration.py),
[reviewed lifecycle](../../tests/test_filter_lifecycle.py),
[production guards](../../tests/test_production_lifecycle.py).

[Contract reference](README.md) · [Documentation home](../README.md)
