# R<selected ID> — <baseline title>

Copy this template to the selected ID's plan. Replace placeholders with evidence;
do not mark template prompts as completed work. Use child plans for approved splits
while retaining the parent's history and unresolved scope.

## Entry review and source authority

- Date, selected ID, reason it is eligible and owner authorization: <evidence>.
- Baseline fingerprints and relevant source sections/card: <reviewed originals>.
- REQ/legacy IDs and decision/change links: <traceability>.
- Repository instructions/template reviewed: <paths/current constraints>.
- Branch, main SHA, worktree state and unrelated changes preserved: <actual state>.
- Dependency PR URLs, merge SHAs, main ancestry and post-merge checks: <verified>.
- Conditional prerequisites/feasibility/activation decisions: <state and evidence>.

## User problem, scope and existing capability

<Trigger and before/after behavior; goals; exact baseline deliverables/acceptance;
what is already delivered and reusable; what remains implementation versus missing
evidence. State exclusions and preserve all displaced work through destinations.>

## File-level design and compatibility

| File/component | Change and purpose | Existing interface to retain | Contract/evidence implications |
| --- | --- | --- | --- |
| <actual or proposed path> | <behavior> | <compatibility> | <version/schema/config/model/source impact> |

<Describe grains, economic populations/units/denominators, time/availability,
missing/conflicting evidence, empty/failure states, budgets, access/privacy and
migration/old-reader behavior where applicable. No contract change means state that.>

## Ordered commits

| Order | Conventional Commit message | Exact files/behavior and purpose | Verification / acceptance evidence |
| --- | --- | --- | --- |
| 1 | <docs/feat/fix(scope): concrete objective> | <bounded change> | <appropriate checks> |

<Save and present the main decisions before implementation. Proceed within selected
clear scope; record routine adjustments. Material changes require the decision
process before affected implementation, preserving original work.>

## Acceptance and validation plan

- Concrete user examples and expected results: <including sparse/unknown/failure>.
- Existing tests/reference fixtures and new tests justified by behavior: <paths>.
- Economic/identity/chronology/leakage/dependence/policy/replay checks: <if relevant>.
- Required current repository lock/lint/format/tests/diff/docs checks: <commands>.
- Notebook/private-data run: <intended snapshot and justification, or inapplicable>.
- Performance/accessibility/browser checks: <workload, target, or inapplicable>.
- Owner checkpoint: <blocking? URL/launch command, publication, selections, steps,
  expected result, limitation and actual confirmation required; or why unnecessary>.

## Rollout, operations and rollback

<Feature/publication/config activation boundary, preserved last usable state,
compatibility/migration, cancellation/retry/recovery, operational ownership,
observability, revert/deactivation and evidence needed. Selected PR merge is not
external deployment/account/purchase/trading authorization.>

## Deviations, questions and retained follow-ups

| Item | Evidence / baseline expectation | Options / proposed effect | State / authority | Original work destination |
| --- | --- | --- | --- | --- |
| <ID or none> | <facts> | <include retaining baseline> | <proposed/approved and actual reference> | <R/child/N requirement> |

## Implementation and validation record

<Append actual commit SHAs, behavior, checks/results, failures/resolutions, owner
review outcomes, versions, limitations and remaining work. Never claim a planned
check, missing source review or pending manual acceptance passed.>

## PR and completion record

- PR URL, target main and tested head: <actual>.
- Current-head remote checks, reviews, owner gates and main movement: <verified>.
- Merge result/SHA, main ancestry and post-merge checks: <actual or pending>.
- Documentation/operations handoff and rollback reference: <paths>.
- Requirement reconciliation and next eligible ID: <do not start automatically>.

Use the current repository PR template. Pre-merge status stays accurate; actual
merge SHA may be reported in the completion response/PR and reconciled in the next
authorized PR, avoiding self-SHA bookkeeping loops. Baseline Word files stay unchanged.
