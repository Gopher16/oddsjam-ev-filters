# Programme decision register

This register tracks delivery decisions, not wagering/research decisions. Existing
[research decision journals](../REVIEW_QUEUES.md) and
[lifecycle records](../REVIEWED_FILTER_LIFECYCLE.md) retain their own schemas and
approval semantics. Architecture choices use [ADRs](../decisions/README.md).
A merge of this register does not approve the roadmap's conditional live capabilities.

## Dispositions and evidence

**Proposed** means awaiting an actual decision; **accepted** requires a named
source of authority/date/scope; **rejected**, **deferred** and **superseded** retain
reason, alternatives and destination. A technical implementation detail within an
already selected PR may be recorded as such; do not call it owner-approved policy.
Record the affected R/REQ IDs, original expectation, new evidence, options including
no change, adjustment/acceptance/dependency effects, original work's disposition,
actual decision and links. Append changes; never silently rewrite an old decision.

| ID | Date / state | Scope and decision | Authority / evidence | Preserved work and effect |
| --- | --- | --- | --- | --- |
| PD001 | 2026-11-23; accepted owner delivery instruction | Keep frozen Word baselines and record evolution in Git; save/present plan then proceed within selected PR, including push/PR/merge after gates | Applied reusable instructions, [source fingerprint](SOURCE_BASELINES.md); [workflow](DELIVERY_WORKFLOW.md) | Material deviations still require agreement; no deployment, purchases, account connections or wagering authority |
| PD002 | 2026-11-23; recorded source clarification | Kelly-sized paper entries and proposed TEST-only pass tags supersede the older unresolved sizing sentence for narrative documentation | Owner clarification and R&D §5.1, documented in [R01](pr-plans/R01.md) | Frozen source intact; no per-row historical reclassification |
| PD003 | 2026-11-23; routine R02 implementation within baseline option | Publish stable screenshot placeholders and A–G case references | R02 permits placeholders; [R02 plan](pr-plans/R02.md), [PR #35](https://github.com/Gopher16/oddsjam-ev-filters/pull/35) | Raw images remain private; future assets still need deliberate review |
| PD004 | 2026-11-23; routine R04 documentation interpretation | Preserve conditional dependency clauses and distinguish current implementation from planned branches | [R04 plan](pr-plans/R04.md), baseline R34/R58–R60, [change log](CHANGE_LOG.md) | No baseline work removed or automatically activated; partial/optional legacy work retains destinations |

| PD005 | 2026-11-24; accepted owner amendment | Adopt R&D §7, PR roadmap §12 and delivery agreement §10; register N001–N014 before R10 | Owner requested tracker addition, supplied expanded documents and applied them for next eligible PR; [versions](SOURCE_BASELINES.md), [N001 plan](pr-plans/N001.md) | Preserve R01–R60/REQ01–REQ50 and completed PRs. N001 is docs only; later N implementation needs selection/readiness. No account, deployment or execution activation |

## Decisions still required at their gates

These are pending baseline questions, not approved architecture decisions:
Streamlit versus API/React and acceptance budgets (R11); new evaluation/sequential
methods (R22); simulation/ruin definitions and probability/dependence assumptions
(R25–R28); feed access/cost/platform architecture (R34); execution access, risk and
unattended activation (R48–R51); fantasy product/rules scope (R52 onward); chosen
hosting and operating ownership (R59). Assess and record each in its selected PR;
this register does not preselect answers.

[Programme index](README.md) · [Change log](CHANGE_LOG.md) · [Opportunity backlog](OPPORTUNITY_BACKLOG.md)
