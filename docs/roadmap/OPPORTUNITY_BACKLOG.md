# Opportunity backlog

Historical state through R04: no N-series additions were prioritized. The dated
N001–N014 addition below is now adopted; only N001 is selected for this PR. Existing
baseline requirements retain R01–R60; missing baseline work is not a new idea.

For future additions use stable IDs such as N001 with discovery date, user problem,
evidence, relationship to baseline requirements, dependencies, disposition and a
linked decision before incorporating material scope. Splits retain their parent
(for example R26a/R26b); moved or deferred work keeps a destination and rationale.

[Delivery register](DELIVERY_REGISTER.md) · [Change log](CHANGE_LOG.md)

## Addition record

For each new N ID record discovery date, source/evidence, user outcome, relationship
to existing REQ/R IDs, options, dependencies, acceptance/operating effects, decision
state/authority and intended destination. Proposed does not mean prioritized or
approved. Baseline missing work stays under its existing IDs; do not relabel it new.

Legacy widening, sensitivity/calibration and replay extensions are preserved in the
[capability crosswalk](CAPABILITY_RECONCILIATION.md). Before material work beyond a
selected baseline card, add its concrete proposal here and link an actual decision.
No extension is silently implemented merely because the crosswalk mentions it.

[Decision register](DECISION_REGISTER.md) · [Delivery workflow](DELIVERY_WORKFLOW.md)

## 2026-11-24 — personal tracker addition adopted

**Discovery/evidence:** owner requested a separate Pikkit-inspired Bet Tracker after
R09 and before R10, supplied 14 private screenshots, then applied the expanded
roadmaps/delivery agreement in the current next-eligible-PR instruction. User outcome:
one source-linked place to review wagers/performance/cash and enter linked research
and risk investigation without copying accounting or hiding missing data.

**Options:** leave the idea unprioritized (would disregard explicit adoption request);
expand/reopen R09 or R10 (would conflict with completed scope and overload refresh);
adopt a separate N stream (selected by owner amendment). **Disposition:** PD005
accepts additive requirements/scheduling, not implementation of all features.
N001 docs is selected now. N002–N014 remain planned/conditional, each requiring
selection and readiness. No original R/REQ work is removed, deferred or renumbered.

| ID | Destination and outcome | Disposition |
| --- | --- | --- |
| [N001](DELIVERY_REGISTER.md#n001) | Adopt the tracker addition and reconcile the programme | Selected adoption PR |
| [N002](DELIVERY_REGISTER.md#n002) | Define the source-linked tracker ledger and annotation extensions | Adopted planned scope; conditional gates apply |
| [N003](DELIVERY_REGISTER.md#n003) | Publish shared tracker accounting queries and filter context | Adopted planned scope; conditional gates apply |
| [N004](DELIVERY_REGISTER.md#n004) | Deliver the dedicated Bet Tracker shell and searchable ledger | Adopted planned scope; conditional gates apply |
| [N005](DELIVERY_REGISTER.md#n005) | Add linked performance, date/season controls and calendars | Adopted planned scope; conditional gates apply |
| [N006](DELIVERY_REGISTER.md#n006) | Add auditable tags, saved views and cohort comparisons | Adopted planned scope; conditional gates apply |
| [N007](DELIVERY_REGISTER.md#n007) | Add account balances, cash movements, at-risk capital and promotions | Adopted planned scope; conditional gates apply |
| [N008](DELIVERY_REGISTER.md#n008) | Deliver qualified closing-reference and actual-return diagnostics | Adopted planned scope; conditional gates apply |
| [N009](DELIVERY_REGISTER.md#n009) | Connect position scenarios, ladders, duplicates and conflicts | Adopted planned scope; conditional gates apply |
| [N010](DELIVERY_REGISTER.md#n010) | Add historical event browsing and conditional quote review | Adopted planned scope; conditional gates apply |
| [N011](DELIVERY_REGISTER.md#n011) | Add evidence-grounded search and an optional tracker assistant | Adopted planned scope; conditional gates apply |
| [N012](DELIVERY_REGISTER.md#n012) | Complete preferences, safe exports and privacy-aware operations | Adopted planned scope; conditional gates apply |
| [N013](DELIVERY_REGISTER.md#n013) | Evaluate and implement supported account/source synchronization | Adopted planned scope; conditional gates apply |
| [N014](DELIVERY_REGISTER.md#n014) | Accept the integrated tracker and reconcile all planned coverage | Adopted planned scope; conditional gates apply |

Dependencies/acceptance/insertion windows and operating limits are in the
[tracker scope/cards](TRACKER_ADDITION.md), [TRK requirements](REQUIREMENTS.md#tracker-requirement-coverage)
and [delivery register](DELIVERY_REGISTER.md#additive-tracker-stream--n001n014).
All references are public-safe prose; raw captures/account amounts remain private.
No UI, new engine, connected account, paid feed, external AI or execution is activated.
Future additions start at the next unused N ID (currently N015); preserve child IDs
and current statuses rather than relabelling baseline work as new opportunities.
