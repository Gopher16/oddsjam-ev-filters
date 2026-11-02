# Chronology and information availability

## Distinct clocks

| Time | What it establishes; what it does not |
| --- | --- |
| Receipt / recorded-at | When the local system received or appended evidence. Not the time a bet was placed or a setting became effective. |
| Knowledge cutoff / as-of | Which evidence is admissible in the selected snapshot/report. Not a reconstruction of pre-bet knowledge absent from source. |
| Effective time | When an evidenced configuration reportedly applied; half-open intervals include the start and exclude the end. Not proof it governed an execution. |
| Placement / observation | Decision or capture time only when semantics and timezone are evidenced. A generic source created timestamp is not automatically placement. |
| Event start | Scheduled source event time; no automatic live/pregame classification from an unverified subtraction. |
| Outcome availability / settlement | When a result became knowable or settlement occurred, separately evidenced. An event ending does not establish either. |
| Generation / publication | Successful build time. A replay can have a new processing attempt without fresher source data. |

## Time, leakage, and feature availability

Unknown source timezone stays unknown: raw timestamps remain available and UTC
values are null. A confirmed timezone or explicit offset allows UTC representation,
but does not establish the timestamp is an execution or quote-observation time.
The builder does not bypass PR 1's preprocessing timezone gate or change its
reconciliation precedence. Ticket timestamps require consensus; no earliest-leg
placement time is invented.

The ingestion manifest, its system as-of cutoff, source metadata, and resolution
history are pinned in the canonical specification. A canonical build only reads
that snapshot; later imports are not consulted. This prevents later snapshots
from silently replacing earlier observed state. It cannot recover pre-bet knowledge
that was never recorded. CLV and settlement outcomes are never declared safe
pre-bet predictors. All catalog entries have decision-time verification false.

`percentage` is catalogued as source-advertised EV percent, separate from the
legacy CLV-derived EV calculation and realized ROI. Raw liquidity remains missing
or untrusted; tags do not verify executable capacity. Player IDs, standalone line
values, settlement/update times, and a verified market taxonomy are not invented.

## Source-calendar views and prospective ordering

Literal source-date aggregation is retrospective attribution, not settlement cash
flow, account balance or bankroll growth. Calendar support, timezone status,
resolution, partial periods and missing-versus-zero periods must remain visible.
Weekday/hour and time-to-event claims need compatible evidenced timestamps and
calendar semantics; do not assume local timezone or DST rules. Active-day and
covered-day denominators answer different activity questions. See the
[visual publication contract](../VISUAL_DIAGNOSTICS.md).

Prospective capture must follow recorded actual launch/resume, occur within the
approved window and be recorded before event start. Every predicate feature has
availability no later than observation. Failed or late quote checks remain visible;
they cannot qualify as timely original-quote availability. Outcome corrections
append later knowledge to the original version; old cutoff reports remain intact.
Current settings never fill historical decision-time gaps.

Verified group-aware temporal folds require explicit windows, information
availability and embargo evidence. Unverified chronology blocks forward-validation
claims; source-calendar graphs and seeded grouped random splits do not repair it.
Scheduled review dates are records, not an installed scheduler.

Implementation: [temporal validation](../../src/oddsjam_ev/analysis/temporal_validation.py),
[visual features](../../src/oddsjam_ev/analysis/visual_features.py),
[prospective event validation](../../src/oddsjam_ev/analysis/prospective.py).
Verification: [temporal gates](../../tests/test_temporal_validation.py),
[visual chronology](../../tests/test_visual_diagnostics.py),
[prospective timing](../../tests/test_prospective_integration.py).

[Contract reference](README.md) · [Documentation home](../README.md)
