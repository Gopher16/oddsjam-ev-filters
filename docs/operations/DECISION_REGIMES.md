# Funding, selection and recording regimes

This guide translates the owner's November 2026 process into analytical boundaries,
reviewed 2026-11-23. Sources are the frozen R&D roadmap §3, the clarified §5.1 and
[dated workflow](ODDSJAM_WORKFLOW.md), with [cases A–G](case-studies/README.md).
It records context, not machine-readable assignments to historical bets. R05 will
own the roadmap's versioned regime evidence; existing publications stay unchanged.

## Keep four questions separate

| Question | Evidence needed | Unsafe shortcut |
| --- | --- | --- |
| Was money actually at risk? | Dated funding/execution evidence, retained with provenance | A PROD name, terminal outcome or positive stake proves a funded wager |
| Which opportunities were selected? | Capture coverage and decision process for that period/filter | All logged TEST bets represent the owner's discretionary trading decisions |
| How was the amount recorded? | Suggested stake, sizing settings and verified applied-policy evidence | A Kelly-sized entry proves executable capacity or a historical bankroll balance |
| What was actually accepted? | Venue order/fill/cancellation records tied to the selected price and time | A displayed blended quote or paper entry proves an executable fill |

An economically eligible candidate can still have unknown funding. A known funded
record does not by itself prove which configuration or sizing policy governed it.
Keep unknown/mixed/conflicting evidence visible; do not relabel it to make a chart
look complete. [Source inventory](SOURCE_FIELD_INVENTORY.md) and
[evidence gaps](EVIDENCE_GAPS.md) describe what must be collected.

## The owner's three processes

| Process | What the owner reports recording | Appropriate interpretation |
| --- | --- | --- |
| Older broad TEST capture | Nearly every surfaced opportunity was logged, generally using displayed price and Kelly-suggested amount, without applying the later discretionary execution review | Broad hypothetical tracking conditional on what was surfaced and recorded; not an exhaustive market/opportunity universe or an executable portfolio |
| Newer discretionary TEST paper evaluation | Apply the same manual reasoning used for funded trades before recording a hypothetical execution; every paper-executed entry uses the Kelly recommendation for sizing | Hypothetical performance of selected entries; suggested amount can exceed available capacity, and a paper entry does not demonstrate a venue fill |
| Actual funded executions | Review the opportunity, inspect the venue, execute if acceptable and reconcile actual price/accepted stake in Bet Tracker | Source-reported execution history subject to evidence/fee/settlement limitations; logged rows are not independent verification of complete fills |

The Core Exchange Conservative strategy was renamed PROD → TEST on **November 10,
2026**, coinciding with stopping **all real-money execution under that strategy,
with no exceptions**. Subsequent use is discretionary paper evaluation. This is an
owner-confirmed date-granularity transition, not an exact intraday timestamp or a
statement that all accounts, other strategies or all TEST recording changed then.
The start/end of older broad TEST practice for each filter remains an evidence task.
Boundary-day rows without sufficient evidence must not receive invented timestamps.

R&D §3 contains an earlier unresolved paper-sizing statement. The owner's direct
clarification and §5.1 specify Kelly-sized paper entries. [R01's source decision](../roadmap/pr-plans/R01.md)
records this resolution; neither frozen Word document is edited. Likewise, the
reported $20,000/quarter-Kelly UI configuration is dated context, not an applied
policy for every historical record. See [sizing history](../SIZING_HISTORY.md).

## Passes, paper executions and later reconsideration

Passes are **not currently logged** in Bet Tracker. The proposed low-friction
improvement is a single **TEST-only** custom tag, such as `Passed` or
`Trade not Executed`. It would describe a decision observation, separate from paper
executions and funded wagers in reporting. Reason tags could follow only if their
use does not create excessive friction. This proposal is not evidence that pass
records already exist or an implemented import/classification rule.

For PROD opportunities the owner wants to leave a pass unlogged, avoiding the green
Previously Placed checkmark so the opportunity can be reconsidered if its outlook
improves. Consequently neither the PROD pass rate nor all reviewed opportunities
can be reconstructed from current tracker history. A later execution is a later
decision; it must not erase an earlier observation if future capture records both.

A paper execution always records Kelly-recommended size under the clarified
practice. Plausibly fillable size, quote time, actual price levels and capacity
uncertainty should be captured separately in future evidence. Do not silently cap
old paper stakes, infer accepted orders or mix proposed passes into wager ROI.

## Practical analytical use

Before comparing regimes, pin the source snapshot, namespace and known coverage;
then inspect funding evidence, selection-period evidence, settings versions and
membership resolution. A source export may span several regimes. Current labels
are insufficient for reassignment: preserve the original snapshot classifications
and publish any later approved evidence/reinterpretation as a new version. Unknown
or contradictory evidence remains unresolved rather than defaulting to paper or
funded. See [ingestion](../INGESTION.md),
[versioned attribution](../VERSIONED_FILTER_ATTRIBUTION.md) and
[chronology](../reference/CHRONOLOGY.md).

Compare funded, paper and unknown cohorts separately. Stratify old broad versus
new discretionary paper cohorts only when period/filter evidence supports that
split, reporting the unresolved remainder. Show eligible and excluded counts,
matched metric denominators, unique-event coverage and uncertainty. Selection,
stake size, sport/venue mix and time period can explain differences; a better
post-transition ROI does not prove manual judgment caused improvement. Overlapping
filter memberships and repeated selections are not additive independent evidence.

Support confirmed that selecting Pinnacle rather than Recommended Weights uses
only Pinnacle's prices in the documented configuration. It does not verify old
per-row settings. Support also confirmed **automated exports are unavailable at
that time**: use manual browser exports with declared provenance. Chrome success
and Python's Cloudflare challenge do not establish a supported automation route;
no bypass or automated acquisition is introduced here.

## Synthetic interpretation checks

These examples are fictional interpretations of existing tests, not owner data.
They illustrate why unresolved evidence must survive into reporting.

| Evidence situation | Required interpretation | Existing verification |
| --- | --- | --- |
| A settled single has reported profit but unknown source timezone | Report supported economics under policy; retain unknown chronology and unverified fees | `test_singles_preserve_profit_and_unknown_time` in [bet-facts tests](../../tests/test_bet_facts.py) |
| Profit is absent, or only an incomplete/conflicting ticket is available | Missing is not zero; do not fabricate complete ticket payout or include unsupported economics | Missing-profit, single-leg and conflicting-ticket cases in [bet-facts tests](../../tests/test_bet_facts.py) |
| A wager has a source-created date inside a sizing-settings interval | Do not infer verified placement or that those settings were applied | `test_placement_required_half_open_and_no_applied_inference` in [sizing-history tests](../../tests/test_sizing_history.py) |
| Two policies overlap, or placement evidence contradicts itself | Retain ambiguous policy/placement state, blocking an exact sizing replay | `test_overlap_and_contradictory_placement_quarantine` in [sizing-history tests](../../tests/test_sizing_history.py) |
| A selection appears across multiple prices/venues or filter labels | Preserve executions/memberships; proxy grouping does not establish independent executable opportunities | [Canonical-link tests](../../tests/test_canonical_links.py) and [attribution tests](../../tests/test_filter_attribution.py) |

[Operating workflow](ODDSJAM_WORKFLOW.md) · [Source inventory](SOURCE_FIELD_INVENTORY.md) ·
[Evidence tasks](EVIDENCE_GAPS.md) · [Documentation home](../README.md)
