# Case G — Small blended quote obscures substantial nearby depth

**Source:** R&D §2.5; November 11, 2026 opportunity on Novig. This follows the
November 10 rename from PROD to **TEST - Exchanges: Core Exchange Conservative
(Low Risk, Slow Build)** and cessation of **all real-money execution under this
strategy, with no exceptions**. This is a paper-evaluation pass, not a funded trade.
Original screenshot capture time is unknown.

## Opportunity and depth outside the blend

UMass Lowell vs. Massachusetts, **Under 150.5**, displayed **5.12% advertised EV**
and approximately **45¢ / +121 American blended odds**. The source describes:

| Scope | Quoted context |
| --- | --- |
| Intended-side contributing levels | $95 at 45.0¢ and $129 at 45.5¢ |
| Nearby intended-side order | **$5,629 at 46.0¢**, slightly worse than the blend |
| Opposing Over depth | Approximately $461 across 56.5–58.0¢ |

The nearby 46.0¢ level matters even though it is outside the small best/blended
quote description. A liquidity bucket alone would omit both distance to that order
and the opposing book.

## Decision, hypothetical tag and source discrepancy

The owner assigned **3/10** and **passed**: the large same-side order raised concern
about taking a quote against a potentially informed participant. That interpretation
is discretionary, not proof that the edge was false or that the other order was sharp.
No paper-executed stake or actual fill is reported for this opportunity.

The source's tagging discussion uses **roughly $210 of blended-price liquidity**
to illustrate how a **Liq <= $500** tag would conceal deeper concentration. Its two
listed component sizes, however, sum to **$224 ($95 + $129)**. Preserve the approximate
narrative amount and quoted components separately: their discrepancy is unresolved,
and neither is a verified filled amount. Both fall within the illustrative bucket,
but this does not reconcile the source or justify replacing either number.

This tag is **hypothetical** for the passed opportunity. Reviewed passes are not
currently logged in Bet Tracker; the TEST-only Passed/Trade not Executed proposal
is distinct from liquidity tags and must remain separate from paper executions.
Do not convert this example into an observed tagged pass or a funded wager.

## Evidence and limits

- [Figure 49 — blended OddsJam listing](../../images/oddsjam-workflow/CATALOGUE.md#figure-49).
- [Figure 50 — substantial nearby same-side depth](../../images/oddsjam-workflow/CATALOGUE.md#figure-50).

All public slots are placeholders. The date-level transition does not populate
historical registries, determine exact intraday cutoffs or relabel every TEST record.
The case motivates recording both sides and nearby levels prospectively; it does
not validate a $5,629 or any other fixed liquidity veto, nor establish settlement.

Context: [dated settings and transition](../ODDSJAM_WORKFLOW.md#evidence-and-terminology),
[quote/warning review](../ODDSJAM_WORKFLOW.md#discover-and-review-an-opportunity),
[blended tags](../ODDSJAM_WORKFLOW.md#execute-reconcile-and-log),
[TEST-only passes](../ODDSJAM_WORKFLOW.md#paper-trading-and-pass-handling).

[All cases](README.md) · [Previous: F](F.md)
