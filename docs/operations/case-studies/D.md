# Case D — Blended listing, partial fill and canceled remainder

**Source:** R&D §2.5, November 6, 2026 opportunity. Original capture timestamps are
not established. Historical PROD Core Exchange Conservative workflow on ProphetX.

## Displayed blend and underlying quotes

Lucrezia Stefanini vs. Anastasia Zakharova, **Anastasia Zakharova moneyline**, displayed
**2.90% advertised EV**, a chain-link blended quote of about **48¢ / +108 American**,
and **$130 displayed combined liquidity**. The visible venue book separately showed:

| Intended Zakharova levels | Opposing Stefanini levels |
| --- | --- |
| $95.38 at 47.39% | $91.87 at 53.49% |
| $19.53 at 47.53% | $368.82 at 53.70% |
| $48.11 at 48.54% | $25.98 at 54.13% |

The three intended-side sizes sum to **$163.02** (approximately $163 in the source),
versus **$486.67** opposing. Other intended-side levels appeared at 49.75% and
57.63%; these were worse prices for an immediate take near the displayed blend.
The reason the displayed $130 differs from the visible-level total is unknown.
Do not equate the entire book with the particular quotes included in a blend.

## Decision and actual order sequence

The owner viewed greater opposing depth as modestly favorable, rated the setup
**6/10**, and followed OddsJam's one-click link to the venue. The slip suggested
**$130**, and the owner submitted at **47.39%**.

| Quantity | Reported value | Interpretation |
| --- | --- | --- |
| Intended/submitted stake | $130 | Suggested amount submitted, not a full fill |
| Immediately filled | $95.38 | Reported actual execution amount |
| Unfilled remainder | $34.62 | Became resting at the same price; owner canceled it |
| Bet Tracker amount | $95.38 | Executed amount, not the intended $130 |

The $95.38 fill plus $34.62 canceled remainder reconciles to $130. A transient resting
remainder does not make this an intentional maker strategy. The actual execution
quote is distinct from OddsJam's approximately 48¢ blended display.

## Tracking, evidence and limits

The source records **Liq <= $500** for the selected blended-price liquidity context.
That tag does not encode individual levels, total depth or the partial-fill sequence.

- [Figures 37–41](../../images/oddsjam-workflow/CATALOGUE.md#figure-37) cover the listing,
  venue book, entry, partial-fill detail and tracking evidence; each figure has its
  own subsequent catalogue anchor.
- [Figure 42](../../images/oddsjam-workflow/CATALOGUE.md#figure-42) is additional generic
  blended-tooltip context; it is not asserted to be this exact order's execution.

All public slots are placeholders. The source order narrative is not an independently
replayed fill ledger. Fees, settlement and the cause of the displayed-liquidity
mismatch remain unknown; the depth heuristic remains unvalidated.

Context: [settings](../ODDSJAM_WORKFLOW.md#dated-configuration-and-display-choices),
[quote/warning review](../ODDSJAM_WORKFLOW.md#discover-and-review-an-opportunity),
[execution and blended tags](../ODDSJAM_WORKFLOW.md#execute-reconcile-and-log).

[All cases](README.md) · [Previous: C](C.md) · [Next: E](E.md)
