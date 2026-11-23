# OddsJam operating case studies

These seven owner-supplied examples explain **discretionary decisions**, not a
validated set of trading rules. They are curated narrative extracts from the frozen
R&D roadmap §§2.1–2.5, reviewed on 2026-11-23. Source figure captions are catalogued;
private account screenshots and original wager records are not published. Amounts
below are selected explanatory quote/fill figures, not a reusable historical dataset.

Start with the [dated settings and evidence labels](../ODDSJAM_WORKFLOW.md#dated-configuration-and-display-choices),
[quote/depth and warning review](../ODDSJAM_WORKFLOW.md#discover-and-review-an-opportunity)
and [logging/tag conventions](../ODDSJAM_WORKFLOW.md#execute-reconcile-and-log).
These general controls supply context; they do not prove the settings or timing of
every source screenshot. All cases used the Core Exchange Conservative strategy;
A–F describe the historical PROD workflow, while G follows the November 10 transition
to TEST-only paper evaluation with no further real-money execution of that strategy.

| Case | Owner's reasoning | Recorded decision | Source figures |
| --- | --- | --- | --- |
| [A](A.md) | Large same-side size despite 13.23% advertised EV | Passed; illustrative 3/10 | 29–30 |
| [B](B.md) | More opposite-side depth at 3.07% advertised EV | Funded $140 wager reported; illustrative 9/10 | 31–33 |
| [C](C.md) | Thin best prices, modestly favorable opposing depth | Funded Under wager reported; illustrative 6/10 | 34–36 |
| [D](D.md) | Blended listing and partial fill at a different exact quote | $130 intended, $95.38 filled, $34.62 canceled; illustrative 6/10 | 37–41; 42 provides blended-tooltip context |
| [E](E.md) | Venue experience substitutes for unavailable book depth | Funded spread wager reported; illustrative 7/10 | 43–45 |
| [F](F.md) | Thin balanced market; rounded display differs from execution | Funded at 50.5¢, logged −102; illustrative 5–6/10 | 46–48 |
| [G](G.md) | Large same-side order near the small blended quote | Passed during TEST evaluation; illustrative 3/10 | 49–50 |

Ratings are the owner's illustrative assessments, not calibrated probabilities or
verified decision-time measurements. The source does not establish settlement,
net fees or realized profit for these examples; an executed decision is not evidence
that the trade was profitable. Unspecified quantities stay unspecified. Cases E/F
are described as “the same day” following D's November 6 example; this contextual
date is distinct from a verified capture timestamp.

## What the examples teach

- A higher advertised EV did not automatically win the owner's preference: A was
  passed while B was executed. This comparison is explanatory, not causal evidence.
- Best-price, contributing blended liquidity, nearby depth, intended stake and filled
  size are distinct. A liquidity tag cannot recreate the full two-sided order book.
- C and F illustrate different judgments under weak depth signals; E illustrates
  missing depth rather than zero depth. G illustrates an important nearby level
  excluded from a small best/blended quote tag.
- D preserves partial fills and canceled remainder; F preserves display rounding
  and an unresolved liquidity discrepancy. Neither is silently “corrected” to a fill.
- G's quoted $95 + $129 sums to $224, while the source's hypothetical tag example
  says roughly $210. Both descriptions remain visible with no invented reconciliation.

[Source figure catalogue](../../images/oddsjam-workflow/CATALOGUE.md) has 50 figure
anchors and 51 reserved slots, including two for Figure 11. All are text placeholders
until actual image/privacy review. The [asset contribution guide](../../images/oddsjam-workflow/README.md)
explains how to supply reviewed images without publishing raw account information.

[Operating workflow](../ODDSJAM_WORKFLOW.md) · [Documentation home](../../README.md)
