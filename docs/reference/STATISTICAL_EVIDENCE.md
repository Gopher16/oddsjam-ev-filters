# Statistical evidence and interpretation

## Separate stages and claims

[Approved exploratory policies](policies/EXPLORATORY.md) define core discovery,
within-filter prioritization and model gates separately. The
[prospective promotion policy](policies/PROMOTION.md) governs a different future,
predeclared family. Equal numeric thresholds do not make policies interchangeable.
The dashboard minimum-wager slider is presentation only, not inferential screening.

Descriptive discovery freezes a bounded candidate ledger with explicit predicates,
parent scope, memberships, exclusions and search budgets. Unsupported/unobserved
intersections are not zero-return findings. Predeclared screening is audited before
testing; budget overflow fails the build rather than silently truncating candidates.

Full eligible point estimates can include wagers lacking event identity. Conditional
core/diagnostic intervals use the event-covered cohort and disclose missing counts,
money and covered point estimates. Within-filter inferential arm gates instead
require complete event IDs. Never present a subset interval as the full-cohort
interval. Shared namespace/event clusters retain related outcomes together;
cross-event, team, player and regime dependence may remain unresolved.

## Contrasts and uncertainty

For subgroup A and complement B, the weighted contrast is
`sum(profit_A)/sum(stake_A) - sum(profit_B)/sum(stake_B)`. The equal-stake contrast
is `mean(profit_A/stake_A) - mean(profit_B/stake_B)` over eligible wagers.

Both arms share draws from the union of namespace/event groups. Events absent
from an arm contribute zero to that arm. The percentile bootstrap supplies
pointwise intervals, using identical draws for both return measures. Stable seeds
are keyed to eligible membership partitions, so equivalent partitions reproduce
identical results regardless of candidate ordering.

Directional marginal tests use a separate approximate cluster influence t
calculation. For a ratio `r = sum(numerator)/sum(denominator)`, each event contributes
`(numerator_g - r * denominator_g) / sum(denominator)`. Subtract the two arm
contributions to obtain `score_g`. With G joint events, the standard error is
`sqrt(G/(G-1) * sum(score_g**2))`; the contrast divided by this error is compared
with a Student t distribution with G−1 degrees of freedom. This retains shared
event covariance. It assumes independent event proxies and adequate ratio
approximation; neither assumption is established by passing the gates. This is
not an exact randomization test or a temporal validation result.

Failed gates, degenerate errors, or invalid resampled denominators block tests.
Their four hypothesis slots remain in the complete Holm family as unavailable
comparisons, using conservative p=1 placeholders for correction. Published
unavailable p-values remain missing. Bootstrap intervals are not simultaneous
intervals and do not substitute for corrected directional tests.

## Presentation and overlap

A positive or loss shortlist entry requires corrected rejection in the same
direction for both return measures, plus matching subgroup return signs.
Relative associations and disagreement between measures have separate labels.
Full statistics retain blocked and unconfirmed candidates. No entry receives a
validated or production-authorized label while chronology is unverified.

Presentation orders candidates by interval availability, arm event breadth,
arm wager support, concentration, and stable ID. It does not reward extreme ROI
or extra dimensions. Exact raw-membership equivalents are suppressed only within
the same parent and category, after correction. Nested, partially overlapping,
and cross-parent memberships are audited without reducing the correction family.
At most ten entries per qualifying category are presented. An empty shortlist is
a valid result and does not justify relaxing gates after seeing outcomes.

## Complete families, repeated looks and labels

Holm correction applies to the complete declared family, not just displayed winners.
For within-filter prioritization this is every frozen candidate × two measures ×
two directions. Unavailable tests retain conservative p=1 correction slots while
published unavailable p-values stay null. The prospective family separately includes
absolute trial return and trial-minus-other-arm claims: eight slots per definition.
Neither overlap suppression nor dashboard filters reduce these families.

Marginal percentile intervals are pointwise, not simultaneous or Holm-adjusted.
A positive interval does not replace a corrected directional test. Holm relies on
valid marginal p-values; grouping and correction do not establish independence,
causality, executable profitability or representative scanner capture.

Repeated cutpoint searches, subset refits and revised reports reuse information.
Single-publication correction does not control repeated looks across publications;
an inspected holdout is not renewed independent evidence. Unknown chronology
blocks temporal-validation labels. Exploratory positive/loss associations, relative
associations, disagreement, blocked and unconfirmed results retain their distinct
statuses. Model utility is not a corrected discovery result. A future study result
is not a settings approval, and an approval is not confirmed external application.

Implementation: [paired contrasts](../../src/oddsjam_ev/analysis/paired_comparison.py),
[complete-family prioritization](../../src/oddsjam_ev/analysis/signal_prioritization.py),
[Holm](../../src/oddsjam_ev/analysis/multiple_testing.py),
[core robustness](../../src/oddsjam_ev/analysis/robustness.py).
Verification: [arm tests/families](../../tests/test_statistical_prioritization.py),
[multiplicity](../../tests/test_multiple_testing.py),
[dependence and degeneracy](../../tests/test_robustness.py),
[prospective claims](../../tests/test_promotion.py).

[Contract reference](README.md) · [Documentation home](../README.md)
