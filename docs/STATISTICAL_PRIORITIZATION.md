# Exploratory statistical prioritization

This opt-in publication compares each frozen within-filter subgroup with its
complement. It consumes verified discovery and sizing-diagnostic publications;
it does not change filters, the default pipeline, or dashboard publications.

## Approved initial policy

The user approved this policy on October 30, 2026. Settings are versioned in
`configs/signal_prioritization.yaml` and included in publication identity.

| Gate or setting | Initial value |
| --- | --- |
| Eligible wagers in each arm | At least 100 |
| Namespace/event groups in each arm | At least 100 |
| Event-ID coverage of eligible wagers | 100% |
| Largest wager / arm stake | At most 5% |
| Largest event / arm stake | At most 10% |
| Stakes | Positive finite values; no additional dollar floor |
| Paired event-bootstrap draws | 5,000 |
| Percentile interval | 95%, pointwise |
| Multiple-comparison correction | Holm, alpha 0.05 |
| Family | Every frozen candidate × two return measures × two directions |

These are operational screening gates, not a power guarantee. Unknown historical
sizing regimes remain unknown and are counted. Eligibility inherits the sizing
diagnostics contract; event coverage does not establish event independence.

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

## Reproducible publication

```bash
poetry run python scripts/build_signal_prioritization.py \
  --discovery data/processed/within_filter_discovery/<publication-id> \
  --diagnostics data/processed/sizing_diagnostics/<publication-id>
```

An optional `--config` selects an explicit policy. The discovery publication must
reference the exact supplied diagnostic manifest. Input hashes, configuration,
implementation hashes and runtime versions determine the output identity.
Outputs include statistics, all hypotheses, overlap audits, presentation
priorities, a shortlist and an audit. Existing publications remain unchanged;
identical builds verify replay integrity. Outputs remain under ignored
`data/processed/signal_prioritization`.

Preflight budgets cover unique resampling partitions and all candidate pairs.
Exceeding a budget fails the entire build and writes a failure audit; it never
publishes a truncated testing family. Defaults permit five billion bootstrap
cells and 250,000 pair comparisons.

## Interpretation and follow-up

Results are exploratory conditional associations in tracked history. They do not
establish causation, prospective availability, removal benefits, independent
opportunities, or calibration of advertised EV. Holm addresses multiplicity only
to the extent marginal p-values are valid under the stated assumptions. Repeated
looks across publications are not controlled by this single-family procedure.

Before stronger labels, verify chronology and event grouping, define independent
temporal evaluation, assess robustness to remaining dependence and sizing
regimes, and validate power and error behavior with simulations. Any later
policy revision requires a new explicit configuration and publication.
