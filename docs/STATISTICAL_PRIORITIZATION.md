# Exploratory statistical prioritization

This opt-in publication compares each frozen within-filter subgroup with its
complement. It consumes verified discovery and sizing-diagnostic publications;
it does not change filters, the default pipeline, or dashboard publications.

## Approved initial policy

The [approved exploratory policy](reference/policies/EXPLORATORY.md#within-filter-prioritization)
is authoritative for arm gates, resampling, correction family and resource limits.
It is separate from prospective promotion eligibility.

## Contrasts, uncertainty and presentation

The [statistical evidence reference](reference/STATISTICAL_EVIDENCE.md#contrasts-and-uncertainty)
defines paired contrasts, event resampling, cluster influence tests, missing tests,
pointwise intervals and post-correction overlap/presentation. Preserve the complete
family even when the visible shortlist is empty.

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

See the [statistical interpretation reference](reference/STATISTICAL_EVIDENCE.md#complete-families-repeated-looks-and-labels)
for exploratory labels, marginal versus corrected claims and repeated-look limits.

Before stronger labels, verify chronology and event grouping, define independent
temporal evaluation, assess robustness to remaining dependence and sizing
regimes, and validate power and error behavior with simulations. Any later
policy revision requires a new explicit configuration and publication.
