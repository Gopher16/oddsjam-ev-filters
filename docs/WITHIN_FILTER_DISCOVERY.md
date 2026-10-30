# Bounded within-filter descriptive discovery

This optional stage enumerates observed feature intersections inside **resolved
saved-filter membership**, then compares each subgroup with its exact parent and
complement. It does not substitute global platform results for within-filter
results or propose automatic filter edits.

## Declared family and operational policy

`configs/multidimensional_discovery.yaml` declares allowed views:

- Platform × market family.
- League × market family × exact advertised EV threshold.
- Platform × sport × market family × exact advertised EV threshold.

The default numeric predicate is advertised EV **> 3 percentage points**, evaluated
from row-level source text using the continuous-EV contract, never inferred from
an aggregate EV band. It is an exploratory predeclared cutpoint, not an optimized
or validated threshold. Operators/endpoints/units are explicit; equivalent decimal
endpoints and reordered dimensions serialize to identical definitions.

Views can use platform (`sportsbook`), sport, league, market family, published odds/EV
bands and exact advertised EV. Only explicit combinations of up to four dimensions
are allowed. At most 32 declared EV thresholds are supported. No unrestricted
Cartesian search, learned threshold or outcome-driven view selection is performed.

Hard budgets bound parents, candidates, membership links and conservative scanned-row
work. They are **resource limits, not research-support gates**. A budget overflow
writes a separate private failure audit and aborts before evaluating economic
summaries. No truncated candidate family is published. Resource-policy changes create
new lineage. Inferential support thresholds remain unset, and formal validation,
comparison inference, multiplicity correction and ranking remain blocked.

## Membership and comparison semantics

Parents are the unique recorded wagers for each resolved saved filter, funding
population and product. Overlapping filters remain separate, non-additive parents.
Unused registered filters do not acquire invented membership. Missing/unresolved
attribution stays outside per-filter parents and is counted in the audit.

Candidate categories come from observed logged features, before examining outcomes,
economic eligibility or support. Missing, invalid, conflicting and unmapped categories
are excluded from positive categorical predicates with an explicit exclusion ledger;
those wagers remain in the parent and therefore in appropriate complements. Missing,
invalid or conflicting EV never satisfies an exact threshold. Empty threshold
subgroups and sparse candidates remain in the frozen family.

For each candidate:

1. Subgroup is the predicate match **within that parent**.
2. Complement is parent minus subgroup; it is not a separate global slice.
3. Set union/disjointness, candidate/eligible counts, eligible stake, source profit
   and fixed-unit notional profit reconcile to the parent.
4. Weighted/equal-stake estimands reuse sizing diagnostics on their common eligible
   cohort. Verified sizing-regime breakdowns and unknown regimes are retained.
5. Descriptive subgroup-minus-complement and complement-minus-parent differences
   are historical associations, not causal gains from excluding that subgroup.

Unknown settings, excluded ticket/product economics and funding limitations remain
unchanged. No widening universe can be established from selected logged bets alone.
A positive complement-minus-parent difference does not authorize removing a subgroup
from production; prospective comparisons and reviewed statistical policy are needed.

## Published artifacts

```sh
poetry run python scripts/build_within_filter_discovery.py \
  --canonical <verified-canonical-directory> \
  --atlas <matching-atlas-directory> \
  --diagnostics <matching-sizing-diagnostics-directory> \
  --config configs/multidimensional_discovery.yaml
```

All three inputs must pin the exact same canonical bundle/manifest. No implicit
latest input is selected. Wager identities and canonical economic/source fields
reconcile before enumeration. Atlas normalization/band definitions remain pinned
by atlas provenance; sizing eligibility and notional-unit policy stay pinned too.

Private output is `data/processed/within_filter_discovery/<publication-id>/`:

| Artifact | Meaning |
| --- | --- |
| `signal_candidates.parquet` | Complete family ledger, exact predicates, parent scope, lineage-linked IDs and explicit blockers. |
| `signal_evidence.parquet` | Parent/subgroup/complement point metrics, support, event coverage, concentration and sizing-regime breakdowns. Aggregate and regime rows are separate views, not additive together. |
| `signal_comparisons.parquet` | Reconciliation, empty-part status, membership fingerprints and descriptive differences. |
| `signal_membership.parquet` | Explicit parent/subgroup wager IDs. Complements are reproducible as parent minus subgroup and their digests are checked. |
| `signal_exclusions.parquet` | Parent/view/category exclusions for missing or invalid positive categorical predicates. |
| `audit.json` | Complete-family status, work usage, missing EV/category coverage, attribution limitations and blocked validation. |
| `manifest.json` | Contract/policy, input manifests, implementation/runtime hashes and output integrity. |

Failed budgets are stored under `failures/` with status
`aborted_budget_exceeded_not_truncated`; these are not consumable result manifests.
A policy with no observed usable predicates fails explicitly instead of inventing
results. Verified loading and identical replay preserve old publications.

No comparison confidence intervals or p-values are estimated here. Reused point
metrics mark comparison inference deferred; prior marginal/event-proxy intervals
are not repurposed as subgroup/complement difference intervals. There is no shortlist,
overlap suppression, production promotion, dashboard rendering, acquisition or
scheduling change in this stage.

Tests cover exact threshold boundaries, predicate/row-order stability, global versus
within-filter membership, overlapping filters, missing categories, empty/sparse
parts, unknown regimes, outcome-independent enumeration, budget audit replay,
source preservation, corrupt publications and mismatched lineage. Statistical
prioritization and interactive evidence views remain focused follow-ups.
