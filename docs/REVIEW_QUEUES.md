# All-history opportunity and production-change review queues

This publication replaces conversation-only selections with reproducible research
queues. It consumes the existing corrected within-filter family, complete decision
sheets and visual candidates. It does not alter their results, render new dashboard
pages, change the scanner or authorize any production change. Dashboard integration
follows in the default-dashboard redesign.

## Explicit build and immutable outputs

```sh
poetry run python scripts/build_review_queues.py \
  --discovery data/processed/within_filter_discovery/<discovery-id> \
  --prioritization data/processed/signal_prioritization/<prioritization-id> \
  --decisions data/processed/decision_records/<decision-id> \
  --visuals data/processed/visual_diagnostics/<visual-id> \
  --config configs/review_queues.yaml \
  --output-root data/processed/review_queues
```

All four inputs are required and must share exact source lineage. The output contains:

- `review.json`: the entire source family plus the new descriptive global catalogue,
  all ordered records, exact-predicate idea groups, complete nonzero wager/event
  overlap pairs, population economics, search contract and missing-category coverage.
- `source_hypotheses.parquet`: a byte-for-byte copy of the original complete corrected
  family, including unavailable tests and every reserved comparison slot.
- `audit.json` and `manifest.json`: counts, evidence tiers, headline counts, sensitivity,
  code/runtime/configuration hashes and exact input manifest identities.

Use `load_review_queues(Path(...))` from `oddsjam_ev.io.review_queues` to verify an
explicit publication. It verifies output integrity, source-correction bytes and
reconciliation counts. Identical replay verifies existing output instead of rewriting
it. Builders never search for “latest”. Output cannot be inside a source publication.
Candidate, membership and pair budgets fail the complete build, with a failure audit;
partial queues are never published. Private output remains ignored by Git.

## All history, without incompatible pooling

All history means **the explicitly pinned reconciled source universe**, not all
possible account history or all opportunities the scanner could have shown. Population
totals count each canonical candidate once. All recorded funding statuses and products
are searched and visible by default, including candidates without resolved saved-filter
attribution. Funding and product economics stay separate. Filter memberships overlap;
filter totals and subgroup totals must not be added into population profit.

The universe ledger contains eligible/excluded counts and economics per recorded
population. Each record includes parent, subgroup and retained complement counts,
stake, source-reported profit, weighted ROI, equal-stake return, $1-unit notional profit
and sizing-regime breakdown. These use the same economic eligibility as the verified
visual/sizing inputs. Source reported profit is not independently verified fee-net.
Returns are fractions; advertised source EV and threshold endpoints use percentage
points. Dollar loss burden is historical context, not a sort score or profit forecast.

## Original inference and new global catalogue are separate

The original candidate family, test statuses, raw/adjusted values, correction family
identity, rejection flags and original shortlist count are preserved. No test is
rerun and no gate, alpha, family size or previous output changes to create winners.
Source discovery blockers are retained under `source_discovery_blockers_json` as
**pre-inference ledger history**, not current queue blocking reasons.

A second, explicitly descriptive search enumerates feature-only global slices within
each funding/product population. Default declared views are platform × market family,
league × market family, and advertised-EV band × odds band. Configuration freezes
finite numeric cutpoints, left-closed/right-open bins and outer bins. Original exact
continuous EV text drives decimal membership. Invalid American odds and unavailable
categories are excluded from positive predicates but retained in parents/complements
and the coverage audit. No search is outcome-filtered or tuned on dashboard clicks.

Global support preflight uses the previously approved operational policy: each arm
needs ≥100 eligible wagers and ≥100 namespace/event proxies, full event coverage,
positive finite stakes, largest wager ≤5% and event ≤10% of arm stake. No bootstrap,
p-value, corrected label or interval is generated for this new catalogue. Global
records have no original hypotheses or source correction family. Passing these gates
is not a power guarantee. Formal testing of an expanded global family requires a
separate reviewed inference policy, full-family accounting and simulation/sensitivity
work; it cannot borrow rejections from overlapping existing candidates.

The search contract names supported features and controls that cannot be reconstructed
from this history. Liquidity/timing, alternate devig/formula/reference settings and
alternative recommended-bet/sizing controls are not silently optimized. A stored EV
number cannot evaluate another devig model; selected bet history is not scanner supply.

## Evidence tiers and queue assignment

| Tier | Required evidence | Interpretation |
| --- | --- | --- |
| Independent prospective evidence | No supported input in this contract; explicitly unavailable | Historical records never acquire this tier |
| Corrected historical comparison | Original tests available; both directional measures reject after the original full-family correction; absolute and relative returns agree | Conditional exploratory historical comparison, not proven or deployable edge |
| Support-qualified exploratory | Existing operational support gates pass and both return measures agree; correction not established or global catalogue not tested | Research association warranting further study |
| Insufficient or conflicting | Gates fail, unavailable economics, zero/relative-only effects, or estimand disagreement | Investigation, not a headline winner |

Opportunities require positive subgroup weighted **and** equal-stake returns, plus
positive subgroup-minus-complement contrasts on both. Loss associations require all
four signs to be negative. Negative weighted returns with nonnegative equal-stake
returns, or conflicting contrast signs, are sizing/estimand investigations. No
absolute negative return is called a leak merely because a relative test is positive.

`opportunities` contains positive-direction records, with insufficient ones retained
but excluded from headlines. `production_changes` requires a negative-direction record,
confirmed effective-dated PROD lifecycle and a pinned prospective-parent baseline.
Unknown lifecycle, missing baseline, global losses and conflicting estimates go to
`investigations`; a PROD prefix or real-money funding label never proves current PROD.

## Ranking, distinct ideas and sensitivity

The frozen presentation policy is independent of test correction:

1. Evidence tier first; do not manufacture a stronger tier when it is empty.
2. Within queue/tier, assign nondominated fronts over the weaker of the two pointwise
   contrast bounds, minimum arm event support and negative worst-arm event stake share.
   For loss associations reverse the upper bounds. Missing intervals are unavailable
   and rank behind finite bounds on that objective. These are **pointwise research
   priorities**, not simultaneous confidence bounds or probability-of-edge scores.
3. Within a front: event breadth descending, worst event share ascending, stable
   candidate identity. Historical dollars and raw ROI do not determine rank.
4. Keep setup readiness separate and visible. Unknown settings do not erase a useful
   hypothesis and do not become implicit defaults or unrestricted scanner settings.
5. Group exact predicates across parents/populations without pooling their evidence.
   Suppress exact parent-and-subgroup equivalence only for presentation, after the
   original full-family correction. The full candidate ledger is always retained.
6. Select at most four headline representatives per actionable queue, never a quota.
   Require distinct predicates and reject headline redundancy when eligible-wager or
   event Jaccard overlap reaches 0.8 in the same economic population. All overlap pairs
   and suppression reasons remain available, including across different parents.

Every record exposes component values, front, rank, explanation and headline blocker.
A sensitivity pass omits the interval objective and records alternative front/order
and rank change. It performs no new tests and does not replace the frozen default
order. Broader logged support, especially for untested global slices, can trade off
against interval evidence; the component values disclose that trade-off.

## Complete setup and controlled modification proposals

Every idea links a complete parameter sheet: baseline, candidate constraints, final
values, explicit setting states, provenance and setup blockers. Existing sheets are
used unchanged. Global candidates get stable **proposed new** identities and require
an explicit new prospective baseline; no existing saved filter is silently inherited.
Unmapped markets, unsupported ranges/endpoints and unknown source settings remain
blocked. These outputs must not be described as executable OddsJam configurations
until all required evidence and UI translations have been resolved.

A production-change proposal retains the exact parent baseline ID/hash, lifecycle
knowledge/effective time, negative subgroup and retained complement, removed counts
and stake share, and the exclusion-comparison parameter sheet from decision records.
The logical change is parent **AND NOT (whole subgroup conjunction)**. It is never
implemented by independently inverting each scanner control. Unsupported exclusions
stay blocked pending a verified equivalent implementation. Alternatives such as
widening a universe, changing reference books or claiming fee-adjusted profit cannot
be inferred from these partitions.

The pinned baseline is prospective configuration evidence, **not proof that those
settings governed historical wagers or were applied externally**. The proposal records
that applied-configuration status is unverified. Confirm actual current configuration
before any external edit; compare baseline versus modified paper definitions first.
Historical loss burden and complement returns are not causal profit-saved estimates.
Setup readiness, review, approval, external application and prospective outcome review
remain distinct steps. This publication performs none of those actions automatically.

## Historical smoke-build expectations

The pinned October source family contains 602 candidates and 2,408 correction slots,
42 available conditional comparisons and zero original corrected shortlist entries.
The default global catalogue adds 953 descriptive slices. The complete queue contains
1,555 records, including 82 support-qualified exploratory associations, with no
corrected-historical or independently validated tier populated. The pinned decision
evidence has no confirmed PROD baseline, so production proposals remain empty and
loss associations stay in investigation. This is explicit missing evidence, not proof
that no production weaknesses exist. All observed outputs remain private and versioned.
