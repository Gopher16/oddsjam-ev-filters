# Follow-up: multidimensional signals and saved-filter optimization

Status: proposed research change, to begin after the stakeholder dashboard changes
are reviewed and merged. This document is an implementation brief, not an implemented
capability or an approval to deploy a filter. Use this descriptive title for the
follow-up pull request rather than a roadmap sequence number.

## Problem and intended outcome

The dashboard consumes existing candidate artifacts. The current discovery policy
limits predicates to at most two categorical dimensions and orders its display by
event-group breadth. Consequently broad cohorts dominate and many user-selected
intersections have no published evidence. Broad platform results also cannot answer
whether a platform/market subset is useful within a particular saved filter.

Publish bounded, reproducible multidimensional candidates and parent/complement
comparisons that explain which exploratory signals deserve prospective TEST work.
Example output: an explicitly defined platform × sport × market × advertised-EV
band, with actual support, exclusions, uncertainty and an identified parent filter.
The outcome is a reviewable TEST hypothesis, never automatic PROD promotion.

## Scope and non-goals

- Work from explicit verified canonical/atlas inputs and manual CSV ingestion.
- Preserve source snapshots, historical outputs, notebooks and preprocessing defaults.
- Separate real-money, paper and unknown funding; partition product economics.
- Preserve reported USD profit. Fee inclusion stays unverified; do not estimate fees.
- Keep existing unsupported-product, ticket, refund and settlement exclusions.
- Preserve unknown timezone, scope, decision-time and outcome chronology.
- No acquisition dependency, scheduling, live scanner, wagering or filter edits.
- Dashboard page loads consume published artifacts, never fit discovery methods.
- Do not infer liquidity, executable capacity, CLV validity or current PROD status
  from filter names. Unchecked paper quotes are signals, not fills.

## Assessment required before coding

Read the merged canonical, atlas, discovery, recommendation, experiment and pipeline
contracts. Inspect `discovery_config.py`, `candidate_review.py`, `robustness.py`,
`multiple_testing.py`, `temporal_validation.py`, the artifact loaders and their tests.
Confirm available row-level fields, membership and event-group identities. Audit how
candidate IDs, family definitions, Holm adjustment and interval assumptions currently
work before extending them. Produce an explicit compatibility/migration proposal.
Inspect October evidence read-only; do not relabel historical funding or rebuild
historical outputs in place. Quantify supported intersections and missing dimensions.

## Proposed policy contract

Add a versioned `configs/multidimensional_discovery.yaml` with a strict parser:

| Setting | Proposed treatment |
| --- | --- |
| Dimensions | Allowlist platform, sport, league, market family, odds band, advertised-EV band; unsupported fields rejected. |
| Predicate depth | Explicit configured combinations; support up to four dimensions initially, not unrestricted Cartesian search. |
| Numeric cutpoints | Predeclared thresholds/bands, with units and open/closed endpoints. EV above 3% requires row-level evaluation, not splitting a 2.5–5% aggregate. |
| Search budget | Maximum candidates, combinations and resampling work; exceeding the budget fails with an audit, never silently truncates the hypothesis family. |
| Missing values | Explicit excluded/unknown categories; never impute a sport, market or product from names. |
| Support | Minimum eligible wagers, event groups, eligible stake, coverage and concentration limits; distinguish display thresholds from inferential gates. |
| Evaluation | Fixed development/holdout windows and embargo only with verified chronology; otherwise exploratory-only. |
| Family | Predeclare hypothesis family and correction scope across dimensions, thresholds, saved filters and positive/loss hypotheses. |
| Reproducibility | Fixed seed, canonical predicate serialization, code/config/input hashes and deterministic tie-breaks. |

Concrete inferential thresholds and work budgets must be justified in assessment and
approved as policy before implementation. Do not reuse the dashboard's 100-wager
presentation convention as a research gate. Log candidate exclusions before testing
so the tested universe and screening rules can be reconstructed.

## Candidate and evidence contracts

Proposed immutable tables (exact names subject to contract review):

1. `signal_candidates`: candidate ID, contract version, canonical/atlas IDs, funding,
   product, predicate JSON, dimensions, thresholds/units, parent scope/filter ID,
   eligibility definition, support counts and search-family ID.
2. `signal_evidence`: eligible wagers/stake/reported profit/ROI, event-group count,
   coverage, largest event/wager stake shares, published group-aware interval,
   raw/adjusted tests and statuses, temporal status and limitation codes. Unknown
   evidence stays null with a reason, not zero.
3. `signal_comparisons`: parent, subgroup and complement identities/counts/economics,
   subset and disjointness checks, descriptive differences, group-aware uncertainty
   for comparisons where supported, and non-causal interpretation status.
4. `signal_overlap`: related candidate pairs, membership overlap/Jaccard, nested or
   equivalent relationship, representative candidate and deterministic suppression
   reason. Suppression affects shortlist presentation, not the test family.
5. `signal_shortlist`: candidate ID, review category, ordering components, explanatory
   reasons, proposed TEST definition, baseline, next evidence and promotion blockers.
6. `signal_audit`: tested/screened/excluded counts by reason, family membership,
   reconciliation, budget usage, randomness, timing evidence and contract versions.

IDs must include semantic policy and source lineage. Equivalent reordered predicates
must have identical identities. Every result must be traceable to eligible wager IDs
or a verified membership digest. Ticket economics count once. Opportunity proxies
remain counts, never wager denominators. No pooling overlapping attributed filters.

## Statistics and shortlist policy

Reuse established group-aware resampling and multiple-testing components where their
assumptions apply. State precisely whether an event ID is verified or a proxy. Preserve
related outcomes in the same resampling/split group; audit cross-platform dependence.
Intervals describe their stated cohort and do not prove executable net returns.

Predeclare screening and all tested hypotheses. Account for selecting favorable
thresholds and parent/filter slices in the multiplicity design. Do not repeatedly
inspect an unchanged holdout and call it independent validation. If chronology is
unverified, produce descriptive evidence with explicit blocked validation; no temporal
pass, significance-based promotion or fabricated holdout result.

Use transparent evidence categories rather than an opaque profitability score:
insufficient evidence, exploratory positive signal, exploratory loss pocket,
prospectively evaluable hypothesis, and validated evidence only when its predeclared
requirements actually pass. Define deterministic ordering within categories using
approved evidence/support/concentration criteria. Missing intervals cannot outrank
verified evidence by default. Show why a candidate was ordered or suppressed. Keep
positive and loss hypotheses separate. Specificity alone earns no priority bonus.

## Saved-filter narrowing and widening

Resolve exact recorded filter membership and attribution status before constructing
within-filter predicates. Compare subgroup and complement against the same eligible
parent, funding/product, source snapshot and time scope. Assert subgroup + complement
reconciles to parent without double counting. A global Novig result cannot replace
a Novig-within-filter result. Describe apparent changes as historical associations,
not causal gains from editing a filter.

Narrowing proposals need an unchanged baseline and prospective exclusion comparison.
Widening proposals require an observed eligible universe outside current settings,
with comparable collection and availability rules. Logged selected bets alone cannot
establish that universe; otherwise publish `widening_unsupported_missing_universe`.
Retain failed/unavailable quote checks in prospective sampling. No action directly
changes OddsJam or authorizes real-money use.

## Integration and proposed files

Reuse canonical wager facts, source features, filter attribution, existing economics,
robustness and multiplicity helpers. Avoid parallel payout or eligibility logic.

- `analysis/multidimensional_config.py`: validated policy and frozen search family.
- `analysis/signal_discovery.py`: bounded enumeration and source reconciliation.
- `analysis/signal_comparisons.py`: within-parent subsets and complements.
- `analysis/signal_prioritization.py`: transparent categories, overlap and explanations.
- `io/signal_discovery.py`: immutable publication, verified loading and lineage.
- `scripts/build_signal_discovery.py`: explicit input/config CLI and replay.
- `pipeline/config.py` and `pipeline/runner.py`: optional pinned stage, manifests,
  gates, resume semantics and code/config hashing; older plans remain loadable.
- `dashboard/artifacts.py`, `dashboard/queries.py`, `dashboard/pages/workspace.py`:
  consume optional verified stage; old reports explicitly lack new evidence.
- `tests/test_multidimensional_*.py`, loader/CLI/pipeline/dashboard integration tests.
- `docs/MULTIDIMENSIONAL_SIGNALS.md`: definitions, interpretation and usage.

Paths above are under `src/oddsjam_ev/` unless a repository path is specified. Confirm
actual module boundaries during assessment; extend existing contracts deliberately
rather than silently changing current bundle schemas.

## Logical, independently testable commit sequence

1. Policy parser, family specification, predicate/ID contracts and boundary tests.
2. Bounded enumeration, eligibility reuse, membership/parent audits and tests.
3. Dependence-aware evidence, multiplicity accounting and missing-chronology gates.
4. Saved-filter subset/complement comparisons, unsupported widening and tests.
5. Overlap reduction, transparent ordering/explanations and adversarial fixtures.
6. Immutable writer/loader, CLI, pipeline stage/retry integration and contract tests.
7. Dashboard optional-stage integration, multi-filter evidence views and UI tests.
8. Usage/methodology docs, migration notes and reproducibility/validation evidence.

Each commit must keep existing behavior/tests intact. No private records, generated
bundles, monetary notebook outputs or credentials belong in commits.

## Synthetic tests and objective acceptance criteria

- Exact AND/OR predicates, numeric endpoints, reordered predicates, missing values,
  invalid thresholds, incompatible dimensions and equivalent predicates.
- Repeated executions, duplicated attribution, ticket legs and repeated exports do
  not inflate eligible wager count or economics; source exclusions reconcile.
- Group-correlated and sparse fixtures retain dependence warnings. Null intervals
  stay unavailable. A two-wager high ROI never becomes a strong evidence signal.
- All tested candidates belong to the declared family; correction includes the
  complete eligible search, not merely visible winners. Budget overflow is explicit.
- Unknown chronology blocks temporal claims; verified fixtures prevent event-group
  leakage across development/holdout and honor embargo and information availability.
- Parent/subgroup/complement reconciliation, within-filter versus global distinction,
  overlapping filters and lack of outside-universe data have explicit tests.
- Overlap suppression is deterministic, explained and does not shrink the test family.
- Fixed seed/input/config/code reproduces identical tables. Changed inputs/policies
  create new bundles; corruption and row-lineage mismatch fail closed.
- Partial runs/resume never publish misleading completeness; old reports and default
  historical research metrics remain unchanged and usable.
- Dashboard renders verified results without research execution, never substitutes
  separate one-dimensional summaries for an intersection, and shows empty/blocked
  states. Funding/product labels and all-data browsing remain distinct.
- No scheduling, acquisition, filter-write, wagering or automatic promotion path.
- Required quality checks pass; benchmark bounded runtime/memory against a declared
  budget and record it. Inspect new artifacts read-only and verify reconciliation.

## Decisions to resolve during assessment

Approve the initial combination allowlist and threshold grid; inferential support,
stake, concentration and coverage gates; family/correction and uncertainty methods;
search/runtime budgets; overlap grouping; prospective sampling/holdout rules; saved
filter parent definitions; and what independent evidence can support widening.
Defaults must remain conservative and explicitly exploratory while historical timing,
fee treatment, availability or scope remain unverified. Present these decisions and
an actual proposed policy before implementation, not after ranking private data.

## Continuous EV: confirmed preservation and proposed extension

Repository assessment confirms `bet_facts.py` preserves source `percentage` in
`SOURCE_FEATURES` and canonical facts. `opportunity_atlas._dimensions` copies its input
and adds `advertised_ev_band` using `numeric_band`; it does not replace `percentage`.
`feature_catalog.py` labels this feature `advertised_ev_percent` with source timing
`source_current_unverified`. Therefore the existing banded summary cannot answer an
exact threshold question, but the source value has not been lost or reduced to a bin.

Maintain two independent representations:

- Original EV text and source observation reference, preserved byte-for-byte through
  ingestion; numeric advertised EV in percentage points, parsed without rounding into
  bands (for example 3.125 means 3.125%, not a probability of 3.125).
- Optional display bands, derived from the numeric field with versioned boundaries.
  They are presentation/aggregation features, never the authoritative EV measurement.

Do not call advertised EV an observed edge. It is the source's theoretical estimate,
not realized return or independently verified true probability. Preserve missing,
invalid and conflicting values and timing provenance. Confirm export-field semantics,
units, odds and decision-time availability before inferential use. Preserve EV values
for source legs even where product economics are excluded; do not assign a leg EV to
a ticket without a justified ticket model.

Proposed research additions:

1. Publish a typed numeric EV field and raw/source provenance, with no historical
   rewrite. Verify decimal precision and round-trip parsing; reject nonfinite values.
2. Support predeclared continuous predicates (for example `ev_percent > 3.0`) directly
   against eligible rows. Define > versus >= explicitly and serialize endpoints/units.
3. Keep the threshold grid finite and declared before evaluation. Searching many
   cutpoints counts as multiple testing; an arbitrary slider cannot create validated
   findings on dashboard load. Publish threshold-sensitivity results separately.
4. Publish calibrated EV-versus-return diagnostics only on a common eligible cohort,
   with funding/product/odds/chronology controls and uncertainty. A gap is descriptive;
   it does not establish model mispricing, fee leakage or executable profit.
5. If using smooth calibration/regression, predeclare model complexity, group-aware
   evaluation and out-of-sample procedure. Treat this as an optional later substage,
   not a prerequisite for exact continuous threshold predicates.

## Wager sizing, bankroll and Kelly policy history

### Current behavior and interpretation

`opportunity_atlas.summarize_wagers` calculates reported ROI as total eligible source
profit divided by total eligible source stake. Source money is preserved; the same
stake/profit eligibility cohort is used. Larger wagers therefore have greater weight.
Dollar profit also reflects capital deployed. Neither metric currently normalizes
historical bankroll amounts or Kelly multipliers, and neither establishes sizing-free
signal quality. Existing concentration fields are useful diagnostics, not a Kelly
policy adjustment. Do not change these historical metric definitions silently.

The user reports historically changing bankroll settings and generally using a
half-Kelly multiplier, with an intention to use quarter Kelly going forward. Exact
historical effective dates, per-bet application, overrides, constraints and future
transition time remain unknown. The account-settings screenshot is evidence of a
setting at capture time, not proof it applied to every historical wager. Do not
commit the private screenshot, bankroll amount or account-specific source records.
Do not assume the planned multiplier is already effective or endorse a sizing level.

Suggested stakes may depend on source EV and odds, along with bankroll settings,
liquidity, limits and manual decisions. Historical weighted returns can consequently
reflect changes in exposure, bet mix and selection as well as signal quality. That is
not automatically bias in the recorded portfolio ROI estimand; it is a confounding
risk when interpreting it as intrinsic filter quality or a future-return estimate.

### Required evidence contract

Add an optional, versioned, append-only sizing-policy history with:

- Policy/regime ID, known effective interval, timezone and recorded-at timestamp.
- Bankroll reference amount/currency, Kelly multiplier and any separately evidenced
  caps, liquidity constraints, overrides or rounding rules.
- Evidence/provenance type (user-confirmed, contemporaneous settings record,
  reconstructed assumption, unknown), source reference and confidence/status.
- Applied per-bet policy ID only when temporal linkage is supported; explicit unknown
  or ambiguous linkage otherwise. Never use export receipt as bet placement time.
- Actual source stake, source suggested stake if available, accepted stake if verified,
  and reason for deviation when known. Never relabel paper stake as an accepted fill.

Keep sensitive policy history local/ignored, with synthetic public fixtures and a
safe documented template. Validate intervals, currency, ranges, contradictions and
overlaps. Separate setting values from proof that they actually governed an execution.
No bankroll-history reconstruction from profit accumulation: deposits, withdrawals,
other wagers and unsettled exposure may be missing. Never infer bankroll/Kelly from
stake alone or backfill unknown dates with today's settings.

### Parallel economics, each with an explicit estimand

Publish side-by-side on the identical eligible positive-stake wager cohort:

| Metric | Definition and interpretation |
| --- | --- |
| Reported profit | Sum of source profit in USD: actual logged capital/exposure dependent, paper hypothetical. |
| Stake-weighted reported ROI | Sum(profit) / sum(stake): preserves existing portfolio-return behavior. |
| Equal-stake unit return | Mean(profit_i / stake_i): every eligible wager receives equal notional stake; arithmetic diagnostic, not a guaranteed replay. |
| Fixed-unit notional profit | Sum(profit_i / stake_i) times a declared unit: hypothetical comparison only. |
| Regime-specific returns | Same metrics split by evidenced sizing-policy regime; unknown regime remains separate. |
| Stake/EV/odds exposure | Distributions and concentration summaries, with missingness and event-group dependence. |
| Bankroll-relative stake | stake / bankroll at decision time only when bankroll and temporal linkage are supported. |

Reject zero/negative/nonfinite stakes for unit-return diagnostics and audit every
exclusion; retain original source records. Use exact settlement conventions and
supported products. Proportional returns assume source profit scales with stake:
fixed charges, minimum stakes, price impact, limits and fee tiers may violate that.
Label normalization as hypothetical and document known limitations. Do not sum unit
returns or weigh opportunities as though overlapping executions were independent.
Provide group-aware uncertainty for both weighted and equal-stake estimands.

A normalized result may differ markedly from source portfolio ROI, including sign
reversals. Explain the difference rather than choosing whichever looks better. Do
not infer causality from a regime split: time, sport mix and execution policy can
change simultaneously. Any standardized odds/EV weighting needs a declared target
population and common-support diagnostics, not an undocumented ranking adjustment.

### Kelly replay is gated and optional

Do not recreate the source's exact Kelly sizing from exported EV/odds without
confirming its probability/fee conventions and implementation. If a later research
scenario uses a documented binary fixed-odds model, identify its assumptions, inputs,
units and exclusions explicitly. Exchange commission, correlated bets, partial wins,
tickets and changing odds may require different models. Do not invent missing true
probabilities or liquidity. Fractional multipliers are scenario parameters, not advice.

Bankroll growth, drawdown or dynamic half/quarter-Kelly simulations require verified
placement/settlement order, concurrent exposure, starting capital, cash flows, fills,
fees and decision-time inputs. Current unknown chronology blocks claims of an exact
historical replay. Publish `sizing_replay_blocked` with reason codes rather than an
attractive but invalid equity curve. A fixed-unit arithmetic diagnostic does not
remove selection bias or establish opportunity availability.

## Within-filter evidence: explicit release requirement

A subgroup exclusion proposal must identify an exact saved-filter parent and its
resolved membership, including overlapping-label policy and effective-setting
uncertainty. On the same snapshot, funding/product, eligibility and evaluation window:

- Compute parent, subgroup and disjoint complement from wager identities once.
- Reconcile counts, stakes and profit; do not average or add subgroup ROI values.
- Compare weighted and equal-stake metrics and sizing-regime composition.
- Show whether the subgroup result is global, within-filter or attribution unresolved.
- Report exclusion/coverage changes and potential sampling differences.
- Present historical complement performance as descriptive, not a counterfactual
  estimate of profit from a filter edit. Include group-aware comparison uncertainty.
- Register a prospective unchanged-baseline versus exclusion TEST before new outcomes.

A global losing tennis/platform subgroup cannot justify editing a particular saved
filter without this evidence. Widening remains unsupported without observations
outside the existing rule and a documented available-opportunity sampling process.

## Minimum-wager display policy

The dashboard default is now 100 analyzed/ROI-eligible wagers per displayed cohort,
configurable through `configs/dashboard.yaml` (`min_analyzed_wagers`) and a sidebar
control. The same threshold applies to signal cards, comparison plots/tables and
saved-filter breakdowns. Show how many small cohorts are hidden and allow explicit
inspection by lowering the value. This is a display rule only, not statistical power,
independence, minimum event-group support or permission to promote a filter.

The follow-up must preserve this separation: published research eligibility/family
membership is fixed by research policy, while changing a dashboard threshold cannot
recompute tests, drop hypotheses from correction or create new claims. Test boundary
counts 99/100/101 and a five-wager -100% result. No high-return small sample may bypass
the display setting, including all-data views and exact saved-filter tables.

## Expanded delivery sequence and proposed components

Extend the logical sequence above with independently reviewable work in this order:

1. Confirm continuous-EV preservation and units; add typed numeric/provenance contract
   and precision/missingness tests without changing historical default metrics.
2. Add optional sizing-history schema, validation, unknown-linkage audits and synthetic
   fixtures (`analysis/sizing_config.py`, `io/sizing_history.py` proposed).
3. Add sizing-aware diagnostic economics (`analysis/sizing_diagnostics.py` proposed):
   weighted, equal-stake, regime and concentration metrics, eligibility reconciliation
   and group-aware uncertainty. Document unavailable exact replay.
4. Extend bounded multidimensional predicates with continuous thresholds and frozen
   hypothesis-family contracts; test endpoints, budgets and stable identities.
5. Build within-filter parent/subgroup/complement evidence using both economic
   estimands and known/unknown sizing regimes before generating exclusion proposals.
6. Add dependence-aware testing, multiplicity, overlap reduction and transparent
   shortlists; include normalized/weighted disagreements rather than hide them.
7. Publish immutable optional evidence stage, CLI, pipeline retries and verified loader;
   include new policy/history hashes in artifact IDs while retaining old readers.
8. Integrate dashboard evidence views, thresholds, explanations and prospective TEST
   proposal references; do not add execution, acquisition or scheduling controls.
9. Document methodology, evidence gaps and reproducible validation; record acceptance
   evidence and benchmark budgets before merging.

This sequence supersedes the earlier coarse commit order where the new sizing work
creates dependencies. Exact file boundaries and version compatibility still require
assessment before coding.

## Additional tests and acceptance criteria for EV and sizing

- Original EV 3.001%, 3%, 2.999% remains distinguishable; >3% differs from >=3%; bands
  do not replace row-level EV. Missing/conflicting data never silently matches.
- Weighted ROI exactly matches existing published metrics on unchanged cohorts.
- Equal stakes produce the same weighted and equal-stake mean return; unequal stakes
  with opposite outcomes demonstrate an expected sign reversal and explicit labels.
- Scaling every stake/profit consistently changes dollars, not either ROI. Changing
  only larger-bet exposure changes weighted results without overwriting unit returns.
- Zero/nonfinite stakes, half-win/loss settlements and unsupported products retain
  precise diagnostic eligibility/exclusion accounting; ticket money is counted once.
- Known, missing, overlapping and ambiguous sizing regimes have deterministic linkage;
  today's settings never fill historical gaps. Unsupported chronology blocks replay.
- Mixed funding/product and sizing regimes stay separate; paper economic outputs are
  hypothetical, not executable-net claims. Private sizing history never enters Git.
- Parent/subgroup/complement reconciliation holds under both estimands and overlapping
  attribution; global and within-filter fixtures deliberately produce different results.
- New ranking exposes stake-concentration and regime sensitivity, all tested hypotheses
  remain in multiplicity accounting, and holdout status cannot pass with unknown timing.
- UI minimum-wager setting hides 99 and shows 100/101 at default, including the small
  screenshot cohort; changing it never alters source artifacts or statistical decisions.

## Additional decisions before research implementation

Confirm whether dated bankroll/Kelly records exist, what transition date can be
supported, whether suggested versus placed/accepted stake is exported, and the exact
source EV semantics. Approve the equal-stake diagnostic estimand, proportional-return
assumptions, regime matching rules and permitted scenario exclusions. Decide whether
continuous calibration modeling belongs in the initial follow-up or a subsequent
study. Missing evidence is an explicit supported state, not a reason to reconstruct
history from present settings or infer a causal improvement from retrospective slices.
