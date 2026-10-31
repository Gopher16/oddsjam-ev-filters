# Dashboard decision workspace: opportunities, leaks and prospective feedback

Status: proposed delivery plan, October 30, 2026. This is a plan, not an implemented
ranking change or a claim that any current candidate is validated. It supplements
`MULTIDIMENSIONAL_SIGNAL_DISCOVERY.md` and replaces its remaining dashboard delivery
priorities with the sequence below. Existing published research is preserved.

## Product objective

Opening the dashboard should immediately answer:

1. Which historical patterns most warrant prospective paper study, why, how strong
   is the evidence, and exactly how should the OddsJam filter be configured?
2. Which confirmed production-filter segments warrant a controlled modification or
   exclusion test, what would change, and what remains uncertain?
3. What happened to previously proposed tests and edits, and what decision is due?

Use a visual decision workspace, with evidence and settings beside each idea.
Methodology, raw IDs, large tables and detailed prose are secondary drill-through.
The default is all reconciled history, all saved filters, all funding labels and all
products, without requiring users to discover a different evidence page. A user can
narrow scope through simultaneous selectors or select a chart element.

The dashboard must not invent statistical support, current production status,
executable capacity, historical settings or future profitability to fill a card.
An empty stronger-evidence tier should coexist with an explicitly exploratory queue
and a useful account of what evidence would change the decision.

## Diagnosis and corrections to the existing experience

- Signals & leaks currently leads with positive historical ROI and orders by event
  breadth/wager count. It is not the corrected shortlist or an opportunity ranking.
- The newer within-filter results are on a separate page, and manually selected ideas
  from conversation were never published as a reproducible recommendation queue.
- A predicate such as odds band × EV band is an analytical slice, not an entire
  scanner configuration. “Evidence and exact definition” overstates its completeness.
- Existing descriptive ROI intervals concern a subgroup's level. They are different
  from subgroup-minus-complement intervals and from multiple-comparison decisions.
- Funding/product badges describe evidence populations, not default UI filters or
  current filter lifecycle. Their prominence currently makes the experience confusing.
- Notebook-inspired visuals were planned but not delivered. They are core scope in
  this sequence, with an explicit checklist rather than a generic future follow-up.
- Paper definitions freeze intent but do not yet connect the new candidate versions
  to the legacy approval/enrollment/outcome workflow. That gap must be closed.

## All-history semantics and provenance

“All history” means a verified, reconciled universe at an explicit knowledge cutoff,
not concatenating every export/report and counting the same wagers repeatedly.

- Inventory available source imports/snapshots and choose explicit inputs. Report
  coverage dates, unique candidates, duplicates, exclusions, unattributed records,
  unavailable dimensions and source-quality gaps. Confirm whether the existing
  snapshot actually covers the user's intended history before calling it complete.
- Search across all supported funding/product populations by default. Do not require
  a real-money selection to surface a paper-origin hypothesis or vice versa.
- Keep population economics distinct. All-history visibility does not justify adding
  paper and real-dollar profit, treating incompatible products alike, or counting
  overlapping saved-filter memberships as independent opportunities.
- Distinguish filter lifecycle (effective-dated confirmed PROD/TEST/retired/unknown),
  historical funding evidence (real/paper/unknown), product, and future test funding.
  Never infer lifecycle or actual funding solely from a PROD/TEST name prefix.
- Default cards lead with idea name and evidence state. Put “Evidence source:
  historical paper observations / recorded funded wagers / mixed-source views” in
  compact secondary context with a tooltip and full counts in drill-through.
- Equivalent ideas across parents/populations may be grouped visually, with their
  constituent evidence shown separately. Formal pooling or meta-analysis requires
  a reviewed model, dependence treatment and a new versioned statistical policy.
- Unique universe totals reconcile once. Filter-level totals are explicitly nonadditive.

## One decision record and complete OddsJam settings

Publish a common decision record consumed by cards, charts, exports, saved plans
and later reviews. Include identity, source publications, parent scope, exact
predicate, historical evidence, comparison estimand, category, rank explanation,
setup readiness, missing evidence, proposed next action and associated test versions.

Represent every relevant OddsJam control, including:

| Group | Controls |
| --- | --- |
| Pricing model | Recommended Filters toggle, EV Formula Builder, required devig books/weights, devig method and model/formula version when known. |
| Universe | Sportsbook/platform, sport, league, market type and exact markets, game/event restrictions. |
| Price and edge | Minimum/maximum American odds, minimum/maximum EV where representable, endpoint operators, units and precision. |
| Execution context | Minimum liquidity with units, minimum recommended bet, market width and warnings/deduplication behavior. |
| Timing | Event horizon/date range, pregame/live eligibility where supported, timezone and any time-to-event constraint. |
| Sizing | Prospective paper sizing policy and effective-dated bankroll/Kelly provenance where applicable; account settings are distinct from scanner predicates. |

Each setting has a value plus one explicit state:

- observed: evidenced value with timestamp/source;
- inherited: copied from a named, pinned prospective parent configuration;
- override: explicit proposed change, with its supporting hypothesis;
- unrestricted: deliberately no additional restriction within the declared universe;
- unset: explicitly confirmed inactive control, not an assumed numeric zero;
- platform_default: explicitly verified platform behavior/version, not a guessed default;
- not_applicable: supported reason for the product/platform;
- unknown: unresolved, with a reason and required next evidence.

Also record evidence quality and how a proposed value was chosen: historical support,
prospective design choice, operational constraint or platform requirement. Unrestricted
within logged history is not evidence that an unrestricted live scanner is profitable.
A prospective design choice can complete a plan without masquerading as historical proof.

Render a settings sheet with baseline value, candidate override, final proposed value,
state, evidence/reference and setup blocker. A raw predicate is supplementary only.
Do not label a filter ready to configure while required settings or UI translations
remain unknown. Unknown settings may remain visible on a useful research idea.

For global slices without one parent, require an explicit new prospective baseline
and mark its untested dimensions. Do not attach a convenient parent or default after
the fact. Current screenshots establish current settings, not past configurations.
Unsupported disjunctions/exclusions, exact > versus >= boundaries, EV upper bounds,
market-family mappings or platform controls must be flagged; provide a manual
capture rule or several child filters only when their equivalence can be demonstrated.
A displayed market family need not map one-to-one to OddsJam's exact markets.
Changing a devig model, formula, reference-book weights or minimum recommended bet
cannot be retrospectively evaluated from a stored EV number alone. Require the
necessary decision-time quotes/model inputs and historical sizing evidence, or mark
the change as a prospective hypothesis with no reconstructed historical performance.

## Evidence and prioritization policy

Replace implicit “largest positive subgroup wins” with an auditable queue. Keep
research strength, economic association, setup readiness and next action distinct.

Suggested evidence tiers:

1. Independent prospective evidence meeting a predeclared review policy, if available.
2. Corrected historical comparison evidence with stated chronology/dependence limits.
3. Support-qualified exploratory association, explicitly unconfirmed.
4. Insufficient or conflicting evidence, available for investigation, not a headline edge.

The current publication has 602 candidates, 42 support-qualified comparisons and no
corrected shortlist entries. A redesign must reproduce that result, not relax gates
or silently relabel exploratory candidates as statistically backed.

Within a tier, publish a deterministic policy considering both return estimands,
subgroup/complement contrast, uncertainty, effective event support, concentration,
sizing sensitivity, configuration completeness and overlap. Do not invent a 0–100
“probability of edge” or multiply significance by ROI as an unexplained score.

Proposed default eligibility and ordering to formalize in PR C:

- Opportunity direction requires positive subgroup returns and positive
  subgroup-minus-complement contrasts under both return measures. Leakage direction
  requires negative subgroup returns and negative contrasts under both measures.
  Disagreement goes to a separate sizing/conflict investigation view.
- Stronger-evidence tiers require their actual corrected/prospective criteria;
  passing a support gate alone never moves an idea into those tiers.
- Within the exploratory tier, publish nondominated fronts over conservative
  contrast evidence (the weaker lower interval bound across both measures, reversing
  direction for leaks), minimum arm event support and worst arm concentration.
  These are pointwise-bound research priorities, not simultaneous confidence claims.
- Within a front use minimum arm event breadth descending, worst event stake share
  ascending, then stable identity. Show historical effect and loss burden alongside;
  do not silently rank by dollar profit. Setup readiness is a visible separate state.
- Apply exact-equivalence suppression after formal correction. Group related ideas
  and prefer distinct representatives; preserve links to the full family and
  cross-event overlap. Do not fill a four-card display with redundant variants.
- Freeze this presentation policy before inspecting its new ranked output, audit
  sensitivity and label its tiers correctly. If a reviewed alternative is selected,
  publish the change rather than silently tune ranks to a preferred anecdote.

Separate “priority for further research” from “evidence of a deployable advantage.”
Use explicit components and short reasons such as “more event support; both return
measures agree; uncertain improvement; quote evidence missing.” Show the trade-offs
instead of making high-dollar historical profit dominate ranking.

Potential impact may include historical loss burden or notional fixed-stake
association, with the relevant denominator. Forecasted profit needs supported future
opportunity frequency, actual availability, fees/capacity and a validated return
estimate; otherwise display unavailable. Logged volume is not scanner supply.

Retain the approved support/concentration gates and full correction family. Any search
expansion, formal ranking/model change or inference policy gets a new predeclared
publication and tests. Simulate null/error behavior and assess power/sensitivity
before stronger inferential claims. Cross-publication repeated looks need a declared
policy; no repeated threshold tuning against the same data and calling it validation.

Catalogue opportunities beyond existing saved-filter parents in a separate global
hypothesis space. The current search is bounded to three declared view types and
EV >3%; it is not an exhaustive optimization of every OddsJam knob. Expand only
supported dimensions (e.g. odds/EV ranges, league/platform/market, later trustworthy
liquidity/timing) with budgets, exact predicates, complete-family accounting and
unavailable-feature audits. Settings absent from the source cannot be optimized from
outcomes. Historical unlogged opportunities cannot be inferred from selected bets.

## Opportunities and production leaks

Opportunities show scope, direction, parent/context, both returns, comparison
uncertainty, evidence tier, setup readiness, complete settings and next test action.
If no idea meets stronger tiers, default to plainly labeled exploratory study ideas.

Leaks must be tied to an exact confirmed production filter/configuration version.
Unknown current status is a confirmation task, not proof of PROD weakness. Present:

- Parent, suspect subgroup and retained complement under the same economics.
- Historical loss burden, weighted/equal-stake comparisons and their uncertainty.
- Exact change: tighten threshold, restrict market/platform, or exclude a subgroup.
- Before/after settings diff, removed population/stake, retained exposure and overlap.
- Whether the scanner can express that change without removing unintended bets.
- A baseline-versus-modified paper test; no causal improvement or profit-saved claim.

Display negative funded profit with nonnegative unit returns as sizing sensitivity,
not an automatic market leak. Global losses can prompt an investigation but cannot
justify changing a particular parent. Widening a filter requires prospective coverage
of the wider universe; the history of a narrower filter cannot establish the gain.

## Default visual experience

The first viewport contains a compact overview and two queues: **Ideas to test** and
**Production changes to investigate**, plus an evidence-versus-impact plot. Three or
four cards per queue are a display cap, not a quota requiring fabricated winners.
Each card contains one sentence explaining priority, a compact comparison graphic,
support/uncertainty, setup status, and actions to inspect settings or draft a test.
No raw JSON, repeated long warnings or methodology wall above the actionable view.

Global controls default to all and combine with AND; values within a field use OR.
Use saved filter, platform, sport/league, market, odds/EV ranges, evidence tier,
configuration status and confirmed lifecycle; add timing/liquidity where supported.
Display selectors do not rerun tests. Arbitrary unsupported intersections require
explicit publication, not summing/intersecting aggregate rows. Persist/reset scope
consistently; show counts before/after filtering and empty results without fallback.

Suggested navigation: Overview; Opportunities; Production health; Explore history;
Tests & reviews. Keep technical methodology/provenance in secondary detail.

### Notebook chart checklist

Reuse notebook questions and pure calculation helpers where contracts agree. Do not
embed screenshots, rerun notebooks on dashboard load, or carry old notebook outputs
into a new report. Initial rendering can use existing Altair/Streamlit dependencies.

| Requested chart | Planned dashboard version and interaction | Release condition |
| --- | --- | --- |
| Bet volume over time | Daily/weekly/monthly line/bars, date brush, platform legend, partial-period indicators. | Declare logged-wager grain/date basis; missing periods differ from confirmed zeros. |
| Average bets by weekday/time bucket | Weekday bars and optional weekday × hour heatmap; hover gives count and covered-day denominator. | Verified calendar/timezone basis, DST/partial-day handling; not scanner opportunity volume. |
| EV ROI versus realized ROI scatter | Matched-cohort bubbles, size legend, selectable labels, zero/identity references, linked settings. | Same weighting/eligible cohort on both axes; distinguish modeled EV from realized results. |
| Filter-level modeled versus realized ROI | Same shared chart component with filter comparison, lifecycle shapes and support/uncertainty. | Source and current lifecycle identified separately; all outliers visible. |
| EV-bucket summary | Linked return/EV/volume/profit panels; hover/click highlights a bucket and candidate. | Versioned display edges preserve original continuous EV; both weighted and unweighted views. |
| Time-to-event summary | Ordered bands with ROI/EV, support and dollars; separate after-start/unknown categories. | Compatible source timestamps and explicit chronology/timezone semantics. |
| Liquidity summary | Platform-specific panels with missingness and source-quality indicators. | Trustworthy units/source/timing; bucket proxies never presented as observed executable dollars. |
| Liquidity × EV summary | Heatmap plus linked returns/support panel; clicking a published cell opens its scope. | Same liquidity gates; no opportunistic new test on click. |
| Cumulative profit by platform | Zoomable line chart with legend isolation; dollars/fixed-unit views and explicit time basis. | Settlement-time curve needs settlement time. Source-date attribution is labeled retrospective, not bankroll balance. |
| Profit by sport/platform/league/market | Diverging horizontal bars with metric switch and support; click through to opportunity/leak evidence. | Unique eligible economics; profit and ROI are separate measures; overlap not summed. |

The two screenshot scatter variants share one reusable component; all requested
analytical questions remain in scope. Additional default visuals: paired
weighted/equal-stake dots; subgroup-versus-complement forest plot; exposure
concentration; overlap/event-dependence view; prospective quote-availability funnel.

Replace the notebook label “Realized EV (CLV)” with “closing-price-derived EV estimate”
only if reference odds, book, transformation and observation timing support it. CLV
is not realized profit or verified true EV. Use matched-cohort denominators or separate
panels. Missing or unsupported series are unavailable, not zero.

Publish chart eligibility and source-quality audit. For blocked charts identify the
specific missing fields, required evidence and a bounded remediation task. Audit
existing files first; unknown timing must not become a blanket excuse to postpone
all descriptive charts. Raw source-date distributions may be possible even when
causal timing or settlement-order claims are blocked, if labeled accurately.

No chopped axes hiding outliers, overlapping permanent labels or mixed % units.
Use full hover names, accessible colors/shapes, zero references, zoom/reset, legend
selection, responsive layout and keyboard-accessible tables for each visual.
Minimum 100 applies to performance cohorts; it must not erase low-volume daily
activity periods. Keep sparse/unknown/unpublished/zero states distinguishable.

## Explainability workstream: global and filtered populations

The objective is to understand which recorded features a model associates with
per-unit returns, where those associations vary and how stable they are. SHAP explains
a fitted prediction relative to its background; it does not identify causes or prove
that changing a control will increase profit. The UI must distinguish model
attribution, observed cohort outcomes, statistical comparisons and prospective evidence.

### Feature inventory and decision-time eligibility

Inventory every exported/canonical/derived field with type, units, grain, availability
at observation time, coverage, missing/conflict status, lineage and exclusion reason.
Expansive coverage means reviewing all fields, not feeding every column into a model.
Candidate families include:

- Platform/exchange, product, sport, league, market type/family and exact market.
- Original continuous advertised EV and offered odds; explicit odds transformations.
- Verified time-to-event, weekday, hour/calendar context and seasonality.
- Sourced liquidity, market width, quote age or other execution context when present
  at decision time and semantically supported.
- Confirmed active filter/configuration version and devig/formula settings only where
  time-correct membership/configuration is actually available.
- Sizing-regime/account context in a separate sensitivity specification; actual stake
  is not silently allowed to dominate the primary equal-stake return model.

Do not infer unavailable features from names, outcome timestamps or current settings.
Exclude outcomes, profit/return/payout, settlement/result status, closing odds/CLV and
post-decision fields from decision-time predictors. They can be targets or separately
labeled diagnostic outcomes, never leakage-prone predictors. IDs/free-text labels
with memorization risk are linkage/context by default, not raw numeric/categorical
predictors. Train-fold-only encoders, missing-value handling and feature selection
must preserve unseen-category and missingness audits.

Correlated representations (sport/league, raw EV/EV bands, odds/odds transforms,
platform/product, filter names/configuration) need grouped summaries and redundancy
checks. Avoid presenting every one-hot level as an independent driver. Prefer raw
continuous variables; bins remain descriptive displays. Missingness itself may reflect
export/regime changes rather than an economic edge.

### Targets, population and weighting

Primary proposed target: per-wager unit return `profit / positive stake` on the
common eligible economic cohort, with equal-wager fitting weights declared explicitly.
Group evaluation/resampling by event rather than counting duplicate exports/legs or
overlapping saved-filter memberships as additional observations. Avoid modeling tiny
aggregate filter ROI rows as independent training examples.

Model-predicted unit returns aggregate to an equal-stake cohort estimate. If a second
stake-weighted analysis is published, declare its training/evaluation weights and
show it separately. Actual dollar profit also depends on stake, bankroll/Kelly policy,
volume and fees; it is not interchangeable with the primary target. Binary win
probability is an optional separate model requiring settlement support, odds-aware
calibration and payout conversion; win rate alone does not establish profitability.

Default all-history scope reviews every supported population, but does not silently
blend incompatible product settlements or paper/funded dollar economics. Use separate
compatible-population models/facets first. Any pooled model needs explicit population
features, shared target definitions and population-specific diagnostics. Funding and
filter lifecycle remain context/provenance, not causal endorsements.

### Model and validation policy

Start with a constant/odds-aware baseline and regularized additive/linear comparator;
add one bounded tree-based regression candidate if warranted. Select the final model,
loss/weights, transformations, feature set and tuning budget before using explanations
to select new hypotheses. SHAP/library/estimator choices get pinned versions and
benchmarks in PR E; no new ML dependencies or training are required in PR A.

Prefer grouped chronological development/evaluation with a final untouched period and
appropriate event grouping/embargo once time semantics are verified. Fit preprocessing,
feature selection and SHAP reference/background selection on training data only.
When chronology is unverified, label any grouped non-temporal evaluation exploratory;
never present it as forward validation. Preserve disjoint event IDs across folds.

Report held-out/cross-fitted prediction error versus baselines, return calibration by
predicted-return bands, residuals, coverage, group/time stability and model sensitivity.
Do not promise individual-bet predictability from a noisy return target. A negative
or inconclusive model result is a supported outcome and limits interpretation.

SHAP output must carry target units, model-output scale, expected/reference prediction,
background identity and feature representation. Demonstrate reconstruction of the
prediction from reference plus contributions. Do not mix log-odds SHAP, probability
SHAP and return-per-dollar SHAP. For cross-fitted explanations retain fold-model identity;
use fold-aware summaries or common-reference comparisons and do not interpret fold
baseline shifts as a cohort effect. Group-aware resampling/repeated splits can assess
ranking stability, but do not invent significance or calibrated causal confidence.

### Two sidebar interaction modes

1. **Explain selected rows with the fixed published model (default).** All eligible
   history is visible initially. Sidebar filters select the matching row IDs and their
   precomputed predictions/SHAP vectors. Summarize mean absolute and signed contributions
   for that selection, compare with the reference population, and keep the original
   model/background fixed. This is cohort exploration, not a refit. A feature fixed by
   the sidebar can still contribute relative to the global reference; explain that.
2. **Fit a model specifically for this scope (explicit optional run).** Export a typed
   scope/configuration request for an offline versioned build. Enforce eligible/event
   support, diversity, compute limits, split rules and tuning budgets. Sparse scopes
   decline fitting with reasons; fixed fields have no estimable within-scope variation.
   A new model has a new reference and cannot be compared to global SHAP bars as if
   the values shared the same baseline. Do not repeatedly fit clicked subsets on an
   unchanged holdout and claim independent validation.

The minimum-wager display threshold is not a sufficient model-training threshold.
Define model eligibility from event support, outcome variation, feature coverage,
learning curves and model complexity in PR E; numerical thresholds need explicit
review rather than copying the existing inference policy without justification.

### Explainability visual catalogue

- Global/grouped mean absolute SHAP importance, with attribution direction separate
  from magnitude and stability/coverage shown.
- Beeswarm/distribution plots, with categorical level displays and grouped original
  features instead of overwhelming one-hot lists.
- Dependence plots for continuous EV, odds and supported timing/liquidity, colored by
  a selected interacting field. Display observed support/density and avoid causal labels.
- Selected-cohort versus reference importance/effect comparison with the fixed model.
- Local waterfall for a chosen eligible observation, with exact context, reference,
  prediction, realized outcome where known, and feature contributions.
- Observed/predicted return calibration and residual plots by platform/market/time,
  plus model quality and missingness panels.
- Held-out grouped permutation importance or feature-group ablations as complementary
  checks; conditional/ALE-style effect plots where correlated features make ordinary
  partial dependence unrealistic, with separate explicit assumptions.
- A small predeclared interaction exploration budget; no unbounded pair mining or
  automatic hypothesis promotion from high SHAP interactions.

Treat model-driven patterns as hypotheses entering the same bounded search, correction,
configuration and prospective test workflow. High SHAP importance neither supplies
unknown scanner settings nor licenses changing a production filter. Explanations must
have an explicit “model association, not proven intervention” interpretation.

### Explainability publications and acceptance

Publish feature inventory, model specification/utility report, model/background IDs,
training/evaluation memberships, immutable preprocessing/model fingerprints,
row-level predictions, SHAP vectors, grouped summaries and scope/model eligibility
audits. Keep private rows/artifacts ignored and never load untrusted executable model
pickle files in the dashboard; the viewer consumes verified tables only.

Use deterministic background sampling, row/model seeds, finite CPU/memory/explanation
budgets and verified replay. Test source/canonical reconciliation, duplicate membership,
leakage columns, unknown chronology, unseen categories, constant/sparse subsets,
additivity/units, feature grouping, all-history versus selected-row identity, artifacts
corruption and responsive browser interactions. Disclose sampling/approximation and
never portray imputed values as observations.

Primary methodological references: SHAP's predictive-versus-causal interpretation
example (https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html),
SHAP grouped importance documentation
(https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/plots/bar.html),
and scikit-learn's leakage and evaluation guidance
(https://scikit-learn.org/stable/common_pitfalls.html,
https://scikit-learn.org/stable/modules/cross_validation.html).

## Proposed PR and Conventional Commit sequence

These are proposed scopes; do not preassign GitHub PR numbers. Each PR uses the
repository template, synthetic tests and explicit local-artifact smoke checks.

### PR A — Decision records, full configuration and lifecycle provenance

- `feat: define complete candidate configuration and evidence states`
- `feat: resolve prospective baselines and publish parameter change sheets`
- `test: cover unknown inherited unset and unsupported scanner settings`

Deliver the shared decision record, per-control states, source-versus-proposed
settings, effective-dated PROD status and complete before/after sheets. Reuse paper
definitions with a versioned extension/adapter; preserve old definitions/readers.
Acceptance: every OddsJam control has an explicit state; global slices cannot inherit
an invented baseline; unavailable settings are visible; no historical relabeling.

### PR B — Publish notebook visual diagnostics and source readiness

- `feat: publish matched-cohort visual diagnostics and coverage audits`
- `feat: add sourced activity timing liquidity and closing-price chart contracts`
- `test: reconcile visual totals units chronology and missingness`

Read notebook sources and existing helpers. Publish immutable optional tables for
all requested charts, supported series and explicit per-series unavailable reasons.
Resolve source/date/liquidity/CLV questions using existing files where possible.
Acceptance: all ten requested chart questions have an implemented dataset or a
specific auditable blocker/remediation item; no private datasets committed.

### PR C — Unified opportunity and within-PROD change queue

- `feat: publish auditable all-history opportunity and leak review queues`
- `feat: prioritize evidence tiers impact uncertainty and distinct hypotheses`
- `test: validate complete-family ranking overlap and production scope`

Consume A and existing inference; publish deterministic tiers/order and explanations.
Add bounded global hypotheses and supported range searches only through explicit
new policy/configuration, never hidden UI optimization. Preserve separate population
evidence; simulations/sensitivity tests cover any new formal inference. Publish
parent/subgroup/complement change proposals and setup readiness.
Acceptance: default suggestions reproduce queue artifacts, no conversation-only list,
no raw-ROI sorting as proof, no forced winners, no inferred current PROD from names.

### PR D — Redesign the default dashboard and deliver interactive charts

- `feat: make opportunity and production review the default dashboard`
- `feat: add linked notebook-inspired charts and complete setup drill-through`
- `test: verify visual interaction scope empty states and accessible layouts`

Consume A–C, replace competing descriptive/corrected recommendation surfaces with one
coherent home, and move legacy descriptive cards into history exploration. Integrate
all supported requested charts, comparison/uncertainty views, complete settings,
shared filters and candidate-to-definition actions. Reuse common chart functions.
Acceptance: a user identifies the leading study idea and leading production concern,
understands why each is shown, and opens the full setup sheet without reading raw
JSON. Default all-history view needs no funding/product selection. Unsupported
charts show their actionable source blocker; they are not silently omitted.

### PR E — Feature contracts, explanatory models and frozen SHAP publications

- `feat: publish decision-time feature eligibility and model evaluation contracts`
- `feat: fit versioned return models and publish stable SHAP explanations`
- `test: guard explanation leakage dependence and unsupported predictive claims`

Audit all available columns/derivations, specify targets and eligible populations,
compare simple baselines with a bounded nonlinear model, and publish predictions,
validation, SHAP values and source quality. Fit preprocessing/background selection
on training data only. Support versioned explicit subset-model requests offline;
no dashboard-triggered fitting or post-selection validation claims. Preserve source
IDs solely for linkage. Timing unknowns block forward-validity claims, not an honest
exploratory-only model with explicit limitations.
Acceptance: null/synthetic signal tests, shared-event separation, unseen categories,
SHAP additivity on the declared output scale, fixed training background, grouped
features, seed/replay stability, population/weighting reconciliation and explicit
model-utility status. A model that cannot improve on simple baselines must display
that result and cannot drive the recommendation queue as established evidence.

### PR F — Interactive global and selected-population explanations

- `feat: add linked SHAP and feature-effect exploration to the dashboard`
- `feat: compare selected cohorts with frozen model explanations`
- `test: preserve model references across filters and explanation empty states`

Add global importance, direction/distribution, dependence and local explanations;
connect the shared sidebar and candidate/filter drill-through. Explain selected
rows from frozen model outputs immediately. Compare population coverage, context,
held-out utility and observed outcomes alongside attribution. Subset-specific fitted
models are explicitly identified, separately published and never implied by a filter
change. Persist the exact model/background/fold/feature/target versions.
Acceptance: filters operate on eligible row identity, aggregate no duplicate
memberships, show fixed/excluded features and sparse/unknown states, and do not
silently change the model/background. No “SHAP proves a profitable setting” text or
automatic production action. Browser checks cover linked selections and full labels.

### PR G — Definition-to-experiment bridge and prospective outcome linkage

- `feat: bind paper definition versions to approved experiment protocols`
- `feat: capture prospective observations availability and versioned outcomes`
- `test: prevent retroactive enrollment cross-version pooling and implicit launches`

Extend rather than bypass the existing approval/register/report workflow. Support
typed numeric predicates and subgroup/complement design with stable observation IDs,
decision-time evidence, source links, late/unsettled outcomes, failed quotes,
shared-event handling and explicit actual launch/end evidence. Publish per-version
metrics and review readiness back to the same decision workspace.
Acceptance: a synthetic candidate completes draft → approved registration → launch →
capture/check/link → settlement/report → scheduled review. Historical exports and
elapsed dates never fabricate enrollment or approval. Missing data stays explicit.

### PR H — Reviewed filter modifications, promotion and monitoring

- `feat: record reviewed paper decisions and effective-dated filter changes`
- `feat: monitor deployed filter versions drift and rollback evidence`
- `test: preserve decision lineage and block unsupported promotion`

Support continue/new version, modify/new test, stop, inconclusive, and explicit human
production approval only under a declared review policy. Pin exact scanner settings,
UI-translation evidence, review report, configuration diff and effective date.
Record confirmed external application separately from approval. Add monitoring and
rollback/retest actions; no wager placement or automatic deployment.
Acceptance: a losing subgroup cannot become an automatic exclusion; insufficient
support cannot become an automatic pass; current status and subsequent economics
remain tied to the applied configuration version. Paper profit does not establish
real fills/capacity. Define review/error-control policy before interpreting outcomes.

### PR I — End-to-end acceptance, documentation and release validation

- `test: exercise the complete discovery test review and modification cycle`
- `docs: document dashboard decisions source coverage and operational limits`
- `chore: verify dashboard performance and release readiness`

Run synthetic positive/negative/ambiguous/no-winner scenarios and explicit private
snapshot smoke checks. Produce visual acceptance evidence, latency/memory benchmarks,
replay/corruption checks and a reconciliation report for every requested feature.
Close the chart checklist with implemented status or specific supported-state blockers.
Acceptance includes a browser walkthrough at normal laptop widths and usability checks
for chart selection, reset, hover, deep links, settings sheets and review actions.
Do not treat passing AppTest alone as proof that the visual design is good.

## Research, release and feedback-loop safeguards

Retain approved current gates: each arm ≥100 eligible wagers and ≥100 namespace/event
proxies; complete event-ID coverage; largest wager ≤5% and event ≤10% of arm stake;
positive finite stakes; 5,000 paired draws; Holm 0.05 across the entire declared
candidate × measure × direction family. These are operational gates, not a power
guarantee. New statistical rules require a reviewed policy and separate publication.

Persist data/decision/code lineage, original snapshots, old registries and published
outputs. Builders run explicitly outside Streamlit. Dashboard interactions read
verified artifacts and do not acquire data, resample, rerank by fitting a model,
activate filters or execute trades. Approved local register actions remain distinct
from immutable dashboard research views and external platform confirmation.

A new export creates a new snapshot and explicit build chain. Reconcile observations,
update source coverage and known lifecycle/configuration periods, publish new
candidate/visual/decision/test reports, and link the dashboard to exact versions.
Track candidate identity, parent/current config, test-definition version, enrollment
and quote records, review report, human decision, applied change and monitoring
period. Historical and prospective evidence stay separately identifiable.

## Project-level completion criteria

- The default view visibly answers both guiding questions using all supported history.
- Promising, corrected, exploratory, conflicting and insufficient are distinguishable.
- Complete settings sheets and specific modification diffs replace ambiguous JSON.
- Requested notebook visuals have shipped or have individual, evidence-based blockers.
- Global scope does not pool incompatible economics or hide unclassified evidence.
- Confirmed production status is independent of a name or historical funding label.
- Global and filtered explainability views disclose model utility, target units, fixed versus refitted scope and reference population.
- A complete synthetic prospective test and modification cycle is reproducible.
- Future exports refresh the feedback loop without rewriting past evidence or tests.
- No dashboard state implies guaranteed profitability, automated promotion or fills.

The research plumbing delivered so far is reusable. The remaining work is the product
and operational connection between evidence, complete settings, prospective outcomes
and reviewed changes; it should no longer be described as merely optional polish.
