# Multidimensional discovery implementation status

Reviewed against the repository after release acceptance merged on November 2, 2026.
This ledger reconciles the [original discovery brief](MULTIDIMENSIONAL_SIGNAL_DISCOVERY.md)
and its subsequent identity/visual requirements with the delivered
[A–I workspace design](DASHBOARD_DECISION_WORKSPACE.md). It describes implemented
capability, not a current account audit or an approval to deploy a filter.

The original brief is retained as historical design context. Its then-current
“two dimensions” and future-tense wording do not describe the present dashboard.
Proposed table/file names were provisional; actual replacements are mapped below.
No historical source, parameter value, publication or contributor note is rewritten
by this status assessment.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **Implemented** | Existing implementation and verification evidence support the stated bounded capability. |
| **Implemented, evidence-blocked** | The capability exists, but the selected historical source lacks required evidence. This does not promise any missing algorithm is implemented. |
| **Partially implemented** | Identify the supported part and remaining implementation explicitly. |
| **Optional extension** | Deliberately deferred, with prerequisites; unnecessary for supported candidate review. |
| **Superseded** | A later reviewed design replaced the proposal; follow the named replacement. |

Evidence links below include component guides, actual modules/configurations and
synthetic tests. The [dated acceptance record](../releases/2026-11-02-acceptance.md)
reports the explicit private snapshot smoke, browser checks, 522-test baseline and
engineering budgets. Counts/readiness there describe that snapshot only. They do
not establish today's account state or additional temporal/statistical validation.

## Intent, safeguards and assessment

| ID | Requirement from the brief | Status | Evidence / interpretation / next action |
| --- | --- | --- | --- |
| S01 | Bounded multidimensional TEST hypotheses and saved-filter context | Implemented | [Discovery](../WITHIN_FILTER_DISCOVERY.md) supports declared views up to depth four with parent/subgroup/complement identities. |
| S02 | Explicit verified canonical/atlas inputs and manual CSV ingestion | Implemented | [Ingestion](../INGESTION.md), [canonical](../CANONICAL_FACTS.md) and explicit builders; no automatic source acquisition. |
| S03 | Preserve sources, historical outputs, notebook inputs and preprocessing defaults | Implemented | Immutable writers/replay; [release checks](../RELEASE_ACCEPTANCE.md). Legacy preprocessing remains independent. |
| S04 | Separate real-money/paper/unknown and product economics | Implemented | Canonical eligibility, sizing and queue population boundaries; no pooled headline ROI. |
| S05 | Preserve reported USD profit and unsupported-product/ticket/refund exclusions | Implemented | Canonical/diagnostic economics and reconciliation; fee inclusion remains unverified rather than estimated. |
| S06 | Preserve unknown chronology, scope, timing and availability | Implemented, evidence-blocked | Explicit source states/gates; historical dates do not establish decision-time availability or forward validation. Obtain source provenance before stronger claims. |
| S07 | No scheduling, live scanner, wagers, automatic filter edits or dashboard fitting | Implemented | [Core configuration](../../configs/weekly_research.yaml) forbids scheduling; viewer consumes publications; approval/application are explicit records. |
| S08 | Do not infer liquidity, CLV validity, fills or PROD from names | Implemented | Visual readiness, configuration/lifecycle evidence and prospective quote checks preserve those distinctions. |
| S09 | Assess existing grains, identities, inference and compatibility before extension | Implemented | Component contracts and tests below separate core discovery, within-filter discovery, prioritization, global catalogue and models; old readers/publications remain supported. |
| S10 | One separate pre-code compatibility/migration proposal and supported-intersection census | Partially implemented | Compatibility, coverage and exclusions are documented/published across components; there is no single standalone assessment artifact matching the initial brief. Current [operator walkthroughs](../walkthroughs/README.md) and [contract references](../reference/README.md) consolidate these links; do not invent an earlier standalone pre-code review. |

## Historical identity requirements

| ID | Requirement | Status | Evidence / next action |
| --- | --- | --- | --- |
| H01 | Preserve original label strings; separate source observations, wagers and attribution counts | Implemented | [Inventory](../FILTER_LABEL_INVENTORY.md) and `label_inventory.py`; repeated exports/legs and embedded commas are covered by synthetic tests. |
| H02 | Source/snapshot lineage, timestamp bounds/limitations and overlap inventory | Implemented | Inventory audits retain count grains and chronology limitations; private inventories remain ignored. |
| H03 | Exact registry/alias matching before proposing stable identities and segmentation | Implemented | [Mapping proposals](../FILTER_MAPPING_PROPOSALS.md); preserve existing IDs and distinguish clones from rename proposals. |
| H04 | Ambiguity-only review, evidence/confidence/reasons, unresolved decisions retained | Implemented | Reviewed mapping evidence; a likely relationship is not silently approved. New ambiguity needs supporting identity evidence. |
| H05 | Versioned attribution, mapping hashes and unchanged-economics before/after audit | Implemented | [Attribution revision](../VERSIONED_FILTER_ATTRIBUTION.md); old snapshots and registry revisions preserved. |
| H06 | Deleted settings stay unknown; identity does not imply historical configuration/funding | Implemented, evidence-blocked | Membership can be analyzed while settings remain unknown. Resolve only with contemporaneous or explicitly scoped evidence. |
| H07 | Dashboard resolved selection and unresolved/missing coverage without invented performance | Implemented | Canonical attribution/atlas coverage and verified dashboard attachments. |

Evidence: [inventory module](../../src/oddsjam_ev/analysis/label_inventory.py),
[mapping module](../../src/oddsjam_ev/analysis/filter_mapping_proposal.py),
[attribution module](../../src/oddsjam_ev/analysis/filter_attribution.py),
[inventory tests](../../tests/test_label_inventory.py),
[mapping tests](../../tests/test_filter_mapping_proposal.py),
[revision tests](../../tests/test_attribution_revision.py).

## Policy, tables and research evidence

| ID | Requirement | Status | Evidence / next action |
| --- | --- | --- | --- |
| D01 | Strict versioned policy, feature allowlist, explicit combinations up to four dimensions | Implemented | [Multidimensional configuration](../../configs/multidimensional_discovery.yaml), strict parser and frozen family. No unrestricted Cartesian or learned search. |
| D02 | Numeric units, finite cutpoints, open/closed endpoints and exact >3% | Implemented | Original decimal EV text drives exact predicates; [EV contract](../CONTINUOUS_EV_CONTRACT.md). Display bands are separate. |
| D03 | Candidate/parent/membership/scan/resampling/overlap budgets, no silent truncation | Implemented | Discovery and prioritization have separate fail-closed budgets and failure audits; changing them creates new lineage. |
| D04 | Missing/invalid/conflicting categories excluded explicitly, retained in parent/complement | Implemented | Discovery membership/exclusion ledger and reconciliation tests. No imputation from filter names. |
| D05 | Reviewed inferential gates distinct from minimum-wager display convention | Implemented | [Prioritization policy](../../configs/signal_prioritization.yaml): each arm ≥100 wagers/events, complete IDs, 5%/10% concentration, positive stakes, no extra dollar floor. |
| D06 | Fixed development/holdout windows, information availability and embargo | Partially implemented | [Core temporal gates](../../src/oddsjam_ev/analysis/temporal_validation.py) exist. New within-filter prioritization remains non-temporal; it has no integrated chronological holdout/embargo evaluator. Requires verified timing, predeclared windows/policy and a scoped research extension. |
| D07 | Complete candidate × measure × direction family, including blocked comparisons | Implemented | Separate prioritization preserves four hypothesis slots per frozen candidate, with unavailable results explicit; global catalogue does not borrow these tests. |
| D08 | Fixed seeds, canonical predicates, stable identities and input/config/code hashes | Implemented | Discovery/prioritization immutable publishers and deterministic partition-based resampling. |
| D09 | `signal_candidates`, evidence, comparisons and wager-level membership/digests | Implemented | [Within-filter publisher](../../src/oddsjam_ev/io/within_filter_discovery.py) emits candidates/evidence/comparisons/membership/exclusions and audit. Ticket economics count once. |
| D10 | Raw/adjusted tests, intervals, coverage, concentration and explicit unknown status | Implemented | [Prioritization](../STATISTICAL_PRIORITIZATION.md) adds `signal_statistics`/hypotheses and pointwise paired-bootstrap intervals; approximate cluster tests are separately identified. |
| D11 | Overlap, equivalence, nesting and deterministic presentation suppression | Implemented | Prioritization overlap plus [review queues](../REVIEW_QUEUES.md) preserve full families while grouping/distinguishing displayed hypotheses. |
| D12 | Shortlists with evidence categories, transparent order, blockers and TEST intent | Implemented | Corrected comparison versus support-qualified exploratory versus insufficient/conflicting tiers; empty queues valid; complete setup/drafts linked. |
| D13 | Search audit: screened/excluded counts, reconciliation, family and budget usage | Implemented | Separate discovery/prioritization/queue audit publications; no single monolithic `signal_audit` table is required. |
| D14 | Dependence-aware shared-event resampling and group separation | Implemented | Namespace/event union draws retain cross-arm covariance; independence of event proxies remains an assumption, not an inferred fact. |
| D15 | Account for favorable thresholds/parents and prohibit reused-holdout validation | Implemented | Frozen family correction and explicit exploratory/non-temporal labels; correction across repeated future publications is not a sequential-testing policy. A new research design is needed for repeated-look control. |
| D16 | Specificity alone earns no priority; missing intervals and disagreement visible | Implemented | Frozen queue tiers/fronts/support/concentration and sensitivity ordering. Rankings do not estimate probability of profit. |
| D17 | Validated evidence only after independent requirements actually pass | Implemented, evidence-blocked | Historical queues cannot gain independent prospective/production labels from source results. Capture and review a real predeclared study; synthetic acceptance is not that evidence. |

Evidence: [discovery implementation](../../src/oddsjam_ev/analysis/multidimensional_discovery.py),
[parser](../../src/oddsjam_ev/analysis/multidimensional_config.py),
[paired comparisons](../../src/oddsjam_ev/analysis/paired_comparison.py),
[prioritization](../../src/oddsjam_ev/analysis/signal_prioritization.py),
[publication](../../src/oddsjam_ev/io/signal_prioritization.py),
[queue ranking](../../src/oddsjam_ev/analysis/review_ranking.py),
[discovery tests](../../tests/test_within_filter_discovery.py),
[publication tests](../../tests/test_within_filter_publication.py),
[inference tests](../../tests/test_statistical_prioritization.py),
[prioritization publication tests](../../tests/test_prioritization_publication.py),
[queue tests](../../tests/test_review_queues.py),
[temporal tests](../../tests/test_temporal_validation.py).

## Within-filter changes, exact EV and sizing

| ID | Requirement | Status | Evidence / next action |
| --- | --- | --- | --- |
| W01 | Exact resolved parent membership; subgroup and disjoint complement reconcile | Implemented | Same snapshot/population/eligibility and wager identities; global results never replace within-filter evidence. |
| W02 | Compare weighted/equal-stake economics, regime composition and coverage changes | Implemented | [Sizing diagnostics](../SIZING_DIAGNOSTICS.md), discovery comparisons and evidence tables. Historical differences remain non-causal. |
| W03 | Unchanged prospective baseline versus whole-subgroup exclusion | Implemented | [Configuration records](../DECISION_CONFIGURATION_RECORDS.md) and [definitions](../PROSPECTIVE_PAPER_DEFINITIONS.md); parent AND NOT whole conjunction, not independent control inversion. |
| W04 | Available observed universe outside current settings for widening | Optional extension | Selected wagers cannot establish it. Collect comparable outside-rule observations/availability before designing a widening evaluator. Current contracts disclose the missing universe. |
| E01 | Preserve original EV text/provenance, exact numeric percent-point/fraction representations | Implemented | [Continuous EV implementation](../../src/oddsjam_ev/analysis/continuous_ev.py), immutable EV publications and precision/conflict audit. |
| E02 | Distinguish 3.001%, 3%, 2.999%, > versus >= and display bands | Implemented | [EV tests](../../tests/test_continuous_ev.py) and discovery endpoint tests. Invalid/nonfinite/missing/conflicting values do not match. |
| E03 | Finite predeclared threshold grid, no inference on slider interaction | Implemented | Frozen discovery family and display-only selectors. |
| E04 | Separate threshold-sensitivity publication across multiple cutpoints | Partially implemented | Multiple declared thresholds can be built/audited; no dedicated sensitivity publication implementing the full original proposal. Requires a chosen grid and separate family-aware design. |
| E05 | Common-cohort EV versus return diagnostics with explicit weighting/coverage | Implemented | Visual datasets expose EV-matched and joint EV/closing cohorts. These are descriptive comparisons, not proof of calibration. |
| E06 | Smooth calibration/regression with group-aware out-of-sample evaluation | Optional extension | Return models exist, but are not a dedicated calibrated-probability/EV calibration study. Needs target, assumptions and reviewed evaluation policy. |
| Z01 | Effective/recorded-time sizing history, bankroll/currency/Kelly/caps/evidence and ambiguity validation | Implemented | [Sizing history](../SIZING_HISTORY.md), validated schema and immutable publication; sensitive history remains local. |
| Z02 | Per-wager regime linkage only with supported chronology; no inferred bankroll | Implemented, evidence-blocked | Unknown periods stay unknown. Need supported effective intervals and placement evidence for actual linkage. |
| Z03 | Actual source stake; suggested/accepted stake and deviation evidence when known | Partially implemented | Source stake preserved; prospective quote/fill evidence is explicit. No generalized historical suggested-versus-accepted/deviation reconstruction exists without fields/provenance. Audit a new supported source before extending it. |
| Z04 | Reported profit, weighted ROI, equal-stake return and fixed-unit notional profit on same eligible cohort | Implemented | Diagnostic eligibility/exclusions and reconciliation; proportional scaling remains hypothetical. |
| Z05 | Regime returns, event/wager concentration, odds/EV exposure and bankroll-relative stake | Implemented | Sizing publication retains known/unknown regime composition, concentration and supported bankroll ratios. UI chart forms are assessed separately below. |
| Z06 | Positive finite stakes; settlement/ticket exclusions; sign reversal and scale invariance | Implemented | [Sizing tests](../../tests/test_sizing_diagnostics.py) and canonical contracts preserve original money and unsupported products. |
| Z07 | Exact Kelly replay, drawdown, concurrent exposure and dynamic bankroll growth | Optional extension | Existing `sizing_replay_blocked` state is implemented; a simulator is not. Requires supported ordering, cash flows, starting capital, fills/fees and sizing conventions. |
| Z08 | User-intended Kelly transition not treated as already effective | Implemented | Policy history distinguishes intent, effective interval and proof of application. Record confirmed evidence before linkage. |

Evidence: [sizing configuration](../../src/oddsjam_ev/analysis/sizing_config.py),
[linkage](../../src/oddsjam_ev/analysis/sizing_linkage.py),
[diagnostics](../../src/oddsjam_ev/analysis/sizing_diagnostics.py),
[history publisher](../../src/oddsjam_ev/io/sizing_history.py),
[history tests](../../tests/test_sizing_history.py),
[configuration tests](../../tests/test_sizing_config.py),
[diagnostic publication tests](../../tests/test_sizing_diagnostics_publication.py),
[decision tests](../../tests/test_decision_records.py).

## Visual catalogue, interface and source gates

This includes the owner's notebook-inspired requirements subsequently incorporated
into the workspace design. Implementation and private-source readiness are separate:
a chart tested with supported synthetic inputs can remain blocked on historical data.

| ID | Requested visual / behavior | Status | Evidence / next action |
| --- | --- | --- | --- |
| V01 | Profit/ROI by sport/platform/league/market; full hover, counts and metric switch | Partially implemented | Chart and controls exist. Generic bar-to-published-candidate drill-through is not implemented; use the idea selector. A chart-to-queue identity handoff is a focused UI extension. |
| V02 | Platform/filter EV-return scatter, matched weighting, reference lines and sizing | Partially implemented | Platform/filter scatter, matched cohort, weighting, hover/size and reference lines exist. Dedicated scatter-to-idea selection and per-point uncertainty overlay are not delivered; add only on matching published evidence. |
| V03 | Linked EV/return/volume/profit bucket diagnostics and exact cutoffs | Implemented | Weighted/equal views, top-point bucket highlighting and lower-panel denominators; source closing series is separately gated. |
| V04 | Liquidity bands | Implemented, evidence-blocked | Implementation/synthetic tests exist. Private acceptance snapshot has no verified decision-time USD liquidity. Obtain source units/timing/capacity evidence. |
| V05 | Liquidity × EV heatmap | Implemented, evidence-blocked | Implemented matched table/interactive heatmap; same liquidity provenance prerequisite plus EV coverage. |
| V06 | Time-to-event categories and after-start exclusions | Implemented, evidence-blocked | Source-gated implementation exists; verify placement/event meanings, timezone/offset and DST before enabling historical timing claims. |
| V07 | Cumulative platform profit, daily/weekly/monthly, reported/fixed units and legends | Implemented | Source-calendar attribution view, not settlement cash balance/drawdown. Unsupported dates/gaps remain explicit. |
| V08 | Volume over time, count grain, resolution, partial periods/missingness | Implemented | Candidate-grain source-calendar periods; activity not suppressed by minimum-performance counts. |
| V09 | Weekday/hour activity bars/heatmap and denominator | Implemented | Active source-day means and literal source-clock limitations. Not scanner supply or verified placement-hour behavior. |
| V10 | Closing-price-derived EV line, fair reference/source/timing and joint cohorts | Implemented, evidence-blocked | Transform/readiness/joint-cohort tests exist. Private source lacks verified semantics; no zero line or raw CLV substitution. |
| V11 | Weighted/equal-stake paired dots with sizing-regime facets | Partially implemented | Paired comparisons and economic/regime tables exist; dedicated regime-faceted paired-dot visualization is not implemented. Use published regime diagnostics before adding a chart. |
| V12 | Uncertainty forest plot and parent/subgroup/complement comparisons | Implemented | Comparison plots retain available pointwise intervals, support, zero and descriptive interpretation. |
| V13 | Platform × market / sport × EV heatmap over prepublished cells | Partially implemented | Published predicates and exact-scope inspection exist; general categorical heatmap is not delivered. Liquidity × EV is a different chart. Add a renderer and supported-cell tests if prioritized. |
| V14 | Exposure concentration distributions and sizing-regime stacked views | Partially implemented | Concentration/composition data and tables exist; dedicated distribution/stacked views are not delivered. Needs chart scope/missingness design, not invented history. |
| V15 | Prospective quote-verification funnel | Partially implemented | [Paper-test page](../../src/oddsjam_ev/dashboard/pages/paper_tests.py) displays availability/outcome tables and returns. A funnel chart is not implemented; use supported enrollment/check statuses and denominator reconciliation when adding it. |
| V16 | Drawdown/rolling performance/bankroll growth | Optional extension | Historical chronology/cash-flow/exposure prerequisites remain unsupported; do not relabel source-date curves. |
| V17 | Versioned series identity, population, eligibility, units, matched coverage and missingness | Implemented | Visual candidates/memberships/cohorts/periods/activity/coverage/readiness and immutable audit; non-additive scopes explicit. |
| V18 | Old reports usable, unavailable intersections explicit, no notebook/page-load research | Implemented | Optional verified attachments and fail-closed loaders; no substituted one-dimensional evidence for a requested intersection. |
| V19 | Visuals primarily under “Filter health”; “Signals & leaks” as the old landing | Superseded | Current **Overview, Opportunities, Production health, Explore history, Explain returns** follow the later decision-workspace design. Legacy descriptive views remain separate. |
| V20 | Shared selectors, minimum 100 display support, all-data population separation | Implemented | UI/query tests include 99/100/101 and five-wager exclusions; display filters do not alter correction families. |
| V21 | Altair theme, hover, reset/zoom/legends, full labels and outlier visibility | Implemented | Chart specs/AppTest plus dated laptop Chrome acceptance; accessible data/download views accompany charts. This is not a WCAG certification or mobile acceptance. |
| V22 | No lines implying continuity between unrelated categories | Partially implemented | Discrete bucket identities/hover are explicit, but `bucket_panel` still connects ordered EV-band points. Review the design and switch to discrete marks where continuity is misleading; preserve matched denominators. |
| V23 | Simultaneous filters, saved memberships, corrupt/old/empty states and no mutation | Implemented | Dashboard tests and release browser run cover the documented supported interactions. No claim that every chart mark provides every suggested drill-through. |

Evidence: [visual contracts](../VISUAL_DIAGNOSTICS.md),
[visual policy](../../src/oddsjam_ev/analysis/visual_config.py),
[visual implementation](../../src/oddsjam_ev/analysis/visual_diagnostics.py),
[chart builders](../../src/oddsjam_ev/dashboard/decision_charts.py),
[history page](../../src/oddsjam_ev/dashboard/pages/visual_history.py),
[dashboard guide](../DASHBOARD_DECISION_WORKSPACE.md),
[visual tests](../../tests/test_visual_diagnostics.py),
[publication tests](../../tests/test_visual_publication.py),
[workspace tests](../../tests/test_dashboard_workspace.py),
[display-boundary tests](../../tests/test_dashboard_signal_filters.py),
[browser harness](../../scripts/acceptance/browser.mjs).

## Integration, files and delivery order

| Original proposal | Status | Actual implementation / dependency |
| --- | --- | --- |
| `analysis/multidimensional_config.py` | Implemented | [Actual parser](../../src/oddsjam_ev/analysis/multidimensional_config.py) and declared configuration; no name change. |
| `analysis/signal_discovery.py` | Superseded | [multidimensional_discovery.py](../../src/oddsjam_ev/analysis/multidimensional_discovery.py) implements the bounded ledger. |
| `analysis/signal_comparisons.py` | Superseded | Discovery/sizing provide descriptive partitions; [paired_comparison.py](../../src/oddsjam_ev/analysis/paired_comparison.py) provides paired inference. |
| `analysis/signal_prioritization.py` | Implemented | [Actual implementation](../../src/oddsjam_ev/analysis/signal_prioritization.py); later review catalogue/queues/ranking add all-history presentation. |
| `io/signal_discovery.py` | Superseded | [within_filter_discovery.py](../../src/oddsjam_ev/io/within_filter_discovery.py), [signal_prioritization.py](../../src/oddsjam_ev/io/signal_prioritization.py) and later component publishers. |
| `scripts/build_signal_discovery.py` | Superseded | [build_within_filter_discovery.py](../../scripts/build_within_filter_discovery.py) and [build_signal_prioritization.py](../../scripts/build_signal_prioritization.py). |
| Optional pinned pipeline stage, retry/resume and policy/history hashes | Partially implemented | Core runner has frozen plans/retries/reuse; newer publishers have immutable lineage/replay but are not integrated core stages. An orchestration PR needs frozen optional-stage configuration, dependency/partial-run tests and backwards compatibility. Explicit builders work now. |
| Dashboard artifact/query/workspace extension | Implemented | [Attachments](../../src/oddsjam_ev/dashboard/decision_artifacts.py), [queries](../../src/oddsjam_ev/dashboard/decision_queries.py), [decision page](../../src/oddsjam_ev/dashboard/pages/decision_workspace.py) and evidence/model/lifecycle pages. |
| `tests/test_multidimensional_*.py` | Superseded | `test_within_filter_*`, `test_statistical_prioritization.py`, `test_prioritization_publication.py`, `test_review_*` and dashboard tests cover the separated contracts. |
| `docs/MULTIDIMENSIONAL_SIGNALS.md` | Superseded | [Within-filter](../WITHIN_FILTER_DISCOVERY.md), [prioritization](../STATISTICAL_PRIORITIZATION.md), [queues](../REVIEW_QUEUES.md) and this ledger replace a monolithic guide. |
| Eight-commit coarse order / ten-step expanded sequence / visual delivery split | Superseded | Delivered across identity, EV, sizing, discovery/prioritization and A–I work. This is historical dependency intent, not outstanding duplicate implementation. Partial items above remain explicit. |
| Policy/history/config hashes and retained old outputs/readers | Implemented | Each explicit publisher retains its own input/evidence/configuration/code lineage; integration into one runner remains partial as stated above. |

## Acceptance and additional decision requirements

| ID | Original acceptance / decision requirement | Status | Verification / next action |
| --- | --- | --- | --- |
| A01 | AND/OR semantics, numeric boundaries, canonical/reordered predicates, missing/invalid fields | Implemented | Discovery/configuration/query tests distinguish OR within fields, AND across fields and whole-conjunction exclusion. |
| A02 | Repeated exports, attribution overlaps, ticket legs and opportunity proxies cannot inflate money/counts | Implemented | Ingestion/canonical/discovery/visual/release reconciliation fixtures. |
| A03 | Sparse/group-correlated cases and null uncertainty never become strong signals | Implemented | Support/concentration/coverage/degenerate-test gates and no-winner cases. No power guarantee. |
| A04 | Complete family/correction despite display suppression; explicit budget overflow | Implemented | Discovery/prioritization/publication/queue tests; no truncated successful family. |
| A05 | Unknown chronology blocks temporal claims; event splits and embargo where supported | Partially implemented | Core temporal and model event-separation tests exist; chronological validation of the newer within-filter family remains D06, not a completed evaluator. |
| A06 | Parent/complement/overlap/global distinctions and unsupported widening tests | Implemented | Set/economics reconciliation and documented missing outside universe; no widening evaluation is claimed. |
| A07 | Replay, changed-policy identities, corruption and row-lineage mismatch | Implemented | Immutable publisher/loaders and byte-reverified caches; checks are not source-truth verification. |
| A08 | Partial runs/resume and default historical metrics preserved | Partially implemented | Core pipeline failure/resume and separate publisher atomicity tested; combined optional-stage resume is the outstanding orchestration item. |
| A09 | Dashboard only consumes verified evidence; explicit missing/unavailable states and no execution | Implemented | AppTest, read-only interaction tests, source hashes and real-browser acceptance. |
| A10 | Runtime/memory benchmarks against declared budgets and reconciliation | Implemented | [Release verifier](../../scripts/verify_release.py), [acceptance tests](../../tests/test_release_acceptance.py) and dated desktop record. Hosting/concurrency remains unbenchmarked. |
| A11 | Weighted/unit-return equivalence, sign reversals, scale invariance, invalid settlements/stakes | Implemented | Sizing/canonical fixtures and common-cohort visual/release reconciliation. |
| A12 | Known/missing/overlapping sizing regimes; unknown dates not backfilled | Implemented | Sizing schema/linkage/history tests; exact replay remains optional Z07. |
| A13 | Separate funding/products, unknown regimes, within/global results and ranking disagreements | Implemented | Sizing/discovery/queue/model population contracts and tests. |
| A14 | Visual unit conversions, matched versus differing coverage, invalid odds/CLV/liquidity | Implemented | Visual tests include source-gated synthetic series, missingness and joint cohorts. |
| A15 | Timezone/DST, partial weeks, missing versus zero and active-day denominator | Implemented | Source-feature/visual tests; supported literal source-calendar semantics do not upgrade chronology. |
| A16 | Visual minimum support, sparse cells, zero references, available uncertainty and full hover | Implemented | Spec/query/UI tests and laptop browser walkthrough within delivered chart forms; partial chart requests remain V01/V02/V11–V15/V22. |
| A17 | Approved grid, support/stake/concentration/coverage/family/budget/overlap decisions | Implemented | Explicit discovery, prioritization and review configurations; new inference policies require separate review and publication. |
| A18 | Historical identity/configuration, sizing dates, EV semantics and accepted stake evidence | Implemented, evidence-blocked | Schemas/review paths preserve unknowns. Obtain candidate/source-specific evidence rather than parameter screenshots for every deleted filter. Z03 describes missing generalized stake reconstruction. |
| A19 | Select continuous calibration scope and outside-universe/holdout design before claims | Optional extension | E06, W04 and D06 define prerequisites and next decisions; existing exploratory outputs do not answer them. |
| A20 | Keep private histories/account evidence out of Git; no acquisition/scheduling/filter-write path | Implemented | Public synthetic fixtures/templates and ignored local storage; explicit external operator action is separate. |

## Remaining work and its next action

| Work item | Reason / prerequisite | Next action and scope |
| --- | --- | --- |
| Candidate scanner settings, baseline and lifecycle evidence | Current names/history alone do not prove exact settings or applied periods | Record evidence through existing decision/lifecycle contracts; publish and attach new versions. No generic feature PR required. |
| Liquidity/timing/closing-price series | Required historical source semantics remain unverified | Obtain provenance; use existing policy gates. Add an adapter only if a supported new source needs one. |
| New within-filter chronological evaluator | Missing verified timing and integrated holdout design | Define windows, information availability, embargo and policy before a separate implementation PR. |
| Combined build orchestration | Optional extensions remain explicit independent commands | Scope a convenience/reliability PR when repeated builds justify frozen optional-stage planning and resume. |
| Deferred chart forms/drill-through and categorical-line design | Data often exists; the exact proposed UI behavior does not | Prioritize V01/V02/V11/V13/V14/V15/V22 in a focused visual enhancement PR, preserving cohort identity and source gates. These are not hidden as evidence-only blockers. |
| Threshold sensitivity / smooth calibration | General model and threshold support do not implement these specific studies | Predeclare research target/grid/family and evaluation before a separate publication. |
| Widening / bankroll replay / additional stake comparison | Required universe, chronology, cash-flow/fill or stake fields are absent | Collect evidence first; do not reconstruct unsupported history. Separate research contracts if pursued. |
| Hosted or multi-user use | Desktop release budgets do not establish deployment capacity | Choose a host/workload and benchmark there before optimization/deployment. |
| Navigation and reference consolidation | Historical intent must remain separate from current instructions | J–M provide entry points/status, tested walkthroughs, shared contracts and [maintenance checks](../DOCUMENTATION_MAINTENANCE.md). Maintain these alongside future contract changes; no further planned documentation PR blocks candidate review. |

## What completion means now

The core workflow supports candidate review and preparation of future paper studies.
It does not require every optional research/visual enhancement above. A specific study
can still be blocked by setup/evidence. The original brief is **not** a blanket fully
implemented checklist: partial engineering work remains visible here, and evidence
gaps do not become verified results merely because the software supports a contract.

[Documentation home](../README.md) · [Architecture](../ARCHITECTURE.md) ·
[Documentation follow-up plan](DOCUMENTATION_OVERHAUL.md)
