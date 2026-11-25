# Personal Bet Tracker — adopted programme addition

Adopted 2026-11-24 under PD005 and N001. Source: applied R&D §7 and PR roadmap §12;
[version/authority record](SOURCE_BASELINES.md). This page describes planned
capabilities, not implemented tracking software. R01–R60 and REQ01–REQ50 retain
all original scope, dependency clauses and acceptance. No original work is retired.
[Fourteen reference captures](../images/pikkit-tracker/CATALOGUE.md) describe the
owner's Pikkit interface; formulas, backend behavior and integration rights are not
established by a screenshot. Public experience reference: [Pikkit](https://app.pikkit.com/).

## Cohesive experience and honest data

A dedicated Bet Tracker workspace supports everyday account/wager review; the
Research workspace supports frozen studies. Both consume the same source-linked
ledger, analysis context and definitions. A selected day, venue, ticket, filter or
tag should link through performance, calendar, CLV, exposure and risk investigation
with its snapshot, funding/product, time basis, currency and unit convention intact.
A private account population differs from a research-selected cohort; show that
change explicitly rather than silently altering a study. Reuse existing R06 venue
capabilities, R07 context, R08 jobs and R09/R10 staging/activation.

Home provides daily navigation, recent-result strip, open/settled counts and a
concise bet feed. Performance adds cumulative profit/return, drawdown, record and
full venue/league/market/bet-type/filter breakdowns, period/custom/season controls,
dollars/units and exact-row drilldowns. Calendar adds monthly/yearly signed heatmaps
and selected-day summaries. Hover/zoom/legend selection, accessible equivalents,
clear scope and honest empty states should replace dense hidden information.
No recorded activity is not verified complete zero activity. Unknown settlement
times cannot supply daily settled-profit chronology. Unit changes need dated rules
or a labelled current-unit restatement. Every metric must expose eligible rows,
denominator, fees treatment and date basis; do not copy unexplained vendor ROI.

Your Bets supports bounded search, structured filters, stable sorting/pagination,
bulk tagging/export, expandable ticket legs and a persistent detail panel. Details
include selection/line, venue/product/account, odds, stake, gross payout versus
profit, status, source timestamps/precision, fees, IDs, revisions and provenance
where available. Ticket cash is counted once; legs supply descriptive membership.
Straight, futures, parlay/SGP, round-robin and fantasy/pick'em rules need explicit
supported contracts. Partial fill, cash-out, push/void/refund, partial settlement,
bonus/boost and resettlement states stay explicit; unsupported reconstruction stays
unknown. Manual entry/corrections and notes use append-only annotations with reason,
author/time, old/new values and reversal; never overwrite imported evidence.

Tags support organization, bulk assignment/removal, saved views and three selectable
cohort comparisons with period/count/stake/ROI/uncertainty where supported. Disclose
overlap; tags are not independent portfolios. Imported tags, local notes, strategy
attribution and experiment decisions retain separate provenance. TEST-only passed
opportunities are decision observations, not settled wagers, cash or funded ROI;
local tags neither place a bet nor trigger OddsJam's Previously Placed warning.
Post-result tags/rules must not contaminate earlier point-in-time evaluation.

Balance/At Risk links evidenced account cash, reserved stake/liability, pending
movements, currency, freshness and open exposure. Reconcile opening balance plus
cash movements and settlement flows to observed closing balance; surface differences.
Deposits/withdrawals/transfers are not wager profit; paired transfers are not counted
twice. Bonus/free-bet/nonwithdrawable balances and stake-return rules remain distinct.
An imported or manual balance is not a linked account, and unavailable is not zero.
Rewards/promotion inventory is a proposal: Rewards is only a navigation label in
the references. No hidden vendor reward workflow is inferred.

Closing Line adds coverage counts, positive/negative/neutral CLV, individual records,
matched actual-versus-reference curves and full breakdowns. Reveal placed/closing
price and line, source, devig, observation/cutoff times, fees assumptions and formula
version. Source-reported CLV is not automatically fair probability, EV or dollar
expectancy. Actual/reference comparisons use the same covered eligible sample;
all-bet profit keeps its own denominator. Closing information is post-decision and
must not leak into predictive inputs. Gap diagnosis links to R30 and risk R25–R29;
a gap alone cannot prove luck, bad execution, edge or a broken filter.

Scenarios groups Open/Futures/Settled ladders, repeated positions, conflicts, hedges
and middles with counts/exposure and feasible payoff tables. Reuse R23/R24 and R46
where pursued; simulations reuse R25–R29. Duplicate imports differ from legitimate
repeated bets/partial fills. Conflicts may be intentional; do not delete exposure.
Ladder rungs share monotone outcomes and cannot independently win impossible joint
states. Probabilities require validated joint models and explicit assumptions.

Events starts with recorded history, date/league/search filters and links to wagers.
Scores, current quote buttons, futures/more-markets and a review slip require actual
supported evidence. A hypothetical review slip is not an order. Independent feeds,
normalization/pricing and executable routing retain R34–R38/R48–R51 access gates.
Freshness, settlement-rule mismatch and residual multi-leg risk remain visible.

Global search and an optional Ask-QPB-inspired evidence assistant return reproducible
queries, source rows and definitions, with permissions and unavailable states. Use
approved deterministic calculations; imported notes are data, not instructions.
A structured local question interface can precede external AI. External model data
sharing/cost is a separate decision. Do not invent correlations, balances or profit
probabilities. Settings cover timezone/date basis, odds conversion/calculator,
dollars/dated unit size, theme/contrast/density, export/privacy and useful reminders.
State/jurisdiction preferences do not establish eligibility. Hosted profile/session
revocation needs actual R59 authentication. Vendor merchandise/social/recruitment
links need no replica; support/profile/preferences retain their useful concepts.

## Data and accounting boundaries

Use separate related source snapshots, observations, execution candidates, tickets,
legs/fills, settlements/revisions, cash movements, balance observations, annotations,
event identities and closing references. Pin namespace, original IDs/payload/hash,
source/export/import/observation times and precision, currency, product and lineage.
Preserve immutable old artifacts and introduce only missing versioned extensions.
R40 retains settlement/cash ownership; earlier imported-data adapters must document
compatibility and later convergence, not create a competing accounting engine.

Repeat exports must not double-count wagers; cross-source identity cannot rely only
on matching team/player/odds. Ambiguity stays unresolved. Paper, funded, unknown,
passed observations and scenarios retain separate economics. Track histories do
not reveal the complete opportunity universe or repair historical missing depth,
passes, fillability, account balances or chronology. Missingness is visible before
any parity claim. Private sources, account amounts and screenshots stay outside Git;
future illustrations use reviewed synthetic/redacted assets. Spreadsheet exports
need formula-safe rendering; local notes are untrusted.

## Selection and recommended insertion points

N001 is the owner-selected adoption checkpoint after R09 and before R10. Once N001
is verified merged, the default next original PR is R10, then the eligible original
R-series order. N002–N014 do not automatically jump that queue: each needs explicit
selection or a recorded owner-authorized scheduling decision and its dependencies.
No completed R PR gains a retroactive prerequisite. R10's upload/preview/progress/
activation remains intact; R11 evaluates framework/design with this new consumer
in mind. Nothing here selects React or changes existing runtime behavior.

| Earliest useful window (not automatic scheduling) | Additive work | Readiness boundary |
| --- | --- | --- |
| Before R10 | N001 | Source adoption only |
| After R10 | N002 | Source/ledger contracts; R06/R07 reused |
| After R13 | N003 | Shared query/metric/chart contract |
| After R12 and N003 | N004 | Dedicated tracker shell and ledger |
| After R14/R15 and N004 | N005 | Linked performance/calendar |
| After N004 | N006, N007, N010 historical, N012 | Individual card prerequisites; cash/source capability gates |
| After analytical dependencies | N008, N009, N011 | Qualified CLV, correlated risk and supported evidence |
| After supported access feasibility | N013 | Conditional read-only synchronization |
| Imported-data core accepted | N014 | Core first; optional activated extensions separately accepted |

## Additive PR cards

These cards summarize public deliverables and acceptance; private source §12.4
remains the detailed baseline. Likely implementation paths are provisional until
that PR inspects current code/R11's architecture. No schema or algorithm is approved
merely by naming a future module. Full commit-level plans are saved on selection.

### N001 — Adopt the tracker addition and reconcile the programme

**Entry/readiness:** R09 merged; attached expanded roadmap versions available. No new UI or live-source access needed.

**Deliverable:** Record the original and expanded source fingerprints with dates and owner authority. Register N001–N014 in delivery records and the opportunity/decision records; preserve R01–R60 and existing requirement IDs. Add public-safe tracker requirements and a screenshot catalogue referencing all 14 filenames without publishing private images. Reconcile R09/#42 merge/main CI evidence. Add the crosswalk and recommended insertion points; keep R10’s scope intact.

**Likely reuse/touchpoints:** docs/roadmap/SOURCE_BASELINES.md, REQUIREMENTS.md, DELIVERY_REGISTER.md, CHANGE_LOG.md, OPPORTUNITY_BACKLOG.md, DECISION_REGISTER.md; docs/roadmap/pr-plans/N001.md; docs/images or operations tracker-reference catalogue. Confirm actual existing layouts before editing.

**Acceptance:** Every original R card remains traceable; each screenshot area maps to a new requirement or explicitly labelled conditional capability. The source-version rule explains how old frozen originals and new amendments coexist. No private balances/screenshots enter Git. Next original item remains R10 after this adoption checkpoint.

**Verification:** Verify source hashes and original-content preservation; local Markdown/CLI checks and required repo checks. Review the full dependency graph for cycles and accidental new prerequisites on completed work.

**Compatibility/rollback:** Documentation-only adoption; rollback by a dated superseding decision, never by deleting the owner’s original programme. No UI owner checkpoint needed unless wording or scope remains materially disputed.

**Provisional commit subjects** (refine in the selected PR plan):

- `docs(roadmap): register the owner tracker amendment`
- `docs(tracker): catalogue references and capability boundaries`
- `docs(roadmap): reconcile R09 delivery and tracker sequencing`

### N002 — Define the source-linked tracker ledger and annotation extensions

**Entry/readiness:** N001, R06, R07 and R10. Reuse R03 source limitations and R09 staging contracts.

**Deliverable:** Audit existing canonical execution/ticket/leg and revision contracts. Define a tracker read model over those artifacts with explicit unsupported states for cash-out, partial settlement, bonuses and uncertain identity. Add only missing versioned local manual-entry/annotation records with provenance, reason, author/time and reversal history. Specify account namespace, currencies, source IDs and time precision; do not change raw source bytes or use leg rows as extra cash tickets.

**Likely reuse/touchpoints:** src/oddsjam_ev/io/ and shared contracts/readers; existing canonical/reconcile/venue modules; proposed tracker contract module and synthetic tests. Reuse experiment/decision journal conventions without overwriting their distinct meaning.

**Acceptance:** The same imported wager has one economic identity across tracker and research. Repeated imports, revisions, account namespaces, ticket/leg links and unsupported product economics remain explicit. Manual edits are append-only and reversible; raw history and old publications still load.

**Verification:** Independent fixtures for straight/parlay/pick’em/round-robin or explicitly unsupported products; repeated imports, uncertain matching, resettlement, free bets, partial fills and annotation reversal. Verify deterministic IDs, schema versions and old-reader compatibility.

**Compatibility/rollback:** Ship additive publications/adapters behind a tracker feature boundary. No arbitrary backfill of balances or timing. If R40 already exists, adapt its ledger; otherwise preserve an explicit compatibility contract for it.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): define versioned ledger and annotation contracts`
- `feat(tracker): adapt verified imports and ticket relationships`
- `test(tracker): verify identity revisions and reversible annotations`

### N003 — Publish shared tracker accounting queries and filter context

**Entry/readiness:** N002, R07 and R13 chart/context contracts. Existing metrics must be inspected first.

**Deliverable:** Build bounded shared queries for date/status/account/venue/sport/league/market/bet-type/tags/filter/funding selection, stable search/sort/pagination and drilldown IDs. Define net profit, gross payout, settled stake, record statuses and weighted/equal-stake returns. Version placement versus settlement date, timezone, currency and unit rules; provide matched-sample coverage and truthful zero/unknown states.

**Likely reuse/touchpoints:** Shared analysis/query modules under src/oddsjam_ev/, dashboard context/query adapters, tracker config and reference tests; actual naming follows R11 architecture decision. No notebook-only duplicate metric implementation.

**Acceptance:** Cards, curves, tables and ledger selection use identical eligible rows and denominators. Cash movements and paper records cannot silently enter funded ROI. A unit preference or date-basis change updates the whole selection consistently and preserves the original amounts.

**Verification:** Independent arithmetic for wins/losses/pushes/voids/refunds/fees; DST/midnight boundaries, absent settlement time, partial periods, ticket–leg joins, overlap and zero denominator. Declare and measure a representative synthetic history and query latency/memory budget.

**Compatibility/rollback:** Query contract is versioned and old research metrics remain unchanged unless an independently justified baseline PR owns the change. UI integrations consume these queries rather than infer values.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): add common selection and accounting queries`
- `feat(tracker): publish coverage and linked row identifiers`
- `test(tracker): reconcile metrics across time and product grains`

### N004 — Deliver the dedicated Bet Tracker shell and searchable ledger

**Entry/readiness:** N003 and R12 workspace/design system. Reuse R10 source/import/activation controls.

**Deliverable:** Add a distinct Bet Tracker navigation destination and Your Bets list with search, filters, saved selection, stable pagination, sort, readable ticket cards/table and detail drawer. Expand parlay legs without duplicating monetary totals. Show source/as-of, status, odds/stake/payout, notes/tags, time evidence and related-ticket/research links. Include Home daily navigation, recent-days results and a no-recorded-activity state; do not imply a source is complete merely because no rows match.

**Likely reuse/touchpoints:** dashboard/app.py and selected R11/R12 route/view framework; tracker queries/components and UI tests. Retain existing research routes and shared context.

**Acceptance:** Owner can find a recorded bet, inspect source and revisions, expand legs, change period/venue and open the same cohort in research without losing scope. Missing fields and errors are readable. No real order or study action is triggered by browsing.

**Verification:** AppTest/browser journeys with synthetic straight and multi-leg records, empty/missing/stale/corrupt sources, keyboard-only navigation, responsive laptop layout and large-history pagination. Required owner review includes exact launch command, synthetic report, selections, steps and expected behavior; blocks merge when needed for acceptance.

**Compatibility/rollback:** Feature-flag tracker routes; keep prior research/report navigation available. Do not expose private screenshots in browser fixtures.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(dashboard): add dedicated tracker navigation and daily overview`
- `feat(tracker): render searchable bets and ticket detail`
- `test(dashboard): verify tracker scope accessibility and empty states`

### N005 — Add linked performance, date/season controls and calendars

**Entry/readiness:** N004, R14 and R15, using N003 accounting definitions.

**Deliverable:** Implement week/month/year/all/custom and sport-season selection, Apply/Clear/Today and dollars/units. Add cumulative settled profit, stake/return summaries, records and full league/venue/bet-type breakdowns with expandable groups. Build monthly/yearly P/L calendars and selected-day details linked to exact wagers. Reuse interactive chart components for hover, zoom, legends and drilldowns.

**Likely reuse/touchpoints:** Tracker performance/calendar views, shared charts/context and account/time query adapters; R14/R15 helpers rather than separate plotting code.

**Acceptance:** Period cards, calendar sums, curves and breakdown totals reconcile with the selected ledger. Overlapping cohorts are labelled; paper/funded/product distinctions remain visible. Placement/settlement basis and unavailable chronology are explicit. Positive ROI versus negative profit cannot occur for an identical scope/definition without an explained adjustment.

**Verification:** Reference accounting totals, boundary dates/DST, period reset, missing versus zero days, partial exports, unit restatement, filtered chart-to-ledger links and readable accessible palettes. Owner visual walkthrough required for interaction/layout acceptance.

**Compatibility/rollback:** Ship imported-history performance first. Historical score images/live data are optional sources, not a prerequisite. Roll back by disabling tracker view additions without deleting source history.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): add performance and period selection views`
- `feat(tracker): link calendar and breakdown drilldowns`
- `test(tracker): verify cross-view totals and date semantics`

### N006 — Add auditable tags, saved views and cohort comparisons

**Entry/readiness:** N002 and N004; integrate R19/R20 decisions when available without creating an artificial dependency for basic personal tags.

**Deliverable:** Add local tag create/edit/assign/remove and descriptions/colors, bulk review actions, saved filters and three-cohort comparison curves. Separate imported tags, personal tags, filter attribution and prospective decision records. Preserve assignment time and reversal history, link to supporting ledger membership and display overlap/counts/coverage.

**Likely reuse/touchpoints:** Tracker annotation persistence, tag query/view modules and tests; shared journal links for research decisions.

**Acceptance:** Tags are reversible and source-linked. An overlapping wager is not double-counted in total performance. TEST-only passed decisions stay outside wager/cash ROI and do not alter OddsJam Previously Placed state. Retrospective tags cannot masquerade as predeclared TEST conditions.

**Verification:** Bulk atomicity/reversal, overlapping cohorts, empty/insufficient tags, imported-name collisions, malicious note text and account isolation; owner review of tag and comparison workflow.

**Compatibility/rollback:** Annotations are an overlay on immutable data. Rollback hides local tag views while retaining their versioned history and imported tags.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): persist personal tags and assignment history`
- `feat(tracker): compare tagged cohorts with overlap disclosure`
- `test(tracker): verify bulk actions and decision separation`

### N007 — Add account balances, cash movements, at-risk capital and promotions

**Entry/readiness:** N002 and N004. Reuse R40 private ledger if merged; when not merged, use a bounded tracker extension and specify its future R40 adapter before implementation.

**Deliverable:** Support manual/imported balance observations, dated deposits/withdrawals/transfers/refunds/fees, pending cash movements, available/reserved/promotional balances and open risk by account/venue. Add explicit promotion/reward lifecycle records where evidence exists. Reconcile opening balance plus cash/settlement movements to observed balance with explained differences. Distinguish currencies and withdrawable cash from other product balances.

**Likely reuse/touchpoints:** Shared cash/position ledger contracts/readers, tracker Balance/promotions views, R40 adapters and reconciliation tests. Do not create a second financial ledger.

**Acceptance:** An unknown or stale balance is not zero; an account is called linked only with a supported connection. Transfers and deposits are not betting profit. Ticket settlements, free-bet rules, cash-out and fees are counted once. Open exposure links to wagers and later risk scenarios.

**Verification:** Independent cash-flow reference cases, paired transfers, bonus expiry/consumption, stale observations, refunds/resettlements, concurrent changes, multiple accounts/currencies and unauthorized cross-account access. Owner review of manual balances and reconciliation differences; no account login required for synthetic acceptance.

**Compatibility/rollback:** Manual/imported evidence first; no inferred live balances. R40’s independent-platform settlement objectives remain fully assigned to R40, including adapters not needed for this increment.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): record balance observations and cash movements`
- `feat(tracker): present reconciled balances and open capital`
- `test(tracker): reconcile promotions settlements and transfers`

### N008 — Deliver qualified closing-reference and actual-return diagnostics

**Entry/readiness:** N003, N005 and R30; verified closing-reference evidence for derived EV. Source-reported CLV display may be delivered as an explicitly limited child slice if documented.

**Deliverable:** Add period-scoped CLV coverage, positive/negative/neutral splits, equal/stake-weighted summaries, matched actual-versus-reference profit curves, venue/market comparisons and sortable individual bets. Display source/line/price/time/devig/formula/fee evidence. Preserve source-reported fields separately from independently supported fair-probability calculations.

**Likely reuse/touchpoints:** R30 gap-investigation publications, shared metric/coverage contracts, Closing Line tracker view and reference-calculation tests.

**Acceptance:** Derived expected profit is unavailable when source semantics are insufficient. Curves share a covered eligible sample, with all-bet actual profit separately labelled. Post-close data never becomes a pre-bet model input. Each displayed value links to supporting bets and calculation evidence.

**Verification:** Independent probability/payoff formulas; differing lines, pushes, fees, missing or post-event quotes, uncertain timestamps, mixed reference sources, sample denominators and coverage bias. Owner review of curves and coverage explanations.

**Compatibility/rollback:** No reproduction of unexplained screenshot percentages or account totals. Missing fair-close data remains an explicit limitation rather than filled with zero.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): expose closing-reference provenance and coverage`
- `feat(tracker): render matched CLV and profit comparisons`
- `test(tracker): validate formulas missingness and temporal boundaries`

### N009 — Connect position scenarios, ladders, duplicates and conflicts

**Entry/readiness:** N004, R23 and R24. R25–R29 are required for probabilistic bankroll/risk simulation; R46 methods reused when available.

**Deliverable:** Add Open/Futures/Settled scenario groups with counts, stake/liability and related tickets. Distinguish import duplicates, repeated economic positions, ladders, conflicting bets, hedges and middles. Show supported feasible-outcome payoff tables and concentration. Link the same group to the risk laboratory without creating a second simulation engine.

**Likely reuse/touchpoints:** Shared exposure/group/payoff publications and tracker scenario views; risk-laboratory context adapters.

**Acceptance:** No legitimate repeated position is deleted as a duplicate. Ladder rungs use feasible joint outcomes. Unsupported dependence/probability stays unknown. The same group has identical economics in tracker and research/risk views.

**Verification:** Nested ladder payoffs, conflicting sides, hedge/middle outcomes, ticket/leg overlap, partial fills, commissions and absent scenario data. Simulation integrations verify pinned seeds/assumptions/context through existing risk tests. Owner scenario walkthrough required.

**Compatibility/rollback:** Descriptive exposure can ship before probabilistic simulation. Each unavailable risk feature names its dependency; do not silently label a descriptive grouping as a modeled forecast.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): surface correlated position and scenario groups`
- `feat(tracker): link payoff views to the shared risk laboratory`
- `test(tracker): verify joint outcome and duplicate distinctions`

### N010 — Add historical event browsing and conditional quote review

**Entry/readiness:** N004 and canonical event identity. Live quote extension additionally needs R34, R36, R37 and R38 capability evidence; execution remains R48–R51.

**Deliverable:** Build event/date/league navigation, search/filter and links to recorded wagers from existing data. Add explicit statuses and available historical event fields. Define an optional non-executing review slip for hypothetical selections. Enrich with current moneyline/spread/total/futures quote comparisons only through supported licensed feeds and truthful freshness/venue coverage.

**Likely reuse/touchpoints:** Shared venue/event matching modules, historical event queries, Events view and review-slip context; future feed adapters owned by baseline access PRs.

**Acceptance:** Historical events remain distinct from currently tradable markets. Quotes disclose provider, observed time, line, odds, fee and stale/missing state. A review slip is not an accepted bet. No credentials, scraping or unsupported order routing is introduced.

**Verification:** Event/market/period identity collisions, home/away matching, stale/suspended quotes, differing settlement rules, missing scores, hypothetical selections and no-order side-effect tests. Owner browsing walkthrough; live integration testing only after access approval.

**Compatibility/rollback:** Historical browser is independently useful. Quote enrichment is a separately activated child slice, preserving all baseline independent scanner/execution work.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): browse recorded events with linked wagers`
- `feat(tracker): add a non-executing review slip contract`
- `test(tracker): verify event identity and conditional quote states`

### N011 — Add evidence-grounded search and an optional tracker assistant

**Entry/readiness:** N003 and N005; explanation/risk links use delivered R31/R29 capabilities when available. Any external model service requires its own actual selection and data-sharing decision.

**Deliverable:** Provide structured question templates and global search over allowed bets, tags, events and investigations. Optionally add a read-only natural-language query assistant that cites frozen source/context, metric definitions and supporting rows. Useful questions cover drawdowns, venue contributions, CLV gaps and concentrated open exposure. Save reproducible investigations through existing journals.

**Likely reuse/touchpoints:** Shared query/context/search modules, local evidence/assistant adapter, investigation links and adversarial tests. Reuse model governance where relevant; no free-form replacement for accounting queries.

**Acceptance:** Numeric answers reproduce approved deterministic queries. Unknown evidence yields an explicit limitation. Imported notes cannot issue instructions; account scope is enforced. No automatic wager, correction, study launch or filter deployment occurs.

**Verification:** Golden questions with reference results, ambiguous date/cohort requests, prompt injection in notes, private source leakage, unsupported causal/profit claims and changed snapshot detection. Owner review of explanations and citations.

**Compatibility/rollback:** Ship local templates first if an external model is unnecessary or unapproved. An Ask label in a screenshot does not establish any proprietary vendor implementation.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): expose reproducible search and question templates`
- `feat(tracker): add optional evidence-cited assistant responses`
- `test(tracker): enforce grounding privacy and read-only actions`

### N012 — Complete preferences, safe exports and privacy-aware operations

**Entry/readiness:** N004 and N003; account/ledger and other tracker features only where delivered. Hosted session controls depend on an actual authenticated topology and R59 as applicable.

**Deliverable:** Add timezone/date basis, odds display/calculator, dollar/unit preferences with dated unit sizing, theme/contrast/table density, notification preferences and source-status diagnostics. Deliver scope-preserving, formula-safe exports with coverage/provenance and redaction controls. Model profile/session/account permissions only for the deployed topology; include supported reminders and cooling-off preferences without nudging more bets.

**Likely reuse/touchpoints:** Versioned settings/preferences store, tracker export/presentation adapters, security diagnostics and UI/accessibility tests; shared auth/session components when selected.

**Acceptance:** Display settings do not rewrite economics, cash promotion rules or funding history. Export totals and selected rows match the ledger. Spreadsheet-facing text is escaped safely; private content is scoped and redacted. No hosted/security capability is claimed for a local-only deployment.

**Verification:** Odds conversion edge cases, unit-history restatement, settings persistence/reset, CSV formula injection, malformed notes, export permissions, session scope when relevant, keyboard/contrast and privacy settings. Owner review of exports/preferences.

**Compatibility/rollback:** Preserve local operation. Appearance and preferences can be disabled without altering raw records; dated economic policies remain versioned separately.

**Provisional commit subjects** (refine in the selected PR plan):

- `feat(tracker): add versioned preferences and display conversions`
- `feat(tracker): export selected evidence safely`
- `test(tracker): verify privacy accessibility and settings semantics`

### N013 — Evaluate and implement supported account/source synchronization

**Entry/readiness:** N002, R34 and relevant R36/R37 adapters. Conditional on supported access, source rights, authentication, cost and capability evidence; manual CSV remains supported.

**Deliverable:** Evaluate per-venue/API/export or owner-supplied tracking-service data methods for bets, settlements and balances. Document supported/missing fields, sync cadence, paging/cursors, latency, credentials storage, reconnect/revoke and partial-failure states. Implement one approved read-only adapter through the existing staging/reconciliation path, then add packs only with capability tests. Pikkit’s banner is not evidence of a public API or permission to automate its account.

**Likely reuse/touchpoints:** Existing acquisition adapters/venue capability registry, private credential/session storage, ingestion/reconciliation receipts and sync status UI. No credentials in Git or documents.

**Acceptance:** Reconnect and repeated sync do not duplicate economics; revisions and source gaps are visible. Source observation time, export time and last successful import remain distinct. Manual fallback works. Balances are not fabricated for unsupported accounts.

**Verification:** Synthetic provider paging, retry/backoff/rate limits, revoked/expired access, partial response, duplicates/corrections, source reconciliation and interruption. Authorized provider sandbox/local checks only when needed; no paid/account access implied by plan inclusion.

**Compatibility/rollback:** Conditional extension, not a blocker for initial manual-import tracker. Full browser account automation or live execution requires separate baseline feasibility and risk gates.

**Provisional commit subjects** (refine in the selected PR plan):

- `docs(tracker): assess supported source sync capabilities`
- `feat(ingestion): add one approved read-only sync adapter`
- `test(tracker): verify sync isolation retries and reconciliation`

### N014 — Accept the integrated tracker and reconcile all planned coverage

**Entry/readiness:** N004–N007 and N012 for initial tracker release. N008–N011 and N013 require acceptance when their feature slices are activated; unresolved conditional features remain explicitly planned. R33 and R60 retain their independent scope.

**Deliverable:** Perform cross-view and cross-workspace acceptance from manual upload to verified source, ledger, Home/performance/calendar, tags, balances and research/risk drilldown. Maintain a parity/coverage matrix for all 14 screenshot references and new N requirements. Produce synthetic walkthroughs, operating/troubleshooting docs, performance measurements, privacy review, feature rollback and a dated remaining-work register.

**Likely reuse/touchpoints:** Tracker acceptance tests/walkthroughs, public-safe fixture screenshots, docs/roadmap coverage/delivery records and operator guides. Preserve historical source and release evidence.

**Acceptance:** Owner can complete named tasks without hidden interpretation changes; accounting reconciles; unsupported capabilities are clearly labelled; original R requirements are still delivered/planned. No “Pikkit parity” claim unless the reviewed requirement matrix supports it. No incomplete optional capability disappears from the roadmap.

**Verification:** Full required checks, representative declared synthetic workload, keyboard/responsive browser journeys, corruption/recovery tests and explicit owner acceptance of required UI workflows before merge. Record exact commands/source/settings/results and outstanding limits.

**Compatibility/rollback:** Release software capability only; production deployment/account connections/trading remain separate actions. Incremental release is allowed with truthful scope and retained conditional work, not silent retirement.

**Provisional commit subjects** (refine in the selected PR plan):

- `test(tracker): verify end-to-end tracking and research journeys`
- `docs(tracker): publish operating and rollback walkthroughs`
- `docs(roadmap): reconcile tracker acceptance and remaining coverage`

## Baseline ownership crosswalk

| Original work (unchanged) | Tracker relationship |
| --- | --- |
| R01–R04; REQ01–REQ04 | Preserve operating/evidence documentation and governance; N001 adds traceability |
| R05–R07; REQ05–REQ07 | Reuse funding/configuration evidence, venue/product capability and shared context |
| R08–R10; REQ08–REQ09 | Reuse reproducible jobs and explicit import/activation; no tracker rewrite in R10 |
| R11–R15; REQ10–REQ11 | Shared workspace/design/chart contracts; N views consume these |
| R16–R22; REQ12–REQ16 | Discovery, decisions and studies remain research-owned; personal tags do not approve experiments |
| R23–R29; REQ17–REQ21 | One exposure and PrimeDope-inspired risk engine, multiple linked views |
| R30–R32; REQ22–REQ24 | Qualified gaps/explanations and auditable actions; no duplicate CLV inference engine |
| R33/R60; REQ25/REQ50 | N014 supplements tracker acceptance, never replaces original acceptance |
| R34–R40; REQ26–REQ30 | Independent access/capture/normalization and cash/settlement converge through adapters |
| R41–R47; REQ31–REQ37 | Tracker inspects supported model/strategy evidence; does not establish executable edge |
| R48–R51; REQ38–REQ41 | Quote/review interfaces do not activate execution |
| R52–R58; REQ42–REQ48 | Product-specific fantasy/contest economics and cross-product exposure where supported |
| R59; REQ49 | Hosting/security/auth and operational ownership require actual deployment decisions |

## Integrated acceptance and retained questions

Use independent synthetic examples covering settled/open straight bets, push/void/
refund/resettlement, supported multi-leg tickets, repeated exports versus repeated
positions, ladders/conflicts, unknown funding/timing/closing semantics, annotation
reversal, deposits/transfers and bonus treatment. Reconcile exact selected rows and
totals across cards/curves/calendar/ledger; explain different samples. Test timezone/
DST, overlap, zero denominators, missing/stale/unsupported states and formula-safe
export. Define realistic row-count/latency/memory budgets when implementing, not
production claims from a small demo. Concrete owner UI walkthroughs block merge
where required; N001 has no UI to review.

Track every TRK family and screenshot region as delivered/planned/conditional or
an explicitly accepted equivalent. First useful imported-data release needs ledger,
Home, performance/calendar, tags, display preferences and truthful source status.
N014's entry also includes N007's cash/balance integration; unknown actual balances
remain unknown. Analytical/integration extensions are accepted when activated;
manual import remains useful while account sync/live data are unavailable. New
splits keep their N parent and linked destination. Preserve every original idea and
amendment; no retirement by omission or automatic authorization of future PRs.

[Requirements](REQUIREMENTS.md#tracker-requirement-coverage) ·
[Delivery register](DELIVERY_REGISTER.md) · [N001 plan](pr-plans/N001.md)
