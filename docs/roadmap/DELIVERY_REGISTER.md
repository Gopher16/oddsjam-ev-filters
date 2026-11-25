# Roadmap delivery register

This register preserves the frozen programme identities; roadmap R numbers are not
GitHub PR numbers. Source fingerprints and authority are in [source baselines](SOURCE_BASELINES.md).
The owner authorizes only the selected PR's delivery through the applied instructions.
Future entries below are planned or conditional, not blanket implementation,
paid-service, account-connection, deployment or wagering authorization.

## Status and dependency rules

**Merged** requires actual remote/main evidence. **Implemented; remote pending** is
pre-merge. **Planned** means retained baseline scope, not eligible until dependencies
and the selected PR's evidence/acceptance gates pass. **Conditional** additionally
requires the specified branch/product feasibility or activation decision. A dependency
being merged never establishes profitable outcomes or complete source evidence.

Dependencies below preserve the baseline wording, including conditional clauses.
R34 can assess feasibility after R04; R33 is required for full product cutover.
R58 needs R52 for fantasy, R59 additionally needs the relevant component release
evidence, and R60 includes R51 only if activated and R54–R59 only if pursued. Do not
turn these into unconditional ordering or silently omit activated prerequisites.
Select the lowest-numbered planned, eligible original R PR unless the owner specifies
another or an accepted additive scheduling decision applies (see N001 below);
complete its merge/verification before starting the next. Re-read sources each time.

## Full programme

| ID | Phase | Baseline scope | Dependencies / conditions | Status | Requirements and delivery evidence |
| --- | --- | --- | --- | --- | --- |
| <a id="r01"></a>R01 | A | Document the actual OddsJam operating workflow | None | Merged; remote/main verified | [REQ01](REQUIREMENTS.md#req01); [PR #34](https://github.com/Gopher16/oddsjam-ev-filters/pull/34), merge `1a0e486c5a4ab0be1d23d71b33216bd0551b7a8e`; [plan](pr-plans/R01.md) |
| <a id="r02"></a>R02 | A | Add a screenshot catalogue and operating case studies | [R01](#r01) | Merged; remote/main verified | [REQ02](REQUIREMENTS.md#req02); [PR #35](https://github.com/Gopher16/oddsjam-ev-filters/pull/35), merge `a72e3aee8dc60c1eab4421c6e346193f3d708f22`; [plan](pr-plans/R02.md) |
| <a id="r03"></a>R03 | A | Publish source limitations and decision-process regimes | [R01](#r01) | Merged; remote/main verified | [REQ03](REQUIREMENTS.md#req03); [PR #36](https://github.com/Gopher16/oddsjam-ev-filters/pull/36), merge `75f12cae9386b8ece8d74014d865a6bbded719ae`; [plan](pr-plans/R03.md) |
| <a id="r04"></a>R04 | A | Land the implementation programme and traceability register | [R01](#r01),[R02](#r02),[R03](#r03) | Merged; remote/main and post-merge CI verified | [REQ04](REQUIREMENTS.md#req04); [PR #37](https://github.com/Gopher16/oddsjam-ev-filters/pull/37), merge `9b7a181fffe1f18258f23ce47b596ea398b4cfea`; [plan](pr-plans/R04.md) |
| <a id="r05"></a>R05 | B | Version operating regimes and prospective configuration evidence | [R03](#r03),[R04](#r04) | Merged; remote/main and post-merge checks verified | [REQ05](REQUIREMENTS.md#req05); [PR #38](https://github.com/Gopher16/oddsjam-ev-filters/pull/38), merge `0572b9e9a5e1d74869de4c6082d0d822ab78bcf9`; [plan/results](pr-plans/R05.md) |
| <a id="r06"></a>R06 | B | Add a versioned venue/product and source-capability registry | [R05](#r05) | Merged; remote/main and post-merge checks verified | [REQ06](REQUIREMENTS.md#req06); [PR #39](https://github.com/Gopher16/oddsjam-ev-filters/pull/39), merge `dcba6376a61be3d29c57fba10ff1a10d55ea233b`; [plan/results](pr-plans/R06.md), [guide](../VENUE_CAPABILITIES.md) |
| <a id="r07"></a>R07 | B | Define shared analysis context and linked selection | [R05](#r05),[R06](#r06) | Merged; remote/main and post-merge checks verified | [REQ07](REQUIREMENTS.md#req07); [PR #40](https://github.com/Gopher16/oddsjam-ev-filters/pull/40), merge `fd53093fa1c4833741b9c6df21d8f28431d9ab10`; [plan](pr-plans/R07.md) |
| <a id="r08"></a>R08 | C | Orchestrate optional research builders with reproducible jobs | [R04](#r04),[R07](#r07) | Merged; remote/main and post-merge CI verified | [PR #41](https://github.com/Gopher16/oddsjam-ev-filters/pull/41), merge `1de23f5486a24d7042d303e4f9329b79e6cd0e50`; [REQ08](REQUIREMENTS.md#req08); [plan/results](pr-plans/R08.md), [guide](../RESEARCH_JOBS.md) |
| <a id="r09"></a>R09 | C | Implement bounded CSV staging and import preview | [R03](#r03),[R08](#r08) | Merged; remote/main and post-merge CI verified | [PR #42](https://github.com/Gopher16/oddsjam-ev-filters/pull/42), merge `e3e89548b7e16c0eed5035b11ea3cc478c1811be`; [REQ09](REQUIREMENTS.md#req09); [plan/results](pr-plans/R09.md) |
| <a id="r10"></a>R10 | C | Add drag-and-drop refresh, progress and publication activation | [R09](#r09) | Implemented; owner UI acceptance and remote CI pending | [REQ09](REQUIREMENTS.md#req09); [plan/results](pr-plans/R10.md) |
| <a id="r11"></a>R11 | D | Design the task-oriented workspace and framework decision record | [R07](#r07),[R10](#r10) | Planned | [REQ10](REQUIREMENTS.md#req10) |
| <a id="r12"></a>R12 | D | Implement workspace shell, clear scope and accessible design system | [R11](#r11) | Planned | [REQ10](REQUIREMENTS.md#req10) |
| <a id="r13"></a>R13 | D | Reconcile notebook and dashboard chart contracts | [R03](#r03),[R07](#r07),[R11](#r11) | Planned | [REQ11](REQUIREMENTS.md#req11) |
| <a id="r14"></a>R14 | D | Deliver interactive EV, return and group diagnostics | [R12](#r12),[R13](#r13) | Planned | [REQ11](REQUIREMENTS.md#req11) |
| <a id="r15"></a>R15 | D | Deliver time, exposure and activity exploration | [R12](#r12),[R13](#r13) | Planned | [REQ11](REQUIREMENTS.md#req11) |
| <a id="r16"></a>R16 | E | Persist optional research briefs and targeting questionnaire | [R06](#r06),[R07](#r07),[R12](#r12) | Planned | [REQ12](REQUIREMENTS.md#req12) |
| <a id="r17"></a>R17 | E | Evaluate broad, coherent hypotheses across supported venues | [R08](#r08),[R13](#r13),[R16](#r16) | Planned | [REQ13](REQUIREMENTS.md#req13) |
| <a id="r18"></a>R18 | E | Build a small overlap-aware research queue with manual guidance | [R17](#r17),[R16](#r16) | Planned | [REQ13](REQUIREMENTS.md#req13) |
| <a id="r19"></a>R19 | E | Connect briefs, suggestions and review decisions to existing journals | [R18](#r18) | Planned | [REQ14](REQUIREMENTS.md#req14) |
| <a id="r20"></a>R20 | F | Expose active TEST/PROD stages and practical decision capture | [R12](#r12),[R19](#r19) | Planned | [REQ15](REQUIREMENTS.md#req15) |
| <a id="r21"></a>R21 | F | Add active-study trends, support progress and review workspace | [R15](#r15),[R20](#r20) | Planned | [REQ15](REQUIREMENTS.md#req15) |
| <a id="r22"></a>R22 | F | Publish a broader-family chronological evaluation protocol | [R03](#r03),[R17](#r17),[R21](#r21) | Planned | [REQ16](REQUIREMENTS.md#req16) |
| <a id="r23"></a>R23 | G | Identify event, player, ladder and cross-filter exposure groups | [R05](#r05),[R06](#r06),[R07](#r07) | Planned | [REQ17](REQUIREMENTS.md#req17) |
| <a id="r24"></a>R24 | G | Quantify correlated exposure and joint ladder payoffs | [R23](#r23),[R22](#r22) | Planned | [REQ17](REQUIREMENTS.md#req17) |
| <a id="r25"></a>R25 | G | Specify and implement reproducible scenario jobs | [R07](#r07),[R08](#r08),[R24](#r24) | Planned | [REQ18](REQUIREMENTS.md#req18) |
| <a id="r26"></a>R26 | G | Implement joint return-path and outcome-distribution simulation | [R25](#r25) | Planned | [REQ18](REQUIREMENTS.md#req18) |
| <a id="r27"></a>R27 | G | Implement Kelly policies and concurrent bankroll accounting | [R26](#r26) | Planned | [REQ19](REQUIREMENTS.md#req19) |
| <a id="r28"></a>R28 | G | Implement drawdown, recovery, ruin and capital-adequacy analysis | [R27](#r27) | Planned | [REQ20](REQUIREMENTS.md#req20) |
| <a id="r29"></a>R29 | G | Deliver the PrimeDope-inspired integrated risk laboratory | [R12](#r12),[R21](#r21),[R28](#r28) | Planned | [REQ21](REQUIREMENTS.md#req21) |
| <a id="r30"></a>R30 | H | Publish matched EV/ROI and performance-gap investigations | [R14](#r14),[R22](#r22),[R24](#r24) | Planned | [REQ22](REQUIREMENTS.md#req22) |
| <a id="r31"></a>R31 | H | Improve explanatory models and forward evaluation integration | [R22](#r22),[R30](#r30) | Planned | [REQ23](REQUIREMENTS.md#req23) |
| <a id="r32"></a>R32 | H | Connect diagnosis, explanation and scenario to auditable action | [R19](#r19),[R29](#r29),[R30](#r30),[R31](#r31) | Planned | [REQ07](REQUIREMENTS.md#req07), [REQ24](REQUIREMENTS.md#req24) |
| <a id="r33"></a>R33 | H | Accept and document the upgraded research workflow | [R10](#r10),[R18](#r18),[R21](#r21),[R29](#r29),[R32](#r32) | Planned | [REQ25](REQUIREMENTS.md#req25) |
| <a id="r34"></a>R34 | I | Assess independent-platform access, costs and architecture | [R04](#r04); [R33](#r33) for full product cutover | Conditional; not activated | [REQ26](REQUIREMENTS.md#req26) |
| <a id="r35"></a>R35 | I | Build immutable market-event capture and replay storage | [R34](#r34),[R06](#r06) | Conditional; not activated | [REQ27](REQUIREMENTS.md#req27) |
| <a id="r36"></a>R36 | I | Implement one authorized read-only venue/feed adapter | [R35](#r35) | Conditional; not activated | [REQ27](REQUIREMENTS.md#req27) |
| <a id="r37"></a>R37 | I | Normalize cross-source events, markets and settlement semantics | [R36](#r36),[R23](#r23) | Conditional; not activated | [REQ27](REQUIREMENTS.md#req27) |
| <a id="r38"></a>R38 | I | Build reference pricing and independent opportunity detection | [R37](#r37) | Conditional; not activated | [REQ28](REQUIREMENTS.md#req28) |
| <a id="r39"></a>R39 | I | Capture the complete opportunity-to-decision funnel | [R38](#r38),[R20](#r20) | Conditional; not activated | [REQ29](REQUIREMENTS.md#req29) |
| <a id="r40"></a>R40 | I | Reconcile settlement, fees and private cash/position ledgers | [R37](#r37),[R39](#r39) | Conditional; not activated | [REQ30](REQUIREMENTS.md#req30) |
| <a id="r41"></a>R41 | J | Add point-in-time features, registry and shadow scoring | [R39](#r39),[R40](#r40),[R31](#r31) | Conditional; not activated | [REQ31](REQUIREMENTS.md#req31) |
| <a id="r42"></a>R42 | J | Develop the first joint game/player outcome simulator | [R41](#r41) | Conditional; not activated | [REQ32](REQUIREMENTS.md#req32) |
| <a id="r43"></a>R43 | J | Validate pricing utility and governed model deployment | [R42](#r42),[R38](#r38),[R22](#r22) | Conditional; not activated | [REQ33](REQUIREMENTS.md#req33) |
| <a id="r44"></a>R44 | J | Introduce feedback, market-microstructure and capacity investigations | [R39](#r39),[R43](#r43) | Conditional; not activated | [REQ34](REQUIREMENTS.md#req34) |
| <a id="r45"></a>R45 | K | Implement indicative arbitrage and middle strategy research | [R38](#r38),[R40](#r40),[R43](#r43) | Conditional; not activated | [REQ35](REQUIREMENTS.md#req35) |
| <a id="r46"></a>R46 | K | Evaluate ladders, correlated portfolios and capital allocation | [R24](#r24),[R43](#r43),[R45](#r45) | Conditional; not activated | [REQ36](REQUIREMENTS.md#req36) |
| <a id="r47"></a>R47 | K | Create a separate maker-strategy feasibility and fill study | [R44](#r44),[R46](#r46) | Conditional; not activated | [REQ37](REQUIREMENTS.md#req37) |
| <a id="r48"></a>R48 | L | Specify execution access and independent risk controls | [R34](#r34),[R40](#r40),[R46](#r46) | Conditional; not activated | [REQ38](REQUIREMENTS.md#req38) |
| <a id="r49"></a>R49 | L | Build sandbox order state machines and reconciliation | [R48](#r48) | Conditional; not activated | [REQ39](REQUIREMENTS.md#req39) |
| <a id="r50"></a>R50 | L | Deliver supervised execution and multi-leg residual-risk handling | [R49](#r49),[R45](#r45) | Conditional; not activated | [REQ40](REQUIREMENTS.md#req40) |
| <a id="r51"></a>R51 | L | Gate unattended operation and automated monitoring | [R50](#r50),[R43](#r43) | Conditional; not activated | [REQ41](REQUIREMENTS.md#req41) |
| <a id="r52"></a>R52 | M | Design the shared fantasy simulation and contest contract | [R42](#r42),[R43](#r43) | Conditional; not activated | [REQ42](REQUIREMENTS.md#req42) |
| <a id="r53"></a>R53 | M | Build contest scoring and joint player/slate projections | [R52](#r52) | Conditional; not activated | [REQ43](REQUIREMENTS.md#req43) |
| <a id="r54"></a>R54 | M | Evaluate DFS fields, duplication and lineup portfolios | [R53](#r53) | Conditional; not activated | [REQ44](REQUIREMENTS.md#req44) |
| <a id="r55"></a>R55 | M | Implement weekly-season simulation and best-ball decision engine | [R52](#r52),[R53](#r53) | Conditional; not activated | [REQ45](REQUIREMENTS.md#req45) |
| <a id="r56"></a>R56 | M | Deliver a supported draft assistant and portfolio view | [R55](#r55) | Conditional; not activated | [REQ46](REQUIREMENTS.md#req46) |
| <a id="r57"></a>R57 | M | Extend to FFPC and managed high-stakes league decisions | [R55](#r55),[R56](#r56) | Conditional; not activated | [REQ47](REQUIREMENTS.md#req47) |
| <a id="r58"></a>R58 | N | Expand sports, venues and products through repeatable adapter packs | [R33](#r33),[R37](#r37),[R43](#r43); [R52](#r52) for fantasy | Conditional; not activated | [REQ06](REQUIREMENTS.md#req06), [REQ48](REQUIREMENTS.md#req48) |
| <a id="r59"></a>R59 | N | Harden hosting, security, observability and disaster recovery | [R34](#r34); component release dependencies | Conditional; not activated | [REQ49](REQUIREMENTS.md#req49) |
| <a id="r60"></a>R60 | N | Complete end-to-end acceptance, governance and sustainable operations | [R33](#r33),[R44](#r44),[R51](#r51) if activated,[R54](#r54)–[R59](#r59) if pursued | Conditional; not activated | [REQ50](REQUIREMENTS.md#req50) |

## Delivery maintenance

R01–R09 are reconciled through N001. R09 tested head
`805d76a85ef8155f1e271f13d337098d1283422c` and merge are present on fetched main;
[main CI 45487190940](https://github.com/Gopher16/oddsjam-ev-filters/actions/runs/45487190940)
passed Ruff, documentation and tests. N001 merged in [PR #43](https://github.com/Gopher16/oddsjam-ev-filters/pull/43)
at `b08275880a2c0ab1ecfe701062575107377592cc`; main CI 45525069997 passed.
R10 is implemented under its saved plan; owner UI acceptance remains pending.
Pending is not merged; no self-SHA bookkeeping commit is required.

[Requirements](REQUIREMENTS.md) identify coverage; [capability reconciliation](CAPABILITY_RECONCILIATION.md)
identifies what already exists and what remains. Use the [delivery workflow](DELIVERY_WORKFLOW.md)
and [PR template](pr-plans/TEMPLATE.md) for each selected entry. New proposals,
splits and moved/deferred work retain their original destination and approved decision.

[Programme index](README.md) · [Change log](CHANGE_LOG.md) · [Opportunity backlog](OPPORTUNITY_BACKLOG.md)

## Additive tracker stream — N001–N014

PD005 adopts the 2026-11-24 R&D §7 / PR roadmap §12 addition. **N001 is merged and verified; R10 is the current selected PR**. R10's original dependency/scope row
above remains unchanged; this is a sequencing checkpoint, not a retroactive change.
N002–N014 require explicit selection or a recorded owner-authorized insertion
and their dependencies. They are not automatically prioritized ahead of eligible
R work. No new dependency is imposed on R01–R09; all R01–R60 remain intact.

The dependency column is the required entry set; conditional capability gates in
the next column apply when pursued. This prevents R40, live quote, hosted-session
or optional simulation readiness from accidentally blocking the imported-data core.
N013 requires relevant adapters, not an unsupported provider chosen by assumption.

| ID | Scope | Required dependencies | Conditional readiness / retained integration | Status | Coverage |
| --- | --- | --- | --- | --- | --- |
| <a id="n001"></a>N001 | Adopt the tracker addition and reconcile the programme | [R09](#r09) | Owner-applied expanded sources; no UI | Merged; PR #43 and main CI verified | [TRK01](REQUIREMENTS.md#trk01); [card](TRACKER_ADDITION.md#n001--adopt-the-tracker-addition-and-reconcile-the-programme) |
| <a id="n002"></a>N002 | Define the source-linked tracker ledger and annotation extensions | [N001](#n001), [R06](#r06), [R07](#r07), [R10](#r10) | R03/R09 semantics retained | Planned; not selected | [TRK02](REQUIREMENTS.md#trk02); [card](TRACKER_ADDITION.md#n002--define-the-source-linked-tracker-ledger-and-annotation-extensions) |
| <a id="n003"></a>N003 | Publish shared tracker accounting queries and filter context | [N002](#n002), [R07](#r07), [R13](#r13) | R13 chart/context contract | Planned; not selected | [TRK03](REQUIREMENTS.md#trk03); [card](TRACKER_ADDITION.md#n003--publish-shared-tracker-accounting-queries-and-filter-context) |
| <a id="n004"></a>N004 | Deliver the dedicated Bet Tracker shell and searchable ledger | [N003](#n003), [R12](#r12) | R10 import/activation reused | Planned; not selected | [TRK04](REQUIREMENTS.md#trk04); [card](TRACKER_ADDITION.md#n004--deliver-the-dedicated-bet-tracker-shell-and-searchable-ledger) |
| <a id="n005"></a>N005 | Add linked performance, date/season controls and calendars | [N004](#n004), [R14](#r14), [R15](#r15) | Shared N003 accounting | Planned; not selected | [TRK05](REQUIREMENTS.md#trk05); [card](TRACKER_ADDITION.md#n005--add-linked-performance-dateseason-controls-and-calendars) |
| <a id="n006"></a>N006 | Add auditable tags, saved views and cohort comparisons | [N002](#n002), [N004](#n004) | R19/R20 integration when available; basic tags do not wait | Planned; not selected | [TRK06](REQUIREMENTS.md#trk06); [card](TRACKER_ADDITION.md#n006--add-auditable-tags-saved-views-and-cohort-comparisons) |
| <a id="n007"></a>N007 | Add account balances, cash movements, at-risk capital and promotions | [N002](#n002), [N004](#n004) | Reuse R40 if delivered; otherwise bound extension and specify future adapter | Planned; not selected | [TRK07](REQUIREMENTS.md#trk07); [card](TRACKER_ADDITION.md#n007--add-account-balances-cash-movements-at-risk-capital-and-promotions) |
| <a id="n008"></a>N008 | Deliver qualified closing-reference and actual-return diagnostics | [N003](#n003), [N005](#n005), [R30](#r30) | Verified closing reference needed for derived EV; limited source-CLV child only if explicitly scoped | Planned; not selected | [TRK08](REQUIREMENTS.md#trk08); [card](TRACKER_ADDITION.md#n008--deliver-qualified-closing-reference-and-actual-return-diagnostics) |
| <a id="n009"></a>N009 | Connect position scenarios, ladders, duplicates and conflicts | [N004](#n004), [R23](#r23), [R24](#r24) | R25–R29 required for probability/risk simulation; R46 reused when available | Planned; not selected | [TRK09](REQUIREMENTS.md#trk09); [card](TRACKER_ADDITION.md#n009--connect-position-scenarios-ladders-duplicates-and-conflicts) |
| <a id="n010"></a>N010 | Add historical event browsing and conditional quote review | [N004](#n004) | Canonical event identity; live quotes additionally R34/R36/R37/R38; execution R48–R51 | Planned; not selected | [TRK10](REQUIREMENTS.md#trk10); [card](TRACKER_ADDITION.md#n010--add-historical-event-browsing-and-conditional-quote-review) |
| <a id="n011"></a>N011 | Add evidence-grounded search and an optional tracker assistant | [N003](#n003), [N005](#n005) | R29/R31 links when delivered; external AI separately decided | Planned; not selected | [TRK11](REQUIREMENTS.md#trk11); [card](TRACKER_ADDITION.md#n011--add-evidence-grounded-search-and-an-optional-tracker-assistant) |
| <a id="n012"></a>N012 | Complete preferences, safe exports and privacy-aware operations | [N004](#n004), [N003](#n003) | Hosted sessions require actual authenticated topology/R59 as applicable | Planned; not selected | [TRK12](REQUIREMENTS.md#trk12); [card](TRACKER_ADDITION.md#n012--complete-preferences-safe-exports-and-privacy-aware-operations) |
| <a id="n013"></a>N013 | Evaluate and implement supported account/source synchronization | [N002](#n002), [R34](#r34), [R36](#r36), [R37](#r37) | Conditional supported source/access/rights/auth/cost; relevant R36/R37 adapters | Conditional; not activated | [TRK13](REQUIREMENTS.md#trk13); [card](TRACKER_ADDITION.md#n013--evaluate-and-implement-supported-accountsource-synchronization) |
| <a id="n014"></a>N014 | Accept the integrated tracker and reconcile all planned coverage | [N004](#n004), [N005](#n005), [N006](#n006), [N007](#n007), [N012](#n012) | Initial core only; N008–N011/N013 acceptance when activated; retain R33/R60 gates | Planned; not selected | [TRK14](REQUIREMENTS.md#trk14); [card](TRACKER_ADDITION.md#n014--accept-the-integrated-tracker-and-reconcile-all-planned-coverage) |

[N001 plan](pr-plans/N001.md) · [Detailed scope and ownership](TRACKER_ADDITION.md) ·
[Screenshot catalogue](../images/pikkit-tracker/CATALOGUE.md)
