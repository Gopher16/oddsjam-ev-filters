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
Select the lowest-numbered planned, eligible PR unless the owner specifies another;
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
| <a id="r07"></a>R07 | B | Define shared analysis context and linked selection | [R05](#r05),[R06](#r06) | Implemented; remote pending | [REQ07](REQUIREMENTS.md#req07); [plan](pr-plans/R07.md) |
| <a id="r08"></a>R08 | C | Orchestrate optional research builders with reproducible jobs | [R04](#r04),[R07](#r07) | Planned | [REQ08](REQUIREMENTS.md#req08) |
| <a id="r09"></a>R09 | C | Implement bounded CSV staging and import preview | [R03](#r03),[R08](#r08) | Planned | [REQ09](REQUIREMENTS.md#req09) |
| <a id="r10"></a>R10 | C | Add drag-and-drop refresh, progress and publication activation | [R09](#r09) | Planned | [REQ09](REQUIREMENTS.md#req09) |
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

R01–R05 delivery is reconciled through R06; the corresponding plans retain their
pre-merge history and subsequent evidence. R06's actual merge will be recorded in
its completion report/PR and reconciled by the next authorized PR. Never label a
pre-merge entry merged to avoid a bookkeeping loop.

[Requirements](REQUIREMENTS.md) identify coverage; [capability reconciliation](CAPABILITY_RECONCILIATION.md)
identifies what already exists and what remains. Use the [delivery workflow](DELIVERY_WORKFLOW.md)
and [PR template](pr-plans/TEMPLATE.md) for each selected entry. New proposals,
splits and moved/deferred work retain their original destination and approved decision.

[Programme index](README.md) · [Change log](CHANGE_LOG.md) · [Opportunity backlog](OPPORTUNITY_BACKLOG.md)
