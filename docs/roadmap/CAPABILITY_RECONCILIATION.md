# Current capability and planned programme

Reviewed against main `75f12cae9386b8ece8d74014d865a6bbded719ae` for R04 on
2026-11-23. This is a planning crosswalk, not a new private-data audit or a rewrite
of prior acceptance. The [historical multidimensional brief](../plans/MULTIDIMENSIONAL_SIGNAL_DISCOVERY.md)
and [implementation ledger](../plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md)
remain authoritative for their detailed S/H/D/W/E/Z/V/A requirement statuses.
The ledger's November 2 acceptance is dated evidence, not today's dataset/account state.

## Reuse before extension

**Existing** means the linked bounded code/contract exists. **Partial** names actual
engineering still needed. **Evidence-gated** means a supported feature lacks required
source evidence; it does not imply an absent algorithm exists. **Planned/conditional**
retains new work under its roadmap gate. Merge status is tracked separately in the
[delivery register](DELIVERY_REGISTER.md); all REQ families are in [coverage](REQUIREMENTS.md).

| Capability / requirements | Existing implementation to reuse | Remaining work / destination |
| --- | --- | --- |
| Operating documentation and evidence context; REQ01–REQ03 | [Workflow](../operations/ODDSJAM_WORKFLOW.md), [cases](../operations/case-studies/README.md), [source inventory](../operations/SOURCE_FIELD_INVENTORY.md), [EG register](../operations/EVIDENCE_GAPS.md) | R01–R03 merged. Public images are placeholders, not reviewed raw assets; historical EG remediation stays open. |
| Evidence/configuration and venue scope; REQ05–REQ06 | [Attribution](../../src/oddsjam_ev/analysis/filter_attribution.py), [sizing linkage](../../src/oddsjam_ev/analysis/sizing_linkage.py), [configuration records](../../src/oddsjam_ev/analysis/decision_config.py), [canonical policy](../../configs/canonical_facts.yaml) | R05 implements an optional [operating-regime ledger](../OPERATING_REGIMES.md), explicit adapters and conservative candidate linkage; activation in shared consumers stays future work. R06 adds an optional [venue/product capability registry](../VENUE_CAPABILITIES.md), historical-policy seed and explicit requests; current vendor contracts/access remain evidence-gated. Existing mappings do not prove account access or current settings. |
| Shared workspace scope; REQ07/REQ24 | [Queries](../../src/oddsjam_ev/dashboard/decision_queries.py), [verified attachments](../../src/oddsjam_ev/dashboard/decision_artifacts.py) | R07 defines shared serialized analysis context; R32 connects investigation/scenario/action. Existing selectors are not that full contract. |
| Immutable import and optional orchestration; REQ08–REQ09 | [Ingestion](../INGESTION.md), [runner](../../scripts/run_weekly.py), separate [visual](../../scripts/build_visual_diagnostics.py) and [queue](../../scripts/build_review_queues.py) builders | R08 integrates optional dependency stages; R09 stages private CSV/preview; R10 adds confirmed upload/progress/activation. Current runner is not an automated export service. |
| Navigation/framework and chart reuse; REQ10–REQ11 | [Workspace](../../src/oddsjam_ev/dashboard/pages/decision_workspace.py), [charts](../../src/oddsjam_ev/dashboard/decision_charts.py), [visual publication](../VISUAL_DIAGNOSTICS.md) | R11 assesses design/framework; R12 implements chosen path; R13 audits Notebook 02 contracts; R14/R15 improve charts. No React migration selected here. |
| Broad candidates, targeting and small queue; REQ12–REQ14 | [Bounded discovery](../../src/oddsjam_ev/analysis/multidimensional_discovery.py), [prioritization](../../src/oddsjam_ev/analysis/signal_prioritization.py), [ranking](../../src/oddsjam_ev/analysis/review_ranking.py), [research journal](../../scripts/record_research_decision.py) | R16 briefs; R17 broad hypothesis evaluation; R18 manageable overlap-aware queue/manual rules; R19 journal integration. Existing many-slice output does not meet the whole new user journey. |
| Active versioned studies; REQ15 | [Definitions](../../scripts/manage_paper_definitions.py), [experiments](../../scripts/manage_experiments.py), [lifecycle](../../scripts/manage_filter_lifecycle.py), [outcome reports](../../scripts/build_experiment_report.py) | R20 stage/capture UI, R21 trends/reviews. Existing approval/launch separation must remain; recording a plan never launches a real study. |
| Temporal evaluation; REQ16 | [Core temporal gates](../../src/oddsjam_ev/analysis/temporal_validation.py), [statistical contract](../reference/STATISTICAL_EVIDENCE.md) | R22 scopes broader-family chronological protocol/evaluation and repeated-look handling. New within-filter holdout integration is partial, not solved by random group splits. |
| Event dependence and ladders; REQ17 | [Paired comparisons](../../src/oddsjam_ev/analysis/paired_comparison.py), [sizing diagnostics](../../src/oddsjam_ev/analysis/sizing_diagnostics.py) | R23 identities/exposure, R24 joint ladder payoffs and correlation diagnostics. Event-proxy resampling does not provide a measured joint portfolio model. |
| PrimeDope-inspired risk; REQ18–REQ21 | [Sizing-history contract](../SIZING_HISTORY.md), [economics](../reference/ECONOMICS.md), [visual source gates](../VISUAL_DIAGNOSTICS.md) | R25 scenario jobs; R26 joint paths; R27 concurrent Kelly accounting; R28 ruin/recovery/capital; R29 integrated laboratory. No current full bankroll replay/ruin simulator is claimed. |
| EV/return diagnosis and explanations; REQ22–REQ24 | [Continuous EV](../../src/oddsjam_ev/analysis/continuous_ev.py), [explanatory models](../../src/oddsjam_ev/analysis/explanatory_models.py), [model builder](../../scripts/build_explanatory_models.py) | R30 matched investigations; R31 improved models/forward evaluation; R32 action links. Fitted explanations are not causal evidence or assured predictive utility. |
| Release and operations; REQ25/REQ49–REQ50 | [Release verifier](../../scripts/verify_release.py), [acceptance record](../releases/2026-11-02-acceptance.md), [maintenance checks](../../scripts/check_documentation.py) | R33 upgraded Section 5 acceptance; R59 chosen-host/security/recovery evidence; R60 activated-branch end-to-end governance. Desktop acceptance does not certify multi-user hosting. |
| Independent capture/scanner; REQ26–REQ30 | Existing immutable publication/ingestion patterns and [source gaps](../operations/EVIDENCE_GAPS.md) | Conditional R34 feasibility, R35 storage, R36 authorized read-only adapter, R37 normalization, R38 pricing, R39 funnel, R40 ledgers. No live feed/account integration currently supplied by this crosswalk. |
| Joint models and feedback; REQ31–REQ34 | Offline explanatory lineage/evaluation and existing reproducible publication patterns | Conditional R41 point-in-time features, R42 simulator, R43 pricing/model validation, R44 feedback/drift/capacity. A model registry is not automatic model promotion. |
| Arbitrage/middles/portfolios/maker; REQ35–REQ37 | Existing exact identity/economics constraints and dependence diagnostics | Conditional R45–R47 require compatible contracts, joint outcomes and fill/capacity evidence. Indicative arbitrage is not guaranteed executed profit. Maker remains a separate study from today's taker workflow. |
| Controlled execution; REQ38–REQ41 | Research lifecycle records provide useful intent/evidence separation only | Conditional R48 access/risk, R49 sandbox orders, R50 supervised execution, R51 separately gated unattended operation. No existing research approval authorizes an order. |
| DFS/best ball/managed fantasy; REQ42–REQ47 | Future validated joint outcome model may be reusable; current prop/explanation code is not a contest engine | Conditional R52–R57 require product/scoring/field/season/roster contracts, source access and independent evaluation. Sidekick is an experience reference, not validation or a licensed implementation. |
| Expansion; REQ48 | R06 registry and later adapter/model contracts are planned reusable foundations | Conditional R58 expands only after relevant release/model gates; fantasy additionally requires R52. Unsupported venue requests stay explicit rather than silently mapping economics. |

## Preserve the older ledger's unfinished work

Implemented S/H/D/W/E/Z/V/A entries continue to be reused, with their original
bounded meaning and linked tests. The following crosswalk accounts specifically for
all entries marked partial, optional or evidence-blocked; it does not promote them
to completed or erase their prerequisites.

| Legacy IDs | Current limit retained | Roadmap destination / disposition |
| --- | --- | --- |
| S06, H06, Z02, A18 | Unknown historical chronology/settings/applied sizing | R03 EG04–EG08/EG18 and R05 evidence; conditional R39–R41 future capture. A new feature cannot recover unseen history. |
| S10 | No single historical pre-code assessment matching the original brief | R04 supplies a present-day crosswalk; future PR plans record their own assessment. Never fabricate a retrospective approval artifact. |
| D06, A05 | Integrated chronological evaluation for the newer within-filter family missing | R22, conditional on verified timing and predeclared evaluation policy. Existing core gates retained. |
| D17 | Independent validation still needs real prospective evidence | R20–R22 collect/review study evidence; R33 accepts software only. No synthetic result establishes a profitable filter. |
| W04 | Missing outside-rule universe prevents widening claims | R17 assesses bounded supported hypotheses; R39 future observed funnel supplies new coverage. A widening evaluator beyond baseline scope remains an explicitly conditional expansion. |
| E04 | No dedicated multi-cutpoint sensitivity publication | R30 assessment, with a separately scoped conditional extension if not covered by its selected plan. Preserve finite family/holdout accounting; no inference from sliders. |
| E06, A19 | Smooth calibration/generalization not implemented by basic EV thresholds | R30/R31 research assessment after R22; any extra calibration method remains conditional on a specific plan/policy, not approved here. |
| Z03 | Generalized suggested/accepted-stake reconstruction missing | R05/R27 assess supported evidence adapters; R39/R40 future capture can retain new facts. Unsupported historical reconstruction remains unavailable. |
| Z07, V16 | No exact bankroll/drawdown/concurrent Kelly replay | R25–R29 scenario/accounting work. Actual historical replay additionally needs EG04–EG07/EG10/EG14 evidence; synthetic scenarios remain labelled assumptions. |
| V01, V02, V11, V13, V14, V22 | Exact chart forms, complete drill-through/facets, concentration views and misleading connecting lines remain partial | R13 catalogue assessment followed by R14/R15; R07 shared scope and R29 scenario views where relevant. Do not mislabel these as data-only blockers. |
| V04, V05, V06, V10 | Liquidity/timing/closing-reference visual contracts exist but source evidence is unverified | R03 EG04/EG09/EG11–EG13, R13 source gate and R14 supported adapter/view work. Missing provenance does not become zero or a verified series. |
| V15 | Prospective quote-verification funnel visualization incomplete | R20/R21 study capture/trends; conditional R39 broader observed funnel. Preserve failed/late checks, not only accepted observations. |
| A08; optional pipeline integration row | Separate immutable publishers not all core runner stages | R08 optional-stage dependency/retry/resume work, preserving existing commands and old reports. |
| Hosted/multi-user remaining item | Only declared desktop workload verified | R11 design/performance assessment and conditional R59 selected-host hardening/benchmarks before deployment. |

The original provisional filenames and coarse delivery sequences remain mapped to
their delivered replacements in the [ledger's integration section](../plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md#integration-files-and-delivery-order).
V19's old navigation is already superseded by the delivered workspace design; R11
may revise the next design with an explicit decision, not revive it by accident.
Current evidence collection for candidate settings uses existing contracts; a new
R05 adapter extends those contracts rather than implying they never existed. R05
implementation was reconciled against main `9b7a181fffe1f18258f23ce47b596ea398b4cfea`
on 2026-11-24; real account evidence remains an owner task, not recovered history.

Conditional expansions above preserve original ideas without expanding an approved
PR by stealth. At selection, either include a justified bounded item within the
baseline scope or record an N-series proposal, impact and owner decision before
material extra implementation. No unfinished item is retired by this crosswalk.

[Programme index](README.md) · [Requirement coverage](REQUIREMENTS.md) ·
[Delivery register](DELIVERY_REGISTER.md)
