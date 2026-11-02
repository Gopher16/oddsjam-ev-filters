# Configuration and publication map

All paths below are relative to the repository root. Builders consume **explicit
verified directories**, not a latest-file convention. Output roots are private local
storage by default. The synthetic adapter overrides every mutable root into its
fresh workspace and retains the actual CLI argv in `commands.jsonl`.

## Research builders

| Script | Explicit input / configuration | Output / effect |
| --- | --- | --- |
| [ingest_bet_tracker.py](../scripts/ingest_bet_tracker.py) | `--config` before import/snapshot/inspect; manual CSV and truthful acquisition metadata | Source archive, import receipts, reconciliation snapshot; no research or acquisition |
| [build_canonical_facts.py](../scripts/build_canonical_facts.py) | `--snapshot`, canonical policy, reviewed registry | Canonical wager facts, source/opportunity links, attribution and eligibility audit |
| [build_opportunity_atlas.py](../scripts/build_opportunity_atlas.py) | `--bundle` canonical directory, atlas policy | Wager dimensions, segments, coverage and opportunity views |
| [build_discovery_report.py](../scripts/build_discovery_report.py) | Explicit atlas, core discovery policy and optional supported temporal evidence | Core discovery family, evidence, gates and audit |
| [build_recommendations.py](../scripts/build_recommendations.py) | Exact canonical/atlas/core discovery plus research policy and capability evidence | Research proposals and review requirements; no approval |
| [run_weekly.py](../scripts/run_weekly.py) | `--config` plus exactly one CSV / snapshot + aware cutoff / resume mode | Frozen core plan, attempts, checkpoints and final report; optional experiment follow-up |
| [build_continuous_ev.py](../scripts/build_continuous_ev.py) | `--canonical` | Exact source EV/provenance tables and missing/conflict audit |
| [build_sizing_history.py](../scripts/build_sizing_history.py) | Canonical plus explicit history or empty-history knowledge cutoff, optional previous publication | Effective-date/unknown-regime linkage publication; no inferred bankroll |
| [build_sizing_diagnostics.py](../scripts/build_sizing_diagnostics.py) | Canonical, exact sizing publication, diagnostic policy | Weighted/equal-stake, concentration, regime and exclusion diagnostics |
| [build_within_filter_discovery.py](../scripts/build_within_filter_discovery.py) | Canonical, atlas, diagnostics, declared multidimensional views | Frozen within-parent candidates/evidence/comparisons/membership/exclusions |
| [build_signal_prioritization.py](../scripts/build_signal_prioritization.py) | Discovery and matching diagnostics, inference policy | Complete-family tests/correction, overlap and exploratory priorities |
| [build_decision_records.py](../scripts/build_decision_records.py) | Discovery, prioritization, explicit baseline/lifecycle/translation evidence, knowledge/effective times | Complete setting sheets, states, setup blockers and proposed change sheets |
| [build_visual_diagnostics.py](../scripts/build_visual_diagnostics.py) | Canonical, atlas, diagnostics and source-evidence visual policy | Visual candidates/cohorts/periods/activity/coverage/readiness |
| [build_review_queues.py](../scripts/build_review_queues.py) | Discovery, prioritization, decisions and visuals, review policy | Frozen all-history opportunity/production/investigation queues, ranks and explanations |
| [build_explanatory_models.py](../scripts/build_explanatory_models.py) | Canonical and visuals; optional explicit JSON `--scope` | Offline frozen predictions, SHAP, utility and feature audits; declined populations remain explicit |
| [verify_release.py](../scripts/verify_release.py) | Explicit dashboard config/run ID and optional performance budgets | Read-only verification/reconciliation JSON; no build, fitting or approval |

The core runner orchestrates its core stages only. Newer optional builders are
separate explicit invocations; they cannot all be enabled by adding arbitrary fields
to `weekly_research.yaml`. See [architecture](ARCHITECTURE.md) and
[the analyst guide](guides/ANALYST.md).

## Identity and prospective lifecycle tools

| Script | Input | Output / effect |
| --- | --- | --- |
| [inventory_filter_labels.py](../scripts/inventory_filter_labels.py) | Explicit verified snapshots/canonical lineage | Raw-label inventory with separate count grains/overlaps |
| [propose_filter_mappings.py](../scripts/propose_filter_mappings.py) | Inventory and canonical publication | Proposed identity/alias/segmentation evidence and review queue |
| [audit_filter_attribution.py](../scripts/audit_filter_attribution.py) | Explicit before/after canonical publications | Coverage/reconciliation audit; no in-place relabeling |
| [manage_paper_definitions.py](../scripts/manage_paper_definitions.py) | Explicit register; draft candidate or save protocol with discovery/prioritization and optional exact decision sheet | Append-only intent versions or frozen definition publication; no approval/launch |
| [build_definition_proposal.py](../scripts/build_definition_proposal.py) | Exact definition publication/version, intended scanner identity and sampling rule | A paper proposal for explicit review |
| [record_research_decision.py](../scripts/record_research_decision.py) | Register, exact proposal bundle/ID, expected status/head, reviewer/rationale | Human proposal review/approval/rejection event; no registration/launch |
| [manage_experiments.py](../scripts/manage_experiments.py) | Approval-bound proposal/protocol, explicit actor/head and event payloads | Registration, launch/end transitions, enrollment/link/check/deviation/settlement/review events |
| [build_experiment_report.py](../scripts/build_experiment_report.py) | Explicit experiment register, canonical publication and aware cutoff | Immutable per-version outcomes, availability and review readiness |
| [manage_filter_lifecycle.py](../scripts/manage_filter_lifecycle.py) | Explicit lifecycle register, actor/rationale/head and typed payload | Policy, review decision, baseline, change, approval/application, monitoring and human follow-up records |
| [build_filter_lifecycle_report.py](../scripts/build_filter_lifecycle_report.py) | Explicit lifecycle register/cutoff | Immutable reviewed changes, applications, observations, economics and drift |

Every mutation requires its own supported evidence and expected state. A published
report is a snapshot of a journal, not a live connection to it. Approval and confirmed
external application are separate records. Templates in `configs/` are examples,
not evidence about your scanner or permission to promote a filter.

## Legacy and acceptance tools

| Tool | Input | Output / boundary |
| --- | --- | --- |
| [preprocess_bet_tracker.py](../scripts/preprocess_bet_tracker.py) | Legacy preprocessing YAML/source | Historical notebook Parquet and QA artifacts; separate from canonical contracts |
| [export_annual_pnl.py](../scripts/export_annual_pnl.py) | Annual P&L configuration and selected historical data | Local configured P&L exports/audits; not canonical pipeline orchestration |
| [acceptance/browser.mjs](../scripts/acceptance/browser.mjs) | Existing local dashboard and isolated Chrome CDP; private output directory | Read-only release-browser evidence; its private-snapshot scenario is not the synthetic tutorial capture |
| [operator_demo.py](examples/operator_demo.py) | Fresh marked temporary workspace and explicit step acknowledgements | Synthetic CLI tutorial inputs/registers/publications; process-local simulated clock |
| [capture_synthetic.mjs](examples/capture_synthetic.mjs) | Synthetic tutorial dashboard, isolated Chrome and marked workspace | Watermarked synthetic documentation screenshots only |

## Dashboard attachment configuration

| Key in [dashboard.yaml](../configs/dashboard.yaml) | Meaning |
| --- | --- |
| `artifact_root` | Core report catalogue/results root |
| `evidence_root` | Trusted root confining optional attached publications |
| `evidence_publications[run_id]` | Exact discovery and prioritization paths matching the selected run |
| `decision_records_publication` | Complete setting-sheet publication |
| `workspace_publications[run_id]` | Exact review queues and visuals |
| `explanation_publications[run_id]` | Explicit frozen global model and separately fitted subset publications |
| `paper_definitions_publication` | Frozen definitions matching the source candidate lineage |
| `prospective_report_publication` | Exact per-version outcome snapshot |
| `lifecycle_report_publication` | Exact reviewed change/application/monitoring snapshot |
| `min_analyzed_wagers`, `max_display_rows` | Display policies; they do not change inference families or research gates |

The tutorial emits separate `dashboard.yaml`, `dashboard-defined.yaml`,
`dashboard-reviewed.yaml`, `dashboard-monitored.yaml` and `dashboard-refresh.yaml`.
The refresh configuration deliberately omits incompatible earlier study attachments.
Keep an earlier configuration/report available to inspect that earlier study.

Source acquisition policy, research policy, scanner baseline evidence, journal heads
and dashboard locators are different kinds of configuration. Changing a dashboard
path cannot supply missing scanner evidence or approve a pending change. For errors,
use [troubleshooting and remediation](TROUBLESHOOTING.md).

[Walkthroughs](walkthroughs/README.md) · [Documentation home](README.md)
