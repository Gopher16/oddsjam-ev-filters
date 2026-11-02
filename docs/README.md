# Documentation home

Choose a task first. The role guides and architecture describe current behavior;
planning documents preserve historical proposals and explicitly deferred work.

| Entry point | What you will find |
| --- | --- |
| [Operator](guides/OPERATOR.md) | Open the dashboard, review evidence, inspect a setup, prepare a study and follow reviewed outcomes |
| [Analyst](guides/ANALYST.md) | Choose source snapshots, build explicit publications, attach evidence and interpret limitations |
| [Contributor](guides/CONTRIBUTOR.md) | Navigate module boundaries, verify changes and preserve contracts |
| [Architecture](ARCHITECTURE.md) | Three diagrams covering research, prospective studies and applied changes |
| [Glossary](GLOSSARY.md) | Grains, artifacts, lifecycle terms and setting states |

## Hands-on workflows

Start the [isolated synthetic tutorial](walkthroughs/README.md) to review a candidate,
define/approve/launch a paper study, review a TEST modification and refresh from a new
export. Use the [configuration/publication map](CONFIGURATION_PUBLICATION_MAP.md) and
[troubleshooting/remediation guide](TROUBLESHOOTING.md) alongside the commands.

## Shared research and data contracts

The [authoritative contract reference](reference/README.md) defines grains,
economics, EV/settings, chronology, statistical evidence, model explanations,
prospective lifecycle and lineage. Approved [exploratory policies](reference/policies/EXPLORATORY.md)
and [prospective promotion policy](reference/policies/PROMOTION.md) remain separate.
Use these explanations across components; the guides below retain commands,
artifact schemas and stage-specific behavior.

## Component implementation guides

| Area | Component guides |
| --- | --- |
| Source selection and grains | [Ingestion](INGESTION.md), [canonical facts](CANONICAL_FACTS.md), [opportunity atlas](OPPORTUNITY_ATLAS.md) |
| Historical identity | [Label inventory](FILTER_LABEL_INVENTORY.md), [mapping proposals](FILTER_MAPPING_PROPOSALS.md), [versioned attribution](VERSIONED_FILTER_ATTRIBUTION.md) |
| EV and sizing | [Continuous EV](CONTINUOUS_EV_CONTRACT.md), [sizing history](SIZING_HISTORY.md), [sizing diagnostics](SIZING_DIAGNOSTICS.md) |
| Discovery and inference | [Core robust discovery](ROBUST_DISCOVERY.md), [within-filter discovery](WITHIN_FILTER_DISCOVERY.md), [separate statistical prioritization](STATISTICAL_PRIORITIZATION.md) |
| Decisions and setup | [Configuration records](DECISION_CONFIGURATION_RECORDS.md), [review queues](REVIEW_QUEUES.md), [research proposals](RECOMMENDATIONS.md) |
| Dashboard and plots | [Decision workspace](DASHBOARD_DECISION_WORKSPACE.md), [visual datasets](VISUAL_DIAGNOSTICS.md), [base viewer](RESEARCH_DASHBOARD.md) |
| Models | [Offline explanatory publications](EXPLANATORY_MODELS.md), [interactive explanations](DASHBOARD_EXPLANATIONS.md) |
| Prospective work | [Paper definitions](PROSPECTIVE_PAPER_DEFINITIONS.md), [experiment protocol](EXPERIMENT_PROTOCOL.md), [definition-to-experiment integration](PROSPECTIVE_EXPERIMENT_INTEGRATION.md) |
| Reviewed changes | [Lifecycle, bounded pilot policy and monitoring](REVIEWED_FILTER_LIFECYCLE.md) |
| Operations and acceptance | [Core pipeline](WEEKLY_RESEARCH.md), [release runbook](RELEASE_ACCEPTANCE.md), [dated acceptance evidence](releases/2026-11-02-acceptance.md) |

Some component guides retain the state at their original delivery. For example,
the descriptive within-filter builder does not perform inference; the separate
statistical-prioritization publication adds approved exploratory comparisons.
Likewise, the older core discovery policy and the later prioritization policy are
separate contracts. Use the [implementation ledger](plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md)
to follow their composition rather than applying one stage's limitations to all stages.

## Planning and status

- [Discovery implementation ledger](plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md):
  current requirement status, actual components, tests, gaps and next actions.
- [Original multidimensional discovery brief](plans/MULTIDIMENSIONAL_SIGNAL_DISCOVERY.md):
  retained design/history; its opening context describes an earlier system.
- [Dashboard decision-workspace plan](plans/DASHBOARD_DECISION_WORKSPACE.md):
  the expanded A–I sequence and original acceptance contract; current operating
  guidance lives in the reference library above.
- [Documentation follow-up sequence](plans/DOCUMENTATION_OVERHAUL.md): J–M navigation,
  walkthroughs, research reference and maintenance; later work remains proposed.
- [Original research plan](RESEARCH_PLAN.md): historical goals and research framing,
  not an approval or a replacement for current versioned policies.

Historical plans do not authorize scanner changes or establish completed capability.
Release evidence is dated: it does not assert that source freshness, deployments or
candidate results are current today. Private/local roadmap notes are not published
as part of these guides.

[Return to the repository README](../README.md)
