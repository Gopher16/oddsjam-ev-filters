# Glossary

These terms describe current contracts. A record at one grain or lifecycle stage
does not establish a record at another. See [architecture](ARCHITECTURE.md) for how
the terms connect. This glossary is a short navigation aid; the
[contract reference](reference/README.md) owns detailed semantics and policy scope.

## Source and research grains

| Term | Meaning and distinction |
| --- | --- |
| **Source observation** | One retained source row/revision with import lineage. Repeated exports or ticket legs are not additional independent wagers. |
| **Import** | Accepted source bytes and declared acquisition metadata, with receipts/audits. Filename or receipt time does not establish event chronology. |
| **Snapshot** | Explicit versioned source selection/resolution at a supported knowledge cutoff. It is not automatically the newest or freshest data. |
| **Canonical wager / wager candidate** | Reconciled wager identity used for ticket-level economics where supported. “Candidate” here does not imply a suggested new filter. Eligibility remains a separate contract. |
| **Leg** | A constituent selection of a ticket. Leg features are preserved without multiplying ticket stake/profit or inventing a ticket EV model. |
| **Opportunity / opportunity proxy** | A source-linked grouping for opportunity analysis. It is not proof of a distinct available quote, executable capacity or an independent trial. |
| **Membership / attribution** | A recorded wager's relationship to a stable filter identity. Membership can be resolved, unresolved or missing; a wager may belong to multiple filters. Filter totals are non-additive. |
| **Filter identity** | Stable registry ID and reviewed names/aliases. Identity alone does not establish historical parameters, funding or actual PROD status. |
| **Signal candidate** | A frozen analytical predicate within a parent/population, with source and policy identity. A global candidate and a similar within-filter candidate have different contexts. |
| **Parent / subgroup / complement** | Parent is the declared study scope; subgroup matches the candidate; complement is parent minus subgroup. Counts and economics reconcile without adding ROI values. |
| **Event group** | Namespace/event identity used for dependence-aware resampling/splits. Proxy grouping does not establish independent underlying events. |
| **Hypothesis family** | Complete predeclared comparison universe for correction. Hiding cards or suppressing overlap does not remove hypotheses from it. |

Sources: [ingestion](INGESTION.md), [canonical facts](CANONICAL_FACTS.md),
[atlas](OPPORTUNITY_ATLAS.md), [within-filter discovery](WITHIN_FILTER_DISCOVERY.md).

## Publications and operations

| Term | Meaning and distinction |
| --- | --- |
| **Publication / bundle** | Immutable versioned output with manifest, policies, input/code lineage and verified payloads. “Published” here means local research output, not public Internet hosting. |
| **Manifest** | Identity, lineage, configuration and integrity metadata for a publication. Hash verification detects altered bytes; it does not prove source truth. |
| **Register / journal** | Append-only sequence of explicit definition, decision, experiment or lifecycle events. A published snapshot freezes selected state; a dashboard view does not write the live journal. |
| **Run / attempt** | Run identifies a frozen core research specification; attempts record execution/reuse/failure. A rerun is not new prospective evidence. |
| **Knowledge cutoff / recorded-at** | Boundary of records available to the system. This is distinct from source placement time, settlement availability and configuration effective time. |
| **Readiness / blocker** | Whether a specific source, setup, chart, model or review requirement is supported. Chart readiness is not statistical validation; setup readiness is not launch approval. |

Sources: [core pipeline](WEEKLY_RESEARCH.md), [visual readiness](VISUAL_DIAGNOSTICS.md),
[release acceptance](RELEASE_ACCEPTANCE.md).

## Definitions and lifecycle

| Term | Meaning and distinction |
| --- | --- |
| **Paper definition** | Versioned intent: exact rules, baseline, future dates, sizing assumptions and review criteria. Saving a definition does not approve, register or launch an experiment. |
| **Baseline** | Pinned comparison configuration with provenance. A prospective baseline is not evidence that the same settings governed historical wagers. |
| **Proposal** | Evidence-linked request for review. A candidate/study draft is not an authorization. |
| **Approval** | Explicit human decision authorizing the specified next step under its policy. Paper approval and settings-change approval are separate decisions. |
| **Registration** | Binds the approved proposal/definition to an experiment protocol and identity. It does not establish actual capture start. |
| **Launch** | Confirmed actual experiment start, separately recorded. Elapsed planned dates do not create launch evidence. |
| **Observation / enrollment** | Prospectively captured parent observation under an exact launched version. It needs timely features/quotes; old exports cannot be enrolled retrospectively. |
| **Availability check** | Evidence of whether the original selection/line/price and evaluated paper stake were available at the declared check time. It is separate from funding and an accepted fill. |
| **Settlement** | Recorded result or amendment with availability/provenance. Late facts stay associated with the original version and can invalidate stale review evidence. |
| **Reviewed decision** | Explicit continue/modify/stop/inconclusive assessment of an exact report/version. It does not itself change the scanner. |
| **Application / applied period** | Operator-confirmed external configuration and effective date, recorded separately from approval. Monitoring links observations to this version. |
| **Promotion / bounded pilot** | Qualifying prospective evidence permits explicit human review of a capped PROD pilot. There is no automatic deployment, profitable-fill guarantee or unlimited production authorization. |
| **Rollback** | A reviewed new change/application restoring specified settings; previous periods and evidence remain intact. |

Sources: [paper definitions](PROSPECTIVE_PAPER_DEFINITIONS.md),
[prospective integration](PROSPECTIVE_EXPERIMENT_INTEGRATION.md),
[reviewed lifecycle](REVIEWED_FILTER_LIFECYCLE.md).

## Setting states

| State | Meaning |
| --- | --- |
| **Observed** | A setting value supported by the stated evidence at its stated time/context. It is not automatically historical or applied evidence. |
| **Inherited** | A candidate carries a value from an explicitly pinned baseline, with that baseline's state and provenance. |
| **Override** | A proposed change to the baseline with explicit value, scope and rationale. |
| **Unset** | Evidence/design explicitly records an unset control. It is not an omitted field or permission to assume a platform default. |
| **Unknown** | Evidence does not establish the setting/value. It remains unresolved rather than becoming zero, unrestricted or today's default. |

The [setting-state reference](reference/EV_SETTINGS.md#states-and-provenance) also validates whether
an analytical restriction has a supported scanner translation. A complete sheet
contains an explicit state for each control; completeness does not mean every value
is known or the setup is executable.

## Economics and explanation terms

| Term | Meaning and distinction |
| --- | --- |
| **Advertised EV** | Source theoretical estimate. Percentage points and fractions have explicit units; display bands do not replace exact numeric predicates. |
| **Stake-weighted ROI** | Eligible source profit divided by eligible source stake. It reflects recorded exposure and is not intrinsic filter quality. |
| **Equal-stake unit return** | Mean of each eligible wager's profit/stake. Fixed-unit profit is a hypothetical scaling diagnostic, not an executable historical replay. |
| **Funding population** | Source real-money, paper or unknown classification; neither a saved-filter name nor a model creates it. |
| **Corrected historical comparison** | A comparison meeting the original family correction contract; it remains conditional exploratory history, not forward validation. |
| **SHAP contribution** | Attribution to a frozen model prediction on its declared output scale/reference. It is not causal effect, proof of a profitable setting or a new corrected hypothesis. |
| **Fixed-model filtered explanation** | Selects rows from frozen model outputs/background. It does not fit a model on the selected subset. |
| **Separately fitted subset model** | Explicit offline build with its own scope, model, background, splits and publication identity. |

Sources: [EV/settings](reference/EV_SETTINGS.md), [economics](reference/ECONOMICS.md),
[statistical evidence](reference/STATISTICAL_EVIDENCE.md),
[models/explanations](reference/MODELS_EXPLANATIONS.md).

[Documentation home](README.md)
