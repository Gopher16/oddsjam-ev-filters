# Operator guide

Use this route to review research and prepare a prospective paper study. All-history
browsing includes supported populations without blending hypothetical and funded returns.

## Open and orient

From the repository root:

```bash
poetry install --with dashboard
poetry run streamlit run dashboard/app.py
```

The viewer reads [the dashboard configuration](../../configs/dashboard.yaml). It needs
local published artifacts. If the intended report is unavailable, use the
[analyst guide](ANALYST.md); do not substitute a similarly named report or data export.
Check the selected report's source scope and dates. Generation time is not data freshness.

## Choose a research question

1. **Overview / Opportunities:** inspect the leading distinct hypotheses and why they
   are ranked. Empty corrected or actionable queues are valid outcomes.
2. **Production health:** review confirmed-PROD concerns and historical investigations
   separately. Funding or a `PROD` prefix alone does not establish deployed status.
3. **Explore history:** inspect supported notebook-inspired charts and source blockers.
4. **Explain returns:** inspect fixed-model associations, coverage and utility. Sidebar
   selections do not retrain the model; separately fitted subset publications are explicit.

Shared selectors narrow published evidence. On the decision/history pages they do
not create unpublished intersections or rerun tests. On explanations they select
recorded rows from fixed model outputs. See [dashboard behavior](../DASHBOARD_DECISION_WORKSPACE.md)
and [explanation behavior](../DASHBOARD_EXPLANATIONS.md).

## Inspect the setup before preparing a test

Open a candidate's **Full OddsJam setup** and review the parent, subgroup and complement,
weighted/equal-stake results, support, concentration, overlap and setting provenance.
An analytical two-field predicate is not a complete scanner configuration. An unknown
setting is not a default; an inherited setting needs a pinned supported baseline.
See [setting states](../GLOSSARY.md#setting-states) and
[complete configuration records](../DECISION_CONFIGURATION_RECORDS.md).

Download the study draft, resolve its blockers and follow the
[paper-definition guide](../PROSPECTIVE_PAPER_DEFINITIONS.md). Future dates, complete
baseline/settings, paper sizing, quote-availability rules and review criteria must be
fixed explicitly. Downloading or registering a definition does not approve or launch it.

## Follow the prospective lifecycle

Use [prospective integration](../PROSPECTIVE_EXPERIMENT_INTEGRATION.md) for approval-bound
registration, actual launch, observations, availability checks, settlements and scheduled
review. Capture the parent and complement, including unavailable quotes. Historical
exports cannot retrospectively enroll a study. Publish results explicitly; the dashboard
reads immutable snapshots rather than live registers.

At review, use [reviewed lifecycle](../REVIEWED_FILTER_LIFECYCLE.md) to record
continue/modify/stop/inconclusive, new versions, proposed settings, separate approval
and confirmed external application, subsequent monitoring and rollback evidence.
The tool does not operate OddsJam. Passing the prospective policy permits human
bounded-pilot review; it does not automatically promote, fund or establish fills.

For the complete first-study checklist and known operational limits, see
[release acceptance](../RELEASE_ACCEPTANCE.md). Reproducible synthetic walkthroughs
and screenshots are planned separately in documentation PR K.

[Documentation home](../README.md) · [Architecture](../ARCHITECTURE.md)
