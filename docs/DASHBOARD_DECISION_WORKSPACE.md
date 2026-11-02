# Using the decision dashboard

The dashboard opens on **Overview**: published ideas to study and confirmed-production
concerns. These replace the old descriptive cards as the default decision surface.
Research artifacts are read-only; selectors and chart clicks do not rerun discovery,
correct tests, fit models, register experiments or change OddsJam settings.

## Start or restart

```bash
poetry install --with dashboard
poetry run streamlit run dashboard/app.py
```

After pulling this change, **restart the Streamlit process**. The repository disables
file watching deliberately, so an already-running process will retain imported code.
The default local URL is loopback only. Use `ODDSJAM_DASHBOARD_CONFIG` for an explicit
alternative trusted configuration file; filesystem paths cannot be entered in the UI.

`configs/dashboard.yaml` links the existing October report to exact queue and visual
publications. Local private artifacts from A–C must exist. An older or differently
published report without attachments shows an unavailable workspace and can still be
explored descriptively. Missing or corrupt attachments never load another queue.

## Identify an idea and its complete setup

1. Leave Funding, Product and Saved filter at **all** to browse the entire linked
   history. The default 100-wager minimum is a display threshold, not a new test gate.
2. Inspect the leading cards. They preserve the published ranking and representatives;
   filtering does not refill four slots with new winners. Empty corrected/production
   tiers remain empty. Historical paper observations and recorded funded wagers are
   context, not a name-based lifecycle or combined ROI.
3. Click **Inspect setup & plan test**, select a point on the **Evidence map**, or use
   the accessible **Inspect a published idea** selector. Point selection links to the
   same exact record and updates its shareable `?candidate=` link.
4. Compare parent, subgroup and complement under weighted/equal-stake returns. The
   contrast panel displays pointwise intervals only where they were published. They
   are not corrected confidence intervals and global descriptive ideas have none.
5. Open **Full OddsJam setup**. Every control has its baseline/final value, explicit
   state, evidence and blocker. Unknown does not mean default, unrestricted or zero.
   A ready setup still requires operator review; it is not production authorization.
6. Use **Why this priority** for tier, components, nondominated front, rank sensitivity
   and original corrected results. **Overlap & economics** retains constituent
   populations and related hypotheses without pooling their evidence.
7. Open **Plan paper study**. A within-filter idea downloads an intentionally
   incomplete, candidate-bound paper draft. Future dates, sizing, baseline and review
   rules remain for the operator to specify. A configuration reference is attached
   only for a verified setup-ready decision sheet.

Global catalogue candidates download a **global study brief**, explicitly not a
registerable definition. Their new prospective baseline and future definition
integration must be resolved first. No historical parent is invented. Downloading
any file performs no registration, approval, enrollment or launch. See
[paper-test definitions](PROSPECTIVE_PAPER_DEFINITIONS.md) for the separate local flow.

## Production review

**Production health** surfaces published change proposals and negative/conflicting
associations needing investigation. A current PROD lifecycle plus a pinned parent
baseline is required for a proposal; name prefixes and historical funding labels
are insufficient. Current October evidence has no confirmed PROD baseline, so the
workspace displays that confirmation task rather than inventing a modification.

For a supported proposal, the displayed setup is the **exclusion-comparison** sheet,
not the positive subgroup sheet. It retains parent AND NOT (the entire conjunction),
removed/retained exposure and implementation blockers. Confirm actual applied
settings before external edits. Historical losses are not causal profit-saved estimates.

## Shared selectors and exact scope

Controls combine **AND across fields** and **OR within a field**. An idea must explicitly
contain each selected predicate dimension. Separate sport and platform summaries
cannot become an invented sport × platform intersection. Exact threshold operators
remain distinct; equivalent numeric band formatting (3 versus 3.0) does not change
membership. No missing or broad slice is silently substituted for an empty result.

Evidence tier, setup and lifecycle controls narrow idea records. Population diagnostics
require published historical scopes, so these advanced idea selectors produce an
explicit instruction to clear them before viewing population charts. They do not
create a pooled cohort from a union of overlapping ideas. Other shared controls
select exact published chart rows; unsupported intersections stay unavailable.

Reset clears predicate/advanced selections, candidate links and chart-point selection.
Funding, product and minimum-support controls remain explicit independent controls.
Changing reports clears display selectors. The app verifies source and attachment
bytes before display, including session-cache reuse; mtimes are not trusted.

## Interactive notebook diagnostics

Choose **Explore history → Interactive diagnostics**. **Legacy descriptive slices**
are secondary and clearly separate from the prioritized queue. Original within-filter
comparisons, filter health, definitions and methodology remain accessible through
navigation; they do not replace the default queue.

| Diagnostic | Interaction and evidence contract |
| --- | --- |
| Profit by sport/platform/league/market | Diverging bars; switch dollars, weighted ROI or equal-stake return. Responsive population panels avoid overlapping economics. |
| Matched advertised EV versus return | Zoomable bubbles, full hover labels, identity/zero reference, matched support/coverage, weighted/equal switch. |
| Saved-filter EV versus return | Same component, full registry names and confirmed-lifecycle shapes; unknown remains unknown. Overlapping filter totals are not additive. |
| EV bands | Linked return/EV and volume panels; click a band to highlight both panels. Exact source EV survives display binning. |
| Liquidity | Verified USD quote bands only; source capacity is not a guaranteed fill. |
| Liquidity × EV | Selectable heatmap with linked matched-return/support detail, separated by population. |
| Time to event | Ordered published bands, explicit after-start/missing states; verified source timing is required. |
| Cumulative profit by platform | Daily/weekly/monthly source-calendar curves; choose reported dollars or fixed $1 units. Legend isolates series; pan/zoom without clipping outliers. |
| Bet volume over time | Same calendar controls; unique logged candidates, unknown gaps and boundary periods. The performance minimum never erases sparse activity periods. |
| Weekday/hour activity | Bars or weekday × hour heatmap; hover gives active-source-day denominator. This is not covered-day average or scanner supply. |

Every diagnostic has an accessible data/coverage table and supported-data download.
Blocked charts show their source remediation instead. Current liquidity, time-to-event
and fair-closing-reference series remain blocked by evidence. Their controls stay
visible so the required work is concrete.

The upper EV/return panels use matching EV rows or supported joint EV/closing rows;
lower activity/profit panels retain their own eligible denominators. Closing-price EV
is a fair-reference estimate, not realized profit. Source-calendar cumulative amounts
are retrospective attribution, not placement order, settlement order, bankroll balance
or drawdown. Unobserved gaps are not zero activity. All limitations and source details
remain available without putting a methodology wall above the decision cards.

## Acceptance and limits

Automated checks cover optional attachment absence, root confinement, integrity on
cache reuse, unchanged source artifacts, empty/multi-selector scope, exact band
formatting, non-refilled headlines, all ten diagnostic choices, valid chart specs,
unknown setting sheets and paper-only draft/brief exports. Browser acceptance checks
cover the real local October publication, keyboard navigation, point-to-detail links,
setup/study tabs, chart containment and laptop layouts. Private screenshots and
historical artifacts are not committed.

Exploratory ideas remain exploratory. No current result is a proven deployable edge.
This release does not add explanatory models/SHAP, global-definition registration,
quote observation capture, outcome review, promotion or external settings application;
those remain the subsequent workstreams in the
[delivery plan](plans/DASHBOARD_DECISION_WORKSPACE.md).
