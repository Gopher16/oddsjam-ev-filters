# How the OddsJam Positive EV tool is used

This guide records the owner's operating practice as documented in November 2026,
reviewed for repository documentation on 2026-11-23. It explains how an advertised
opportunity becomes a discretionary execution, pass or paper record before a manual
Bet Tracker export enters this offline repository. It is dated operating context,
not a live settings feed, a recommendation to wager or a verified profitable strategy.

Use the [operator guide](../guides/OPERATOR.md) for the research dashboard and
prospective study tools. This repository does not operate OddsJam, submit orders,
apply scanner settings or automatically acquire exports. Current UI descriptions
and owner-reported behavior do not verify historical CSV semantics.

## Evidence and terminology

The private, owner-supplied **OddsJam Research & Development Roadmap** is the source
for this narrative. Sections and figure numbers below refer to that frozen Word
baseline, not public image files. Its fingerprint and review decisions are in the
[R01 delivery plan](../roadmap/pr-plans/R01.md). No private document, account
identifier, screenshot or raw bet export is distributed with this guide.

| Label | What it establishes | What it does not establish |
| --- | --- | --- |
| Owner practice | The owner's reported configuration, decisions and logging conventions at the documented time | Universally effective settings, a complete historical configuration or verified fills |
| Support confirmation | A specific answer supplied by OddsJam Support and reproduced in the owner document | Independent API/formula verification or the answer's indefinite applicability |
| Platform UI description | A displayed control, tooltip, icon or default value in the source screenshots | That an inactive value is enforced, or that platform claims are empirically established |
| Working hypothesis | A discretionary explanation or suspected market/venue effect | Counterparty identity, causality or a validated trading edge |
| Proposed practice | A desired future logging or research step | A deployed feature or a complete historical observation set |

The historical filter name is **PROD - Exchanges: Core Exchange Conservative (Low
Risk, Slow Build)**. On **November 10, 2026**, it was renamed **TEST - Exchanges:
Core Exchange Conservative (Low Risk, Slow Build)** and **all real-money execution
under this strategy stopped, with no exceptions** (R&D Case G). Subsequent use is
paper evaluation using the same discretionary reasoning. Historical references to
PROD in the source describe the earlier workflow; they are not evidence that this
strategy is funded today. This transition is documentation only: no private registry,
configuration history, funding classification or prior publication is changed here.

## Dated configuration and display choices

These are owner-reported settings from the November 2026 source, not repository
defaults or a verified per-wager history. Record later changes with their evidence
and effective dates; do not backfill old settings from this table.

| Control | Documented setting or state | Evidence and interpretation |
| --- | --- | --- |
| Bankroll / Kelly multiplier | $20,000 sizing configuration; 0.25 (quarter Kelly) | Owner practice, §2 / Figure 1. Dated recommendation inputs, not a current account balance or a prescribed stake policy |
| Saved filter | Core Exchange Conservative; historical PROD, TEST since November 10 | Owner practice, §2 and Case G. Names alone do not establish funding or settings |
| Recommended Filters | Disabled | Owner practice, Figures 2–3. Owner believes presets may suit recreational books better; that belief remains untested |
| EV Formula Builder | No custom configuration explicitly selected; panel displays 100% OddsJam Algo | Owner practice / UI description, Figures 4–5. The default panel display does not establish Algo contribution under the separate devig selection |
| Devig Book | Pinnacle | Owner practice; Alex from OddsJam Support confirms selecting Pinnacle instead of Recommended Weights uses only Pinnacle's prices, not Recommended Weights (§2) |
| Devig Method | Worst Case | Owner practice, Figures 10–11. UI describes the least favorable supported devig estimate; this limited conservatism does not guarantee a safer or profitable wager |
| Sportsbook selector | Accessible exchanges for the owner's Minnesota-based workflow | Owner practice, Figure 6 / §2.3. Do not infer current account access, product availability or an exhaustive selected list from brand names |
| American odds | Minimum −150; maximum +150 | Owner practice, Figure 7. Do not infer exact historical export boundary semantics from UI labels |
| Advertised EV Percentage | 2.5% minimum | Owner practice, Figure 8. Screening threshold, not true edge or a guarantee; exact evaluated predicates retain their own contracts |
| Minimum Liquidity | Unselected / inactive | Owner practice / UI description, Figure 9. Grey $300 is an inactive suggested value, not a minimum; absence of a restriction is not zero observed liquidity |
| League | No selected league restriction | Owner practice, Figures 12–13. Qualifying leagues remain within the platform/filter universe, not every possible sporting market |
| Market | Main Markets focused on spreads, moneylines and totals | Owner practice, Figures 14–15. Source does not enumerate a complete exact market-ID set; do not translate the family description into one |
| Date Range | Next 24 hours | Owner practice, Figures 16–17. Time until event, not historical analysis lookback |
| Game | No selected game/team restriction | Owner practice, Figure 18 |
| Min Rec Bet | Unselected / inactive | Owner practice / UI description, Figure 19. Grey $15 is not an enforced minimum and does not change bankroll/Kelly |
| Minimum Market Width | Unselected / inactive | Owner practice / UI description, Figure 20. Grey 25 is not an active minimum |
| Maximum Market Width | 25 | Owner practice, Figure 20. Screening guideline; no validated universal cutoff or inferred cross-venue units |
| Bet Warnings | Same Game, Conflicting and Previously Placed enabled | Owner practice, Figures 21–24. Alerts prompt review; they do not independently approve or reject a trade |
| Sorting | Percentage (advertised EV) | Owner practice, Figure 26. Ranks discovery attention; highest EV need not be the best executable trade |
| General display settings | Source describes odds formats, no-vig odds, probability, width, supporting books and floating widget | UI descriptions, Figure 25. Exact enabled states beyond those explicitly documented remain unspecified |
| Trading style | Take existing quotes; historically almost entirely pre-match | Owner practice, §2.1. No intentional maker-order strategy; a partial fill can still leave a resting remainder requiring review |

**Reference-price clarification:** Alex's support answer resolves the current
Pinnacle-versus-Recommended-Weights question for this configuration. It does not
recover the reference inputs, methods, timestamps or model version for every old
Bet Tracker row. Likewise, a platform tooltip calling no-vig odds “true” odds or
suggesting confidence based on width is a platform description, not evidence of
objective truth. See the [EV and settings contract](../reference/EV_SETTINGS.md)
for exact units, setting states and scanner translation boundaries.

## Discover and review an opportunity

1. Select the intended saved filter and inspect its actual controls. The owner has
   historically used one filter at a time. The owner's understanding is that selecting
   multiple saved filters surfaces their union; that behavior and whether Bet Tracker
   records every matching filter remain unconfirmed. Do not assume additive labels.
2. Sort by Percentage to prioritize attention, then identify the exact venue, event,
   market, side, line, event timing, offered price and advertised EV. An item appearing
   in a PROD-labelled filter does not mean it was accepted or executed.
3. Inspect the venue quote itself before acting. Confirm whether OddsJam shows a
   single price or blended odds. The chain-link icon and its hover detail describe
   the contributing levels for a displayed blended price and combined size. A useful
   average quote is not a guarantee that every contributing level is still available.
4. Compare **same-side and opposite-side liquidity** at the intended price and at
   nearby deeper levels. Consider distance to large resting orders, width, staleness,
   rapid changes, intended stake, fees and settlement mechanics. Missing visible depth
   is an evidence gap, not zero depth or automatic approval; Case E describes a venue
   without a visible order book in this workflow.
5. Review Same Game, Conflicting and Previously Placed warnings against existing
   exposure and records. Consider correlated markets and repeated bets; alerts are
   inputs to discretionary judgment rather than an independently enforced risk policy.
6. Decide whether the apparent edge and available execution are sufficiently credible
   to act under the applicable funded or paper process. A large EV estimate alone
   cannot replace quote, context and feasibility review. Pass or reconsider later if
   the outlook changes.

The order-book heuristic is deliberately provisional: large resting size on the
intended side can raise concern about trading against an informed participant;
less same-side size and greater opposing-side depth can appear more favorable.
Neither pattern identifies sharp/recreational participants or proves an edge. The
owner's more favorable impressions of Novig, BetOpenly and some basketball markets
are hypotheses subject to selection, exposure and time-period effects, not a
validated ranking of venues or sports. No numerical depth-based veto is specified.

An illustrative 1–10 judgment rubric in §2.4 ranges from substantial concerns
(1–3), through marginal/conflicting evidence (4–5), to plausible candidates (6–7)
and compelling setups (8–10), always subject to final checks. Ratings are optional,
subjective reasoning aids, not win probabilities or required historical labels.
The [A–G case studies](case-studies/README.md) link each decision to the
[source figure catalogue](../images/oddsjam-workflow/CATALOGUE.md). The catalogue
currently provides text placeholders; no private account images are published or
claimed to have passed visual review.

## Execute, reconcile and log

For the historical funded workflow, the owner follows the opportunity to the venue,
checks the actual price/size and submits a taker order only after discretionary
review. Confirm what was filled rather than recording the entire suggested stake
as though it executed. In Case D, a suggested order partially filled and the owner
canceled the remainder that had become resting liquidity. This is distinct from
intentionally launching a maker strategy. Do not infer a universal automatic
cancellation rule or venue-wide behavior from that example.

Displayed, intended, submitted and filled prices/sizes can differ. Case F explains
that OddsJam rounded a displayed price while the owner executed at the venue's
actual price and recorded the corresponding American odds. Neither rounding nor a
liquidity discrepancy proves slippage, order-book consumption or a blended trade.
Reconcile venue execution and Bet Tracker records using available evidence; retain
unknown causes rather than explaining discrepancies by assumption.

Custom liquidity tags are a lightweight record of **posted quote context**:

- For a single-price opportunity, the tag represents liquidity at that selected price.
- For a blended opportunity, it represents the combined liquidity across the quotes
  included in the calculated blended odds.
- Neither means the entire order book, nearby large same-side orders, opposite-side
  depth, price persistence or verified filled size. Missing tags are not zero liquidity.

Actual execution amount/price belongs with funded execution evidence. A displayed
Kelly recommendation is a suggested stake; a later settlement result does not
verify the original liquidity, fee treatment or order acceptance. Preserve these
limitations when reviewing Bet Tracker history.

## Paper trading and pass handling

The source describes three materially different observation processes:

| Process | Selection and sizing convention | Interpretation |
| --- | --- | --- |
| Historical funded executions | Discretionary acceptance plus actual venue execution constraints | Recorded outcomes for selected wagers, not the full surfaced universe |
| Older broadly logged TEST opportunities | Nearly every surfaced opportunity logged, generally with displayed quote and Kelly-suggested stake | Broad paper capture; not a portfolio the owner necessarily would have executed |
| Newer discretionary TEST evaluation | Apply the same reasoning as the funded workflow before hypothetical execution; paper-executed stake follows the Kelly recommendation | Hypothetical results for selected opportunities; recommended size may exceed executable capacity |

The November 10 strategy transition above is confirmed at date granularity; no
exact intraday timestamp is supplied. It does not date every historical TEST filter's
selection regime. Keep the new practice distinct from an assumption that all paper
records before/after a date share the same process.

**Clarified paper convention:** R&D §3 retains an older sentence saying sizing still
needs clarification. The owner's direct clarification and the same document's §5.1
specify **Kelly-recommended sizing for every paper-executed entry**. This guide uses
that clarified convention without editing the frozen Word baseline. Review plausible
capacity separately: do not silently replace the recorded paper stake with fillable
size or describe the recommendation as a verified hypothetical fill.

Reviewed passes are **not currently logged in Bet Tracker**. The proposed low-friction
`Passed` or `Trade not Executed` tag would apply **only to TEST filters**, with tagged
passes reported separately from paper executions and funded wagers. It is a proposal,
not an implemented pass dataset, and no settlement treatment for passes is assumed.
For PROD opportunities the owner wants passes left unlogged in Bet Tracker to avoid
a Previously Placed green checkmark and preserve later reconsideration. Do not log
PROD passes using that proposed tag or infer that an unlogged opportunity was unseen.

Capturing discovery/decision timestamps, rationale, quote context and hypothetical
capacity alongside the Kelly amount is a desired improvement; current history is
incomplete. Future independent pass capture can be designed separately without
changing OddsJam's Previously Placed state. A small queue of two or three coherent
TEST hypotheses is the intended research direction, not a currently delivered
selection system or a reason to invent missing observations.

## Manual export and research handoff

The owner downloads Bet Tracker CSV through a signed-in browser. The documented
[CSV export endpoint](https://oddsjam.com/api/backend/bets/export-csv) works in Chrome;
Python requests received a Cloudflare HTTP 403 browser challenge. Alex from OddsJam
Support subsequently confirmed that automated exports are not available at that time
(R&D §3). Manual browser export followed by explicit repository ingestion remains
the documented route; draft acquisition PR #3 and disabled weekly scheduling do not
establish unattended acquisition. Do not infer an undocumented authentication flow
or try to bypass the browser challenge from these instructions.

Keep raw exports in ignored local storage and follow the existing
[analyst refresh guide](../guides/ANALYST.md) and
[ingestion contract](../INGESTION.md). Pin the intended export/snapshot and preserve
its provenance; do not substitute a conveniently named dataset. A recent import or
report generation time does not prove export freshness or decision-time knowledge.

Later review should trace the displayed opportunity, actual or hypothetical decision,
reported economics, funding/selection regime and available quote evidence separately.
Avoid pooling funded and paper ROI, counting overlapping filters as independent
opportunities, reconstructing deleted settings from today's UI or treating missing
fields as zero. The [data-grain](../reference/DATA_GRAINS.md),
[economics](../reference/ECONOMICS.md), [chronology](../reference/CHRONOLOGY.md) and
[prospective lifecycle](../reference/PROSPECTIVE_LIFECYCLE.md) references define the
repository's existing analytical boundaries; this guide changes none of them.

The repository dashboard reads verified publications. It cannot infer all reviewed
passes, fill a historical order book, register a real study from a narrative or
change the live OddsJam filter. Use supported historical evidence to propose
prospective tests, then record complete versioned evidence and follow the existing
review/approval/launch process separately. This owner workflow is context for future
work, not a machine-readable baseline that silently clears configuration blockers.

[Documentation home](../README.md) · [Research operator guide](../guides/OPERATOR.md) ·
[Delivery register](../roadmap/DELIVERY_REGISTER.md)
