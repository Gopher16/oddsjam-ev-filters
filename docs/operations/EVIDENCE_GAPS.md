# Historical evidence gaps and remediation tasks

Reviewed 2026-11-23. Sources: the frozen owner R&D roadmap §3 historical limitations,
§5.1 clarified practice and §6.2 future capture; [operating workflow](ODDSJAM_WORKFLOW.md),
[source inventory](SOURCE_FIELD_INVENTORY.md) and [decision regimes](DECISION_REGIMES.md).
This register describes known evidence boundaries, not measured missingness rates in
a newly audited private snapshot. EG IDs are stable references, not new roadmap PR IDs.

Each entry connects an analytical consequence to a concrete task and functional
owner. These roles describe responsibility, not assigned people or promised delivery
dates. **All remediation tasks remain open** unless a later dated record supplies
actual evidence and an approved disposition. Existing safeguards are already
implemented contracts, not claims that missing historical evidence has been recovered.

## How to close or carry a gap

Record the source/snapshot, field coverage and denominator, supporting evidence
reference, namespace/filter/period scope, observation and effective times, reviewer,
limitations and versioned destination. Partial evidence closes only the bounded
part it supports. Retain conflicts and the original record. If history is
unrecoverable, mark it as such with a reason and a separate prospective capture
task; do not fill it using current settings or future outcomes. R04 owns programme
traceability and R05 owns machine-readable regime/configuration evidence. Broader
capture requirements remain the R&D §6.2 future platform direction, not work shipped
by this documentation PR. The [delivery register](../roadmap/DELIVERY_REGISTER.md)
tracks implementation separately.

## Acquisition, identity and time

### EG01

**Export scope and completeness.** A tracker export is not proof of all account
wagers, all reviewed opportunities or all scanner opportunities. Missing passes,
source filters and export restrictions can bias the denominator and apparent edge.
The current boundary is a pinned recorded-history cohort, with no inferred global
coverage. **Owner/operator task:** retain each manual export's actual acquisition
scope, account alias, date restrictions and source/export time where known; reconcile
bounded counts against the tracker and document what cannot be checked. Future
capture engineering must record monitored-feed coverage and outages, not claim an
exhaustive market universe. See [ingestion](../INGESTION.md).

### EG02

**Acquisition support and freshness.** Alex's support answer states automated exports
are unavailable at that time; browser downloads work, while Python meets a Cloudflare
challenge. A weekly runner, recent import or new report date does not make the
source current. **Owner/operator task:** perform manual exports, record known export
metadata and retain the receipt; ask for a supported route if vendor capabilities
change. **Data maintainer task:** preserve unknown times/scope and surface freshness
limitations in the selected snapshot. No unsupported acquisition bypass is proposed.
See [workflow handoff](ODDSJAM_WORKFLOW.md) and [chronology](../reference/CHRONOLOGY.md).

### EG03

**Stable execution identity and revisions.** The two assessed exports lack a
standalone execution ID. Event IDs, repeated signatures, quote grouping and ticket
IDs cannot establish independent executable opportunities or safely collapse every
repeated row. Conflicting revisions can make current status unresolved.
**Data maintainer task:** inspect source namespaces and reconciliation audits; retain
explicit identity-resolution evidence for ambiguous records and quantify unresolved
coverage. **Future capture engineer task:** store venue execution/order/fill IDs and
append revisions with source and receipt times. Preserve legitimate repeated fills.
See [data grains](../reference/DATA_GRAINS.md) and [ingestion](../INGESTION.md).

### EG04

**Decision chronology and timezone.** `created_at` is not verified placement time;
event starts can change and timezone may be unknown. Source-date charts or timezone
conversion cannot establish forward validation or information available before a
bet. **Owner/operator task:** obtain bounded source-time semantics/timezone evidence.
**Data maintainer task:** retain raw times and provenance; use verified placement,
quote-observation and feature-availability timestamps only where supported, leaving
UTC/chronology unresolved elsewhere. Review DST and boundary-day ambiguity. See
[chronology](../reference/CHRONOLOGY.md) and [temporal tests](../../tests/test_temporal_validation.py).

### EG05

**Outcome availability and settlement chronology.** A terminal status/profit does
not tell when it became knowable or when cash settled; later corrections can leak
into retrospective evaluation and distort bankroll paths. **Data maintainer task:**
retain dated revisions and explicit outcome-availability/settlement evidence;
block chronological claims when these are missing. **Future capture engineer task:**
append settlement/correction events and reconcile a cash ledger separately from
source-calendar profit. See [economics](../reference/ECONOMICS.md).

## Operating regimes and reference inputs

### EG06

**Funding and selection-regime boundaries.** A PROD/TEST label is insufficient for
funding, and older broad TEST capture differs from newer discretionary paper
selection. Pooling them can produce misleading performance comparisons.
**Owner/operator task:** document filter/account-scoped effective periods and funding
sources, including the Core strategy's Nov 10 date-granularity transition and any
unresolved boundary rows. **Data maintainer task:** carry these into reviewed,
versioned evidence under R05; retain unknown/mixed/conflicting records and prior
snapshot classifications. See [decision regimes](DECISION_REGIMES.md).

### EG07

**Suggested, applied and accepted sizing.** Clarified paper entries follow Kelly's
recommendation, but current bankroll/multiplier settings do not prove a historical
applied policy or executable size. Exact Kelly replay and bankroll-growth attribution
can therefore be unavailable. **Owner/operator task:** supply dated bankroll/Kelly
settings and distinct suggested/accepted amounts, with verified application evidence
where available. **Data maintainer task:** use existing sizing-history linkage;
keep unknown placement, overlapping intervals and unverifiable application blocked.
Do not cap or rewrite old stakes by assumption. See [sizing history](../SIZING_HISTORY.md).

### EG08

**Historical devig/configuration inputs.** Support confirms Pinnacle-only reference
prices for the documented selection instead of Recommended Weights; it does not
recover each historical row's book weights, method, model version or filter settings.
A current default panel cannot reproduce old advertised EV.
**Owner/operator task:** retain dated settings/support evidence, exact controls and
known effective periods. **Data maintainer task:** represent unknown separately from
unset/default/unrestricted and attach reviewed configuration versions under R05.
Future capture should store quote-level calculation inputs and model versions. See
[EV/settings](../reference/EV_SETTINGS.md).

## Quotes, execution and economics

### EG09

**Displayed quote versus selected execution price.** Rounded displays, blended
levels, refreshed quotes and manual tracker entries can differ. Attributing a gap
to slippage without matched observations is unsupported. **Owner/operator task:**
retain displayed quote/time/icon/levels and selected venue price separately; identify
which price was recorded. **Future capture engineer task:** link quote versions to
order requests and accepted fills. Case F records 50.5¢/−102 execution against a
51.0¢ display; the liquidity discrepancy's cause stays unresolved. See
[Case F](case-studies/F.md).

### EG10

**Order acceptance, partial fills and cancellations.** A tracker stake or suggested
amount does not verify full acceptance at that price. Paper ROI can overstate
capacity and funded exposure can differ from intent. **Owner/operator task:** retain
venue receipts for requested/accepted/canceled amounts and prices when available.
**Future capture engineer task:** link order-state/fill/cancellation events and
unfilled remainders without treating a canceled remainder as a loss or execution.
Case D's $130 intent, $95.38 fill and $34.62 canceled remainder illustrate the
separation. See [Case D](case-studies/D.md).

### EG11

**Book depth and liquidity tags.** Tags cover a single price or aggregate contributing
blended quotes, not a full same/opposite-side book, historical depth or verified
capacity. Missing depth is not zero; liquidity-conditioned results cannot be
extended automatically to missing-liquidity bets. **Owner/operator task:** record tag
meaning and uncertainty consistently and preserve unresolved source discrepancies.
**Future capture engineer task:** store timestamped levels, sides, sizes, depth
coverage and quote expiry/refresh evidence. Do not reconstruct omitted levels from
a blended average. See [source inventory](SOURCE_FIELD_INVENTORY.md).

### EG12

**Width, staleness and market comparability.** Exported width and a favorable-looking
book do not establish matched timing, consistent units or a universal threshold.
Observed performance differences may reflect product mix or stale quotes.
**Data maintainer task:** audit raw units/taxonomy and same-time quote support by
product; keep unresolved comparisons descriptive. **Future capture engineer task:**
retain bid/ask, timestamps and stale-feed/latency flags with venue-specific semantics.
See [EV/settings](../reference/EV_SETTINGS.md).

### EG13

**Advertised EV, closing-reference EV and realized ROI.** They are distinct measures;
CLV fairness/source timing is unverified and missingness can change cohorts. Their
difference alone does not identify execution quality or a profitable edge.
**Research reviewer task:** compare matched eligible cohorts, disclose EV/CLV coverage
and investigate calibration, selection and sampling separately. **Data maintainer
task:** retain raw/exact advertised EV and reference-price provenance; never use
post-decision outcomes/CLV as default decision-time predictors. See
[EV/settings](../reference/EV_SETTINGS.md) and [economics](../reference/ECONOMICS.md).

### EG14

**Fees, currency and verified net returns.** USD is an owner-confirmed source-level
assumption; source profit lacks a verified fee/fill breakdown. Reported ROI is not
fee-verified net ROI or a reconciled bankroll ledger. **Owner/operator task:** obtain
bounded venue statements and fee/currency rules, identifying whether reported profit
already includes fees. **Data maintainer task:** reconcile that evidence before any
new net-return metric; preserve raw reported amounts and avoid deducting estimated
fees twice. See [economics](../reference/ECONOMICS.md).

### EG15

**Eligibility, payouts and ticket completeness.** Unknown flags, unsupported
products, pending/missing profit and inconsistent ticket amounts prevent trustworthy
ordinary-wager economics. Treating legs as tickets or null as zero changes the
profit/stake denominator. **Data maintainer task:** inspect exclusion/audit counts,
obtain complete ticket/settlement evidence when available and preserve unresolved
rows. Apply existing eligibility; introduce any future payout support through a
reviewed contract change, not this documentation. See
[bet-facts tests](../../tests/test_bet_facts.py) and [economics](../reference/ECONOMICS.md).

## Attribution, dependence and decision coverage

### EG16

**Market/player/venue taxonomy.** Labels alone can conflate different contracts,
products or alternate lines; standalone player/line fields may be unavailable.
Cross-market models and ladder detection can be unreliable. **Data maintainer task:**
review explicit mappings with versioned identity/contract evidence and quantify
unmapped/ambiguous coverage. **Future capture engineer task:** retain canonical
player/team/event/market IDs, exact threshold, side and settlement rules alongside
raw labels. See [canonical facts](../CANONICAL_FACTS.md).

### EG17

**Same-event correlation and ladder exposure.** Shared event/selection proxies signal
possible dependence, not a measured correlation matrix or complete ladder. Counts
of bets or inverse stake HHI are not independent samples; independent-bet simulations
can understate tail risk. **Research reviewer task:** audit event-ID coverage,
repeated selections and exposure concentration; bound unavailable dependence and
state simulation assumptions. **Future capture engineer task:** collect exact rungs,
market identities, joint outcomes and concurrent exposure for validated portfolio
models. See [statistical evidence](../reference/STATISTICAL_EVIDENCE.md).

### EG18

**Saved-filter membership and settings attribution.** The tracker label may not
record every matching filter or the settings applied. Selecting filters together is
understood by the owner as a union but is unconfirmed; overlapping memberships are
not additive portfolios. **Owner/operator task:** verify bounded vendor behavior and
retain historical settings/name evidence. **Data maintainer task:** resolve only
supported aliases/memberships using the versioned attribution process, retaining
unknown/ambiguous cases. Never classify a new export solely from current names.
See [versioned attribution](../VERSIONED_FILTER_ATTRIBUTION.md).

### EG19

**Unrecorded passes and changing selection.** Current passes are unlogged; proposed
TEST-only Passed tags do not recover historical rejected opportunities. PROD passes
remain unlogged so the owner can reconsider later. Acceptance/pass rates and
counterfactual performance of all reviewed opportunities are unavailable.
**Owner/operator task:** decide whether to adopt the low-friction TEST tag and
specify its meaning separately from paper execution. **Future capture engineer
task:** append every observed decision/revisit, including no action, within declared
coverage; reporting must keep decisions distinct from wager economics. See
[decision regimes](DECISION_REGIMES.md).

### EG20

**Discretionary reasons and counterparty hypotheses.** Manual depth/context review
and subjective confidence are not consistently structured historical fields.
Large orders do not identify sharp/recreational traders or establish why returns
changed. **Owner/operator task:** optionally record concise decision reasons and
available quote context without excessive friction. **Research reviewer task:**
treat retrospective stories and feature importance as hypotheses, test prospective
rules and avoid causal/counterparty claims unsupported by evidence. See
[model explanations](../reference/MODELS_EXPLANATIONS.md).

### EG21

**Validation and prospective generalization.** Selected historical bets, changing
regimes, multiple comparisons and missing decision chronology limit claims about
new opportunities or deployable filters. Replaying data or explaining a fitted model
does not provide independent forward evidence. **Research reviewer task:** declare
scope, candidate family, support/uncertainty and admissible validation strategy;
use the existing temporal and prospective gates. **Owner/operator task:** approve
and launch complete prospective definitions separately, recording review outcomes.
Future data expands evidence only within monitored coverage; it cannot silently
turn historical exclusions into causal improvements. See
[statistical evidence](../reference/STATISTICAL_EVIDENCE.md) and
[prospective lifecycle](../reference/PROSPECTIVE_LIFECYCLE.md).

## Reviewing the register in practice

For any proposed filter, start with its regime and eligible population, then follow
its inventory fields to the relevant EG entries. Report which limitations actually
affect that population and which evidence tasks remain open. Keep descriptive
research, prospective tests and production authorization distinct. The current
repository's preservation and blocking behavior is illustrated by the
[synthetic interpretation examples](DECISION_REGIMES.md#synthetic-interpretation-checks).
No task here silently changes a publication, source data or live scanner setting.

[Source inventory](SOURCE_FIELD_INVENTORY.md) · [Decision regimes](DECISION_REGIMES.md) ·
[Operating workflow](ODDSJAM_WORKFLOW.md) · [Documentation home](../README.md)
