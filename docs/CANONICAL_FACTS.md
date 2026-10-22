# Canonical bets, opportunities, and economics

Canonical bundles are opt-in research artifacts built from a pinned PR 1 ingestion
snapshot. They do not acquire data, place bets, change filters, update the legacy
snapshot resolver, or execute notebooks. The acquisition draft is not a dependency.

## Build and inspect

Run from the repository root with the existing Poetry environment:

```bash
poetry run python scripts/build_canonical_facts.py \
  --snapshot data/raw/ingestion/snapshots/861924428766080b4468adad998805037901532c68f8bdd041e1aa722994c7d1
```

The snapshot must already exist locally. No automatic latest selection or ingestion
reconciliation takes place. Optional `--config`, `--registry`, and `--output-root`
arguments select explicit alternate inputs/destinations. The default outputs are
private, Git-ignored `data/processed/canonical/<bundle-id>/` directories.

The command prints the bundle path and reconciliation counts. Read its manifest
for exact inputs, policies, code hashes, versions, exclusions, and table counts.
The Python reader verifies output hashes before returning tables:

```python
from pathlib import Path
from oddsjam_ev.io.canonical import load_canonical_bundle

tables = load_canonical_bundle(Path("data/processed/canonical/BUNDLE_ID"))
bets = tables["bet_facts"]
reported_roi_population = bets.loc[bets["reported_roi_eligible"]]
```

Use `economic_summary` for precomputed reported-return cohorts. Do not sum the
source features, source audit, ticket legs, opportunities, or attribution bridge
as a wager portfolio. Actual source records remain recoverable from ingestion.

## Contract version 1

| Table | Grain and contract |
| --- | --- |
| `bet_facts` | One current non-ticket source candidate, or one namespace/parlay-ID ticket candidate. These are not all economically eligible wagers. |
| `bet_legs` | Current and unresolved explicit ticket leg observations. No ticket stake, profit, or payout columns are carried here. |
| `source_audit` | Every ingestion observation, including unresolved and historical rows, with canonical disposition and linkage where known. |
| `source_features` | Every current ingestion source record, plus its canonical bet ID. Original source field values remain strings; monetary fields here are source-grain audit values, not additive wager facts. |
| `opportunity_facts` | Distinct labelled selection/quote proxies derived from current logged source records. No stake, profit, ROI, or selected representative execution. |
| `bet_opportunity_links` | One link per current source observation. A ticket can link to several leg opportunities. |
| `filter_attribution` | Unique resolved bet/filter memberships, plus explicit missing/unresolved label diagnostics. No economic fields. |
| `feature_catalog` | Actual current-source column availability, denominator, meaning, and temporal limitations. |
| `economic_summary` | Matching eligible profit/stake aggregates by product and assumed execution mode. This is reported ROI, not fee-verified net ROI. |

Every table carries ingestion snapshot ID and canonical contract version. Bet IDs
for singles reuse ingestion record identities. Ticket IDs hash namespace plus
source parlay ID. Source observation IDs link back to exact import/row provenance.
Namespaced ticket IDs do not assert the source identifier is globally unique or
stable across providers/accounts. Conflicting tickets remain ineligible.

All source observations reconcile to `current_single`, `current_ticket_leg`,
`unresolved_source`, or `historical_observation`. Revisions do not create extra
current wagers. Unresolved singles remain in the audit, rather than being promoted
into accepted economic facts. Unresolved explicit legs appear in `bet_legs` and
flag their ticket. Ticket grouping never silently removes an ambiguous leg.

## Wagers, explicit tickets, and candidate legs

A nonempty `parlay_id` is explicit ticket evidence even if the source bet type is
inconsistent. Repeated ticket amounts are recorded once only when all current
legs agree numerically. Missing or conflicting ticket amounts remain null; no
first-row, maximum, or sum rule resolves them. Mixed venue/execution evidence is
not replaced with a guessed value.

Individual statuses/odds belong to legs. They are not used to determine ticket
status or combine odds. All ticket candidates have unverified completeness and
settlement eligibility; even one exported leg or all terminal leg statuses does
not establish a complete settled ticket. Current and unresolved leg counts are
reported separately. Historical superseded observations are source audit history,
not extra legs in the current ticket.

User clarification: Pick'em plays can be logged as individual bets before being
assembled into a real ticket. Therefore `bet_type=positive_ev` is not sufficient
to establish standalone-wager economics. Exact configured Pick'em products are
ineligible for standalone ROI regardless of their row label. Dated Pick'em filter
IDs/aliases provide additional negative eligibility evidence: they cannot make a
candidate leg eligible or establish payout mechanics. No tickets are reconstructed
from timestamps, shared events, nearby rows, or filter names.

`configs/canonical_facts.yaml` declares exact product mappings. These mappings
are research policy, not independently verified platform mechanics. Any unlisted
product remains `unknown`, including venues that may support multiple products.
The legacy exchange/sportsbook inference remains untouched; it is not used as a
fallback for canonical payout classification.

## USD, reported profit, and fee uncertainty

USD is the user's confirmed source-level currency assumption, recorded as
`user_confirmed_2026-10-22`. Source amounts are parsed into separate reported USD
columns without rounding or modifying the raw strings. No conversion is applied.

OddsJam's `bet_profit` is retained as reported profit. There is no explicit source
fee/fill breakdown or fee-verification field. `fee_inclusion` remains `unverified`
and `fee_verified_net_roi_eligible` is false for every record. No estimated fees
are deducted, avoiding a potential double charge.

`reported_roi_eligible` permits supported ordinary pre-match single candidates on
configured fixed-odds or exchange products with finite positive stake, finite
profit, nonzero finite odds, supported terminal status, and explicit false values
for free-bet, odds-boost, and live-bet flags. Missing/unknown flags do not establish
ordinary wagers. Middle/arbitrage/unknown types remain unsupported. Execution mode
is retained as an independent evidence dimension; unknown execution is never
silently described as funded. Economic summaries separate assumed real, paper,
and unknown cohorts.

For fixed-odds singles, modeled profit is a consistency check using the existing
American-odds multiplier:

- Won: stake × profit multiplier.
- Lost: −stake.
- Half won: half the modeled winning profit.
- Half lost: −half the stake.

The default mismatch tolerance is USD 0.01. A mismatch excludes the row from the
reported ROI cohort but never overwrites reported profit. No model is applied to
exchange returns: they remain source-reported diagnostics with unknown fee status.
No actual payout is inferred from `potential_payout`; it remains potential.

Pending/open/unknown settlement, refunds/cancellations, unlinked Pick'em plays,
explicit tickets, unknown products, promotions, invalid amounts, and unsupported
bet types remain visible with explicit exclusion reasons. Refund/cancellation
amounts are preserved but excluded from the new denominator by policy. Legacy
notebook metric definitions are unchanged.

Reported ROI equals the sum of eligible reported profit divided by the sum of
stake for those exact same eligible candidates. Excluded amounts never enter
either side of that ratio. Empty eligible populations have null totals and ROI,
not a false zero result. Do not call these estimates verified bankroll returns.

## Opportunities and filter attribution

Selection proxies use namespace, event identity, raw market label, and raw bet
selection. When game ID is missing, event name requires a raw event-start value;
otherwise an observation-specific fallback prevents unrelated rows collapsing.
Quote proxies add source venue, odds, created timestamp, and timestamp timezone.
Missing quote identity evidence also uses an observation-specific fallback.
These are grouping proxies, not confirmed scanner quotes or independent samples.

Different executions may link to the same proxy; their wagers are preserved.
Across prices/books, selection-group membership does not combine their economic
facts. Only invariant descriptive attributes are retained on shared proxies;
conflicting metadata stays in source features rather than choosing an execution.
The denominator is logged source records, not all available market opportunities.

Filter resolution gives precedence to exact names/aliases, preserving commas
inside names. Multiple memberships resolve only when the entire string has one
unambiguous segmentation using known names. Unknown/ambiguous labels are retained
as diagnostics. Repeated leg labels or aliases do not duplicate bet/filter links.

Registry validation rejects duplicate IDs, conflicting aliases, missing clone
references, and clone cycles. The Onyx clone reference now targets the already
committed `test_dk_pick6_player_props` ID; stable IDs were not renamed. Registry
snapshot date is provenance, not proof its settings applied to an earlier bet.
Canonical attribution never supplies historical UI settings or execution funding.

## Time, leakage, and feature availability

Unknown source timezone stays unknown: raw timestamps remain available and UTC
values are null. A confirmed timezone or explicit offset allows UTC representation,
but does not establish the timestamp is an execution or quote-observation time.
The builder does not bypass PR 1's preprocessing timezone gate or change its
reconciliation precedence. Ticket timestamps require consensus; no earliest-leg
placement time is invented.

The ingestion manifest, its system as-of cutoff, source metadata, and resolution
history are pinned in the canonical specification. A canonical build only reads
that snapshot; later imports are not consulted. This prevents later snapshots
from silently replacing earlier observed state. It cannot recover pre-bet knowledge
that was never recorded. CLV and settlement outcomes are never declared safe
pre-bet predictors. All catalog entries have decision-time verification false.

`percentage` is catalogued as source-advertised EV percent, separate from the
legacy CLV-derived EV calculation and realized ROI. Raw liquidity remains missing
or untrusted; tags do not verify executable capacity. Player IDs, standalone line
values, settlement/update times, and a verified market taxonomy are not invented.

## Publication and research compatibility

Bundle identity hashes its complete source manifest, economic policy, registry,
transformation code hashes, pandas version, and PyArrow version. Tables and counts
have output hashes. Rebuilding the same specification checks logical table equality
and reuses the bundle. Library/code/policy changes produce a different bundle ID.
Use the pinned code/lock version for reproducible historical replay.

Writes use a single-writer guard, staging directory, and atomic publication.
Failed builds leave no published partial bundle. Historical bundles are never
overwritten, and no latest pointer or research default is updated.

Notebook 01 remains legacy QA/readiness; Notebook 02 remains legacy descriptive
filter evaluation. No cells, IDs, metadata, outputs, or historical metric functions
were modified or executed. Future notebook integration must explicitly opt into
canonical grains and explain exclusions, rather than replace its input silently.
The existing row-bootstrap confidence intervals are not made group-aware by this
change. Opportunity discovery, statistical promotion, fee/payout inference,
acquisition, scheduling, and dashboard changes remain outside this PR.

## Validation

Synthetic tests cover source accounting, repeated ticket amounts, contradictory
leg statuses, unresolved members, one-leg incompleteness, missing money, explicit
settlement formulas, exchange fee uncertainty, Pick'em exclusions, literal commas,
registry errors, proxy fallbacks, matching ROI populations, empty inputs,
as-of settlement separation, output corruption, atomic failure, and replay.
Tests require no private data, network, acquisition branch, or notebook execution.

The October 22 ingestion snapshot is the private-data validation input. Record
counts and known limitations may be shared; raw records and monetary reports
remain under ignored data paths. `.local/` is independently ignored on this branch;
no credential files are read or needed by canonical processing.

### October snapshot validation evidence

The final implementation was built and replayed against ingestion snapshot
`861924428766080b4468adad998805037901532c68f8bdd041e1aa722994c7d1`.

| Result | Count |
| --- | ---: |
| Source audit observations | 106,246 |
| Current source records / opportunity links | 106,168 |
| Non-ticket wager candidates | 61,598 |
| Explicit ticket candidates | 9,037 |
| Total canonical bet candidates | 70,635 |
| Current plus unresolved ticket legs | 44,615 |
| Tickets affected by unresolved observations | 42 |
| Reported-ROI eligible single candidates | 53,853 |
| Fee-verified net-ROI eligible candidates | 0 |
| Opportunity proxies (not verified unique quotes) | 106,111 |

All nine tables reproduced with logical equality. Original raw files, ingestion
history, root-level processed Parquet datasets, and notebook hashes remained
unchanged. The 138-test suite passed with pandas FutureWarnings treated as errors;
Poetry lock validation, Ruff lint/format checks, and `git diff --check` passed.

All 70,635 bet candidates have unknown source timezone and null canonical UTC
creation time in this input. No complete temporal validation or decision-time
feature availability is claimed. Eligibility is reported-return arithmetic and
product policy, not proof of funded execution, net fees, or profitability.
