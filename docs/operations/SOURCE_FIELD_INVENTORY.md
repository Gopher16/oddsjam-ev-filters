# Bet Tracker source fields and provenance

Reviewed 2026-11-23 against the frozen owner roadmap (R&D §3 limitations and §6.2
future capture), [operating workflow](ODDSJAM_WORKFLOW.md) and current repository
contracts. This is a semantic inventory, not a fresh audit of a private export.
Optional columns below are supported or inspected concepts; their inclusion here
does not assert that any particular CSV contains them or that values are complete.
Use the snapshot's feature catalogue and audit counts to establish actual coverage.

## Read the evidence at its own grain

Original CSV bytes and raw field strings are source observations. Ingestion resolves
wager candidates and revisions; canonical facts separately qualify economics.
Ticket legs, filter memberships, opportunity proxies and event groups are distinct
views, not additional independent wagers. A hash proves which bytes were imported,
not that all bets or opportunities were exported. An imported row is not independent
verification of a venue fill. See [data grains](../reference/DATA_GRAINS.md),
[ingestion](../INGESTION.md) and [canonical facts](../CANONICAL_FACTS.md).

| Provenance layer | Evidence to retain | Safe interpretation / remaining gap |
| --- | --- | --- |
| Acquisition receipt | Non-secret namespace, ingestion time, original file hash/size and declared export time, scope and timezone when known | Receipt is local knowledge time; unknown acquisition metadata stays unknown. [EG01](EVIDENCE_GAPS.md#eg01), [EG02](EVIDENCE_GAPS.md#eg02) |
| Immutable source object and import manifest | Exact archived bytes, parser/schema/config/code versions, source row position, quality counts | Repeated identical bytes do not create new wagers; archive integrity is not export completeness. [EG01](EVIDENCE_GAPS.md#eg01), [EG03](EVIDENCE_GAPS.md#eg03) |
| Observation/revision audit | `_ingest_` IDs, current/disposition/reason metadata and linked source records | Preserve unresolved conflicts; later ingestion alone does not prove later settlement state. [EG03](EVIDENCE_GAPS.md#eg03) |
| Pinned snapshot/publication | Snapshot ID, knowledge cutoff, input/policy hashes and build versions | Reproducible processing does not recover missing pre-decision knowledge; build time is not source freshness. [EG02](EVIDENCE_GAPS.md#eg02), [EG04](EVIDENCE_GAPS.md#eg04) |
| Interpretation evidence | Owner narrative, dated support answer, settings evidence and source references | Narrative context is separate from per-wager classification and verified application. [Decision regimes](DECISION_REGIMES.md); [EG07](EVIDENCE_GAPS.md#eg07), [EG08](EVIDENCE_GAPS.md#eg08) |

## Field inventory

Names in backticks are source fields used or inspected by current code, rather than
promised vendor API fields. The [feature catalogue](../../src/oddsjam_ev/analysis/feature_catalog.py)
reports presence, nonempty counts, denominator and temporal role. Its current
`decision_time_verified` values are false: nonempty is not verified decision-time
availability. Canonical metadata/derived values must remain distinguishable from
source columns. Missing, invalid, conflicting and unavailable are not numeric zero.

| Source field or concept | Current interpretation and bounded use | Evidence boundary / gap |
| --- | --- | --- |
| `sportsbook`, `sport`, `league` | Recorded venue and category labels; product mappings require the declared configuration | Brand/category labels do not verify historical product, account access or consistent taxonomy. [EG16](EVIDENCE_GAPS.md#eg16) |
| `game_id`, `event_name`, `event_start_date` | Event identity context; scheduled event-start value, with raw timestamp retained | `game_id` is not a bet ID. Start time may be revised; timezone/semantics and event-link coverage matter. [EG03](EVIDENCE_GAPS.md#eg03), [EG04](EVIDENCE_GAPS.md#eg04), [EG17](EVIDENCE_GAPS.md#eg17) |
| `market_name`, `bet_name`; optional `player`, `line` | Raw market/selection labels; standalone player/line fields remain unavailable unless observed | Do not invent a player ID, alternate-line ladder or normalized contract from ambiguous text. [EG16](EVIDENCE_GAPS.md#eg16), [EG17](EVIDENCE_GAPS.md#eg17) |
| `created_at`; optional `updated_at`, `settled_at` | Source timestamps; update is a revision role and settlement is an outcome role | Created time is not automatically placement. A timezone conversion alone proves no decision/settlement chronology. [EG04](EVIDENCE_GAPS.md#eg04), [EG05](EVIDENCE_GAPS.md#eg05) |
| `odds` | Recorded American odds, separately parsed; used for supported payout consistency and quote diagnostics | May be a manually recorded execution price rather than the earlier displayed quote; no quote-depth or acceptance guarantee. [EG09](EVIDENCE_GAPS.md#eg09), [EG10](EVIDENCE_GAPS.md#eg10) |
| `percentage` | Source-advertised EV in percentage points under the declared contract; exact/raw representations retained where available | Not realized ROI or true probability. Original historical formula inputs and availability remain unverified. [EG08](EVIDENCE_GAPS.md#eg08), [EG13](EVIDENCE_GAPS.md#eg13) |
| `clv`, `clv_source` | Closing-reference odds/source context; legacy modeled EV uses its own CLV-derived calculation | Fairness, observation time and completeness are not established by presence. Never a default pre-decision predictor. [EG13](EVIDENCE_GAPS.md#eg13) |
| `market_width` | Recorded source width; descriptive screening context | Do not infer common units, matched timestamps or a universal sharpness threshold across products. [EG12](EVIDENCE_GAPS.md#eg12) |
| `liquidity`, `tags` | Raw liquidity is untrusted; owner liquidity tags describe single-price or aggregated contributing blended quotes | Tags do not retain full same/opposite-side depth, future fillability or actual accepted size. Missing depth is not zero. [EG09](EVIDENCE_GAPS.md#eg09), [EG11](EVIDENCE_GAPS.md#eg11) |
| `stake` | Reported stake, separately parsed as USD under the owner's source-level currency assumption | Paper stake follows Kelly recommendation; this is not accepted size or a historical bankroll/Kelly replay. [EG07](EVIDENCE_GAPS.md#eg07), [EG10](EVIDENCE_GAPS.md#eg10) |
| `bet_profit`, `status`, `potential_payout` | Source-reported profit/status; potential payout is potential, not observed settlement | Require explicit economic eligibility and same-cohort denominators; missing profit is not zero and fees remain unverified. [EG05](EVIDENCE_GAPS.md#eg05), [EG14](EVIDENCE_GAPS.md#eg14), [EG15](EVIDENCE_GAPS.md#eg15) |
| `bet_type`, `parlay_id` | Type and possible explicit ticket linkage; ticket economics are not repeated per leg | One leg or a shared ID does not establish complete ticket structure or consistent settlement. [EG15](EVIDENCE_GAPS.md#eg15) |
| `is_live_bet`, `is_free_bet`, `is_odds_boost` | Source flags used in ordinary pre-match economic eligibility | Unknown flags do not qualify as confirmed false; explicit flags still do not prove actual funding or feature availability. [EG06](EVIDENCE_GAPS.md#eg06), [EG15](EVIDENCE_GAPS.md#eg15) |
| `saved_filter_names` | Recorded filter label context with separate versioned attribution/membership resolution | Current names do not establish historical rules, all matching filters, funding or independent opportunities. [EG06](EVIDENCE_GAPS.md#eg06), [EG18](EVIDENCE_GAPS.md#eg18) |
| Decision reasons, reviewed passes and subjective confidence | Owner narrative/case examples; a proposed TEST-only pass tag is separate from an execution | No complete historical decision/pass denominator or consistently recorded decision rating is established. [EG19](EVIDENCE_GAPS.md#eg19), [EG20](EVIDENCE_GAPS.md#eg20) |

## Context the export does not reconstruct

A CSV alone does not establish the complete surfaced-opportunity universe, each
quote's lifetime, full book depth, accepted/rejected orders, partial fills, canceled
remainders, fee ledger, historical devig configuration, applied sizing policy or
counterparty identity. [Gaps EG01–EG21](EVIDENCE_GAPS.md) specify the consequences
and evidence tasks rather than filling those facts by assumption. Future capture
must state its monitored feeds, accounts, products and periods; it cannot claim a
complete global market universe or repair history that was never observed.

For safe interpretation use [economics](../reference/ECONOMICS.md),
[EV/settings](../reference/EV_SETTINGS.md), [chronology](../reference/CHRONOLOGY.md)
and [lineage/reproducibility](../reference/LINEAGE_REPRODUCIBILITY.md). These are
existing analytical contracts; this inventory changes neither schemas nor policy.

[Operating workflow](ODDSJAM_WORKFLOW.md) · [Decision regimes](DECISION_REGIMES.md) ·
[Evidence gaps](EVIDENCE_GAPS.md) · [Documentation home](../README.md)
