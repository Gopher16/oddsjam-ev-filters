# Sizing-history provenance

This optional contract records historical bankroll references and fractional Kelly
settings without altering recorded portfolio ROI or assuming they governed every
wager. Account history stays local/ignored. No account-specific amount or screenshot
is included in Git. The safe template is
`configs/examples/sizing_history.example.json`; its unknown settings are placeholders,
not evidence about any account.

## Policy history

Copy the template into an ignored local directory such as
`data/processed/private_sizing_history.json`. Supply only supported evidence:

- Stable policy ID and ingestion namespace. Namespace prevents settings from one
  account/source being linked to another.
- `effective_from` inclusive and `effective_to` exclusive, with aware ISO offsets,
  an IANA timezone, and `effective_interval_verified`. Adjacent intervals do not
  overlap. An indefinite known interval requires `open_ended: true`; an unknown
  end must not be silently interpreted as indefinite. Undated observations have
  both dates null and remain unlinked. Unknown timezone is allowed for unverified
  observations; a verified interval requires a known timezone and matching offsets.
- `recorded_at` and a history `knowledge_as_of` cutoff distinguish when evidence was
  recorded from when settings reportedly applied. No receipt/export timestamp is
  substituted for wager placement.
- Nullable exact-decimal strings for `bankroll_amount` and `kelly_multiplier`.
  Supported currency is USD; bankroll must be positive when known, and this version
  supports fractional multipliers from 0 through 1. This is a schema range, not a
  recommendation. Missing values remain null, not zero or today's setting.
- Evidence type (`user_confirmed`, `contemporaneous_record`,
  `reconstructed_assumption`, `unknown`, `planned`), confidence
  (`confirmed`, `assumed`, `unknown`) and source reference. Only confirmed,
  user-confirmed/contemporaneous records with verified intervals can match settings.
- Optional product/funding scope; null means all populations in that namespace.
  Constraints are separately evidenced notes with `kind` (cap/liquidity/rounding/
  override), description and source reference. They are recorded, not modeled.

The account screenshot and a general recollection of half Kelly do not establish
historical interval boundaries. An intention to move to quarter Kelly is not an
already-effective setting. No bankroll is reconstructed from stakes or cumulative
profit; cash flows and other exposure may be absent.

## Independent wager evidence

Optional `wager_evidence` records carry a unique evidence ID, canonical `bet_id`,
recorded-at time and source reference, plus:

- Explicit `placement_at` and `placement_verified`. Canonical `created_at_utc` is
  still timing-unverified and is deliberately not a fallback.
- Optional `applied_policy_id` and `application_verified`. A unique settings-interval
  match alone never populates an applied policy. Verified application requires
  placement evidence and a unique matching, trusted interval. Contradictory claims
  and overlapping intervals are quarantined rather than resolved by record order.
- Optional decimal `suggested_stake`, `accepted_stake`, `accepted_verified` and
  deviation reason. Suggested amounts are source evidence, not a Kelly calculation.
  Accepted-fill claims require real-money classification; paper/unknown wagers cannot
  become accepted fills. The canonical reported stake remains unchanged.

Repeated evidence IDs are rejected. Multiple distinct verified placement times or
conflicting application/stake evidence remain visible as conflicts. Equivalent
numeric stake strings are not contradictions. Evidence for unknown wager IDs is
rejected rather than silently discarded. Settings scope does not infer funding.

## Explicit publication

With no dated history, publish an unknown-coverage audit:

```sh
poetry run python scripts/build_sizing_history.py \
  --canonical <explicit-canonical-bundle> \
  --knowledge-as-of <aware-knowledge-cutoff>
```

With a private ledger:

```sh
poetry run python scripts/build_sizing_history.py \
  --canonical <explicit-canonical-bundle> \
  --history data/processed/private_sizing_history.json
```

Artifacts are private under `data/processed/sizing_history/<publication-id>/`:
`history.json`, `linkage.parquet`, `audit.json` and an integrity manifest. The manifest
pins canonical/snapshot identity, exact input-file and predecessor hashes, logical
history ID, code and runtime versions. Loaders verify bytes; replay reproduces the
same publication. No canonical, atlas, default pipeline or dashboard output is
rewritten, and no acquisition/scheduling is introduced.

For a revision, set `previous_history_id` to the predecessor manifest's logical
history ID and pass `--previous <prior-publication-directory>`. Preserve both prior
record arrays exactly and append new policies/evidence; knowledge cutoff cannot
move backward. Prior evidence cannot be edited, reordered or deleted. Corrections
are new evidence, not in-place overwrites; overlapping/contradictory records stay
quarantined. Explicit supersession/retraction semantics are a future contract change.

## Audit interpretation and limits

There is one linkage row per canonical candidate, including tickets, paper records
and economically ineligible products. Do not expand these rows by overlapping saved
filter memberships and sum them as independent wagers. The audit reports unknown,
matched, ambiguous and verified-application counts, overlapping trusted policy pairs,
stake conflicts and undated/untrusted policies. Half-open intervals, scope and
verified timestamps determine matches deterministically; latest-record-wins is never
used. Partial settings can match while bankroll or Kelly stays unknown.

Policy fields attached to an interval match describe that setting, not proof of
execution-level application. Future bankroll-relative diagnostics must require
verified application and known bankroll. This stage computes no normalized returns,
Kelly stakes, bankroll growth or drawdown. Every row reports `sizing_replay_blocked`
with missing evidence for settlement order, exposure, cash flows, source algorithm,
fees and fillability. Evidence recorded here is not sufficient for exact replay.

The next research change can add weighted/equal-stake and concentration diagnostics
on reconciled eligible cohorts. Unknown regimes must remain a separate population;
current settings must never fill those gaps.
