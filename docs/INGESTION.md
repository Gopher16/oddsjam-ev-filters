# Historical Bet Tracker ingestion

PR 1 adds a local, CSV-first lineage layer before the existing research transform.
It does not change live filters, acquire account data, calculate new profitability
metrics, or execute notebooks. Future authorized acquisition adapters can download
a file and call `oddsjam_ev.io.ingestion.import_csv` with source metadata.

## Quick start

Run commands from the repository root on the existing branch. Install the locked
environment with `poetry install` if needed. The defaults in
`configs/data_sources.yaml` place all private artifacts beneath the already
Git-ignored `data/raw/ingestion/` directory. Use a non-secret account/source alias
for `namespace`; different accounts must use different namespaces.

```bash
poetry run python scripts/ingest_bet_tracker.py import data/raw/oddsjam-bet-tracker-03-21-2026.csv
poetry run python scripts/ingest_bet_tracker.py inspect
poetry run python scripts/ingest_bet_tracker.py snapshot
poetry run python scripts/ingest_bet_tracker.py rebuild
```

The last two commands both replay the verified archive and print the published
snapshot directory and manifest. Rebuild checks logical table equality if that
snapshot already exists. No latest pointer is silently updated. Reimporting the
same bytes under another filename adds an attempt receipt but no observations or
revisions. The original and repeated filenames remain in private provenance.

Global options precede the subcommand:

```bash
poetry run python scripts/ingest_bet_tracker.py --config configs/data_sources.yaml inspect
```

Only provide `--exported-at`, `--timestamp-timezone`, or `--scope` on import when
you know them. Export time must include an offset; timezone must be an IANA name.
Do not derive export time from the filename or guess a timezone to enable matching.
Different metadata for already imported bytes fails rather than rewriting history.
Use an independently reviewed store to replay corrected acquisition metadata;
retain the original store and record why its interpretation was superseded.

```bash
# A system knowledge cutoff, not an assertion of what was known when bets occurred:
poetry run python scripts/ingest_bet_tracker.py snapshot --as-of 2026-10-22T16:00:00Z
```

Command failures return a nonzero exit code with an actionable message. Malformed
sources do not publish accepted imports. Failed input bytes remain at their
original location; successful and repeated attempts get receipts. Inspect/rebuild
verify accepted sources and fail on missing or corrupted archives.

## Storage and reproducibility

```text
data/raw/ingestion/
  objects/<source-sha256>.csv             original bytes, including encoding/BOM
  imports/<import-id>/manifest.json       immutable accepted source metadata
  attempts/<attempt-id>.json              successful/repeated acquisition receipts
  snapshots/<snapshot-id>/
    manifest.json                        pinned imports, decisions, policy, hashes
    observations.parquet                 every source row plus reconciliation audit
    current.parquet                      resolved latest state, one row per record
```

Imports preserve original field names, values, ordering, empty strings, quoted
newlines, and literal `NA` strings. All source columns remain in observation
Parquet. The `_ingest_` prefix is reserved for lineage metadata. Row numbers are
one-based logical data rows (a multiline CSV record is one row).

Manifests include source byte hash/size, filename, UTC ingestion time, declared
export time, namespace, export scope, timestamp policy, parser/schema version,
configuration hash, code version, row count, raw timestamp bounds, schema drift,
invalid numeric counts, duplicate counts, and status counts. Raw bounds are
lexical diagnostics, not a safe acquisition watermark. Optional field loss and
extra columns are surfaced without dropping data. Required-column loss, repeated
headers, malformed rows, unsupported encoding, reserved headers, and empty data
fail. CSV is UTF-8 with optional BOM; XLSX and network acquisition are deferred.

Snapshot IDs pin import manifests, explicit identity resolutions, execution
policy, reconciliation version/code hash, and pandas version. Output hashes detect
corruption. Determinism means logical table/ID equality under the pinned algorithm,
not identical operational timestamps or Parquet bytes across library versions.
Keep the code and Poetry lock version alongside archived data for later replay.

Publication uses staging directories, exclusive writes, atomic local renames, and
a fail-fast single-writer lock. Interrupted staging is ignored. A crash can leave
an unreferenced content object or `.writer.lock`; inspect running processes before
manually removing a stale lock. Atomic publication is not a backup system or a
cross-filesystem/object-storage transaction. Keep private backups separately.

## Identity, repeated observations, and updates

No standalone execution ID exists in the two assessed OddsJam exports. `game_id`
is an event identifier and `parlay_id` can identify multiple leg rows. Neither is
used alone as a bet ID. Existing `opportunity_id` and notebook execution-row IDs
remain research grouping proxies and are not reused for ingestion identity.

Automatic candidates require sportsbook, event identity (`game_id`, or event name
when absent), market, selection, odds, positive finite stake, bet type, and a
parseable created timestamp. Parlay ID participates when available. Numeric fields
are not rounded and original strings remain unchanged. Missing evidence produces
an unresolved observation instead of shortening the key silently.

- A source hash identifies bytes, not a bet.
- An observation ID identifies a source row within one namespace/import.
- A payload fingerprint identifies full field content, not execution identity.
- A reconciled record ID identifies an accepted candidate or reviewed link.
- Opportunity grouping is a separate research problem, deferred to PR 2.

Identical rows within a file retain their multiplicity and are flagged as identity
collisions. Different timestamps can represent separate executions in a single
source; they are not automatically dropped. Across exports, uncertain timezone
changes, possible stake/odds corrections, or ambiguous candidate multiplicity
quarantine the affected groups. This intentionally sacrifices coverage rather
than silently adding or merging uncertain executions.

For confirmed timezone metadata, timestamps are normalized to UTC for matching.
Unknown timezone values are never shifted; identical raw timestamps can match,
but different timestamps for overlapping candidates remain unresolved. The local
February/March exports show five/six-hour shifts; the origin of those shifts is
not confirmed. The importer therefore defaults to `unknown` for both.

Accepted links retain every observation. Changed payloads require known distinct
export timestamps; ingestion order never selects the winning settlement. Same-time
conflicting payloads are unresolved. Outcomes, CLV, tags, and filter attribution
are not identity fields. The audit includes dispositions, revision numbers,
changed field names, current-state selection, and exclusion reasons. Summary
counts reconcile every observation. Later exports omitting a bet do not delete it.
Pending rows and their reported profit remain intact; profit presence is not proof
of settlement. No economics or parlay payout is reconstructed.

## Explicit reviewed resolutions

Use a private JSON list to map ambiguous observation IDs from the audit to record
IDs. Assign different IDs for confirmed distinct bets; use the same ID for
confirmed repeated observations. Include all observations that should share a
reviewed identity. Export chronology is still required for conflicting payloads.

```json
[
  {
    "observation_id": "COPY_FROM_OBSERVATIONS_PARQUET",
    "record_id": "reviewed-bet-001",
    "reason": "Explain independent evidence for this identity decision",
    "decided_at": "2026-10-22T16:00:00Z"
  }
]
```

```bash
poetry run python scripts/ingest_bet_tracker.py snapshot --decisions data/raw/identity-resolutions.json
```

Resolution timestamps must not predate the corresponding import. Future decisions
are excluded from an earlier `--as-of` snapshot. Resolutions are embedded in each
snapshot manifest; changing a resolution creates a different snapshot. No mutable
spreadsheet is an implicit dependency. Cross-namespace links are rejected.

## Execution evidence and historical filters

Classification is explicitly a **user-supplied assumption dated October 22, 2026**,
not independently verified execution evidence:

- Before October 22, only the exact attribution
  `PROD - Exchanges: Core Exchange Conservative (Low Risk, Slow Build)` is real
  money. Other nonempty historical filter attribution is paper.
- Missing attribution, or attribution mixing that real filter with other labels,
  is unknown. Filter names containing commas are not split indiscriminately.
- After October 22, recognizable PROD designations are real money and TEST
  designations are paper; mixed PROD/TEST or unrecognized labels are unknown.
- On October 22 itself, retain the historical exact-name rule; otherwise classify
  unknown because the intraday policy/timezone boundary is not established.
- Unparseable dates are unknown. These assumptions do not verify fillability,
  liquidity, fees, or executable odds. Paper verification is unavailable.

The original filter string and classification evidence are preserved on every
observation. `prod_filter` remains the existing research flag; it is not a funded
execution indicator. The dated filter registry provides contextual settings and
aliases, never retroactively inferred configurations. PROD/TEST names are not
opportunity identity or the principal discovery unit.

## Connecting to research notebooks

The legacy command and default March research snapshot remain unchanged:

```bash
poetry run python scripts/preprocess_bet_tracker.py --config configs/preprocess.yaml
```

The script now delegates to `oddsjam_ev.io.preprocess.run_preprocessing`; metric,
settlement inference, cutoff, and optional dedupe behavior remain the same.

For new ingestion snapshots, first resolve timestamp provenance. The bridge
requires a single confirmed source timezone matching
`timestamps.assume_tz_for_naive` in the preprocessing configuration. It refuses
unknown or mixed metadata; importing and auditing unknown-timezone sources is
still supported. Do not bypass this gate for the two existing exports without
additional evidence.

```bash
poetry run python scripts/preprocess_bet_tracker.py --config configs/preprocess.yaml --ingestion-snapshot data/raw/ingestion/snapshots/SNAPSHOT_ID
```

Outputs publish atomically under a separate
`data/processed/ingestion-RESEARCH_ID/` directory with existing dated Parquet/QA
filenames and `research-manifest.json`. The research manifest pins input lineage,
configuration, code hash, exclusion counts, and output hashes. Existing bundles
are never overwritten. Source strings are converted using the legacy CSV reader's
inference/NA behavior only in the derived research view; the source observations
remain lossless. This distinction preserves existing research missingness semantics.

To opt both notebooks into a reviewed research bundle without modifying cells:

```bash
export ODDSJAM_RESEARCH_MANIFEST="/absolute/path/to/data/processed/ingestion-RESEARCH_ID/research-manifest.json"
```

Start the notebook kernel with that environment. The shared resolver verifies the
bundle, selects its dated outputs and sidecars, and exposes `provenance_path`.
Unset the variable to return to existing snapshot selection. No notebook has been
modified or executed for PR 1. Notebook 01 remains readiness/QA; Notebook 02 remains
descriptive evaluation. Additional lineage presentation cells can be a later,
separately reviewed notebook change.

System as-of time limits imported knowledge and identity decisions; it does not
recover historical decision-time features. `created_at` is source metadata, not
proof of execution time. No source settlement/update timestamp is available.
Later CLV and settlement revisions must not be used as pre-bet predictors. PR 2
can build canonical economics/opportunities from pinned snapshots; weekly jobs
can call these Python functions without running notebooks.

## Validation and privacy

`tests/test_ingestion.py` uses only small synthetic sources and temporary stores.
It covers idempotency, source preservation, malformed/schema inputs, collisions,
chronology, timestamp shifts, namespaces, reviewed resolutions, as-of selection,
archive/output integrity, interrupted staging, lock contention, execution-policy
boundaries, preprocessing equivalence, and explicit research selection. Existing
research tests remain applicable. CI already runs Ruff, formatting, lock checks,
and pytest; no private data or new network capability is required.

Never commit store contents, resolution files, private reports, raw exports, or
credentials. Use the default ignored storage locations. Alternate configured
storage roots require equivalent privacy controls. Existing tracked notebooks can
contain historical outputs; this PR does not refresh them or publish new data.

## Optional canonical facts

After manual ingestion, the [canonical builder](CANONICAL_FACTS.md) can consume an
explicit published snapshot without rerunning ingestion or acquisition. It keeps
unknown timestamps explicit, separates ticket legs, and audits economic eligibility.
It does not bypass the legacy preprocessing timezone gate or update notebook
defaults. Canonical outputs remain private under `data/processed/canonical/`.
