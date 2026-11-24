# Private CSV staging and import preview

R09 adds a local service and CLI for reviewing a manually exported OddsJam CSV
before import. R10 retains drag/drop, dashboard progress, cancellation/retry and
explicit report activation. Staging alone never imports, builds, activates a report
or launches a study. Existing direct import and weekly runner commands still work.

## Review and confirm

Use a private directory under ignored `.local/` or `data/raw/`. Never commit raw
exports, previews or receipts. The service creates staging directories with mode
0700 and files with mode 0600; an existing staging root must already be owner-only.
Confirmation also requires a private ingestion root. Private parent directories
protect existing ingestion artifacts; this is local filesystem isolation, not a
multi-user authorization or encryption service. No hosted deployment is included.

Run from the repository root, replacing the uppercase placeholders with your local
values. Each command takes the same two separate store roots:

```bash
poetry run python scripts/stage_csv.py --staging-root .local/csv-stages --ingestion-root data/raw/ingestion stage /PRIVATE/PATH/bets.csv --namespace owner --scope full --timestamp-timezone America/Chicago --exported-at 2026-11-24T12:00:00-06:00
poetry run python scripts/stage_csv.py --staging-root .local/csv-stages --ingestion-root data/raw/ingestion preview STAGE_ID
poetry run python scripts/stage_csv.py --staging-root .local/csv-stages --ingestion-root data/raw/ingestion confirm STAGE_ID --preview-sha256 REVIEWED_PREVIEW_SHA256 --confirm-import
```

Use the returned stage ID and the exact preview digest. `full`, `partial` and
`unknown` are owner declarations, never inferred completeness. Unknown coverage
can be staged and inspected but must be restaged with an explicit declaration
before confirmation. Export time is optional and must include an offset if supplied;
unknown timestamp timezone remains unknown. Do not invent either from filenames.
Dropping or selecting a file in a future adapter must call staging only.

Inspect row count, detected/added/missing columns, raw timestamp bounds, status
counts, invalid numeric counts, duplicate rows and overlap states. Raw lexical date
bounds do not establish event coverage, a safe cursor, source freshness or actual
execution time. The original filename is validated as a simple CSV basename but
is not persisted; immutable IDs are the authoritative handles.

## Preview meanings

- **New:** no conservatively linked prior observation; not proof of a new fill.
- **Unchanged:** a matched payload or an already imported identical file.
- **Revised:** a linked record has a different payload. Settlement-related changes
  are counted separately. An older export is not automatically the current revision.
- **Unresolved:** ambiguous identity, collision or missing revision chronology;
  existing reconciliation rules preserve the uncertainty and exclude ambiguous
  economics rather than guess identity. A preview does not resolve records.

Full and partial exports both append observations. Missing rows never delete prior
history. Identical namespace/content imports reuse the immutable importer and cannot
duplicate economics. Conflicting acquisition metadata for identical imported bytes
fails. Namespace aliases isolate accounts; they must not contain secrets.

## Limits, integrity and failures

The version-1 service bounds each source to 20,000,000 bytes, 100,000 rows and 100
columns. It accepts UTF-8 (optional BOM), a `.csv` basename and the existing required
schema. Unknown columns remain literal text. NULs, malformed quoting/row widths,
duplicate headers, path separators and invalid encodings fail before staging.
Formula-like cells are counted and preserved literally, never evaluated. No sample
cells or spreadsheet download is emitted by this service; any future spreadsheet
export must separately escape formula injection. Negative numeric text may also
contribute to the conservative formula-like count.

Stage IDs bind source bytes and metadata. Reads verify the receipt, source hash and
schema; corrupt or symlinked stage files are rejected. The local trusted operator
controls these stores; hashes detect inconsistency, not malicious replacement by
someone with the same filesystem permissions. Stream input is read at most the byte
budget plus one. CSV parser resource use is also bounded by that source budget.

The preview digest includes prior import identities and manifest hashes. A changed
store requires a new preview. The importer checks those identities again under its
writer lock, preventing an intervening writer from invalidating confirmation.
Concurrent writers fail fast; wait for completion and preview again. Review stale
locks using existing ingestion guidance, never delete an active writer's lock.
Unconfirmed, stale or corrupt inputs create no accepted import. Staged files remain
available for inspection/retry. Retrying after an interrupted successful import
returns the existing receipt after a fresh preview.

## Integration and rollback

Adapters can call `stage_csv`, `preview_csv` and `confirm_import` in
`oddsjam_ev.io.staging`. Confirmation returns the existing immutable import manifest;
report generation remains a separate explicit runner action. Model fitting remains
independently opt-in under [research jobs](RESEARCH_JOBS.md). No metric, funding,
source-regime or canonical publication contract changes here.

Rollback by ceasing to use staging and retaining the prior verified report. Do not
rewrite or delete accepted history. Cleanup of unneeded private stages is an explicit
local operator action after checking that no staging/confirmation operation is active.
