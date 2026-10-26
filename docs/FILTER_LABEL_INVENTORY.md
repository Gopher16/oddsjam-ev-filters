# Historical source-label inventory

Inventory exact historical `saved_filter_names` values before proposing aliases.
This command requires an explicit ingestion snapshot and the canonical bundle
that pins that exact snapshot. Both inputs are verified before analysis.

```sh
poetry run python scripts/inventory_filter_labels.py \
  --snapshot data/raw/ingestion/snapshots/<snapshot-id> \
  --canonical data/processed/canonical/<bundle-id>
```

Outputs are private, ignored files under
`data/processed/filter_label_inventory/<inventory-id>/`: `inventory.json`,
`report.md`, and a manifest containing input identities, manifest hashes, pinned
registry identity, import lineage, code hashes and dependency versions. Identical
replays verify and reuse the result; changed inputs or implementation produce a
new identity. Existing snapshots and canonical attribution remain unchanged.

## Counts and interpretation

- **Source observations** include repeated exports, superseded rows and unresolved
  records. Their total is not a wager count.
- **Current observations** reconcile to the snapshot's current table. Ticket legs
  remain source rows, not separate wagers.
- **Linked wager candidates** count distinct canonical `bet_id` values per exact
  label, across all observations and separately for current observations. This
  reuses canonical identity; it does not establish independent funded executions.
- **Unlinked observations** remain visible rather than receiving invented wager
  identities. Source dispositions explain their canonical treatment.
- **Attribution statuses** report both bridge rows and distinct candidate counts
  for resolved, unresolved and missing attribution. Status populations can overlap.

Null, empty strings, whitespace, punctuation and Unicode remain distinct source
values. A raw string can contain multiple memberships or commas inside a name;
the inventory never splits it into a fabricated filter catalogue. Resolution uses
only the registry pinned by the canonical bundle and its existing resolver.

Per-label import counts preserve namespace/import identity. Timestamp bounds are
raw lexical bounds, partitioned by recorded timezone; they are explicitly **not
verified chronological bounds**. Unknown timezone, export scope or timestamp
meaning remains unknown. The snapshot knowledge cutoff is recorded separately.

Pairwise overlap ledgers count shared distinct candidates for raw label values and
resolved filter memberships. Per-label, per-import and membership counts must not
be added to obtain independent wager totals. Ticket candidates and repeated imports
are deduplicated at the candidate grain within each group.

## Review boundary

This inventory does not propose or apply new aliases, infer funding, change
economics, or reconstruct historical settings. Historical settings remain unknown
unless contemporaneous evidence establishes them. Deleted filters can still have
analyzable recorded membership. Missing labels cannot be recovered from current
filter parameters or sportsbook/EV similarity.

The next step is a separate proposed name/alias mapping for review, with questions
limited to ambiguous cases. Approved changes can then publish a new versioned
attribution report while retaining existing snapshots and reports.
