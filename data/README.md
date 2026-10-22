# Data

- `raw/`: untouched exports (OddsJam Bet Tracker CSVs). Never edit these.
- `processed/`: cleaned/typed datasets and feature tables derived from `raw/`.

- `raw/ingestion/`: immutable source objects, accepted import manifests, attempt
  receipts, and reconciled snapshots. All contents remain private and Git-ignored.
- `processed/ingestion-*/`: explicitly selected research bundles and provenance;
  these never replace the default historical snapshot.

See [the ingestion guide](../docs/INGESTION.md). Preserve the source archive,
manifest history, resolution decisions, source code, and lockfile together for
reproducible replay. Do not commit private identity-resolution files or manifests.
