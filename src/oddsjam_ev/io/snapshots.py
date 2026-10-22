"""
snapshots.py
===============================================================================
Purpose
-------
Materialize and verify explicit ingestion snapshots for downstream research.

What this module provides
-------------------------
Atomic versioned Parquet snapshots with complete import and decision provenance.

Design Notes
------------
Rebuild means replaying archived bytes and pinned metadata, not relying on caches.
Logical table equality is the reproducibility contract, not Parquet byte equality.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from oddsjam_ev.io.ingestion import load_observations, utc_time
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.reconcile import POLICY, reconcile
from oddsjam_ev.io.source_csv import fingerprint


def materialize_snapshot(
    root: Path, *, as_of: str | None = None, decisions: list[dict] | None = None
) -> Path:
    """Replay a store into an immutable, content-addressed snapshot directory.

    Parameters
    ----------
    root : pathlib.Path
        Private ingestion store.
    as_of : str or None
        Aware system knowledge cutoff; does not backdate historical knowledge.
    decisions : list[dict] or None
        Reviewed identity resolutions, recorded with decision timestamps.

    Returns
    -------
    pathlib.Path
        Published snapshot directory containing current.parquet,
        observations.parquet, and manifest.json.
    """
    cutoff = utc_time(as_of) if as_of else None
    selected_decisions = sorted(
        [d for d in decisions or [] if not cutoff or utc_time(d["decided_at"]) <= cutoff],
        key=lambda d: d["observation_id"],
    )
    with writer_lock(root):
        observations, imports = load_observations(root, as_of=cutoff)
        if not imports:
            raise ValueError("No accepted imports available for snapshot")
        for decision in selected_decisions:
            if utc_time(decision["decided_at"]) > utc_time(datetime.now(UTC).isoformat()):
                raise ValueError("Resolution cannot be future-dated")
            matching = observations[
                observations["_ingest_observation_id"].eq(decision["observation_id"])
            ]
            if matching.empty:
                raise ValueError("Resolution refers to an unavailable observation")
            if utc_time(decision["decided_at"]) < utc_time(matching.iloc[0]["_ingest_imported_at"]):
                raise ValueError("Resolution cannot predate system knowledge")
        current, audit, summary = reconcile(observations, selected_decisions)
        specification = {
            "schema_version": 1,
            "reconciliation_version": 1,
            "reconciliation_code_sha256": hashlib.sha256(
                Path(__file__).with_name("reconcile.py").read_bytes()
            ).hexdigest(),
            "pandas_version": pd.__version__,
            "as_of": cutoff,
            "imports": imports,
            "decisions": selected_decisions,
            "execution_policy": POLICY,
        }
        snapshot_id = fingerprint(specification)
        parent = root / "snapshots"
        parent.mkdir(exist_ok=True)
        destination = parent / snapshot_id
        if destination.exists():
            from pandas.testing import assert_frame_equal

            assert_frame_equal(current, load_snapshot(destination))
            assert_frame_equal(audit, pd.read_parquet(destination / "observations.parquet"))
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
            stage = Path(staging)
            current.to_parquet(stage / "current.parquet", index=False)
            audit.to_parquet(stage / "observations.parquet", index=False)
            hashes = {
                name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                for name in ["current.parquet", "observations.parquet"]
            }
            write_json(
                stage / "manifest.json",
                {
                    "snapshot_id": snapshot_id,
                    "specification": specification,
                    "summary": summary,
                    "output_sha256": hashes,
                },
            )
            stage.rename(destination)
        return destination


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load current rows only after verifying pinned output hashes.

    Parameters
    ----------
    path : pathlib.Path
        Explicit published snapshot directory.

    Returns
    -------
    pandas.DataFrame
        Current reconciled source records, including lineage columns.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["snapshot_id"]:
        raise ValueError("Snapshot specification integrity failure")
    for name in ("current.parquet", "observations.parquet"):
        verify_bytes(path / name, manifest["output_sha256"][name])
    return pd.read_parquet(path / "current.parquet")
