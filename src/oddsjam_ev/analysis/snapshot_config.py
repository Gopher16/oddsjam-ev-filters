"""
snapshot_config.py
===============================================================================

Purpose
-------
Resolve historical OddsJam processed snapshots consistently across notebooks,
independently of the current working directory.

What this module provides
-------------------------
1. SnapshotConfig
   Immutable configuration containing the repository root, selected snapshot
   date, and paths to processed data and preprocessing QA artifacts.

2. resolve_snapshot_config(...)
   Resolves the project root, validates a dated snapshot identifier, and
   optionally accepts an explicit snapshot override from the environment.

Design Notes
------------
- The default snapshot is explicit and historical, not the latest available.
- ODDSJAM_SNAPSHOT_DATE can override the default using MM-DD-YYYY format.
- ODDSJAM_PROJECT_ROOT can identify a nonstandard repository location.
- No filesystem writes occur during configuration resolution.
- Snapshot filename dates do not establish the actual range of bet timestamps.
===============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_SNAPSHOT_DATE = "03-21-2026"


@dataclass(frozen=True)
class SnapshotConfig:
    """
    Paths and identifiers for one processed research snapshot.

    Attributes
    ----------
    project_root
        Absolute path to the repository root.
    snapshot_date
        Filename date suffix in MM-DD-YYYY format.
    processed_path
        Path to the processed Bet Tracker Parquet snapshot.
    artifact_dir
        Directory containing preprocessing QA artifacts.
    """

    project_root: Path
    snapshot_date: str
    processed_path: Path
    artifact_dir: Path


def resolve_snapshot_config(
    snapshot_date: str | None = None,
    project_root: Path | str | None = None,
) -> SnapshotConfig:
    """
    Resolve a reproducible processed snapshot and associated artifact paths.

    Parameters
    ----------
    snapshot_date
        Optional explicit snapshot suffix. Otherwise reads
        ODDSJAM_SNAPSHOT_DATE, falling back to DEFAULT_SNAPSHOT_DATE.
    project_root
        Optional repository root. Otherwise reads ODDSJAM_PROJECT_ROOT or
        resolves the root from this installed source module's location.

    Returns
    -------
    SnapshotConfig
        Validated snapshot identifier and absolute dataset/artifact paths.

    Raises
    ------
    ValueError
        If the snapshot suffix is not a valid MM-DD-YYYY date.
    FileNotFoundError
        If the resolved project root lacks the expected data/processed folder.

    Notes
    -----
    This function does not require the dated Parquet file to exist. Callers
    should check ``processed_path.is_file()`` before loading data.
    """
    selected = snapshot_date or os.environ.get("ODDSJAM_SNAPSHOT_DATE", DEFAULT_SNAPSHOT_DATE)
    try:
        parsed = datetime.strptime(selected, "%m-%d-%Y")
    except ValueError as exc:
        raise ValueError("Snapshot date must use MM-DD-YYYY format.") from exc
    if parsed.strftime("%m-%d-%Y") != selected:
        raise ValueError("Snapshot date must use zero-padded MM-DD-YYYY format.")

    override_root = project_root or os.environ.get("ODDSJAM_PROJECT_ROOT")
    root = (
        Path(override_root).expanduser().resolve()
        if override_root is not None
        else Path(__file__).resolve().parents[3]
    )
    processed_dir = root / "data" / "processed"
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    return SnapshotConfig(
        project_root=root,
        snapshot_date=selected,
        processed_path=processed_dir / f"oddsjam-bet-tracker-processed-{selected}.parquet",
        artifact_dir=processed_dir / "artifacts",
    )
