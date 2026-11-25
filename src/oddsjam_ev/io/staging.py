"""
staging.py
===============================================================================
Purpose
-------
Stage bounded private CSV uploads for review before immutable import.

What this module provides
-------------------------
Content-addressed staging and conservative overlap previews using existing rules.

Design Notes
------------
Coverage is a declaration, not inferred completeness. No preview executes formulas
or mutates the import store. Preview digests bind confirmation to reviewed evidence.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from oddsjam_ev.io.ingestion import SourceMetadata, import_csv, load_observations, utc_time
from oddsjam_ev.io.manifests import write_json, writer_lock
from oddsjam_ev.io.reconcile import reconcile
from oddsjam_ev.io.source_csv import fingerprint, read_source

MAX_BYTES = 50_000_000
MAX_ROWS = 100_000
MAX_COLUMNS = 100


def _private_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError("Staging root must be a private owner-only directory")


def _parse(data: bytes) -> tuple[pd.DataFrame, dict]:
    if len(data) > MAX_BYTES or b"\x00" in data:
        raise ValueError("CSV exceeds byte limit or contains NUL bytes")
    frame, diagnostics = read_source(data)
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError("CSV exceeds row or column limit")
    return frame, diagnostics


def stage_csv(stream: BinaryIO, root: Path, *, filename: str, metadata: SourceMetadata) -> dict:
    """Stage a bounded UTF-8 CSV without importing it.

    Parameters
    ----------
    stream : BinaryIO
        Binary upload stream, read only up to the byte budget plus one.
    root : Path
        Private staging directory, separate from the ingestion store.
    filename : str
        Simple CSV basename used only for display.
    metadata : SourceMetadata
        Explicit namespace, time context and full/partial/unknown scope.

    Returns
    -------
    dict
        Immutable staging receipt. Identical bytes/context reuse the same stage.
    """
    if (
        not isinstance(filename, str)
        or not filename.lower().endswith(".csv")
        or any(c in filename for c in "/\\\x00\r\n")
        or filename in {".csv", "..csv"}
        or len(filename) > 255
    ):
        raise ValueError("Upload requires a simple .csv filename")
    if metadata.scope not in {"full", "partial", "unknown"}:
        raise ValueError("Declare scope as full, partial or unknown")
    data = stream.read(MAX_BYTES + 1)
    if not isinstance(data, bytes):
        raise ValueError("Upload must be a binary stream")
    _, diagnostics = _parse(data)
    context = asdict(metadata)
    if context["exported_at"]:
        context["exported_at"] = utc_time(context["exported_at"])
    digest = hashlib.sha256(data).hexdigest()
    stage_id = fingerprint([digest, context])
    receipt = {
        "schema_version": 1,
        "stage_id": stage_id,
        "source_sha256": digest,
        "source": context,
        "byte_size": len(data),
        "diagnostics": diagnostics,
    }
    _private_root(root)
    with writer_lock(root):
        destination = root / stage_id
        if destination.exists():
            return load_stage(root, stage_id)[1]
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as directory:
            temporary = Path(directory)
            source = temporary / "source.csv"
            source.write_bytes(data)
            source.chmod(0o600)
            write_json(temporary / "receipt.json", receipt)
            (temporary / "receipt.json").chmod(0o600)
            temporary.rename(destination)
    return receipt


def load_stage(root: Path, stage_id: str) -> tuple[bytes, dict]:
    """Verify a bounded staged source and receipt.

    Parameters
    ----------
    root : Path
        Private staging directory.
    stage_id : str
        SHA-256 stage identity; paths are not accepted.

    Returns
    -------
    tuple[bytes, dict]
        Verified source bytes and receipt.
    """
    if not re.fullmatch(r"[a-f0-9]{64}", stage_id):
        raise ValueError("Invalid stage identity")
    _private_root(root)
    directory = root / stage_id
    if directory.is_symlink():
        raise ValueError("Stage cannot be a symlink")
    for name in ("source.csv", "receipt.json"):
        if (directory / name).is_symlink():
            raise ValueError("Stage files cannot be symlinks")
    with (directory / "source.csv").open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    with (directory / "receipt.json").open("rb") as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("Stage receipt exceeds limit")
    receipt = json.loads(raw)
    _, diagnostics = _parse(data)
    metadata = SourceMetadata(**receipt["source"])
    expected = {
        "schema_version": 1,
        "stage_id": fingerprint([hashlib.sha256(data).hexdigest(), asdict(metadata)]),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source": asdict(metadata),
        "byte_size": len(data),
        "diagnostics": diagnostics,
    }
    if receipt != expected or receipt["stage_id"] != stage_id:
        raise ValueError("Stage integrity failure")
    return data, receipt


def preview_csv(root: Path, stage_id: str, ingestion_root: Path) -> dict:
    """Preview overlap without importing or claiming verified execution identity.

    Parameters
    ----------
    root : Path
        Private staging directory.
    stage_id : str
        Verified stage identity.
    ingestion_root : Path
        Existing private immutable import store.

    Returns
    -------
    dict
        Review digest, coverage/schema/date diagnostics and candidate dispositions.
    """
    data, receipt = load_stage(root, stage_id)
    frame, diagnostics = _parse(data)
    observations, manifests = load_observations(ingestion_root)
    source = receipt["source"]
    import_id = fingerprint([source["namespace"], receipt["source_sha256"]])
    repeated = next((m for m in manifests if m["import_id"] == import_id), None)
    counts = {"new": 0, "unchanged": 0, "revised": 0, "unresolved": 0}
    settlement_changes = 0
    if repeated:
        if repeated["source"] != source:
            raise ValueError("Identical source already imported with different metadata")
        counts["unchanged"] = len(frame)
    else:
        candidate = frame.copy()
        candidate["_ingest_payload_hash"] = [fingerprint(r) for r in frame.to_dict("records")]
        candidate["_ingest_observation_id"] = [
            fingerprint([import_id, i]) for i in range(1, len(frame) + 1)
        ]
        candidate["_ingest_source_row"] = range(1, len(frame) + 1)
        candidate["_ingest_import_id"] = import_id
        candidate["_ingest_namespace"] = source["namespace"]
        candidate["_ingest_imported_at"] = "preview_only"
        candidate["_ingest_exported_at"] = source["exported_at"] or ""
        candidate["_ingest_timestamp_timezone"] = source["timestamp_timezone"]
        _, audit, _ = reconcile(pd.concat([observations, candidate], ignore_index=True))
        selected = audit[audit["_ingest_import_id"].eq(import_id)]
        prior = audit[~audit["_ingest_import_id"].eq(import_id)]
        for row in selected.to_dict("records"):
            if row["_ingest_disposition"] == "unresolved":
                counts["unresolved"] += 1
                continue
            matches = prior[prior["_ingest_record_id"].eq(row["_ingest_record_id"])]
            if matches.empty:
                counts["new"] += 1
            elif matches["_ingest_payload_hash"].eq(row["_ingest_payload_hash"]).any():
                counts["unchanged"] += 1
            else:
                counts["revised"] += 1
                settlement_changes += int(
                    any(
                        any(
                            str(old.get(key, "")) != str(row.get(key, ""))
                            for key in ("status", "bet_profit", "potential_payout")
                        )
                        for old in matches.to_dict("records")
                    )
                )
    preview = {
        "schema_version": 1,
        "stage_id": stage_id,
        "source_sha256": receipt["source_sha256"],
        "source": source,
        "diagnostics": diagnostics,
        "coverage_state": "unknown_requires_declaration"
        if source["scope"] == "unknown"
        else "owner_declared_" + source["scope"],
        "repeated_import": repeated is not None,
        "candidate_rows": counts,
        "settlement_revision_rows": settlement_changes,
        "formula_like_cells": sum(
            str(v).lstrip().startswith(("=", "+", "-", "@")) for v in frame.to_numpy().flat
        ),
        "prior_imports": sorted([m["import_id"], fingerprint(m)] for m in manifests),
        "interpretation": "Conservative candidate overlap, not verified fills; raw dates are not freshness or complete coverage. Formula-like cells remain literal text.",
    }
    return {**preview, "preview_sha256": fingerprint(preview)}


def confirm_import(
    root: Path,
    stage_id: str,
    ingestion_root: Path,
    *,
    preview_sha256: str,
    confirmed: bool = False,
) -> dict:
    """Import only an explicitly confirmed, unchanged preview.

    Parameters
    ----------
    root : Path
        Private staging directory.
    stage_id : str
        Verified stage identity.
    ingestion_root : Path
        Separate immutable ingestion store.
    preview_sha256 : str
        Digest of the exact preview reviewed by the operator.
    confirmed : bool
        Must be explicitly true; unknown coverage must first be restaged with a
        full or partial declaration. This does not verify that declaration.

    Returns
    -------
    dict
        Existing immutable import receipt and repeated-import flag. No build or
        report activation is performed. Failures preserve the staged source.
    """
    if confirmed is not True:
        raise ValueError("Explicit import confirmation required")
    if root.resolve() == ingestion_root.resolve():
        raise ValueError("Staging and ingestion stores must be separate")
    _private_root(root)
    with writer_lock(root):
        preview = preview_csv(root, stage_id, ingestion_root)
        if preview["preview_sha256"] != preview_sha256:
            raise ValueError("Preview changed; review a fresh preview")
        if preview["source"]["scope"] == "unknown":
            raise ValueError("Restage with an explicit full or partial declaration")
        data, receipt = load_stage(root, stage_id)
        _private_root(ingestion_root)
        # Import the verified bytes through a private temporary file, never a
        # caller-controlled filename or mutable upload path.
        with tempfile.TemporaryDirectory(prefix=".confirm-", dir=root) as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(data)
            path.chmod(0o600)
            return import_csv(
                path,
                ingestion_root,
                SourceMetadata(**receipt["source"]),
                expected_imports=preview["prior_imports"],
            )
