"""
ingestion.py
===============================================================================
Purpose
-------
Archive manually acquired CSVs with immutable, reproducible import manifests.

What this module provides
-------------------------
Validated source metadata, import idempotency, and verified observation loading.

Design Notes
------------
Acquisition adapters can call import_csv with a downloaded local file. Import time
is system knowledge time, never a fabricated historical observation timestamp.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint, read_source


@dataclass(frozen=True)
class SourceMetadata:
    """Explicit acquisition context, independently of filenames.

    Attributes
    ----------
    namespace : str
        Non-secret stable source/account alias; prevents cross-account matching.
    exported_at : str or None
        User-supplied timezone-aware export time, never inferred from filename.
    timestamp_timezone : str
        Confirmed IANA timezone for naive source times, or 'unknown'.
    scope : str
        Declared export coverage, default unknown.
    """

    namespace: str
    exported_at: str | None = None
    timestamp_timezone: str = "unknown"
    scope: str = "unknown"

    def __post_init__(self) -> None:
        for field in ("namespace", "timestamp_timezone", "scope"):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"{field} must be a string")
        if self.exported_at is not None and not isinstance(self.exported_at, str):
            raise ValueError("exported_at must be an aware ISO timestamp string or null")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.namespace):
            raise ValueError(
                "namespace must be a non-secret letters/digits/underscore/hyphen alias"
            )
        if self.exported_at is not None:
            utc_time(self.exported_at)
        if self.timestamp_timezone != "unknown":
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            try:
                ZoneInfo(self.timestamp_timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError("timestamp_timezone must be an IANA timezone or unknown") from exc
        if not self.scope.strip():
            raise ValueError("scope must be supplied or unknown")


def utc_time(value: str) -> str:
    """Validate and normalize an aware ISO timestamp to UTC.

    Parameters
    ----------
    value : str
        ISO timestamp including an offset.

    Returns
    -------
    str
        UTC ISO timestamp.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone offset")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def import_csv(path: Path, root: Path, metadata: SourceMetadata) -> dict:
    """Archive a CSV once per namespace and content hash.

    Parameters
    ----------
    path : pathlib.Path
        Manually acquired source, never modified.
    root : pathlib.Path
        Private store, normally data/raw/ingestion.
    metadata : SourceMetadata
        Explicit source context.

    Returns
    -------
    dict
        Import manifest and repeated-import flag. Repeated bytes with conflicting
        metadata fail instead of changing historical interpretation.

    Notes
    -----
    Malformed sources fail before publication. Publication uses a staging directory
    and atomic rename; interrupted staging directories are never read as imports.
    """
    data = path.read_bytes()
    source_hash = hashlib.sha256(data).hexdigest()
    _, diagnostics = read_source(data)
    context = asdict(metadata)
    if context["exported_at"]:
        context["exported_at"] = utc_time(context["exported_at"])
    import_id = fingerprint([metadata.namespace, source_hash])
    with writer_lock(root):
        objects = root / "objects"
        imports = root / "imports"
        objects.mkdir(exist_ok=True)
        imports.mkdir(exist_ok=True)
        destination = imports / import_id
        if destination.exists():
            existing = json.loads((destination / "manifest.json").read_text())
            if existing["source"] != context:
                raise ValueError("Identical source already imported with different metadata")
            verify_bytes(objects / f"{source_hash}.csv", source_hash)
            _record_attempt(root, existing, path.name, repeated=True)
            return {**existing, "repeated_import": True}
        archive = objects / f"{source_hash}.csv"
        if archive.exists():
            verify_bytes(archive, source_hash)
        else:
            with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
                temporary = Path(staging) / "source.csv"
                temporary.write_bytes(data)
                temporary.rename(archive)
        manifest = {
            "schema_version": 1,
            "import_id": import_id,
            "source_sha256": source_hash,
            "source_filename": path.name,
            "byte_size": len(data),
            "imported_at": datetime.now(UTC).isoformat(),
            "source": context,
            "source_config_sha256": fingerprint(context),
            "code_version": _code_version(),
            "parser_version": 1,
            "diagnostics": diagnostics,
            "outcome": "accepted",
        }
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
            write_json(Path(staging) / "manifest.json", manifest)
            Path(staging).rename(destination)
        _record_attempt(root, manifest, path.name, repeated=False)
        return {**manifest, "repeated_import": False}


def _record_attempt(root: Path, manifest: dict, filename: str, *, repeated: bool) -> None:
    attempts = root / "attempts"
    attempts.mkdir(exist_ok=True)
    write_json(
        attempts / f"{uuid.uuid4().hex}.json",
        {
            "import_id": manifest["import_id"],
            "source_filename": filename,
            "attempted_at": datetime.now(UTC).isoformat(),
            "outcome": "already_imported" if repeated else "accepted",
        },
    )


def load_observations(root: Path, *, as_of: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Rebuild observations from verified archived bytes and import manifests.

    Parameters
    ----------
    root : pathlib.Path
        Private ingestion store.
    as_of : str or None
        System knowledge cutoff, including offset. Later imports are excluded.

    Returns
    -------
    tuple[pandas.DataFrame, list[dict]]
        Lossless observations and selected manifests in deterministic order.
    """
    cutoff = utc_time(as_of) if as_of else None
    manifests = []
    frames = []
    for path in sorted((root / "imports").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if cutoff and utc_time(manifest["imported_at"]) > cutoff:
            continue
        source_hash = manifest["source_sha256"]
        context = manifest["source"]
        SourceMetadata(**context)
        expected_id = fingerprint([context["namespace"], source_hash])
        if expected_id != manifest["import_id"] or path.parent.name != expected_id:
            raise ValueError("Import manifest identity mismatch")
        if fingerprint(context) != manifest["source_config_sha256"]:
            raise ValueError("Import metadata integrity failure")
        data = verify_bytes(root / "objects" / f"{source_hash}.csv", source_hash)
        frame, diagnostics = read_source(data)
        if diagnostics != manifest["diagnostics"] or len(data) != manifest["byte_size"]:
            raise ValueError("Source schema/row count differs from import manifest")
        columns = list(frame.columns)
        frame["_ingest_payload_hash"] = [
            fingerprint(row) for row in frame[columns].to_dict("records")
        ]
        frame["_ingest_observation_id"] = [
            fingerprint([expected_id, i]) for i in range(1, len(frame) + 1)
        ]
        frame["_ingest_source_row"] = range(1, len(frame) + 1)
        frame["_ingest_import_id"] = expected_id
        frame["_ingest_namespace"] = context["namespace"]
        frame["_ingest_imported_at"] = manifest["imported_at"]
        frame["_ingest_exported_at"] = context["exported_at"] or ""
        frame["_ingest_timestamp_timezone"] = context["timestamp_timezone"]
        frames.append(frame)
        manifests.append(manifest)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), manifests
