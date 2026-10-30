"""
sizing_history.py
===============================================================================
Purpose
-------
Publish private sizing-policy history and non-economic linkage audits immutably.

What this module provides
-------------------------
Explicit input validation, append-only revisions, verified loading and replay.

Design Notes
------------
No current account settings are backfilled. Source economics and prior snapshots
remain unchanged. An absent history explicitly produces unknown linkage.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.sizing_config import validate_history
from oddsjam_ev.analysis.sizing_linkage import build_sizing_linkage
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.source_csv import fingerprint


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sizing_history(path: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Verify an explicit history publication and its linkage/audit bytes.

    Parameters
    ----------
    path : pathlib.Path
        Private immutable publication directory.

    Returns
    -------
    tuple[dict, pandas.DataFrame, dict]
        History, one-row-per-candidate linkage and coverage/contradiction audit.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec.get("contract_version") != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Sizing publication specification integrity failure")
    for name, digest in manifest["output_sha256"].items():
        if name not in {"history.json", "linkage.parquet", "audit.json"}:
            raise ValueError("Unsupported sizing publication artifact")
        verify_bytes(path / name, digest)
    if set(manifest["output_sha256"]) != {"history.json", "linkage.parquet", "audit.json"}:
        raise ValueError("Incomplete sizing publication")
    history = json.loads((path / "history.json").read_text())
    frame = read_parquet_file(path / "linkage.parquet")
    audit = json.loads((path / "audit.json").read_text())
    if (
        fingerprint(history) != spec["history_id"]
        or len(frame) != audit["candidate_count"]
        or frame.bet_id.duplicated().any()
        or audit != manifest["summary"]
    ):
        raise ValueError("Sizing history coverage/identity mismatch")
    return history, frame, audit


def publish_sizing_history(
    canonical: Path,
    output_root: Path,
    *,
    history_path: Path | None = None,
    previous: Path | None = None,
    knowledge_as_of: str | None = None,
) -> Path:
    """Publish optional policy history without altering default research metrics.

    Parameters
    ----------
    canonical : pathlib.Path
        Explicit verified canonical bundle.
    output_root : pathlib.Path
        Private destination outside source publications.
    history_path : pathlib.Path or None
        Local JSON policy/evidence ledger; never committed account data.
    previous : pathlib.Path or None
        Verified predecessor publication required for noninitial history revisions.
    knowledge_as_of : str or None
        Explicit aware cutoff for an empty-history unknown-coverage audit. Mutually
        exclusive with history_path, which carries its own knowledge cutoff.

    Returns
    -------
    pathlib.Path
        Content-addressed publication or verified identical replay.
    """
    canonical, output_root = Path(canonical), Path(output_root)
    if (history_path is None) == (knowledge_as_of is None):
        raise ValueError("Choose explicit history or empty-history knowledge cutoff")
    for source in (canonical, previous):
        if source is not None and output_root.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError("Sizing output must not modify a source publication")
    tables = load_canonical_bundle(canonical)
    cm = json.loads((canonical / "manifest.json").read_text())
    prior = load_sizing_history(Path(previous))[0] if previous else None
    history = (
        json.loads(Path(history_path).read_text())
        if history_path
        else {
            "contract_version": 1,
            "knowledge_as_of": knowledge_as_of,
            "previous_history_id": None,
            "policies": [],
            "wager_evidence": [],
        }
    )
    history_id = validate_history(history, prior)
    linkage, audit = build_sizing_linkage(tables["bet_facts"], history)
    analysis = Path(__file__).parents[1] / "analysis"
    files = [Path(__file__), analysis / "sizing_config.py", analysis / "sizing_linkage.py"]
    spec = {
        "contract_version": 1,
        "history_id": history_id,
        "knowledge_as_of": history["knowledge_as_of"],
        "canonical_bundle_id": cm["bundle_id"],
        "canonical_manifest_sha256": _digest(canonical / "manifest.json"),
        "snapshot_id": cm["specification"]["ingestion_manifest"]["snapshot_id"],
        "previous_publication_manifest_sha256": _digest(Path(previous) / "manifest.json")
        if previous
        else None,
        "history_source_sha256": _digest(Path(history_path)) if history_path else None,
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "interval_semantics": "[effective_from,effective_to); explicit open end; no naive or export-receipt fallback",
    }
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old, frame, summary = load_sizing_history(destination)
            if old != history or summary != audit:
                raise ValueError("Sizing publication replay mismatch")
            assert_frame_equal(frame, linkage)
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as temporary:
            stage = Path(temporary)
            write_json(stage / "history.json", history)
            write_json(stage / "audit.json", audit)
            linkage.to_parquet(stage / "linkage.parquet", index=False)
            write_json(
                stage / "manifest.json",
                {
                    "publication_id": ident,
                    "specification": spec,
                    "summary": audit,
                    "output_sha256": {
                        name: _digest(stage / name)
                        for name in ("history.json", "audit.json", "linkage.parquet")
                    },
                },
            )
            _, frame, _ = load_sizing_history(stage)
            assert_frame_equal(frame, linkage)
            stage.rename(destination)
    return destination
