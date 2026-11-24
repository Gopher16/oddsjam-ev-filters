"""
operating_regimes.py
===============================================================================
Purpose
-------
Publish optional operating evidence and non-economic linkage immutably.

What this module provides
-------------------------
Pinned canonical/registry lineage, verified reads, append-only revisions and replay.

Design Notes
------------
Existing publications are read-only. No default consumer or classification changes.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.operating_regimes import COLUMNS, build_regime_linkage, validate_regimes
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.source_csv import fingerprint

ARTIFACTS = {"history.json", "registry.json", "linkage.parquet", "audit.json"}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_operating_regimes(path: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Read a byte-verified optional publication without current-config fallback.

    Parameters
    ----------
    path : pathlib.Path
        Explicit content-addressed publication directory.

    Returns
    -------
    tuple[dict, pandas.DataFrame, dict]
        Ledger, non-economic candidate linkage and coverage audit.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec.get("contract_version") != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Operating publication specification integrity failure")
    if set(manifest["output_sha256"]) != ARTIFACTS:
        raise ValueError("Incomplete operating publication")
    for name, digest in manifest["output_sha256"].items():
        verify_bytes(path / name, digest)
    history = json.loads((path / "history.json").read_text())
    registry = json.loads((path / "registry.json").read_text())
    validate_regimes({**history, "previous_history_id": None}, registry)
    frame = read_parquet_file(path / "linkage.parquet")
    audit = json.loads((path / "audit.json").read_text())
    if (
        fingerprint(history) != spec["history_id"]
        or fingerprint(registry) != spec["registry_id"]
        or list(frame.columns) != COLUMNS
        or frame.bet_id.isna().any()
        or frame.bet_id.duplicated().any()
        or len(frame) != audit["candidate_count"]
        or audit != manifest["summary"]
        or frame.linkage_status.value_counts().sort_index().to_dict()
        != audit["linkage_status_counts"]
    ):
        raise ValueError("Operating publication coverage/identity mismatch")
    return history, frame, audit


def publish_operating_regimes(
    canonical: Path, output_root: Path, history_path: Path, *, previous: Path | None = None
) -> Path:
    """Publish an explicit private ledger without modifying canonical tables.

    Parameters
    ----------
    canonical : pathlib.Path
        Verified immutable canonical input.
    output_root : pathlib.Path
        Separate private destination; must not overlap an input publication.
    history_path : pathlib.Path
        Explicit version-1 JSON input, pinned by exact byte hash.
    previous : pathlib.Path or None
        Verified predecessor required for append revisions.

    Returns
    -------
    pathlib.Path
        New publication or identical verified replay.
    """
    canonical, output_root, history_path = map(Path, (canonical, output_root, history_path))
    for source in (canonical, previous):
        if source is not None:
            source = Path(source).resolve()
            if output_root.resolve().is_relative_to(source) or (
                source == canonical.resolve() and source.is_relative_to(output_root.resolve())
            ):
                raise ValueError("Operating output must not overlap source publications")
    tables = load_canonical_bundle(canonical)
    cm = json.loads((canonical / "manifest.json").read_text())
    registry = cm["specification"]["filter_registry"]
    history = json.loads(history_path.read_text())
    prior = load_operating_regimes(previous)[0] if previous else None
    if previous:
        pm = json.loads((Path(previous) / "manifest.json").read_text())
        if pm["specification"]["canonical_manifest_sha256"] != _digest(canonical / "manifest.json"):
            raise ValueError("Revision must retain the same canonical input; migrate explicitly")
    history_id = validate_regimes(history, registry, prior)
    linkage, audit = build_regime_linkage(
        tables["bet_facts"], tables["filter_attribution"], history, registry
    )
    base = Path(__file__).parents[1]
    code = [
        Path(__file__),
        base / "analysis/operating_regimes.py",
        base / "analysis/decision_config.py",
        base / "analysis/filter_attribution.py",
        base / "analysis/sizing_config.py",
    ]
    spec = dict(
        contract_version=1,
        history_id=history_id,
        registry_id=fingerprint(registry),
        canonical_bundle_id=cm["bundle_id"],
        canonical_manifest_sha256=_digest(canonical / "manifest.json"),
        history_source_sha256=_digest(history_path),
        previous_manifest_sha256=_digest(Path(previous) / "manifest.json") if previous else None,
        code_sha256={str(p.relative_to(base)): _digest(p) for p in code},
        runtime=dict(python=platform.python_version(), pandas=pd.__version__),
    )
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old, frame, summary = load_operating_regimes(destination)
            if old != history or summary != audit:
                raise ValueError("Operating replay mismatch")
            assert_frame_equal(frame, linkage)
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as temp:
            stage = Path(temp)
            write_json(stage / "history.json", history)
            write_json(stage / "registry.json", registry)
            write_json(stage / "audit.json", audit)
            linkage.to_parquet(stage / "linkage.parquet", index=False)
            write_json(
                stage / "manifest.json",
                dict(
                    publication_id=ident,
                    specification=spec,
                    summary=audit,
                    output_sha256={name: _digest(stage / name) for name in sorted(ARTIFACTS)},
                ),
            )
            _, frame, _ = load_operating_regimes(stage)
            assert_frame_equal(frame, linkage)
            stage.rename(destination)
    return destination
