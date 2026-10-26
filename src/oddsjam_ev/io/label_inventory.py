"""
label_inventory.py
===============================================================================
Purpose
-------
Publish private exact-label inventories from explicit verified source bundles.

What this module provides
-------------------------
Content-addressed JSON/report publication, input lineage checks and verified replay.

Design Notes
------------
No aliases, classifications, economics or historical snapshots are changed.
===============================================================================
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow

from oddsjam_ev.analysis.label_inventory import build_label_inventory
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.snapshots import load_snapshot
from oddsjam_ev.io.source_csv import fingerprint


def _report(payload):
    lines = [
        "# Historical label inventory",
        "",
        "Exact strings, not an inferred filter catalogue. No alias mappings were changed.",
        "",
        "## Count definitions",
        "",
        "Source observations include repeated imports and unresolved records. Current observations are reconciled source rows, not wagers. Wager counts are distinct canonical candidates, including ticket candidates counted once per label. Candidate identity is not proof of a distinct funded execution.",
        "",
        "Resolved, unresolved and missing attribution are counted separately. Status counts may overlap when a wager has both resolved and unresolved labels. Per-label/import counts and filter memberships are not additive across groups.",
        "",
        "Timestamp bounds are lexical raw source strings, not verified placement or settlement chronology. Export scope/timezone remain as recorded; no current settings are applied retrospectively.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value:,}")
    lines += [
        "",
        "## Exact labels",
        "",
        "| Raw label (JSON string) | Observations | Current | Linked wager candidates | Registry resolution |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(
        payload["labels"], key=lambda r: (-r["linked_wager_candidate_count"], r["label_id"])
    ):
        label = (
            json.dumps(row["raw_label"], ensure_ascii=True).replace("|", "\\|").replace("`", "\\`")
        )
        lines.append(
            f"| {label} | {row['source_observation_count']} | {row['current_observation_count']} | {row['linked_wager_candidate_count']} | {row['registry_resolution_status']} |"
        )
    lines += [
        "",
        "Full import lineage, timestamp bounds, dispositions, status counts and overlap ledgers are in inventory.json. Deleted-filter settings may remain unknown. Missing labels cannot be reconstructed from sportsbook, EV or current parameters.",
    ]
    return "\n".join(lines) + "\n"


def load_label_inventory(path: Path) -> dict:
    """Verify one published inventory's specification and output bytes.

    Parameters
    ----------
    path : pathlib.Path
        Explicit content-addressed inventory directory.

    Returns
    -------
    dict
        Verified inventory payload with counts and source lineage in its manifest.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["specification"].get("inventory_contract_version") != 1:
        raise ValueError("Unsupported label inventory contract")
    if fingerprint(manifest["specification"]) != manifest["inventory_id"]:
        raise ValueError("Label inventory specification integrity failure")
    data = verify_bytes(path / "inventory.json", manifest["output_sha256"]["inventory.json"])
    verify_bytes(path / "report.md", manifest["output_sha256"]["report.md"])
    payload = json.loads(data)
    if payload["summary"] != manifest["summary"]:
        raise ValueError("Label inventory summary mismatch")
    return payload


def publish_label_inventory(snapshot: Path, canonical: Path, output_root: Path) -> Path:
    """Inventory a matched snapshot/canonical pair without updating research defaults.

    Parameters
    ----------
    snapshot, canonical : pathlib.Path
        Explicit verified ingestion snapshot and canonical bundle using that snapshot.
    output_root : pathlib.Path
        Private output directory; must not lie inside either source bundle.

    Returns
    -------
    pathlib.Path
        New immutable inventory directory, or verified identical replay result.
    """
    snapshot, canonical, output_root = map(Path, (snapshot, canonical, output_root))
    for source in (snapshot, canonical):
        if output_root.resolve().is_relative_to(source.resolve()):
            raise ValueError("Inventory output must not modify a source bundle")
    current = load_snapshot(snapshot)
    sm = json.loads((snapshot / "manifest.json").read_text())
    tables = load_canonical_bundle(canonical)
    cm = json.loads((canonical / "manifest.json").read_text())
    if cm["specification"]["ingestion_manifest"] != sm:
        raise ValueError("Canonical bundle does not pin this exact source snapshot")
    observations = read_parquet_file(snapshot / "observations.parquet")
    payload = build_label_inventory(
        observations, current, tables, cm["specification"]["filter_registry"]
    )
    analysis = Path(__file__).resolve().parents[1] / "analysis"
    files = [
        Path(__file__),
        analysis / "label_inventory.py",
        analysis / "filter_attribution.py",
    ] + [
        Path(__file__).with_name(n + ".py")
        for n in ("canonical", "snapshots", "parquet", "source_csv")
    ]
    specification = {
        "inventory_contract_version": 1,
        "snapshot_id": sm["snapshot_id"],
        "canonical_bundle_id": cm["bundle_id"],
        "source_manifest_sha256": {
            "snapshot": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
            "canonical": hashlib.sha256((canonical / "manifest.json").read_bytes()).hexdigest(),
        },
        "registry_sha256": fingerprint(cm["specification"]["filter_registry"]),
        "source_imports": sm["specification"]["imports"],
        "snapshot_knowledge_cutoff": sm["specification"]["as_of"],
        "timestamp_bounds_semantics": "lexical_raw_strings_not_verified_chronology",
        "code_sha256": {
            p.name if p.parent.name != "analysis" else "analysis/" + p.name: hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in files
        },
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
    }
    identity = fingerprint(specification)
    with writer_lock(output_root):
        destination = output_root / identity
        if destination.exists():
            if load_label_inventory(destination) != payload:
                raise ValueError("Label inventory replay mismatch")
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            write_json(stage / "inventory.json", payload)
            (stage / "report.md").write_text(_report(payload), encoding="utf-8")
            hashes = {
                name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                for name in ("inventory.json", "report.md")
            }
            write_json(
                stage / "manifest.json",
                {
                    "inventory_id": identity,
                    "specification": specification,
                    "summary": payload["summary"],
                    "output_sha256": hashes,
                },
            )
            load_label_inventory(stage)
            stage.rename(destination)
    return destination
