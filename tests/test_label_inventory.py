"""Exact label counts preserve source grain, overlaps and unresolved identities."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical

from oddsjam_ev.analysis.label_inventory import build_label_inventory
from oddsjam_ev.io.label_inventory import load_label_inventory, publish_label_inventory


def fixtures():
    labels = [
        "A (Slow, Build)",
        "A (Slow, Build)",
        "B",
        "A (Slow, Build), B",
        "",
        None,
        "  ",
        "Unknown λ",
    ]
    obs = pd.DataFrame(
        {
            "_ingest_observation_id": [f"o{i}" for i in range(8)],
            "_ingest_record_id": ["r0", "r0", "r2", "r3", "r4", "r5", "r6", "r7"],
            "_ingest_import_id": ["import1", "import2", *["import1"] * 6],
            "_ingest_namespace": ["namespace"] * 8,
            "_ingest_timestamp_timezone": ["unknown"] * 8,
            "_ingest_is_current": [False, *[True] * 7],
            "saved_filter_names": labels,
            "created_at": ["2026-9-2", "2026-10-1", *[""] * 6],
        }
    )
    audit = pd.DataFrame(
        {
            "_ingest_observation_id": obs._ingest_observation_id,
            "bet_id": ["ticket", "ticket", "ticket", "ticket", "single", None, None, None],
            "canonical_disposition": [
                "superseded",
                *["current_ticket_leg"] * 3,
                "current_single",
                *["unresolved_source"] * 3,
            ],
        }
    )
    bridge = pd.DataFrame(
        {
            "bet_id": ["ticket", "ticket", "single"],
            "filter_id": ["a", "b", None],
            "attribution_status": ["resolved", "resolved", "missing"],
        }
    )
    tables = {
        "source_audit": audit,
        "bet_facts": pd.DataFrame({"bet_id": ["ticket", "single"]}),
        "filter_attribution": bridge,
    }
    registry = {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [{"id": "a", "name": "A (Slow, Build)"}, {"id": "b", "name": "B"}],
    }
    return obs, obs.loc[obs._ingest_is_current].reset_index(drop=True), tables, registry


def test_exact_counts_repeated_imports_ticket_legs_missing_and_overlaps():
    obs, current, tables, registry = fixtures()
    before = obs.copy(deep=True)
    result = build_label_inventory(obs, current, tables, registry)
    rows = {r["raw_label"]: r for r in result["labels"]}
    assert rows["A (Slow, Build)"]["source_observation_count"] == 2
    assert rows["A (Slow, Build)"]["distinct_source_record_count"] == 1
    assert rows["A (Slow, Build)"]["current_observation_count"] == 1
    assert rows["A (Slow, Build)"]["linked_wager_candidate_count"] == 1
    assert rows["A (Slow, Build), B"]["resolved_filter_ids"] == ["a", "b"]
    assert rows[""]["label_presence"] == "empty"
    assert rows[None]["label_presence"] == "null"
    assert rows["  "]["registry_resolution_status"] == "unresolved"
    assert rows["Unknown λ"]["raw_label"] == "Unknown λ"
    assert rows["A (Slow, Build)"]["raw_created_at_lexical_min"] == "2026-10-1"
    assert result["summary"]["source_observations"] == 8
    assert result["summary"]["current_source_observations"] == 7
    assert result["summary"]["canonical_wager_candidates"] == 2
    assert result["summary"]["wagers_with_multiple_raw_label_values"] == 1
    assert result["resolved_filter_overlaps"] == [
        {"left_filter_id": "a", "right_filter_id": "b", "shared_wager_candidates": 1}
    ]
    assert len(result["raw_label_overlaps"]) == 3
    assert sum(r["source_observation_count"] for r in result["labels"]) == 8
    assert sum(r["source_observation_count"] for r in result["label_import_counts"]) == 8
    assert sum(r["source_observation_count"] for r in result["label_timestamp_bounds"]) == 8
    assert_frame_equal(before, obs)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["duplicate", "audit_gap", "current_mismatch", "orphan"])
def test_inventory_rejects_invalid_lineage(case):
    obs, current, tables, registry = fixtures()
    if case == "duplicate":
        obs.loc[1, "_ingest_observation_id"] = "o0"
    elif case == "audit_gap":
        tables["source_audit"] = tables["source_audit"].iloc[1:]
    elif case == "current_mismatch":
        current.loc[0, "saved_filter_names"] = "changed"
    else:
        tables["source_audit"].loc[0, "bet_id"] = "orphan"
    with pytest.raises(ValueError):
        build_label_inventory(obs, current, tables, registry)


def test_empty_inventory():
    obs, current, tables, registry = fixtures()
    result = build_label_inventory(
        obs.iloc[:0], current.iloc[:0], {k: v.iloc[:0] for k, v in tables.items()}, registry
    )
    assert result["summary"]["source_observations"] == 0
    assert result["labels"] == []


def source_pair(tmp_path):
    bundle = canonical(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text())
    return tmp_path / "store" / "snapshots" / manifest["specification"]["ingestion_manifest"][
        "snapshot_id"
    ], bundle


def test_publication_replay_cli_and_source_immutability(tmp_path):
    snapshot, bundle = source_pair(tmp_path)
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for base in (snapshot, bundle)
        for p in base.iterdir()
        if p.is_file()
    }
    destination = publish_label_inventory(snapshot, bundle, tmp_path / "inventory")
    assert publish_label_inventory(snapshot, bundle, tmp_path / "inventory") == destination
    assert load_label_inventory(destination)["summary"]["canonical_wager_candidates"] > 0
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/inventory_filter_labels.py"),
        "--snapshot",
        str(snapshot),
        "--canonical",
        str(bundle),
        "--output-root",
        str(tmp_path / "inventory"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"raw_label":' not in result.stdout
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for base in (snapshot, bundle)
        for p in base.iterdir()
        if p.is_file()
    }
    (destination / "inventory.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_label_inventory(destination)


def test_wrong_snapshot_corruption_and_source_output_are_rejected(tmp_path):
    snapshot, bundle = source_pair(tmp_path)
    with pytest.raises(ValueError, match="source bundle"):
        publish_label_inventory(snapshot, bundle, snapshot / "output")
    # Keep the snapshot internally valid but change provenance, not archived data.
    from oddsjam_ev.io.source_csv import fingerprint

    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["specification"]["inventory_test_marker"] = "different snapshot"
    manifest["snapshot_id"] = fingerprint(manifest["specification"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact source snapshot"):
        publish_label_inventory(snapshot, bundle, tmp_path / "output")
    (snapshot / "current.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        publish_label_inventory(snapshot, bundle, tmp_path / "output")
