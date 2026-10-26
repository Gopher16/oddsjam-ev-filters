"""Mapping proposals preserve exact evidence without approving inferred identities."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_label_inventory import fixtures, source_pair

from oddsjam_ev.analysis.filter_mapping_proposal import build_mapping_proposal
from oddsjam_ev.analysis.label_inventory import build_label_inventory
from oddsjam_ev.io.filter_mapping_proposal import load_mapping_proposal, publish_mapping_proposal
from oddsjam_ev.io.label_inventory import publish_label_inventory
from oddsjam_ev.io.source_csv import fingerprint


def evidence():
    obs, current, tables, registry = fixtures()
    return build_label_inventory(obs, current, tables, registry), registry


def add_label(data, registry, raw):
    from oddsjam_ev.analysis.filter_attribution import resolve_labels, validate_registry

    row = copy.deepcopy(data["labels"][0])
    row.update(label_id=fingerprint({"raw_label": raw}), raw_label=raw)
    ids, status = resolve_labels(raw or "", validate_registry(registry))
    row.update(resolved_filter_ids=ids, registry_resolution_status=status)
    data["labels"].append(row)


def test_exact_alias_segmentation_unknown_and_clone_evidence():
    data, registry = evidence()
    registry["filters"][0]["former_names"] = ["Old A"]
    registry["filters"][1]["clone_of"] = "a"
    add_label(data, registry, "Old A")
    add_label(data, registry, "TEST - B")
    add_label(data, registry, "Unknown λ, B")
    before = copy.deepcopy((data, registry))
    result = build_mapping_proposal(data, registry)
    rows = {r["raw_label"]: r for r in result["proposals"]}
    assert rows["Old A"]["proposed_filter_ids"] == ["a"]
    assert rows["Old A"]["proposed_display_name"] == "A (Slow, Build)"
    assert rows["A (Slow, Build), B"]["proposed_filter_ids"] == ["a", "b"]
    assert rows["Unknown λ"]["proposed_filter_ids"] == [
        "historical_label_" + fingerprint({"raw_label": "Unknown λ"})
    ]
    assert rows["TEST - B"]["relationship_candidates"][0]["filter_id"] == "b"
    assert rows["TEST - B"]["approval_status"] == "not_approved"
    assert set(rows["Unknown λ, B"]["segmentation_candidate_ids"]) == {
        "b",
        rows["Unknown λ"]["proposed_filter_ids"][0],
    }
    assert rows[""]["proposed_filter_ids"] == []
    assert rows["  "]["status"] == "unassigned"
    assert all(not r["proposed_former_names"] for r in rows.values())
    assert result["registered_identities"][1]["registered_clone_of"] == "a"
    assert (data, registry) == before
    shuffled = copy.deepcopy(data)
    shuffled["labels"].reverse()
    assert build_mapping_proposal(shuffled, registry) == result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["duplicate", "wrong_label_id", "wrong_resolution", "collision"])
def test_invalid_evidence_rejected(case):
    data, registry = evidence()
    if case == "duplicate":
        data["labels"].append(data["labels"][0])
    elif case == "wrong_label_id":
        data["labels"][0]["label_id"] = "bad"
    elif case == "wrong_resolution":
        data["labels"][0]["resolved_filter_ids"] = ["invented"]
    else:
        registry["filters"].append(
            {"id": "historical_label_" + fingerprint({"raw_label": "Unknown λ"}), "name": "Other"}
        )
    with pytest.raises(ValueError):
        build_mapping_proposal(data, registry)


def test_empty_inventory():
    data, registry = evidence()
    data["labels"] = []
    assert build_mapping_proposal(data, registry)["proposals"] == []


def test_publication_cli_replay_integrity_and_source_preservation(tmp_path):
    snapshot, canonical = source_pair(tmp_path)
    inventory = publish_label_inventory(snapshot, canonical, tmp_path / "inventory")
    before = {
        p: p.read_bytes()
        for base in (inventory, canonical, snapshot)
        for p in base.iterdir()
        if p.is_file()
    }
    output = publish_mapping_proposal(inventory, canonical, tmp_path / "proposals")
    assert publish_mapping_proposal(inventory, canonical, tmp_path / "proposals") == output
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/propose_filter_mappings.py"),
            "--inventory",
            str(inventory),
            "--canonical",
            str(canonical),
            "--output-root",
            str(tmp_path / "proposals"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert all(p.read_bytes() == content for p, content in before.items())
    assert load_mapping_proposal(output)["proposal_contract_version"] == 1
    with pytest.raises(ValueError, match="source bundle"):
        publish_mapping_proposal(inventory, canonical, inventory / "output")
    (output / "proposal.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_mapping_proposal(output)


def test_wrong_canonical_and_changed_registry_rejected(tmp_path):
    snapshot, canonical = source_pair(tmp_path)
    inventory = publish_label_inventory(snapshot, canonical, tmp_path / "inventory")
    path = inventory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["specification"]["registry_sha256"] = "different"
    manifest["inventory_id"] = fingerprint(manifest["specification"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        publish_mapping_proposal(inventory, canonical, tmp_path / "proposals")
