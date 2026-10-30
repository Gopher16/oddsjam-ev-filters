"""Confirmed aliases preserve old IDs and economic tables; tentative names stay unknown."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_label_inventory import source_pair

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.filter_attribution import resolve_labels, validate_registry
from oddsjam_ev.io.attribution_audit import publish_attribution_audit, reconcile_tables
from oddsjam_ev.io.canonical import build_canonical_bundle

ROOT = Path(__file__).resolve().parents[1]


def revision():
    return yaml.safe_load(
        (ROOT / "configs/filter_history/oddsjam_filter_configurations_2026-10-30.yaml").read_text()
    )


def test_reviewed_revision_preserves_ids_and_only_confirmed_aliases():
    old = yaml.safe_load(
        (ROOT / "configs/filter_history/oddsjam_filter_configurations_2026-10-22.yaml").read_text()
    )
    new = revision()
    assert {r["id"] for r in old["filters"]} <= {r["id"] for r in new["filters"]}
    mapping = validate_registry(new)
    assert (
        mapping["PROD - DFS Platforms"] == mapping["TEST - DFS Platforms"] == "test_dfs_platforms"
    )
    for relation in new["identity_review"]["confirmed_relationships"]:
        assert {mapping[name] for name in relation["names"]} == {relation["attribution_target_id"]}
    for raw in new["identity_review"]["unresolved"]:
        assert resolve_labels(raw, mapping) == ([], "unresolved")
    for row in new["filters"]:
        if row["id"].startswith("historical_label_"):
            assert row["settings"] is None
    assert new["identity_review"]["private_evidence_sha256"]


@pytest.mark.parametrize("targets", [{"a": "missing"}, {"a": "b", "b": "a"}, {"a": None}])
def test_invalid_redirects(targets):
    reg = {"schema_version": 1, "filters": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]}
    for row in reg["filters"]:
        if row["id"] in targets:
            row["attribution_target_id"] = targets[row["id"]]
    with pytest.raises(ValueError):
        validate_registry(reg)


def test_redirect_chain_and_conflicting_names():
    reg = {
        "schema_version": 1,
        "filters": [
            {"id": "a", "name": "A", "attribution_target_id": "b"},
            {"id": "b", "name": "B", "attribution_target_id": "c"},
            {"id": "c", "name": "C"},
        ],
    }
    assert set(validate_registry(reg).values()) == {"c"}
    reg["filters"].append({"id": "d", "name": "A"})
    with pytest.raises(ValueError):
        validate_registry(reg)


def test_reconciliation_rejects_economic_change_and_counts_memberships_once():
    old = {
        "bet_facts": pd.DataFrame({"bet_id": ["a"], "stake": [100]}),
        "filter_attribution": pd.DataFrame(
            {"bet_id": ["a"], "filter_id": [None], "attribution_status": ["unresolved"]}
        ),
    }
    new = copy.deepcopy(old)
    new["filter_attribution"] = pd.DataFrame(
        {
            "bet_id": ["a", "a"],
            "filter_id": ["x", "y"],
            "attribution_status": ["resolved", "resolved"],
        }
    )
    result = reconcile_tables(old, new)
    assert result["coverage"]["resolved"]["after_distinct_candidates"] == 1
    assert result["membership_pairs_added"] == 2
    new["bet_facts"].loc[0, "stake"] = 101
    with pytest.raises(ValueError, match="bet_facts"):
        reconcile_tables(old, new)


def test_versioned_build_audit_replay_and_old_bytes(tmp_path):
    snapshot, before = source_pair(tmp_path)
    cm = json.loads((before / "manifest.json").read_text())
    reg = cm["specification"]["filter_registry"]
    reg["filters"][0].setdefault("former_names", []).append("Previously unknown alias")
    reg["identity_review"] = {"source": "synthetic decision", "evidence_sha256": "synthetic"}
    old_bytes = {
        p: p.read_bytes() for base in (snapshot, before) for p in base.iterdir() if p.is_file()
    }
    after = build_canonical_bundle(
        snapshot,
        tmp_path / "revised",
        CanonicalConfig(**cm["specification"]["economic_policy"]),
        reg,
    )
    audit = publish_attribution_audit(before, after, tmp_path / "audits")
    assert publish_attribution_audit(before, after, tmp_path / "audits") == audit
    result = json.loads((audit / "audit.json").read_text())
    assert len(result["unchanged_tables"]) == 8
    assert all(p.read_bytes() == value for p, value in old_bytes.items())
    with pytest.raises(ValueError, match="source bundle"):
        publish_attribution_audit(before, after, after / "audit")
