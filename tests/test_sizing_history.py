"""Sizing linkage never substitutes export chronology or presumed account settings."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical
from test_sizing_config import evidence, history, policy

from oddsjam_ev.analysis.sizing_config import validate_history
from oddsjam_ev.analysis.sizing_linkage import build_sizing_linkage
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.sizing_history import load_sizing_history, publish_sizing_history


def bets():
    return pd.DataFrame(
        {
            "bet_id": ["b1", "b2", "b3"],
            "namespace": ["synthetic"] * 3,
            "product": ["exchange"] * 3,
            "execution_status": ["real_money", "paper", "unknown"],
            "reported_stake_usd": [10.0, 20.0, 30.0],
            "created_at_utc": ["2026-01-15T00:00:00Z"] * 3,
        }
    )


def test_placement_required_half_open_and_no_applied_inference():
    data = history()
    data["policies"].append(policy("p2", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"))
    data["wager_evidence"] = [evidence(), evidence("b2", "e2", "2026-02-01T00:00:00Z")]
    validate_history(data)
    source = bets()
    before = source.copy(deep=True)
    frame, audit = build_sizing_linkage(source, data)
    assert frame.setting_policy_id.tolist() == ["p1", "p2", None]
    assert frame.applied_policy_id.isna().all()
    assert frame.linkage_status.tolist() == ["matched_setting_interval"] * 2 + ["unknown_placement"]
    assert audit["overlapping_policy_pairs"] == []
    assert frame.sizing_replay_blocked.all()
    assert_frame_equal(source, before)


def test_unknown_undated_planned_assumed_and_scoped_settings():
    data = history()
    data["wager_evidence"] = [evidence()]
    for updates in (
        {"effective_from": None, "effective_to": None, "effective_interval_verified": False},
        {"evidence_type": "planned"},
        {"confidence": "assumed"},
        {"scope": {"product": "fixed_odds", "execution_status": None}},
        {"namespace": "other"},
    ):
        current = copy.deepcopy(data)
        current["policies"][0].update(updates)
        validate_history(current)
        frame, _ = build_sizing_linkage(bets(), current)
        assert frame.setting_policy_id.isna().all()
        assert frame.applied_policy_id.isna().all()


def test_overlap_and_contradictory_placement_quarantine():
    data = history()
    data["policies"].append(policy("p2"))
    data["wager_evidence"] = [evidence()]
    frame, audit = build_sizing_linkage(bets(), data)
    assert frame.linkage_status.iloc[0] == "ambiguous_policy_interval"
    assert json.loads(frame.candidate_policy_ids_json.iloc[0]) == ["p1", "p2"]
    assert audit["overlapping_policy_pairs"] == [["p1", "p2"]]
    data["wager_evidence"].append(evidence("b1", "e2", "2026-01-16T00:00:00Z"))
    frame, _ = build_sizing_linkage(bets(), data)
    assert frame.linkage_status.iloc[0] == "ambiguous_placement"


def test_verified_application_partial_settings_and_stake_evidence():
    data = history()
    data["policies"][0]["bankroll_amount"] = None
    record = evidence()
    record.update(
        applied_policy_id="p1",
        application_verified=True,
        suggested_stake="12",
        accepted_stake="10",
        accepted_verified=True,
        deviation_reason="synthetic manual override",
    )
    data["wager_evidence"] = [record]
    validate_history(data)
    frame, audit = build_sizing_linkage(bets(), data)
    assert frame.applied_policy_id.iloc[0] == "p1"
    assert frame.bankroll_amount.iloc[0] is None
    assert frame.kelly_multiplier.iloc[0] == "0.5"
    assert frame.accepted_stake_verified.iloc[0] == "10"
    assert audit["applied_policy_count"] == 1
    bad = copy.deepcopy(data)
    bad["wager_evidence"][0]["bet_id"] = "b2"
    with pytest.raises(ValueError, match="real-money"):
        build_sizing_linkage(bets(), bad)


def test_conflicting_application_and_stake_records():
    data = history()
    data["policies"].append(policy("p2", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"))
    record = evidence()
    record.update(applied_policy_id="p2", application_verified=True, suggested_stake="10")
    second = evidence("b1", "e2")
    second["suggested_stake"] = "11"
    data["wager_evidence"] = [record, second]
    validate_history(data)
    frame, audit = build_sizing_linkage(bets(), data)
    assert frame.linkage_status.iloc[0] == "application_conflict"
    assert frame.applied_policy_id.iloc[0] is None
    assert frame.suggested_stake_recorded.iloc[0] is None
    assert audit["stake_evidence_conflicts"] == 1


def test_orphan_duplicate_empty_and_open_ended():
    data = history()
    data["wager_evidence"] = [evidence("missing")]
    with pytest.raises(ValueError, match="Orphan"):
        build_sizing_linkage(bets(), data)
    with pytest.raises(ValueError, match="Unique"):
        build_sizing_linkage(pd.concat([bets(), bets()]), history())
    frame, audit = build_sizing_linkage(bets().iloc[:0], history())
    assert frame.empty and audit["candidate_count"] == 0
    data = history()
    data["policies"][0].update(effective_to=None, open_ended=True)
    data["wager_evidence"] = [evidence()]
    validate_history(data)
    assert build_sizing_linkage(bets(), data)[0].setting_policy_id.iloc[0] == "p1"


def test_immutable_publication_revision_replay_cli_and_unknown_history(tmp_path):
    bundle = canonical(tmp_path)
    before = {p: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    root = tmp_path / "publications"
    output = publish_sizing_history(bundle, root, knowledge_as_of="2026-03-02T00:00:00Z")
    assert publish_sizing_history(bundle, root, knowledge_as_of="2026-03-02T00:00:00Z") == output
    old, frame, audit = load_sizing_history(output)
    assert frame.linkage_status.eq("unknown_placement").all()
    assert audit["candidate_count"] == len(load_canonical_bundle(bundle)["bet_facts"])
    new = history()
    new["previous_history_id"] = json.loads((output / "manifest.json").read_text())[
        "specification"
    ]["history_id"]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(new))
    revised = publish_sizing_history(bundle, root, history_path=path, previous=output)
    assert revised != output
    assert load_sizing_history(output)[0] == old
    assert all(p.read_bytes() == data for p, data in before.items())
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/build_sizing_history.py"),
            "--canonical",
            str(bundle),
            "--history",
            str(path),
            "--previous",
            str(output),
            "--output-root",
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "bankroll" not in result.stdout
    new["previous_history_id"] = None
    path.write_text(json.dumps(new))
    with pytest.raises(ValueError, match="predecessor"):
        publish_sizing_history(bundle, root, history_path=path, previous=output)
    with pytest.raises(ValueError, match="source publication"):
        publish_sizing_history(bundle, bundle / "output", knowledge_as_of="2026-03-02T00:00:00Z")
    (revised / "history.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_sizing_history(revised)
