"""Explicit version-bound review transitions, immutable history and local locking."""

import json

import pytest
from test_recommendation_bundle import proposal_bundle

from oddsjam_ev.io.decision_register import read_decisions, record_decision
from oddsjam_ev.io.recommendations import load_recommendations


def test_review_approval_and_stale_events(tmp_path):
    out, *_ = proposal_bundle(tmp_path)
    proposal = load_recommendations(out)["proposals"].query("action == 'COLLECT_MORE_DATA'").iloc[0]
    root = tmp_path / "decisions"
    with pytest.raises(ValueError, match="transition"):
        record_decision(
            root,
            out,
            proposal.proposal_id,
            "APPROVED",
            "synthetic reviewer",
            "synthetic rationale",
            "PROPOSED",
        )
    first = record_decision(
        root,
        out,
        proposal.proposal_id,
        "UNDER_REVIEW",
        "synthetic reviewer",
        "Review evidence requests only",
        "PROPOSED",
    )
    with pytest.raises(ValueError, match="Stale"):
        record_decision(
            root,
            out,
            proposal.proposal_id,
            "APPROVED",
            "synthetic reviewer",
            "Review only",
            "PROPOSED",
        )
    second = record_decision(
        root,
        out,
        proposal.proposal_id,
        "APPROVED",
        "synthetic reviewer",
        "Approve research follow-up only",
        "UNDER_REVIEW",
        first["event_id"],
    )
    assert not second["external_execution_performed"]
    assert second["approval_scope"] == "research_review_only"
    assert read_decisions(root) == [first, second]
    with pytest.raises(ValueError, match="transition"):
        record_decision(
            root,
            out,
            proposal.proposal_id,
            "REJECTED",
            "synthetic reviewer",
            "Cannot rewrite approved history",
            "APPROVED",
            second["event_id"],
        )
    record_decision(
        root,
        out,
        proposal.proposal_id,
        "SUPERSEDED",
        "synthetic reviewer",
        "Supersede this version only",
        "APPROVED",
        second["event_id"],
    )
    assert len(read_decisions(root)) == 3


def test_actor_lock_unknown_proposals_and_corruption(tmp_path):
    out, *_ = proposal_bundle(tmp_path)
    p = load_recommendations(out)["proposals"].iloc[0].proposal_id
    root = tmp_path / "decisions"
    with pytest.raises(ValueError, match="Reviewer"):
        record_decision(root, out, p, "UNDER_REVIEW", "", "reason", "PROPOSED")
    with pytest.raises(ValueError, match="not found"):
        record_decision(
            root, out, "unknown", "UNDER_REVIEW", "synthetic reviewer", "reason", "PROPOSED"
        )
    root.mkdir()
    (root / ".writer.lock").write_text("synthetic lock")
    with pytest.raises(ValueError, match="locked"):
        record_decision(root, out, p, "UNDER_REVIEW", "synthetic reviewer", "reason", "PROPOSED")
    (root / ".writer.lock").unlink()
    event = record_decision(
        root, out, p, "UNDER_REVIEW", "synthetic reviewer", "reason", "PROPOSED"
    )
    path = root / "00000001.json"
    event["rationale"] = "changed without new event"
    path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="integrity"):
        read_decisions(root)


def test_changed_bundle_cannot_inherit_approval(tmp_path):
    from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
    from oddsjam_ev.io.recommendations import publish_recommendations

    out, d, a, c, r, cap = proposal_bundle(tmp_path)
    root = tmp_path / "decisions"
    p = load_recommendations(out)["proposals"].iloc[0].proposal_id
    first = record_decision(
        root, out, p, "UNDER_REVIEW", "synthetic reviewer", "original version", "PROPOSED"
    )
    new = publish_recommendations(
        d,
        a,
        c,
        tmp_path / "recommendations",
        RecommendationConfig(draft_paper_hypotheses=False),
        r,
        cap,
    )
    new_id = load_recommendations(new)["proposals"].iloc[0].proposal_id
    with pytest.raises(ValueError, match="not found"):
        record_decision(
            root,
            new,
            p,
            "APPROVED",
            "synthetic reviewer",
            "old approval",
            "UNDER_REVIEW",
            first["event_id"],
        )
    with pytest.raises(ValueError, match="Stale"):
        record_decision(
            root,
            new,
            new_id,
            "APPROVED",
            "synthetic reviewer",
            "old state",
            "UNDER_REVIEW",
            first["event_id"],
        )
    assert len(read_decisions(root)) == 1


def test_incomplete_paper_protocol_cannot_be_approved(tmp_path):
    from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
    from oddsjam_ev.io.recommendations import publish_recommendations

    _, d, a, c, r, _ = proposal_bundle(tmp_path)
    cap = {
        "schema_version": 1,
        "exact_values": {
            "sportsbook": {
                "Book": {
                    "ui_control": "books",
                    "ui_value": ["Book"],
                    "semantics_verified": True,
                    "evidence_reference": "synthetic fixture",
                }
            }
        },
    }
    out = publish_recommendations(
        d, a, c, tmp_path / "recommendations", RecommendationConfig(), r, cap
    )
    p = load_recommendations(out)["proposals"].query("action == 'CREATE_TEST'").iloc[0]
    root = tmp_path / "decisions"
    first = record_decision(
        root,
        out,
        p.proposal_id,
        "UNDER_REVIEW",
        "synthetic reviewer",
        "Review draft paper protocol",
        "PROPOSED",
    )
    with pytest.raises(ValueError, match="incomplete"):
        record_decision(
            root,
            out,
            p.proposal_id,
            "APPROVED",
            "synthetic reviewer",
            "Incomplete protocol",
            "UNDER_REVIEW",
            first["event_id"],
        )
    assert len(read_decisions(root)) == 1


def test_decision_atomic_failure_and_cli(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    import oddsjam_ev.io.decision_register as module

    out, *_ = proposal_bundle(tmp_path)
    p = load_recommendations(out)["proposals"].iloc[0].proposal_id
    root = tmp_path / "decisions"
    original = module.os.link

    def fail(*args, **kwargs):
        raise OSError("synthetic link failure")

    monkeypatch.setattr(module.os, "link", fail)
    with pytest.raises(OSError):
        record_decision(root, out, p, "UNDER_REVIEW", "synthetic reviewer", "reason", "PROPOSED")
    assert read_decisions(root) == []
    assert not list(root.iterdir())
    monkeypatch.setattr(module.os, "link", original)
    script = Path(__file__).resolve().parents[1] / "scripts/record_research_decision.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--register",
            str(root),
            "record",
            "--bundle",
            str(out),
            "--proposal-id",
            p,
            "--to-status",
            "UNDER_REVIEW",
            "--reviewer",
            "synthetic reviewer",
            "--rationale",
            "Synthetic explicit review only",
            "--expected-status",
            "PROPOSED",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    inspected = subprocess.run(
        [sys.executable, str(script), "--register", str(root), "inspect"],
        capture_output=True,
        text=True,
    )
    assert inspected.returncode == 0
    assert json.loads(inspected.stdout)[0]["to_status"] == "UNDER_REVIEW"
