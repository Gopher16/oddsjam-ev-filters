"""Future definitions, exact source rules, immutable revisions and frozen visibility."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_prioritization_publication import family_sources

from oddsjam_ev.analysis.paper_definitions import draft_definition, validate_definition
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.io.paper_definitions import (
    load_definition_publication,
    publish_definitions,
    read_definitions,
    register_definition,
)
from oddsjam_ev.io.signal_prioritization import publish_prioritization
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery


def complete(candidate):
    draft = draft_definition(candidate)
    p = draft["protocol"]
    p.update(
        name="Synthetic candidate",
        hypothesis="Prospective descriptive comparison",
        planned_start_utc="2030-01-02T00:00:00Z",
        planned_end_utc="2030-02-02T00:00:00Z",
        review_dates_utc=["2030-02-04T00:00:00Z"],
        settlement_followup_days=2,
        review_criteria={
            "continue": "Review both measures and availability; new version only",
            "stop": "Missing quote evidence or configuration deviation",
            "inconclusive": "Insufficient predeclared arm support",
        },
        stop_rules=["Pause capture on baseline or rule mismatch"],
        overlap_policy="Shared events recorded once per arm and reported across tests",
        quote_availability_policy="Record decision-time UTC selection, line, odds, book, event ID and failed or unknown availability checks; never assume a fill",
    )
    p["baseline"] = {
        "kind": "unchanged_parent",
        "filter_config": {"book": ["Book"]},
        "evidence_reference": "synthetic contemporaneous baseline settings",
    }
    p["sizing"]["stake_usd"] = 1.0
    return draft


@pytest.fixture
def definition_sources(tmp_path, monkeypatch):
    family, diagnostics = family_sources(tmp_path)
    priorities = publish_prioritization(
        family, diagnostics, tmp_path / "priorities", PrioritizationConfig()
    )
    candidate = load_within_filter_discovery(family)["signal_candidates"].iloc[0].to_dict()
    import oddsjam_ev.io.paper_definitions as module

    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T00:00:00Z")
    return family, priorities, candidate, complete(candidate), tmp_path / "register"


def save(inputs, **kwargs):
    family, priority, candidate, document, register = inputs
    return register_definition(
        register,
        family,
        priority,
        candidate["candidate_id"],
        "synthetic-test",
        document["protocol"],
        "synthetic actor",
        "Explicit exploratory intent despite no shortlist",
        **kwargs,
    )


def test_immutable_versions_asof_and_published_replay(definition_sources, tmp_path, monkeypatch):
    import oddsjam_ev.io.paper_definitions as module

    family, priority, candidate, document, register = definition_sources
    before = {
        p: p.read_bytes() for base in (family, priority) for p in base.iterdir() if p.is_file()
    }
    first = save(definition_sources)
    original = (register / "00000001.json").read_bytes()
    assert first["status"] == "definition_only_not_launched"
    assert not first["external_execution_performed"]
    assert first["source"]["exact_predicate"] == json.loads(candidate["predicate_json"])
    output = publish_definitions(register, tmp_path / "published", "2030-01-01T00:00:00Z")
    assert publish_definitions(register, tmp_path / "published", "2030-01-01T00:00:00Z") == output
    with pytest.raises(ValueError, match="Stale"):
        save(definition_sources)
    with pytest.raises(ValueError, match="No material"):
        save(definition_sources, expected_version_id=first["version_id"])
    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T01:00:00Z")
    document["protocol"]["sizing"]["stake_usd"] = 2
    second = save(definition_sources, expected_version_id=first["version_id"])
    assert second["version"] == 2 and second["previous_version_id"] == first["version_id"]
    assert (register / "00000001.json").read_bytes() == original
    assert len(read_definitions(register)) == 2
    assert len(read_definitions(register, "2030-01-01T00:30:00Z")) == 1
    assert load_definition_publication(output)["events"] == [first]
    assert before == {p: p.read_bytes() for p in before}
    (output / "definitions.json").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_definition_publication(output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("planned_start_utc", "2020-01-01T00:00:00Z"),
        ("planned_start_utc", "2030-01-02"),
        ("funding", "real_money"),
        ("max_funded_stake_usd", 1),
        ("min_wagers_per_arm", True),
        ("primary_metric", "unplanned_metric"),
        ("review_dates_utc", []),
        ("stop_rules", []),
        ("review_criteria", {"continue": "Keep looking"}),
    ],
)
def test_invalid_intent_never_registers(definition_sources, field, value):
    definition_sources[3]["protocol"][field] = value
    with pytest.raises(ValueError):
        save(definition_sources)
    assert not list(definition_sources[4].glob("*.json"))


def test_invalid_sizing_draft_and_tampering(definition_sources):
    family, priority, candidate, document, register = definition_sources
    with pytest.raises(ValueError):
        validate_definition(draft_definition(candidate)["protocol"])
    for stake in (0, float("nan"), float("inf"), True):
        changed = copy.deepcopy(document["protocol"])
        changed["sizing"]["stake_usd"] = stake
        with pytest.raises(ValueError):
            validate_definition(changed)
    first = save(definition_sources)
    first["protocol"]["name"] = "changed"
    (register / "00000001.json").write_text(json.dumps(first))
    with pytest.raises(ValueError, match="integrity"):
        read_definitions(register)


def test_source_mismatch_empty_register_and_cli(definition_sources, tmp_path):
    family, priority, candidate, document, register = definition_sources
    cli = Path(__file__).resolve().parents[1] / "scripts/manage_paper_definitions.py"
    draft = subprocess.run(
        [
            sys.executable,
            str(cli),
            "draft",
            "--discovery",
            str(family),
            "--candidate-id",
            candidate["candidate_id"],
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert draft.returncode == 0, draft.stderr
    assert "planned_start_utc: null" in draft.stdout
    empty = publish_definitions(
        tmp_path / "empty-register", tmp_path / "empty-publications", "2030-01-01T00:00:00Z"
    )
    assert load_definition_publication(empty)["events"] == []
    manifest = priority / "manifest.json"
    data = json.loads(manifest.read_text())
    data["specification"]["discovery_manifest_sha256"] = "wrong"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        save(definition_sources)
    assert not list(register.glob("*.json"))


def test_cli_save_rule_guard_and_version_conflict(definition_sources, tmp_path):
    import yaml

    family, priority, candidate, document, register = definition_sources
    cli = Path(__file__).resolve().parents[1] / "scripts/manage_paper_definitions.py"
    file = tmp_path / "definition.yaml"
    file.write_text(yaml.safe_dump(document))
    args = [
        sys.executable,
        str(cli),
        "--register",
        str(register),
        "save",
        "--discovery",
        str(family),
        "--prioritization",
        str(priority),
        "--definition",
        str(file),
        "--test-key",
        "cli-test",
        "--actor",
        "synthetic",
        "--rationale",
        "Explicit paper intent",
    ]
    result = subprocess.run(args, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "definition_only_not_launched"
    assert subprocess.run(args, text=True, capture_output=True, timeout=30).returncode == 2
    document["exact_predicate_json"] = "[]"
    file.write_text(yaml.safe_dump(document))
    assert subprocess.run(args, text=True, capture_output=True, timeout=30).returncode == 2
    assert len(read_definitions(register)) == 1


def test_review_schedule_boundaries_and_parent_design(definition_sources):
    protocol = definition_sources[3]["protocol"]
    for dates in (
        ["2030-02-03T00:00:00Z"],
        ["2030-02-04T00:00:00Z"] * 2,
        ["2030-02-04T00:00:00Z", "2030-01-03T00:00:00Z"],
    ):
        changed = copy.deepcopy(protocol)
        changed["review_dates_utc"] = dates
        with pytest.raises(ValueError):
            validate_definition(changed)
    protocol["design"] = "exclusion_comparison"
    registered = save(definition_sources)
    assert registered["source"]["trial_arm"] == "parent_minus_subgroup"
    assert registered["source"]["baseline_arm"] == "unchanged_parent"
