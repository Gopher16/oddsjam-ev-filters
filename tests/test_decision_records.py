"""Complete settings, baseline intersection and lifecycle evidence never imply approval."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_paper_definitions import complete
from test_prioritization_publication import family_sources

from oddsjam_ev.analysis.decision_config import CONTROLS, normalize_settings, validate_evidence
from oddsjam_ev.analysis.decision_records import resolve_decision
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.io.decision_records import load_decisions, publish_decisions
from oddsjam_ev.io.paper_definitions import register_definition
from oddsjam_ev.io.signal_prioritization import publish_prioritization
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery

STAMP = "2030-01-01T00:00:00Z"


def value(v, state="observed"):
    return {
        "state": state,
        "value": v,
        "reason": "Synthetic explicit evidence",
        "basis": "prospective_design",
    }


def configuration(candidate):
    # No omission silently becomes default: irrelevant controls explicitly confirmed.
    settings = {k: value(None, "unset") for k in CONTROLS}
    settings.update(
        sportsbooks=value(["Book"]),
        paper_sizing=value(
            {"method": "fixed_paper_stake", "stake_usd": 1.0, "applies_to": "both_arms"}
        ),
        min_ev=value({"value": 2.5, "operator": ">=", "units": "percentage_points"}),
    )
    baseline = {
        "baseline_id": "synthetic-parent-v1",
        "filter_id": candidate["filter_id"],
        "kind": "prospective_parent",
        "recorded_at": STAMP,
        "effective_from": STAMP,
        "effective_until": None,
        "evidence_reference": "Synthetic current settings capture",
        "settings": settings,
    }
    translations = []
    for term in json.loads(candidate["predicate_json"]):
        if term["dimension"] == "sportsbook":
            translations.append(
                {
                    "term": term,
                    "control": "sportsbooks",
                    "value": [term["value"]],
                    "recorded_at": STAMP,
                    "evidence_reference": "Synthetic exact UI mapping",
                    "platform_version": "synthetic-ui-v1",
                    "semantics_verified": True,
                }
            )
    return {
        "contract_version": 1,
        "baselines": [baseline],
        "lifecycle": [],
        "translations": translations,
    }


def candidate():
    return {
        "candidate_id": "candidate:synthetic",
        "parent_id": "parent:synthetic",
        "filter_id": "PROD - Synthetic",
        "execution_status": "paper",
        "product": "exchange",
        "predicate_json": json.dumps(
            [{"dimension": "sportsbook", "operator": "==", "value": "Book"}]
        ),
    }


def resolve(c, config, design="subgroup_tracking", as_of=STAMP, effective_at=STAMP):
    return resolve_decision(c, validate_evidence(config), design, as_of, effective_at)


def test_complete_catalog_unknown_unset_and_inheritance():
    c = candidate()
    empty = {"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []}
    record = resolve(c, empty)
    assert len(record["settings"]) == len(CONTROLS)
    assert all(r["state"] == "unknown" for r in record["settings"])
    assert record["setup_status"] == "blocked"
    assert record["lifecycle"]["status"] == "unknown"  # PROD prefix is not evidence.
    config = configuration(c)
    record = resolve(c, config)
    assert record["setup_status"] == "ready_for_operator_configuration_review"
    row = next(r for r in record["settings"] if r["control"] == "minimum_liquidity")
    assert (
        row["state"] == "inherited"
        and row["baseline_state"] == "unset"
        and row["final_value"] is None
    )
    assert not record["production_change_authorized"]
    assert record["baseline"]["baseline_sha256"]
    config["baselines"][0]["settings"].pop("market_width")
    assert "market_width:setting_unknown" in resolve(c, config)["setup_blockers"]


def test_numeric_bounds_keep_strictness_and_do_not_weaken_parent():
    c = candidate()
    terms = [
        {
            "dimension": "advertised_ev_percent",
            "operator": ">",
            "endpoint": "3",
            "units": "percentage_points",
        }
    ]
    c["predicate_json"] = json.dumps(terms)
    config = configuration(c)
    config["translations"] = [
        {
            "term": terms[0],
            "control": "min_ev",
            "value": {"value": 3, "operator": ">=", "units": "percentage_points"},
            "recorded_at": STAMP,
            "evidence_reference": "UI includes boundary",
            "platform_version": "v1",
            "semantics_verified": True,
        }
    ]
    assert "min_ev:exact_scanner_translation_required" in resolve(c, config)["setup_blockers"]
    config["translations"][0]["value"]["operator"] = ">"
    assert (
        next(r for r in resolve(c, config)["settings"] if r["control"] == "min_ev")["final_value"][
            "operator"
        ]
        == ">"
    )
    config["baselines"][0]["settings"]["min_ev"]["value"]["value"] = 5
    assert (
        next(r for r in resolve(c, config)["settings"] if r["control"] == "min_ev")["final_value"][
            "value"
        ]
        == 5
    )


def test_lifecycle_half_open_known_time_conflicts_and_exclusion():
    c = candidate()
    config = configuration(c)
    lifecycle = {
        "filter_id": c["filter_id"],
        "status": "PROD",
        "recorded_at": STAMP,
        "effective_from": STAMP,
        "effective_until": "2030-01-02T00:00:00Z",
        "evidence_reference": "Synthetic operator confirmation",
        "actor": "Synthetic actor",
        "confirmation": "operator_confirmed",
    }
    config["lifecycle"] = [lifecycle]
    record = resolve(c, config, "exclusion_comparison")
    assert record["lifecycle"]["status"] == "PROD"
    assert record["setup_status"] == "blocked"
    assert record["logical_change"]["operator"] == "NOT"
    assert "compound_exclusion_requires_verified_implementation" in record["setup_blockers"]
    assert "production_lifecycle_not_confirmed" not in record["setup_blockers"]
    assert (
        resolve(c, config, as_of="2030-01-02T00:00:00Z", effective_at="2030-01-02T00:00:00Z")[
            "lifecycle"
        ]["status"]
        == "unknown"
    )
    late = copy.deepcopy(lifecycle)
    late["recorded_at"] = "2030-01-02T00:00:00Z"
    late["status"] = "TEST"
    config["lifecycle"].append(late)
    assert resolve(c, config)["lifecycle"]["status"] == "PROD"
    assert (
        resolve(c, config, as_of="2030-01-02T00:00:00Z")["lifecycle"]["confirmation_status"]
        == "conflicting"
    )


def test_baseline_ambiguity_scope_and_universe_conflicts():
    c = candidate()
    config = configuration(c)
    config["baselines"][0]["settings"]["sportsbooks"] = value(["Other Book"])
    assert "sportsbooks:candidate_baseline_conflict" in resolve(c, config)["setup_blockers"]
    config = configuration(c)
    c["parent_id"] = None
    assert "baseline_scope_mismatch" in resolve(c, config)["setup_blockers"]
    config["baselines"][0]["kind"] = "new_prospective"
    assert resolve(c, config)["setup_status"] == "ready_for_operator_configuration_review"
    config["baselines"].append({**copy.deepcopy(config["baselines"][0]), "baseline_id": "another"})
    assert "ambiguous_prospective_baselines" in resolve(c, config)["setup_blockers"]


@pytest.mark.parametrize(
    "control,item",
    [
        ("minimum_liquidity", value(0, "unset")),
        ("min_ev", value({"value": 3, "operator": ">", "units": "fraction"})),
        ("kelly_multiplier", value(True)),
        ("market_width", value(float("inf"))),
        ("devig_method", value("default", "platform_default")),
        ("sportsbooks", value(["Book", "Book"])),
    ],
)
def test_invalid_settings(control, item):
    with pytest.raises(ValueError):
        normalize_settings({control: item}, "synthetic", STAMP)


@pytest.fixture
def published_sources(tmp_path):
    family, diagnostics = family_sources(tmp_path)
    priority = publish_prioritization(
        family, diagnostics, tmp_path / "priorities", PrioritizationConfig()
    )
    candidates = load_within_filter_discovery(family)["signal_candidates"]
    c = candidates[candidates.predicate_json.str.contains("sportsbook")].iloc[0].to_dict()
    return family, priority, c


def test_publication_replay_binding_and_tampering(published_sources, tmp_path, monkeypatch):
    family, priority, c = published_sources
    config = configuration(c)
    before = {
        p: p.read_bytes() for base in (family, priority) for p in base.iterdir() if p.is_file()
    }
    root = tmp_path / "decisions"
    output = publish_decisions(family, priority, config, root, STAMP, STAMP)
    assert publish_decisions(family, priority, config, root, STAMP, STAMP) == output
    publication = load_decisions(output)
    assert len(publication["records"]) == 4
    record = next(
        r
        for r in publication["records"]
        if r["candidate_id"] == c["candidate_id"] and r["design"] == "subgroup_tracking"
    )
    assert record["setup_status"] == "ready_for_operator_configuration_review"
    protocol = complete(c)["protocol"]
    protocol["baseline"]["filter_config"] = {
        r["control"]: r["baseline_value"] for r in record["settings"]
    }
    protocol["baseline"]["evidence_reference"] = record["baseline"]["evidence_reference"]
    protocol["configuration_reference"] = {
        "contract_version": 1,
        "publication_id": output.name,
        "decision_id": record["decision_id"],
        "record_sha256": record["record_sha256"],
    }
    import oddsjam_ev.io.paper_definitions as module

    monkeypatch.setattr(module, "utc_now", lambda: STAMP)
    kwargs = dict(
        root=tmp_path / "paper",
        discovery=family,
        prioritization=priority,
        candidate_id=c["candidate_id"],
        test_key="bound-test",
        protocol=protocol,
        actor="synthetic",
        rationale="paper only",
    )
    with pytest.raises(ValueError, match="exact decision publication"):
        register_definition(**kwargs)
    saved = register_definition(**kwargs, decision_publication=output)
    assert saved["protocol"]["configuration_reference"] == protocol["configuration_reference"]
    protocol["sizing"]["stake_usd"] = 2
    with pytest.raises(ValueError, match="sizing differs"):
        register_definition(
            **kwargs, decision_publication=output, expected_version_id=saved["version_id"]
        )
    assert before == {p: p.read_bytes() for p in before}
    (output / "decisions.json").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_decisions(output)


def test_empty_evidence_cli_and_source_protection(published_sources, tmp_path):
    family, priority, c = published_sources
    file = tmp_path / "evidence.json"
    file.write_text(
        json.dumps({"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []})
    )
    cli = Path(__file__).resolve().parents[1] / "scripts/build_decision_records.py"
    result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--discovery",
            str(family),
            "--prioritization",
            str(priority),
            "--evidence",
            str(file),
            "--as-of",
            STAMP,
            "--effective-at",
            STAMP,
            "--output-root",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert load_decisions(Path(result.stdout.strip()))["audit"]["setup_ready_count"] == 0
    with pytest.raises(ValueError, match="source publications"):
        publish_decisions(
            family, priority, json.loads(file.read_text()), family / "new", STAMP, STAMP
        )


@pytest.mark.parametrize(
    "state", ["unknown", "unset", "unrestricted", "not_applicable", "platform_default", "observed"]
)
def test_state_roundtrip_false_and_default_provenance(state):
    item = value(False if state in {"observed", "platform_default"} else None, state)
    if state == "platform_default":
        item["platform_version"] = "synthetic-v2"
    normalized = normalize_settings({"recommended_filters": item}, "synthetic", STAMP)
    assert normalized["recommended_filters"]["state"] == state
    assert normalized["recommended_filters"]["value"] is item["value"]


def test_band_endpoints_and_unsupported_dimensions_stay_blocked():
    c = candidate()
    term = {"dimension": "odds_band", "operator": "==", "value": "[100.0, 150.0)"}
    c["predicate_json"] = json.dumps([term])
    config = configuration(c)
    config["translations"] = []
    for key, operator, number in (("min_odds", ">=", 100), ("max_odds", "<", 150)):
        config["translations"].append(
            {
                "term": term,
                "control": key,
                "value": {"value": number, "operator": operator, "units": "american_odds"},
                "recorded_at": STAMP,
                "evidence_reference": "Exact synthetic band UI mapping",
                "platform_version": "v1",
                "semantics_verified": True,
            }
        )
    sheet = resolve(c, config)
    assert sheet["setup_status"] == "ready_for_operator_configuration_review"
    assert (
        next(r for r in sheet["settings"] if r["control"] == "max_odds")["final_value"]["operator"]
        == "<"
    )
    c["predicate_json"] = json.dumps(
        [{"dimension": "unknown_feature", "operator": "==", "value": "x"}]
    )
    assert "unsupported_candidate_dimension:unknown_feature" in resolve(c, config)["setup_blockers"]
