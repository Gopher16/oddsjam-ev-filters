"""Frozen allowlisted jobs preserve policies and explicit model-build intent."""

import json

import pytest
import yaml

from oddsjam_ev.pipeline.research_jobs import dependency_order, freeze_job, stage_blockers


def write_job(tmp_path, **changes):
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({"contract_version": 1, "stages": ["queues"], **changes}))
    return path


def test_dependency_closure_and_complete_defaults(tmp_path):
    job = freeze_job(write_job(tmp_path))
    assert job["order"] == [
        "sizing",
        "sizing_diagnostics",
        "within_filter_discovery",
        "prioritization",
        "sheets",
        "visuals",
        "queues",
    ]
    assert job["stage_configs"]["prioritization"]["bootstrap_resamples"] == 5000
    assert not job["fit_models"] and job["model_policy"] is None
    assert stage_blockers("sheets", job, {}) == ["missing_decision_effective_time"]
    assert stage_blockers("queues", job, {"sheets": {"state": "blocked"}}) == [
        "blocked_dependency:sheets"
    ]
    assert json.loads(json.dumps(job)) == job


@pytest.mark.parametrize(
    "stages", [[], ["arbitrary_command"], ["models", "models"], "models", [False]]
)
def test_invalid_graph(stages):
    with pytest.raises(ValueError):
        dependency_order(stages)


@pytest.mark.parametrize(
    "changes",
    [
        {"contract_version": True},
        {"extra": "ignored"},
        {"stage_configs": {"models": {}}},
        {"stage_configs": {"visuals": {"max_rows": 0}}},
        {"model_policy": {}},
        {"decision_effective_at": "2026-01-01"},
    ],
)
def test_invalid_request(tmp_path, changes):
    with pytest.raises((ValueError, TypeError)):
        freeze_job(write_job(tmp_path, **changes))


def test_frozen_evidence_and_model_intent(tmp_path):
    evidence = tmp_path / "evidence.yaml"
    evidence.write_text("contract_version: 1\nbaselines: []\nlifecycle: []\ntranslations: []\n")
    request = write_job(tmp_path, decision_evidence_file=evidence.name, stages=["models"])
    frozen = freeze_job(request)
    evidence.write_text("changed: true")
    assert frozen["decision_evidence"]["content"]["baselines"] == []
    assert stage_blockers("models", frozen, {}) == ["model_build_not_requested"]
    built = freeze_job(request, fit_models=True)
    assert built["fit_models"] and built["model_runtime"]["scikit-learn"]
    assert not stage_blockers("models", built, {})
    with pytest.raises(ValueError):
        freeze_job(write_job(tmp_path), fit_models=True)
    with pytest.raises(ValueError):
        freeze_job(write_job(tmp_path, stages=["models"], model_scope={"profit": [1]}))


def test_future_evidence_and_changed_model_runtime_block(tmp_path, monkeypatch):
    from oddsjam_ev.pipeline.research_jobs import verify_job_inputs

    job = freeze_job(write_job(tmp_path, stages=["models"]))
    job["model_runtime"]["scikit-learn"] = "changed"
    with pytest.raises(ValueError, match="runtime"):
        verify_job_inputs(job, "2026-01-01T00:00:00Z")
    job["model_runtime"] = None
    job["sizing_history"] = {"content": {"knowledge_as_of": "2027-01-01T00:00:00Z"}}
    with pytest.raises(ValueError, match="cutoff"):
        verify_job_inputs(job, "2026-01-01T00:00:00Z")


def test_input_size_bound_and_invalid_predecessor(tmp_path):
    evidence = tmp_path / "large.yaml"
    evidence.write_text("x" * 2_000_001)
    with pytest.raises(ValueError, match="bound"):
        freeze_job(write_job(tmp_path, sizing_history_file=evidence.name))
    with pytest.raises(ValueError, match="explicit history"):
        freeze_job(write_job(tmp_path, sizing_previous="nonexistent"))
