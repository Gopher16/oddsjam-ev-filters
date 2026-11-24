"""Real optional publishers, reproducible receipts, failure isolation and replay."""

import csv
import errno
import json
from pathlib import Path

import pytest
import test_weekly_pipeline as base
import yaml
from test_research_jobs import write_job
from test_weekly_pipeline import result_path

from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.pipeline import research_jobs
from oddsjam_ev.pipeline.manifests import load_result
from oddsjam_ev.pipeline.runner import run_pipeline


@pytest.fixture
def setup(tmp_path):
    cfg, original, _, _ = base.setup.__wrapped__(tmp_path)
    policy = yaml.safe_load(Path(cfg.canonical_config).read_text())
    policy["products"]["fixed_odds"].append("Other")
    Path(cfg.canonical_config).write_text(yaml.safe_dump(policy))
    with original.open() as stream:
        template = next(csv.DictReader(stream))
    template.update(percentage="3", sport="basketball", league="NBA")
    source = tmp_path / "diverse.csv"
    with source.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(template))
        writer.writeheader()
        for i in range(8):
            writer.writerow(
                {**template, "game_id": str(i), "sportsbook": "Book" if i % 2 else "Other"}
            )
    from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
    from oddsjam_ev.io.snapshots import materialize_snapshot
    from oddsjam_ev.pipeline.runner import utc_now

    import_csv(source, tmp_path / "diverse-store", SourceMetadata("synthetic"))
    return cfg, source, materialize_snapshot(tmp_path / "diverse-store"), utc_now()


def job(tmp_path, **changes):
    return write_job(
        tmp_path,
        decision_effective_at="2020-01-01T00:00:00Z",
        stage_configs={
            "sizing_diagnostics": {"bootstrap_resamples": 100},
            "prioritization": {"bootstrap_resamples": 100},
            "within_filter_discovery": {"views": [["sportsbook"]]},
        },
        **changes,
    )


def test_real_chain_manifest_replay_and_frozen_request(setup, tmp_path):
    cfg, _, snapshot, cutoff = setup
    request = job(tmp_path)
    first = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, research_job=request)
    assert first["exit_code"] == 0, first
    path = result_path(cfg, first)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    result = load_result(path)
    manifest = result["job_manifest"]
    assert all(
        manifest["receipts"][s]["state"] == "completed" for s in manifest["request"]["order"]
    )
    assert manifest["receipts"]["models"]["state"] == "not_requested"
    assert result["summaries"]["sheets"]["coverage_audit"]["blocked_count"] > 0
    assert result["summaries"]["sizing"]["coverage_audit"]["sizing_replay_blocked"]
    request.write_text("broken: request")
    resumed = run_pipeline(cfg, resume=first["run_id"])
    assert resumed["exit_code"] == 0, resumed
    assert all(p.read_bytes() == before[p.name] for p in path.iterdir())
    assert len([e for e in resumed["stages"] if e["state"] == "reused"]) == 11
    (path / "job.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        load_result(path)
    assert run_pipeline(cfg, resume=first["run_id"])["exit_code"] == 2


def test_missing_optional_evidence_and_model_optin_are_visible(setup, tmp_path, monkeypatch):
    cfg, _, snapshot, cutoff = setup
    request = write_job(
        tmp_path,
        stages=["queues", "models"],
        stage_configs={"within_filter_discovery": {"views": [["sportsbook"]]}},
    )
    calls = []
    original = research_jobs.build_stage

    def spy(name, *args):
        calls.append(name)
        return original(name, *args)

    monkeypatch.setattr(research_jobs, "build_stage", spy)
    out = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, research_job=request)
    assert out["exit_code"] == 0, out
    receipts = load_result(result_path(cfg, out))["job_manifest"]["receipts"]
    assert receipts["sheets"]["reasons"] == ["missing_decision_effective_time"]
    assert receipts["queues"]["state"] == "blocked"
    assert receipts["models"]["reasons"] == ["model_build_not_requested"]
    assert "models" not in calls
    assert "model_build_not_requested" in load_result(result_path(cfg, out))["report"]


def test_requested_model_uses_existing_publisher(setup, tmp_path):
    cfg, _, snapshot, cutoff = setup
    out = run_pipeline(
        cfg,
        snapshot=snapshot,
        as_of=cutoff,
        research_job=write_job(tmp_path, stages=["models"]),
        fit_models=True,
    )
    assert out["exit_code"] == 0, out
    manifest = load_result(result_path(cfg, out))["job_manifest"]
    assert manifest["receipts"]["models"]["state"] == "completed"
    assert manifest["request"]["fit_models"]
    assert run_pipeline(cfg, resume=out["run_id"], fit_models=True)["exit_code"] == 2


def test_interrupted_stage_retries_resume_and_corruption(setup, tmp_path, monkeypatch):
    cfg, _, snapshot, cutoff = setup
    prior = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    prior_bytes = (result_path(cfg, prior) / "manifest.json").read_bytes()
    original = research_jobs.build_stage

    def fail(name, *args):
        if name == "visuals":
            raise RuntimeError("SECRET_MARKER private account")
        return original(name, *args)

    monkeypatch.setattr(research_jobs, "build_stage", fail)
    failed = run_pipeline(
        cfg, snapshot=snapshot, as_of=cutoff, research_job=write_job(tmp_path, stages=["visuals"])
    )
    assert failed["exit_code"] == 2 and not result_path(cfg, failed).exists()
    assert (result_path(cfg, prior) / "manifest.json").read_bytes() == prior_bytes
    assert all(
        "SECRET_MARKER" not in p.read_text()
        for p in (Path(cfg.output_root) / "attempts").rglob("*.json")
    )
    calls = []

    def transient(name, *args):
        if name == "visuals":
            calls.append(name)
            if len(calls) == 1:
                raise OSError(errno.EINTR, "SECRET_MARKER")
        return original(name, *args)

    monkeypatch.setattr(research_jobs, "build_stage", transient)
    resumed = run_pipeline(cfg, resume=failed["run_id"])
    assert resumed["exit_code"] == 0 and len(calls) == 2, resumed
    assert any(e["state"] == "retry" for e in resumed["stages"])
    record = load_result(result_path(cfg, resumed))["outputs"]["visuals"]
    publication = Path(record["path"])
    (publication / "visual_candidates.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_result(result_path(cfg, resumed))
    assert run_pipeline(cfg, resume=failed["run_id"])["exit_code"] == 2


def test_dependency_receipt_hash_and_frozen_spec_tamper(setup, tmp_path):
    cfg, _, snapshot, cutoff = setup
    first = run_pipeline(
        cfg,
        snapshot=snapshot,
        as_of=cutoff,
        research_job=write_job(tmp_path, stages=["sizing_diagnostics"]),
    )
    assert first["exit_code"] == 0, first
    checkpoint = Path(cfg.output_root) / "checkpoints" / first["run_id"] / "sizing_diagnostics.json"
    data = json.loads(checkpoint.read_text())
    data["dependencies"]["sizing"] = "other"
    data["checkpoint_sha256"] = fingerprint(
        {k: v for k, v in data.items() if k != "checkpoint_sha256"}
    )
    checkpoint.write_text(json.dumps(data))
    assert run_pipeline(cfg, resume=first["run_id"])["exit_code"] == 2
    plan = Path(cfg.output_root) / "plans" / (first["run_id"] + ".json")
    data = json.loads(plan.read_text())
    data["specification"]["research_job"]["fit_models"] = True
    plan.write_text(json.dumps(data))
    assert run_pipeline(cfg, resume=first["run_id"])["exit_code"] == 2


def test_cli_requires_explicit_job_and_core_only_stays_compatible(setup, tmp_path):
    import subprocess
    import sys
    from dataclasses import asdict

    cfg, _, snapshot, cutoff = setup
    core = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    assert core["exit_code"] == 0
    assert "research_job" not in load_result(result_path(cfg, core))["specification"]
    assert not (result_path(cfg, core) / "job.json").exists()
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    cli = Path(__file__).resolve().parents[1] / "scripts/run_weekly.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--config",
            str(config),
            "--snapshot",
            str(snapshot),
            "--as-of",
            cutoff,
            "--research-job",
            str(write_job(tmp_path, stages=["visuals"])),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert (
        load_result(result_path(cfg, result))["job_manifest"]["receipts"]["visuals"]["state"]
        == "completed"
    )
    rejected = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, fit_models=True)
    assert rejected["exit_code"] == 2 and rejected["run_id"] is None


def test_job_receipt_cannot_hide_a_blocked_stage(setup, tmp_path):
    from oddsjam_ev.pipeline.research_jobs import verify_job_manifest

    cfg, _, snapshot, cutoff = setup
    out = run_pipeline(
        cfg, snapshot=snapshot, as_of=cutoff, research_job=write_job(tmp_path, stages=["models"])
    )
    result = load_result(result_path(cfg, out))
    result["job_manifest"]["receipts"]["models"] = {"state": "not_requested"}
    with pytest.raises(ValueError, match="Blocked stage"):
        verify_job_manifest(result)
