"""Synthetic end-to-end refresh, cancellation and frozen-retry contracts."""

import io
import json
from pathlib import Path

import pytest
from test_canonical_bundle import source
from test_weekly_pipeline import setup as pipeline_setup

from oddsjam_ev.dashboard import refresh
from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.staging import preview_csv, stage_csv
from oddsjam_ev.pipeline.runner import run_pipeline


@pytest.fixture
def setup(tmp_path):
    value = pipeline_setup.__wrapped__(tmp_path)
    (tmp_path / "store").chmod(0o700)
    return value


def request(setup, tmp_path):
    cfg, csv, _, _ = setup
    # Distinct namespace avoids silently mixing the pre-existing fixture import.
    staging, jobs = tmp_path / "staging", tmp_path / "jobs"
    receipt = stage_csv(
        io.BytesIO(csv.read_bytes()),
        staging,
        filename="history.csv",
        metadata=SourceMetadata("upload", scope="full"),
    )
    preview = preview_csv(staging, receipt["stage_id"], tmp_path / "store")
    job = refresh.create_job(jobs, cfg, staging, preview)
    return cfg, jobs, job, preview


def test_build_repeat_and_frozen_retry(setup, tmp_path):
    cfg, jobs, job, preview = request(setup, tmp_path)
    assert preview["candidate_rows"]["new"] > 0
    path = refresh.job_path(jobs, job)
    first = refresh.execute(path)
    assert first["status"] == "ready", first
    run = first["outcome"]["run_id"]
    assert (Path(cfg.output_root) / "results" / run / "manifest.json").is_file()
    receipt = (path / "import.json").read_bytes()
    again = refresh.execute(path)
    assert again["status"] == "ready" and again["outcome"]["run_id"] == run
    assert (path / "import.json").read_bytes() == receipt
    assert any(e.get("state") == "reused" for e in again["events"])
    p = preview_csv(
        Path(json.loads((path / "request.json").read_text())["staging"]),
        preview["stage_id"],
        tmp_path / "store",
    )
    assert p["repeated_import"]
    assert refresh.status(jobs, job)["status"] == "ready"


def test_cancel_before_import_and_safe_pipeline_boundary(setup, tmp_path):
    cfg, jobs, job, _ = request(setup, tmp_path)
    refresh.cancel(jobs, job)
    path = refresh.job_path(jobs, job)
    assert refresh.execute(path)["status"] == "cancelled"
    assert not (path / "import.json").exists()
    _, _, snapshot, cutoff = setup
    events = []
    outcome = run_pipeline(
        cfg,
        snapshot=snapshot,
        as_of=cutoff,
        on_event=events.append,
        cancelled=lambda: any(
            e["stage"] == "canonical" and e["state"] == "completed" for e in events
        ),
    )
    assert outcome["status"] == "cancelled"
    assert not list((Path(cfg.output_root) / "results").glob("*"))
    retry = run_pipeline(cfg, resume=outcome["run_id"])
    assert retry["exit_code"] == 0
    assert any(e["stage"] == "canonical" and e["state"] == "reused" for e in retry["stages"])


def test_stale_preview_fails_without_import(setup, tmp_path):
    _, jobs, job, _ = request(setup, tmp_path)
    csv = source(tmp_path, profit="55", name="changed.csv")
    import_csv(csv, tmp_path / "store", SourceMetadata("another"))
    path = refresh.job_path(jobs, job)
    assert refresh.execute(path)["status"] == "failed"
    assert not (path / "import.json").exists()


def test_request_tampering_and_traversal(setup, tmp_path):
    _, jobs, job, _ = request(setup, tmp_path)
    with pytest.raises(ValueError):
        refresh.job_path(jobs, "../secret")
    path = refresh.job_path(jobs, job)
    raw = json.loads((path / "request.json").read_text())
    raw["preview_sha256"] = "changed"
    (path / "request.json").write_text(json.dumps(raw))
    assert refresh.execute(path)["status"] == "failed"
    assert not (path / "import.json").exists()


def test_changed_config_blocks_retry(setup, tmp_path):
    cfg, jobs, job, _ = request(setup, tmp_path)
    Path(cfg.atlas_config).write_text("contract_version: 2\n")
    assert refresh.execute(refresh.job_path(jobs, job))["status"] == "failed"


def test_failed_build_preserves_import_and_can_retry(setup, tmp_path, monkeypatch):
    _, jobs, job, _ = request(setup, tmp_path)
    actual = refresh.run_pipeline
    monkeypatch.setattr(
        refresh, "run_pipeline", lambda *a, **k: {"status": "failed", "exit_code": 2}
    )
    path = refresh.job_path(jobs, job)
    assert refresh.execute(path)["status"] == "failed"
    receipt = (path / "import.json").read_bytes()
    monkeypatch.setattr(refresh, "run_pipeline", actual)
    assert refresh.execute(path)["status"] == "ready"
    assert (path / "import.json").read_bytes() == receipt
