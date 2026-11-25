"""Private local refresh jobs; inputs and events are immutable, activation is separate."""

import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from oddsjam_ev.io.manifests import writer_lock
from oddsjam_ev.io.snapshots import materialize_snapshot
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.staging import _private_root, confirm_import
from oddsjam_ev.pipeline.config import PipelineConfig, configuration_bundle
from oddsjam_ev.pipeline.manifests import atomic_json, code_provenance, load_result
from oddsjam_ev.pipeline.runner import run_pipeline


def job_path(root: Path, job_id: str) -> Path:
    """Resolve only an opaque session-owned job ID beneath the trusted root."""
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid job identity")
    _private_root(root)
    path = root / job_id
    if path.is_symlink():
        raise ValueError("Symlink jobs forbidden")
    return path


def create_job(root: Path, cfg: PipelineConfig, staging: Path, preview: dict) -> str:
    """Freeze a reviewed request; no import or processing until explicitly launched."""
    job_id = uuid.uuid4().hex
    path = job_path(root, job_id)
    path.mkdir(mode=0o700)
    request = {
        "version": 1,
        "job_id": job_id,
        "pipeline": asdict(cfg),
        "configuration": configuration_bundle(cfg),
        "code": code_provenance(),
        "staging": str(staging.resolve()),
        "stage_id": preview["stage_id"],
        "preview_sha256": preview["preview_sha256"],
    }
    atomic_json(path / "request.json", {**request, "sha256": fingerprint(request)})
    return job_id


def read_request(path: Path) -> dict:
    """Reject changed request/configuration/code before every attempt."""
    request = json.loads((path / "request.json").read_text())
    if fingerprint({k: v for k, v in request.items() if k != "sha256"}) != request["sha256"]:
        raise ValueError("Request integrity failure")
    cfg = PipelineConfig(**request["pipeline"])
    if (
        request["configuration"] != configuration_bundle(cfg)
        or request["code"] != code_provenance()
    ):
        raise ValueError("Changed code/configuration requires a fresh reviewed request")
    return request


def status(root: Path, job_id: str) -> dict:
    """Return latest durable attempt; absence is distinct from success."""
    path = job_path(root, job_id)
    attempts = sorted(path.glob("attempt-*.json"))
    if not attempts:
        return {"status": "queued"}
    return json.loads(attempts[-1].read_text())


def cancel(root: Path, job_id: str) -> None:
    """Request a stop; committed imports and already published results survive."""
    path = job_path(root, job_id)
    (path / "cancel").touch(mode=0o600, exist_ok=True)


def launch(root: Path, job_id: str) -> subprocess.Popen:
    """Start a fixed Python worker without a shell, credentials or arbitrary commands."""
    path = job_path(root, job_id)
    read_request(path)
    if (path / ".writer.lock").exists():
        raise ValueError("Job is already running")
    (path / "cancel").unlink(missing_ok=True)
    return subprocess.Popen(
        [sys.executable, "-m", "oddsjam_ev.dashboard.refresh_worker", str(path.resolve())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )


def execute(path: Path) -> dict:
    """Import once, pin an as-of snapshot, and retry the same frozen pipeline plan."""
    with writer_lock(path):
        sequence = len(list(path.glob("attempt-*.json")))
        events = []

        def record(value):
            nonlocal sequence
            atomic_json(path / f"attempt-{sequence:06d}.json", value)
            sequence += 1
            return value

        def progress(event):
            events.append(event)
            record({"status": "running", "events": events})
            if event.get("run_id") and not (path / "run.json").exists():
                atomic_json(path / "run.json", {"run_id": event["run_id"]})

        def stopped():
            return (path / "cancel").exists()

        try:
            request = read_request(path)
            cfg = PipelineConfig(**request["pipeline"])
            if stopped():
                return record({"status": "cancelled", "events": events})
            progress({"stage": "import", "state": "started"})
            receipt_path = path / "import.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
            else:
                receipt = confirm_import(
                    Path(request["staging"]),
                    request["stage_id"],
                    Path(request["configuration"]["source_config"]["store"]),
                    preview_sha256=request["preview_sha256"],
                    confirmed=True,
                )
                atomic_json(receipt_path, receipt)
            progress({"stage": "import", "state": "completed"})
            if stopped():
                return record({"status": "cancelled", "events": events})
            snapshot = materialize_snapshot(
                Path(request["configuration"]["source_config"]["store"]),
                as_of=receipt["imported_at"],
            )
            args = {"snapshot": snapshot, "as_of": receipt["imported_at"]}
            if (path / "run.json").exists():
                args = {"resume": json.loads((path / "run.json").read_text())["run_id"]}
            outcome = run_pipeline(cfg, **args, on_event=progress, cancelled=stopped)
            if outcome["exit_code"] == 0:
                load_result(Path(cfg.output_root) / "results" / outcome["run_id"])
            return record(
                {
                    "status": "ready" if outcome["exit_code"] == 0 else outcome["status"],
                    "outcome": outcome,
                    "events": events,
                }
            )
        except Exception:
            return record(
                {
                    "status": "failed",
                    "error_code": "refresh_validation_or_io_failure",
                    "events": events,
                }
            )
