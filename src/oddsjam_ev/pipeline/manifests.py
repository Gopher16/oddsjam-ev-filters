"""Atomic private run records and source-code provenance, separate from attempts."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from oddsjam_ev.io.manifests import verify_bytes, write_json
from oddsjam_ev.io.source_csv import fingerprint


def atomic_json(path: Path, value: dict) -> None:
    """Publish a new complete JSON file exclusively; never overwrite history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=path.parent) as temporary:
        staged = Path(temporary) / "record.json"
        write_json(staged, value)
        os.link(staged, path)


def code_provenance() -> dict:
    """Hash all package source and locked dependencies, including dirty source edits."""
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    paths = sorted(package.rglob("*.py")) + [repo / "poetry.lock", repo / "scripts/run_weekly.py"]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "revision": revision,
        "python_version": platform.python_version(),
        "sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "versions": {
            name: version(name) for name in ("pandas", "numpy", "scipy", "pyarrow", "pyyaml")
        },
    }


def read_plan(path: Path) -> dict:
    """Reject edited frozen specifications before resuming."""
    plan = json.loads(path.read_text())
    if fingerprint(plan["specification"]) != plan["run_id"]:
        raise ValueError("Run specification integrity failure")
    return plan


def load_result(path: Path) -> dict:
    """Verify result identity and deterministic report bytes."""
    result = read_plan(path / "manifest.json")
    if (
        fingerprint({k: v for k, v in result.items() if k != "payload_sha256"})
        != result["payload_sha256"]
    ):
        raise ValueError("Result payload integrity failure")
    verify_bytes(path / "report.md", result["report_sha256"])
    if result["report_sha256"] != fingerprint_report(result["report"]):
        raise ValueError("Report manifest mismatch")
    return result


def fingerprint_report(report: str) -> str:
    """Hash report bytes without embedding attempt timestamps."""
    return hashlib.sha256(report.encode()).hexdigest()
