"""Run the documented tutorial through actual CLIs without private account inputs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from oddsjam_ev.analysis.experiments import replay_experiments
from oddsjam_ev.analysis.filter_lifecycle import replay_lifecycle
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.experiment_register import read_experiment_events
from oddsjam_ev.io.filter_lifecycle import load_lifecycle_report, read_lifecycle
from oddsjam_ev.io.paper_definitions import read_definitions

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/examples/operator_demo.py"
STEPS = (
    "init",
    "build",
    "define",
    "approve-study",
    "launch",
    "capture-review",
    "propose-change",
    "approve-change",
    "confirm-application",
    "monitor",
    "refresh",
)


def run(workspace, step, acknowledge=True):
    args = [sys.executable, str(SCRIPT), "--workspace", str(workspace), step]
    if acknowledge:
        args.append("--acknowledge-synthetic-decision")
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_documented_cli_workflow_keeps_decisions_versions_and_refresh_isolated(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "demo"
    workspace.mkdir()
    before = None
    for step in STEPS:
        result = run(workspace, step)
        assert result.returncode == 0, (step, result.stdout, result.stderr)
        state = json.loads((workspace / "state.json").read_text())
        assert state["completed"][-1] == step  # catches swallowed CLI SystemExit(0)
        if step == "build":
            before = {
                p: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (workspace / "first").rglob("*")
                if p.is_file()
            }
        if step == "define":
            assert not (workspace / "experiments").exists()
            assert (
                read_definitions(workspace / "definitions-register")[0]["status"]
                == "definition_only_not_launched"
            )
        if step == "approve-study":
            states = replay_experiments(read_experiment_events(workspace / "experiments"))
            assert next(iter(states.values()))["status"] != "RUNNING"
            assert state["protocol"]["funding"] == "paper_only"
            assert (
                read_definitions(workspace / "definitions-register")[0]["protocol"][
                    "min_wagers_per_arm"
                ]
                == 100
            )
        if step == "approve-change":
            lifecycle = replay_lifecycle(read_lifecycle(workspace / "lifecycle"))
            assert len(lifecycle["applications"]) == 1  # baseline only, approval is not application
        if step == "monitor":
            report = load_lifecycle_report(Path(state["lifecycle_report"]))
            assert len(read_definitions(workspace / "definitions-register")) == 2
            assert len(report["tables"]["applications"]) == 2
            assert report["tables"]["applications"].status.eq("TEST").all()
            assert "market_width" in (workspace / "rollback-evidence.json").read_text()
            cfg = workspace / "dashboard-monitored.yaml"
            monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(cfg))
            app = AppTest.from_file(str(ROOT / "dashboard/app.py")).run(timeout=30)
            assert not app.exception and not app.error
            app.sidebar.selectbox(key="page").set_value("Paper-test definitions").run(timeout=30)
            assert not app.exception and not app.error
    state = json.loads((workspace / "state.json").read_text())
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    old = load_canonical_bundle(Path(state["previous_paths"]["canonical"]))["bet_facts"]
    new = load_canonical_bundle(Path(state["paths"]["canonical"]))["bet_facts"]
    assert len(old) == 240 and len(new) == 241
    refreshed = yaml.safe_load((workspace / "dashboard-refresh.yaml").read_text())
    assert refreshed["paper_definitions_publication"] is None
    assert (
        refreshed["lifecycle_report_publication"] is None
    )  # old definitions do not attach to new lineage
    transcript = [
        json.loads(line) for line in (workspace / "commands.jsonl").read_text().splitlines()
    ]
    assert all(row["synthetic_only"] for row in transcript)
    assert {r["script"] for r in transcript} >= {
        "run_weekly.py",
        "manage_experiments.py",
        "manage_filter_lifecycle.py",
        "verify_release.py",
    }


def test_tutorial_refuses_reused_roots_out_of_order_and_unacknowledged_decisions(tmp_path):
    workspace = tmp_path / "demo"
    workspace.mkdir()
    (workspace / "keep.txt").write_text("preserve")
    assert run(workspace, "init").returncode != 0
    assert (workspace / "keep.txt").read_text() == "preserve"
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert run(fresh, "init").returncode == 0
    assert run(fresh, "launch").returncode != 0
    assert run(fresh, "init").returncode != 0
    # An in-order decision still requires an explicit acknowledgement.
    state = json.loads((fresh / "state.json").read_text())
    state["completed"] = list(STEPS[:3])
    (fresh / "state.json").write_text(json.dumps(state))
    rejected = run(fresh, "approve-study", acknowledge=False)
    assert rejected.returncode != 0 and "acknowledge" in rejected.stderr
    assert not (fresh / "approvals").exists()
    state["paths"] = {"outside": str(ROOT)}
    (fresh / "state.json").write_text(json.dumps(state))
    assert "external path" in run(fresh, "approve-study").stderr
