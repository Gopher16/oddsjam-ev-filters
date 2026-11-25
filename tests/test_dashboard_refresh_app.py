"""Explicit activation in the opt-in refresh UI."""

from pathlib import Path

import pytest
from test_dashboard_refresh import request
from test_dashboard_refresh import setup as refresh_setup

from oddsjam_ev.dashboard import refresh


@pytest.fixture
def setup(tmp_path):
    return refresh_setup.__wrapped__(tmp_path)


def test_empty_catalogue_requires_explicit_activation(setup, tmp_path, monkeypatch):
    import yaml
    from streamlit.testing.v1 import AppTest

    cfg, jobs, job, _ = request(setup, tmp_path)
    # Use the controller directory configured by the UI.
    root = Path(cfg.output_root)
    ui_jobs = root / "dashboard_refresh" / "jobs"
    ui_jobs.mkdir(parents=True, mode=0o700)
    path = refresh.job_path(jobs, job)
    path.rename(ui_jobs / job)
    pipeline = tmp_path / "pipeline.yaml"
    from dataclasses import asdict

    pipeline.write_text(yaml.safe_dump(asdict(cfg)))
    dashboard = tmp_path / "dashboard.yaml"
    dashboard.write_text(
        yaml.safe_dump({"artifact_root": str(root), "refresh_pipeline_config": str(pipeline)})
    )
    monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(dashboard))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py")).run(
        timeout=30
    )
    assert not app.exception and not app.error
    assert app.sidebar.selectbox(key="selected_run").value is None
    assert refresh.execute(ui_jobs / job)["status"] == "ready"
    app.session_state["refresh_job"] = job
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert app.sidebar.selectbox(key="selected_run").value is None
    next(b for b in app.button if b.label == "Activate verified report").click().run(timeout=30)
    assert not app.exception and not app.error
    assert (
        app.sidebar.selectbox(key="selected_run").value
        == refresh.status(ui_jobs, job)["outcome"]["run_id"]
    )
