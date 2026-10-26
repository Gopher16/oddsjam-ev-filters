"""Shared isolated published-run fixture for read-only dashboard verification."""

from pathlib import Path

import pytest
import test_weekly_pipeline

from oddsjam_ev.pipeline.runner import run_pipeline

pipeline_setup = test_weekly_pipeline.setup


@pytest.fixture
def dashboard_run(pipeline_setup, tmp_path):
    cfg, _, snapshot, cutoff = pipeline_setup
    result = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, register=tmp_path / "empty")
    assert result["exit_code"] == 0, result
    return Path(cfg.output_root), result["run_id"]
