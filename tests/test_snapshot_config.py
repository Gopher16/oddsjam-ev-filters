"""
test_snapshot_config.py
===============================================================================

Purpose
-------
Verify deterministic snapshot selection and path resolution for notebook inputs.

What this module provides
-------------------------
1. Snapshot resolution tests
   Cover default selection, explicit overrides, and invalid dates.

Design Notes
------------
- Tests use temporary directories and do not require real betting data.
- Environment overrides are isolated with pytest's monkeypatch fixture.
===============================================================================
"""

import pytest

from oddsjam_ev.analysis.snapshot_config import (
    DEFAULT_SNAPSHOT_DATE,
    resolve_snapshot_config,
)


def test_default_snapshot_and_paths(tmp_path, monkeypatch):
    """Resolve the default snapshot relative to an explicit project root."""
    (tmp_path / "data" / "processed").mkdir(parents=True)
    monkeypatch.delenv("ODDSJAM_SNAPSHOT_DATE", raising=False)
    cfg = resolve_snapshot_config(project_root=tmp_path)
    assert cfg.snapshot_date == DEFAULT_SNAPSHOT_DATE
    assert cfg.processed_path.parent == tmp_path / "data" / "processed"
    assert cfg.artifact_dir == tmp_path / "data" / "processed" / "artifacts"


def test_environment_and_explicit_snapshot_precedence(tmp_path, monkeypatch):
    """Allow an explicit snapshot date to override the environment."""
    (tmp_path / "data" / "processed").mkdir(parents=True)
    monkeypatch.setenv("ODDSJAM_SNAPSHOT_DATE", "04-01-2026")
    assert resolve_snapshot_config(project_root=tmp_path).snapshot_date == "04-01-2026"
    assert (
        resolve_snapshot_config("05-02-2026", project_root=tmp_path).snapshot_date == "05-02-2026"
    )


@pytest.mark.parametrize("date", ["2026-03-21", "2-03-2026", "02-30-2026"])
def test_invalid_snapshot_dates(tmp_path, date):
    """Reject ambiguous, noncanonical, and impossible date strings."""
    (tmp_path / "data" / "processed").mkdir(parents=True)
    with pytest.raises(ValueError):
        resolve_snapshot_config(date, project_root=tmp_path)


def test_missing_processed_directory(tmp_path):
    """Report missing repository data directories explicitly."""
    with pytest.raises(FileNotFoundError):
        resolve_snapshot_config(project_root=tmp_path)
