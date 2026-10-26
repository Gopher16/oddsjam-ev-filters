"""Automatic report loading, read-only all-data browsing and fail-closed caching."""

import hashlib
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from oddsjam_ev.dashboard.artifacts import load_published_run

APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


def app_for(root, tmp_path, monkeypatch):
    config = tmp_path / "dashboard.yaml"
    config.write_text(yaml.safe_dump({"artifact_root": str(root)}))
    monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(config))
    return AppTest.from_file(str(APP)).run(timeout=30)


def test_auto_open_all_data_and_read_only_navigation(dashboard_run, tmp_path, monkeypatch):
    root, run_id = dashboard_run
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    app = app_for(root, tmp_path, monkeypatch)
    assert not app.exception and not app.error
    assert app.sidebar.selectbox(key="selected_run").value == run_id
    assert app.sidebar.selectbox(key="mode").value == "all"
    assert app.sidebar.selectbox(key="filter_product").value == "all"
    assert {b.label for b in app.button} == {"Reset signal filters"}
    for page in ("Signals & leaks", "Filter health", "Evidence & methodology"):
        app.sidebar.radio(key="page").set_value(page).run(timeout=30)
        assert not app.exception and not app.error, [e.value for e in app.error]
    app.sidebar.radio(key="page").set_value("Signals & leaks").run(timeout=30)
    app.sidebar.selectbox(key="mode").select("paper").run(timeout=30)
    app.sidebar.selectbox(key="filter_product").select("fixed_odds").run(timeout=30)
    assert not app.exception and not app.error
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


def test_corruption_clears_results_without_fallback(dashboard_run, tmp_path, monkeypatch):
    root, run_id = dashboard_run
    app = app_for(root, tmp_path, monkeypatch)
    run = load_published_run(root, run_id)
    next(p for p in run.expected_files if p.name == "wager_segments.parquet").write_bytes(
        b"SECRET_MARKER"
    )
    app.run(timeout=30)
    assert not app.exception and app.error and not app.metric
    assert "published_run" not in app.session_state
    assert "SECRET_MARKER" not in str([e.value for e in app.error])


def test_missing_and_incomplete_reports_do_not_load_fallback(dashboard_run, tmp_path, monkeypatch):
    root, run_id = dashboard_run
    (root / "results" / run_id / "report.md").unlink()
    app = app_for(root, tmp_path, monkeypatch)
    assert not app.exception and app.error and not app.metric
    absent = app_for(tmp_path / "absent", tmp_path, monkeypatch)
    assert not absent.exception and not absent.metric
    assert any("No published results" in x.value for x in absent.info)


def test_simultaneous_signal_controls_and_reset(dashboard_run, tmp_path, monkeypatch):
    app = app_for(dashboard_run[0], tmp_path, monkeypatch)
    platform = app.sidebar.multiselect(key="filter_signal_sportsbook")
    sport = app.sidebar.multiselect(key="filter_signal_sport")
    assert platform.options and sport.options
    platform.select(platform.options[0]).run(timeout=30)
    app.sidebar.multiselect(key="filter_signal_sport").select(sport.options[0]).run(timeout=30)
    assert not app.exception and not app.error
    assert app.sidebar.multiselect(key="filter_signal_sportsbook").value
    assert app.sidebar.multiselect(key="filter_signal_sport").value
    assert any("Active signal filters" in c.value for c in app.caption)
    app.sidebar.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert all(not widget.value for widget in app.sidebar.multiselect)


def test_minimum_wager_control_defaults_and_small_sample_empty_state(
    dashboard_run, tmp_path, monkeypatch
):
    app = app_for(dashboard_run[0], tmp_path, monkeypatch)
    assert app.sidebar.number_input(key="filter_min_wagers").value == 100
    assert any("No published signal cohorts meet the minimum" in value.value for value in app.info)
    app.sidebar.number_input(key="filter_min_wagers").set_value(1).run(timeout=30)
    assert not app.error and not app.exception
    assert not any(
        "No published signal cohorts meet the minimum" in value.value for value in app.info
    )


def test_saved_filter_selector_keeps_scope_explicit(dashboard_run, tmp_path, monkeypatch):
    app = app_for(dashboard_run[0], tmp_path, monkeypatch)
    selector = app.sidebar.selectbox(key="filter_signal_saved")
    assert selector.value == "all"
    if len(selector.options) > 1:
        selector.select(selector.options[1]).run(timeout=30)
        assert not app.error and not app.exception
        assert any("Saved-filter scope:" in c.value for c in app.caption)
        assert any("Global discovery results are not substituted" in c.value for c in app.info)
