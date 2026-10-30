"""Optional evidence remains pinned, read-only and separate from descriptive cards."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig
from oddsjam_ev.dashboard.artifacts import DashboardConfig, load_published_run
from oddsjam_ev.dashboard.evidence import candidate_rows, load_evidence
from oddsjam_ev.io.signal_prioritization import publish_prioritization
from oddsjam_ev.io.sizing_diagnostics import publish_sizing_diagnostics
from oddsjam_ev.io.sizing_history import publish_sizing_history
from oddsjam_ev.io.within_filter_discovery import publish_within_filter_discovery


@pytest.fixture
def attached(dashboard_run, tmp_path):
    root, run_id = dashboard_run
    run = load_published_run(root, run_id)
    canonical = Path(run.result["outputs"]["canonical"]["path"])
    atlas = Path(run.result["outputs"]["atlas"]["path"])
    history = publish_sizing_history(
        canonical, tmp_path / "history", knowledge_as_of="2030-01-01T00:00:00Z"
    )
    diagnostics = publish_sizing_diagnostics(
        canonical, history, tmp_path / "diagnostics", SizingDiagnosticsConfig()
    )
    discovery = publish_within_filter_discovery(
        canonical,
        atlas,
        diagnostics,
        tmp_path / "family",
        MultidimensionalConfig(views=[["sportsbook"]]),
    )
    priority = publish_prioritization(
        discovery, diagnostics, tmp_path / "priorities", PrioritizationConfig()
    )
    cfg = DashboardConfig(
        artifact_root=str(root),
        evidence_root=str(tmp_path),
        evidence_publications={
            run_id: {"discovery": str(discovery), "prioritization": str(priority)}
        },
        min_analyzed_wagers=1,
    )
    return run, cfg, discovery, priority


def test_explicit_link_lineage_and_display_only_selection(attached):
    run, cfg, discovery, priority = attached
    before = {
        p: p.read_bytes() for base in (discovery, priority) for p in base.iterdir() if p.is_file()
    }
    evidence = load_evidence(run, cfg)
    rows = candidate_rows(evidence, minimum=1)
    assert len(rows) > 0
    assert candidate_rows(evidence, mode="no-such-funding", minimum=1).empty
    assert candidate_rows(evidence, product="no-such-product", minimum=1).empty
    assert candidate_rows(evidence, minimum=1000000).empty
    assert len(evidence["priority"]["signal_hypotheses"]) == 4 * len(
        evidence["family"]["signal_candidates"]
    )
    assert evidence["priority"]["signal_hypotheses"].holm_pvalue.isna().all()
    assert before == {p: p.read_bytes() for p in before}
    assert load_evidence(run, replace(cfg, evidence_publications={})) is None
    wrong = replace(cfg, evidence_root=str(discovery))
    with pytest.raises(ValueError, match="outside"):
        load_evidence(run, wrong)
    # Same IDs without exact manifest bytes must not link to another report.
    run.result["outputs"]["canonical"]["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="lineage"):
        load_evidence(run, cfg)


def test_symlink_and_corruption_rejected(attached, tmp_path):
    run, cfg, discovery, priority = attached
    target = priority / "signal_statistics.parquet"
    external = tmp_path.parent / (tmp_path.name + "-outside")
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        load_evidence(run, cfg)
    target.unlink()
    target.write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_evidence(run, cfg)


def test_dashboard_render_empty_shortlist_and_reverify(attached, tmp_path, monkeypatch):
    from dataclasses import asdict

    run, cfg, discovery, priority = attached
    config = tmp_path / "dashboard-evidence.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(config))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py")).run(
        timeout=30
    )
    app.sidebar.radio(key="page").set_value("Within-filter evidence").run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert any("No candidate qualified" in x.value for x in app.info)
    assert any("Blocked gates" in x.value for x in app.markdown)
    assert len(app.dataframe) >= 5
    app.sidebar.number_input(key="filter_min_wagers").set_value(1000000).run(timeout=30)
    assert not app.exception and not app.error
    assert any("No published candidates" in x.value for x in app.info)
    (priority / "signal_hypotheses.parquet").write_bytes(b"bad")
    app.run(timeout=30)
    assert not app.exception and app.error
    assert (
        app.session_state["verified_evidence"] is None
        if "verified_evidence" in app.session_state
        else True
    )
    assert not app.dataframe


def test_config_rejects_partial_links():
    with pytest.raises(ValueError, match="Explicit discovery"):
        DashboardConfig(evidence_publications={"a" * 64: {"discovery": "family"}})
    with pytest.raises(ValueError, match="run-ID"):
        DashboardConfig(evidence_publications=[])


def test_populated_corrected_views_and_exact_selectors(tmp_path):
    from dataclasses import asdict
    from types import SimpleNamespace

    from test_statistical_prioritization import prepared

    from oddsjam_ev.analysis.signal_prioritization import build_prioritization

    family, data, cfg = prepared()
    priority, audit = build_prioritization(family, data, cfg)
    evidence = {
        "family": family,
        "priority": priority,
        "manifests": {"prioritization": {"audit": audit, "specification": {"policy": asdict(cfg)}}},
    }
    rows = candidate_rows(evidence, minimum=1)
    assert len(rows) == 4 and rows.included_shortlist.sum() == 2
    assert priority["signal_hypotheses"].holm_pvalue.notna().all()
    ids = family["signal_candidates"].filter_id.unique()
    run = SimpleNamespace(
        manifests={
            "canonical": {
                "specification": {
                    "filter_registry": {"filters": [{"id": v, "name": v} for v in ids]}
                }
            }
        }
    )
    app = AppTest.from_string("""
import streamlit as st
from oddsjam_ev.dashboard.pages.evidence import evidence_page
evidence_page(st.session_state.test_run, "all", 20)
""")
    app.session_state["test_run"] = run
    app.session_state["verified_evidence"] = evidence
    app.session_state["filter_min_wagers"] = 1
    app.run(timeout=30)
    assert not app.exception
    assert any("Other saved filter" in frame.value.columns for frame in app.dataframe)
    options = app.multiselect(key="evidence_term_sportsbook").options
    app.multiselect(key="evidence_term_sportsbook").set_value([options[0]]).run(timeout=30)
    assert not app.exception
    assert len(priority["signal_hypotheses"]) == 16
    app.multiselect(key="evidence_term_sport").set_value(["== sportB"]).run(timeout=30)
    assert not app.exception
    assert any("No published candidates" in x.value for x in app.info)


def test_frozen_definition_visibility_and_corruption(attached, tmp_path, monkeypatch):
    from dataclasses import asdict, replace

    from test_paper_definitions import complete

    import oddsjam_ev.io.paper_definitions as module
    from oddsjam_ev.dashboard.evidence import load_paper_definitions
    from oddsjam_ev.io.paper_definitions import publish_definitions, register_definition

    run, cfg, discovery, priority = attached
    evidence = load_evidence(run, cfg)
    candidate = evidence["family"]["signal_candidates"].iloc[0].to_dict()
    protocol = complete(candidate)["protocol"]
    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T00:00:00Z")
    register = tmp_path / "definition-register"
    first = register_definition(
        register,
        discovery,
        priority,
        candidate["candidate_id"],
        "test-example",
        protocol,
        "synthetic",
        "Prospective intent only",
    )
    protocol["name"] = "Changed version"
    second = register_definition(
        register,
        discovery,
        priority,
        candidate["candidate_id"],
        "test-example",
        protocol,
        "synthetic",
        "Versioned revision",
        first["version_id"],
    )
    published = publish_definitions(
        register, tmp_path / "definition-publications", "2030-01-01T00:00:00Z"
    )
    cfg = replace(cfg, paper_definitions_publication=str(published))
    snapshot = load_paper_definitions(run, cfg, evidence)
    assert len(snapshot["events"]) == 2
    assert snapshot["latest"]["test-example"] == second["version_id"]
    assert load_paper_definitions(run, cfg, None)["events"] == []
    assert (
        load_paper_definitions(run, replace(cfg, paper_definitions_publication=None), evidence)
        is None
    )
    config = tmp_path / "paper-dashboard.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(config))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py")).run(
        timeout=30
    )
    app.sidebar.radio(key="page").set_value("Paper-test definitions").run(timeout=30)
    assert not app.error and not app.exception
    assert len(app.dataframe[0].value) == 2
    assert any("Superseded definition" in x.value for x in app.warning)
    (register / "00000001.json").write_bytes(b"bad")
    app.run(timeout=30)
    assert not app.error and not app.exception  # Live register is never read.
    (published / "definitions.json").write_bytes(b"bad")
    app.run(timeout=30)
    assert app.error and not app.exception and not app.dataframe
