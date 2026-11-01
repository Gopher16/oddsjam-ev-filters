"""Verified default workspace, exact selectors, chart schemas and paper-only actions."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import test_dashboard_evidence
import yaml
from streamlit.testing.v1 import AppTest
from test_review_queues import review_inputs
from test_visual_diagnostics import inputs as visual_inputs

from oddsjam_ev.analysis.review_config import ReviewConfig
from oddsjam_ev.analysis.review_queues import build_review_queues
from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.analysis.visual_diagnostics import build_visual_diagnostics
from oddsjam_ev.dashboard import decision_charts as charts
from oddsjam_ev.dashboard.decision_artifacts import load_workspace
from oddsjam_ev.dashboard.decision_queries import (
    chart_rows,
    select_records,
    study_draft,
    visible_headlines,
)
from oddsjam_ev.dashboard.evidence import load_decision_sheets, load_evidence
from oddsjam_ev.dashboard.pages.visual_history import CHARTS
from oddsjam_ev.io.decision_records import publish_decisions
from oddsjam_ev.io.review_queues import publish_review_queues
from oddsjam_ev.io.visual_diagnostics import publish_visual_diagnostics

attached = test_dashboard_evidence.attached

APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


@pytest.fixture
def workspace_sources(attached, tmp_path):
    run, cfg, discovery, priority = attached
    canonical = Path(run.result["outputs"]["canonical"]["path"])
    atlas = Path(run.result["outputs"]["atlas"]["path"])
    diagnostics = next((tmp_path / "diagnostics").glob("*/manifest.json")).parent
    decisions = publish_decisions(
        discovery,
        priority,
        {"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []},
        tmp_path / "sheets",
        "2030-01-01T00:00:00Z",
        "2030-01-01T00:00:00Z",
    )
    visuals = publish_visual_diagnostics(
        canonical, atlas, diagnostics, tmp_path / "visuals", VisualConfig()
    )
    queues = publish_review_queues(
        discovery, priority, decisions, visuals, tmp_path / "queues", ReviewConfig()
    )
    cfg = replace(
        cfg,
        decision_records_publication=str(decisions),
        workspace_publications={
            run.result["run_id"]: {"queues": str(queues), "visuals": str(visuals)}
        },
    )
    return run, cfg, queues, visuals


def get_workspace(run, cfg, cache=None):
    evidence = load_evidence(run, cfg)
    sheets = load_decision_sheets(run, cfg, evidence)
    return load_workspace(run, cfg, evidence, sheets, cache)


def test_attachment_integrity_cache_and_selected_report_scope(workspace_sources, tmp_path):
    run, cfg, queues, visuals = workspace_sources
    workspace = get_workspace(run, cfg)
    assert get_workspace(run, cfg, workspace) is workspace
    assert (
        load_workspace(run, replace(cfg, workspace_publications={}), None, None, workspace) is None
    )
    with pytest.raises(ValueError, match="lineage"):
        load_workspace(run, cfg, None, None)
    with pytest.raises(ValueError, match="outside"):
        get_workspace(run, replace(cfg, evidence_root=str(queues)))
    # Cache reuse must hash bytes, not trust metadata/mtime or a previous render.
    (visuals / "visual_readiness.parquet").write_bytes(b"private-corrupt-marker")
    with pytest.raises(ValueError, match="integrity"):
        get_workspace(run, cfg, workspace)


def test_ui_default_navigation_selection_drafts_and_fail_closed(
    workspace_sources, tmp_path, monkeypatch
):
    run, cfg, queues, visuals = workspace_sources
    config = tmp_path / "dashboard.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(config))
    before = {
        p: p.read_bytes() for root in (queues, visuals) for p in root.iterdir() if p.is_file()
    }
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception and not app.error
    assert app.sidebar.selectbox(key="page").value == "Overview"
    assert app.sidebar.selectbox(key="mode").value == "all"
    assert any(m.label == "Corrected historical comparisons" and m.value == "0" for m in app.metric)
    assert not any(
        x.label == "Inspect setup & plan test" for x in app.button
    )  # no fabricated winners
    candidate = app.selectbox(key="decision_selected").options[0]
    app.selectbox(key="decision_selected").select(candidate).run(timeout=30)
    assert not app.exception and not app.error
    assert any("Full OddsJam setup" in t.label for t in app.tabs)
    assert len(app.dataframe) > 1
    for page in ("Opportunities", "Production health", "Explore history"):
        app.sidebar.selectbox(key="page").set_value(page).run(timeout=30)
        assert not app.exception and not app.error
    for name in CHARTS:
        app.selectbox(key="diagnostic_chart").set_value(name).run(timeout=30)
        assert not app.exception and not app.error, name
    app.sidebar.selectbox(key="page").set_value("Overview").run(timeout=30)
    options = app.sidebar.multiselect(key="filter_signal_sportsbook").options
    if options:
        app.sidebar.multiselect(key="filter_signal_sportsbook").select(options[0]).run(timeout=30)
        app.sidebar.multiselect(key="filter_signal_league").set_value([]).run(timeout=30)
        assert not app.exception and not app.error
    app.sidebar.button[0].click().run(timeout=30)
    assert all(not w.value for w in app.sidebar.multiselect)
    assert before == {p: p.read_bytes() for p in before}
    (queues / "review.json").write_bytes(b"private-corrupt-marker")
    app.run(timeout=30)
    assert app.error and not app.exception
    assert "decision_workspace" not in app.session_state
    assert "private-corrupt-marker" not in str([e.value for e in app.error])
    assert not app.metric


def test_exact_shared_selectors_no_implicit_intersection_or_refill():
    source = review_inputs()
    queue, _ = build_review_queues(*source, ReviewConfig())
    records = queue["records"]
    original = copy.deepcopy(records)
    selected = select_records(
        records, minimum=1, selections={"sportsbook": ["A"], "sport": ["sportA"]}
    )
    assert selected == []  # separate A and sportA slices are not their intersection
    selected = select_records(records, minimum=1, selections={"sportsbook": ["A", "B"]})
    assert selected and all(
        "sportsbook" in json.loads(r["predicate_json"])[0].get("dimension", "")
        or "sportsbook" in r["predicate_json"]
        for r in selected
    )
    assert select_records(records, mode="missing", minimum=1) == []
    assert select_records(records, minimum=1000000) == []
    heads = visible_headlines(records, "opportunities")
    assert len(heads) <= 4
    assert visible_headlines([r for r in records if not r["headline"]], "opportunities") == []
    for r in records:
        draft = study_draft(
            r,
            {
                "queues": {
                    **queue,
                    "manifest": {
                        "publication_id": "queue",
                        "specification": {"input_publication_ids": {"decisions": "sheets"}},
                    },
                }
            },
        )
        if r["source_kind"] == "global_descriptive_catalogue":
            assert draft["kind"] == "global_study_brief_not_registerable_definition"
            assert draft["max_funded_stake_usd"] == 0
        else:
            assert draft["protocol"]["funding"] == "paper_only"
            assert draft["protocol"]["planned_start_utc"] is None
            assert "configuration_reference" not in draft["protocol"]
    assert records == original


def test_chart_sources_match_cohorts_and_all_notebook_specs_validate():
    config = VisualConfig(
        placement_time_verified=True,
        event_time_verified=True,
        chronology_evidence="synthetic",
        liquidity_usd_verified=True,
        liquidity_evidence="synthetic",
        closing_fair_american_verified=True,
        closing_sources=("fair",),
        closing_evidence="synthetic",
    )
    visual, _ = build_visual_diagnostics(*visual_inputs(), config)
    workspace = {"visuals": visual}
    co = chart_rows(workspace, "visual_cohorts")
    assert co.scope_kind.eq("population").all()
    assert chart_rows(
        workspace, "visual_periods", selections={"sportsbook": ["A"], "league": ["NBA"]}
    ).empty
    platform_periods = chart_rows(workspace, "visual_periods", selections={"sportsbook": ["A"]})
    assert not platform_periods.empty and platform_periods.platform.eq("A").all()
    single = co.loc[co.view.eq("sport") & co.execution_status.eq("real_money")]
    assert (
        charts.profit_groups(single, "reported_profit_usd").to_dict()["encoding"]["y"]["axis"][
            "labelOverlap"
        ]
        is False
    )
    ranges = chart_rows(
        workspace, "visual_cohorts", selections={"advertised_ev_band": ["[3, 3.5)"]}
    )
    assert not ranges.empty  # numeric endpoint representation is equivalent
    charts_to_test = [
        charts.profit_groups(co.query("view == 'sport'"), "reported_profit_usd"),
        charts.ev_scatter(co.query("view == 'platform'")),
        charts.ev_scatter(co.query("view == 'platform'"), equal=True),
        charts.bucket_panel(co.query("view == 'ev_bucket'")),
        charts.bucket_panel(co.query("view == 'liquidity_bucket'")),
        charts.liquidity_heatmap(co.query("view == 'liquidity_ev'")),
        charts.bucket_panel(co.query("view == 'time_to_event_bucket'")),
        charts.calendar(chart_rows(workspace, "visual_periods"), "candidate_count"),
        charts.calendar(
            chart_rows(workspace, "visual_periods"), "cumulative_attributed_reported_profit_usd"
        ),
        charts.activity(chart_rows(workspace, "visual_activity"), "weekday_hour"),
    ]
    for chart in charts_to_test:
        assert chart is not None
        spec = chart.to_dict()
        assert spec.get("$schema")
    source = review_inputs()
    queue, _ = build_review_queues(*source, ReviewConfig())
    assert charts.evidence_map(queue["records"]).to_dict()["params"]
    assert charts.comparison(queue["records"][0]).to_dict()
    assert charts.comparison(queue["records"][0], contrast=True).to_dict()
