"""Frozen model selection, unit-preserving charts and no implicit UI fitting."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_dashboard_workspace
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest
from test_explanatory_models import candidates

from oddsjam_ev.analysis.explanatory_models import ModelPolicy, audit_features, fit_population
from oddsjam_ev.dashboard import explanation_charts as charts
from oddsjam_ev.dashboard import explanation_queries as queries
from oddsjam_ev.dashboard.artifacts import DashboardConfig
from oddsjam_ev.dashboard.explanation_artifacts import FEATURES, load_explanations
from oddsjam_ev.dashboard.pages.decision_workspace import sidebar
from oddsjam_ev.dashboard.pages.explanations import render
from oddsjam_ev.io.explanatory_models import publish_explanatory_models

attached = test_dashboard_workspace.attached
workspace_sources = test_dashboard_workspace.workspace_sources
UI_FIXTURE = None


@pytest.fixture
def model_rows():
    source = candidates(n=600).head(24).copy()
    source["advertised_ev_percent_exact"] = (source.advertised_ev_fraction * 100).map(str)
    source["event_name"] = "Synthetic event"
    result = fit_population(
        source,
        ModelPolicy(
            min_wagers=1,
            min_events=3,
            min_train_events=1,
            min_validation_events=1,
            min_test_events=1,
            background_rows=2,
        ),
    )
    assert result["status"] == "published"
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            result[key] = value.assign(funding="paper", product="fixed_odds")
    attribution = pd.DataFrame(
        {
            "bet_id": [*source.bet_id, *source.bet_id],
            "filter_id": "a",
            "attribution_status": "resolved",
        }
    )
    publication = {
        "predictions": result["records"],
        "publication_id": "global-frozen",
        "kind": "global",
        "specification": {
            "scope": {},
            "source_manifest_sha256": {"canonical": "synthetic", "visuals": "synthetic"},
            "limitations": ["Synthetic; model associations are not causal evidence."],
        },
        "audit": {
            "populations": [
                {
                    "funding": "paper",
                    "product": "fixed_odds",
                    **{
                        k: v
                        for k, v in result.items()
                        if not isinstance(v, pd.DataFrame) and k != "safe_model"
                    },
                }
            ]
        },
        "feature_inventory": audit_features({"model_candidates": source}),
    }
    for key in (
        "utility",
        "coverage",
        "training_features",
        "learning_curves",
        "permutation",
        "redundancy",
        "source_periods",
    ):
        publication[key] = result[key]
    return source, attribution, publication


def test_selector_identity_bands_boundaries_and_overlap(model_rows):
    source, attribution, publication = model_rows
    selected = queries.select_candidates(
        source,
        attribution,
        saved="a",
        selections={
            "sportsbook": ["A", "B"],
            "sport": ["basketball"],
            "advertised_ev_percent": ["> 3"],
        },
    )
    assert selected.bet_id.is_unique
    expected = source.loc[source.advertised_ev_percent_exact.map(float).gt(3)]
    assert set(selected.bet_id) == set(expected.bet_id)
    rows = queries.explanation_rows(publication, selected)
    assert_frame_equal(
        rows,
        publication["predictions"].loc[publication["predictions"].bet_id.isin(selected.bet_id)],
    )
    boundary = source.head(4).copy()
    boundary["advertised_ev_percent_exact"] = ["3", "3.0000000000000001", "2.5", "5"]
    exact = queries.select_candidates(
        boundary, attribution, selections={"advertised_ev_percent": ["> 3"]}
    )
    assert exact.bet_id.tolist() == boundary.bet_id.iloc[[1, 3]].tolist()
    band = queries.select_candidates(
        boundary, attribution, selections={"advertised_ev_band": ["[2.5, 5.0)"]}
    )
    assert band.bet_id.tolist() == boundary.bet_id.iloc[:3].tolist()
    assert queries.select_candidates(
        source, attribution, selections={"sportsbook": ["missing"]}
    ).empty
    with pytest.raises(ValueError):
        queries.select_candidates(source, attribution, selections={"profit": ["positive"]})
    with pytest.raises(ValueError):
        queries.select_candidates(
            source, attribution, selections={"advertised_ev_percent": ["approximately 3"]}
        )


def test_idea_intersection_and_reference_identity(model_rows):
    source, attribution, publication = model_rows
    idea = {
        "candidate_id": "idea",
        "execution_status": "paper",
        "product": "fixed_odds",
        "source_kind": "within_filter_corrected_family",
        "filter_id": "a",
        "predicate_json": json.dumps([{"dimension": "sportsbook", "operator": "==", "value": "A"}]),
    }
    selected = queries.select_candidates(source, attribution, idea=idea)
    assert selected.sportsbook.eq("A").all()
    rows = queries.explanation_rows(publication, selected, "test")
    assert rows.split.eq("test").all()
    if rows.empty:
        rows = queries.explanation_rows(publication, selected)
    reference = publication["predictions"]
    before = reference.copy(deep=True)
    summary = queries.importance(rows, reference)
    assert set(summary.model_id) == set(rows.model_id)
    assert set(summary.background_id) == set(rows.background_id)
    assert summary.query("cohort == 'Published reference'").wagers.eq(len(reference)).all()
    assert_frame_equal(reference, before)
    conflicts = queries.select_candidates(
        source, attribution, idea=idea, selections={"sportsbook": ["B"]}
    )
    assert conflicts.empty


def test_charts_preserve_units_and_bound_display_samples(model_rows):
    _, _, publication = model_rows
    rows = publication["predictions"]
    summary = queries.importance(rows, rows)
    sample = queries.chart_sample(rows, limit=10)
    assert len(sample) == 10
    assert_frame_equal(sample, queries.chart_sample(rows.sample(frac=1, random_state=5), limit=10))
    long = queries.distribution(sample)
    assert len(long) == len(sample) * len(FEATURES)
    local = queries.waterfall(rows.iloc[0])
    assert np.isclose(local.iloc[1:-1].change.sum() + local.iloc[0].end, local.iloc[-1].end)
    for chart in (
        charts.importance(summary),
        charts.importance(summary, True),
        charts.distribution(long),
        charts.dependence(sample, "advertised_ev_fraction", "Advertised EV", "sportsbook"),
        charts.dependence(sample, "sportsbook", "Platform", "sportsbook"),
        charts.waterfall(local),
        charts.utility(publication["utility"]),
        charts.calibration(rows),
        charts.residuals(rows, "sportsbook"),
    ):
        spec = chart.to_dict(validate=True)
        assert "data" in spec or "layer" in spec
    assert not queries.explanation_rows(publication, rows.iloc[:0]).shape[0]


def test_offline_requests_never_silently_broaden():
    scope, blockers = queries.offline_scope(
        {"sportsbook": ["A"], "advertised_ev_band": ["[2.5, 5)"], "advertised_ev_percent": ["> 3"]},
        mode="paper",
        product="exchange",
    )
    assert not blockers
    assert scope["advertised_ev_fraction"] == {"ge": 0.025, "lt": 0.05, "gt": 0.03}
    bounded, blocked = queries.offline_scope(
        {}, base_scope={"sportsbook": ["A"], "advertised_ev_fraction": {"gt": 0.03}}
    )
    assert not blocked and bounded["sportsbook"] == ["A"]
    assert bounded["advertised_ev_fraction"] == {"gt": 0.03}
    assert queries.offline_scope({"sportsbook": ["B"]}, base_scope={"sportsbook": ["A"]})[1]
    assert queries.offline_scope({}, saved="a")[1]
    assert queries.offline_scope({"odds_band": ["[100, 150)"]})[1]
    assert queries.offline_scope({"advertised_ev_band": ["[0, 3)", "[5, 10)"]})[1]


def render_fixture():
    import streamlit as st

    run, workspace, attachment = UI_FIXTURE
    st.session_state.setdefault("page", "Explain returns")
    st.sidebar.selectbox("Explore", ["Overview", "Explain returns"], key="page")
    st.session_state.explanation_bundle = attachment
    st.session_state.decision_workspace = workspace
    sidebar(workspace, run)
    render(run, "all", 20)


def test_ui_fixed_global_subset_selection_local_and_no_training(model_rows, monkeypatch):
    import test_dashboard_explanations as module

    import oddsjam_ev.analysis.explanatory_models as modeling
    import oddsjam_ev.io.explanatory_models as publishing

    source, attribution, publication = model_rows
    subset = copy.deepcopy(publication)
    subset["publication_id"] = "separate-subset"
    subset["kind"] = "subset"
    subset["specification"]["scope"] = {"sportsbook": ["A"]}
    subset["predictions"] = subset["predictions"].query("sportsbook == 'A'").copy()
    subset["predictions"]["model_id"] = "separate-fitted-model"
    subset["predictions"]["background_id"] = "separate-training-reference"
    subset["audit"]["populations"][0]["model_id"] = "separate-fitted-model"
    subset["audit"]["populations"][0]["background_id"] = "separate-training-reference"
    run = SimpleNamespace(
        manifests={
            "canonical": {
                "specification": {"filter_registry": {"filters": [{"id": "a", "name": "Saved A"}]}}
            }
        },
        tables={
            "canonical": {"bet_facts": source, "filter_attribution": attribution},
            "atlas": {
                "wager_dimensions": source.assign(
                    odds_band="[-150, -100)", advertised_ev_band="[2.5, 10)"
                )
            },
        },
    )
    workspace = {"queues": {"records": []}, "visuals": {"visual_candidates": source}}
    bundle = {
        "publications": {"global-frozen": publication, "separate-subset": subset},
        "global_id": "global-frozen",
    }
    monkeypatch.setattr(module, "UI_FIXTURE", (run, workspace, bundle))

    def forbidden(*args, **kwargs):
        raise AssertionError("UI must not fit or publish models")

    monkeypatch.setattr(modeling, "fit_population", forbidden)
    monkeypatch.setattr(publishing, "publish_explanatory_models", forbidden)
    app = AppTest.from_string(
        "from test_dashboard_explanations import render_fixture\nrender_fixture()"
    ).run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key="explanation_publication").value == "global-frozen"
    original_ids = set(publication["predictions"].model_id)
    app.sidebar.multiselect(key="filter_signal_sportsbook").select("A").run(timeout=30)
    app.sidebar.multiselect(key="filter_signal_sport").select("basketball").run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key="explanation_publication").value == "global-frozen"
    assert set(publication["predictions"].model_id) == original_ids
    app.sidebar.selectbox(key="page").set_value("Overview").run(timeout=30)
    assert app.sidebar.multiselect(key="filter_signal_sportsbook").value == ["A"]
    assert app.sidebar.multiselect(key="filter_signal_sport").value == ["basketball"]
    app.sidebar.selectbox(key="page").set_value("Explain returns").run(timeout=30)
    assert app.selectbox(key="explanation_publication").value == "global-frozen"

    ids = app.selectbox(key="explanation_local").options
    app.selectbox(key="explanation_local").set_value(ids[0]).run(timeout=30)
    assert any(m.label == "Frozen prediction" for m in app.metric)
    app.selectbox(key="explanation_publication").set_value("separate-subset").run(timeout=30)
    assert not app.exception and not app.error
    assert any("Separately fitted subset model" in item.value for item in app.warning)
    app.sidebar.multiselect(key="filter_signal_sportsbook").unselect("A").select("B").run(
        timeout=30
    )
    assert app.selectbox(key="explanation_publication").value == "separate-subset"
    assert any("No published explanation rows" in item.value for item in app.info)


def test_verified_attachments_cache_and_no_fallback(workspace_sources, tmp_path):
    run, cfg, _, visuals = workspace_sources
    canonical = __import__("pathlib").Path(run.result["outputs"]["canonical"]["path"])
    output = publish_explanatory_models(canonical, visuals, tmp_path / "models", ModelPolicy())
    cfg = replace(
        cfg, explanation_publications={run.result["run_id"]: {"global": str(output), "subsets": []}}
    )
    workspace = test_dashboard_workspace.get_workspace(run, cfg)
    bundle = load_explanations(run, cfg, workspace)
    assert load_explanations(run, cfg, workspace, bundle) is bundle
    assert (
        load_explanations(run, replace(cfg, explanation_publications={}), workspace, bundle) is None
    )
    with pytest.raises(ValueError, match="outside"):
        load_explanations(run, replace(cfg, evidence_root=str(visuals)), workspace)
    with pytest.raises(ValueError, match="lineage"):
        load_explanations(run, cfg, None)
    (output / "predictions.parquet").write_bytes(b"CORRUPT_PRIVATE_MARKER")
    with pytest.raises(ValueError, match="integrity"):
        load_explanations(run, cfg, workspace, bundle)


def test_config_rejects_implicit_or_duplicate_subset_attachments():
    with pytest.raises(ValueError, match="Explicit global"):
        DashboardConfig(explanation_publications={"a" * 64: {"global": "x", "subsets": ["x"]}})


def test_app_clears_corrupt_models_and_keeps_other_research(
    workspace_sources, tmp_path, monkeypatch
):
    from dataclasses import asdict
    from pathlib import Path

    import yaml

    run, cfg, _, visuals = workspace_sources
    canonical = Path(run.result["outputs"]["canonical"]["path"])
    output = publish_explanatory_models(canonical, visuals, tmp_path / "models", ModelPolicy())
    cfg = replace(
        cfg, explanation_publications={run.result["run_id"]: {"global": str(output), "subsets": []}}
    )
    config = tmp_path / "viewer.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    monkeypatch.setenv("ODDSJAM_DASHBOARD_CONFIG", str(config))
    app = AppTest.from_file(str(test_dashboard_workspace.APP)).run(timeout=30)
    app.sidebar.selectbox(key="page").set_value("Explain returns").run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["explanation_bundle"] is not None
    (output / "predictions.parquet").write_bytes(b"PRIVATE_CORRUPTION_MARKER")
    app.run(timeout=30)
    assert not app.exception and app.error
    assert "explanation_bundle" not in app.session_state
    assert "published_run" in app.session_state
    assert "decision_workspace" in app.session_state
    assert not app.metric
    assert all("PRIVATE_CORRUPTION_MARKER" not in error.value for error in app.error)


def test_model_symlinks_and_wrong_source_lineage_rejected(workspace_sources, tmp_path):
    from pathlib import Path

    from oddsjam_ev.io.source_csv import fingerprint

    run, cfg, _, visuals = workspace_sources
    output = publish_explanatory_models(
        Path(run.result["outputs"]["canonical"]["path"]),
        visuals,
        tmp_path / "models",
        ModelPolicy(),
    )
    cfg = replace(
        cfg, explanation_publications={run.result["run_id"]: {"global": str(output), "subsets": []}}
    )
    workspace = test_dashboard_workspace.get_workspace(run, cfg)
    target = output / "predictions.parquet"
    original = target.read_bytes()
    outside = tmp_path.parent / (tmp_path.name + "-external.parquet")
    outside.write_bytes(original)
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        load_explanations(run, cfg, workspace)
    target.unlink()
    target.write_bytes(original)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["specification"]["source_manifest_sha256"]["canonical"] = "another-source"
    manifest["publication_id"] = fingerprint(manifest["specification"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lineage"):
        load_explanations(run, cfg, workspace)
