"""Stakeholder labels preserve identities, provenance dates and missing evidence."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.dashboard import charts, queries
from oddsjam_ev.dashboard.artifacts import catalogue_labels, load_published_run
from oddsjam_ev.dashboard.labels import (
    pretty,
    report_dates,
    report_label,
    sample_label,
    segment_label,
)


def test_manifest_dates_are_not_decoded_from_hashes(dashboard_run):
    root, run_id = dashboard_run
    labels = catalogue_labels(root)
    assert run_id in labels and run_id not in labels[run_id]
    assert run_id[:8] not in labels[run_id]
    assert "Generated " in labels[run_id]
    value = report_label(
        {"mode": "historical_reanalysis", "as_of": "2026-10-23T16:00:00Z"}, "20250101" + "a" * 56
    )
    assert "Oct 23, 2026" in value and "11:00 AM CDT" in value
    run = load_published_run(root, run_id)
    dates = report_dates(run)
    assert dates["Export date"] == "Not confirmed for all sources"
    assert dates["Data last received"]
    with pytest.raises(ValueError):
        report_label({"mode": "historical_reanalysis", "as_of": "2026-10-23"}, run_id)


def test_readable_segments_and_bands_are_not_saved_filter_names():
    assert pretty("[2.5, 5.0)", "advertised_ev_band") == "2.5% to under 5%"
    assert pretty("[20.0, inf)", "advertised_ev_band") == "20% and above"
    assert pretty("[-inf, 0.0)", "advertised_ev_band") == "Below 0%"
    assert pretty("[100.0, 150.0)", "odds_band") == "+100 to under +150"
    assert pretty("<unmapped>", "market_family") == "Market not mapped"
    assert pretty("<missing>", "advertised_ev_band") == "Advertised EV not recorded"
    row = {
        "product": "exchange",
        "predicate_json": '{"sportsbook":"Novig","market_family":"player_prop","advertised_ev_band":"[5.0, 10.0)"}',
    }
    label = segment_label(row)
    assert "Novig" in label and "Player props" in label and "5% to under 10%" in label
    assert "_json" not in label and "candidate:" not in label
    assert "very small sample" in sample_label(2)


def test_one_product_and_mode_only_and_no_source_metric_mutation(dashboard_run):
    run = load_published_run(*dashboard_run)
    source = run.tables["atlas"]["wager_segments"].copy(deep=True)
    selected = queries.product_cohort(run.tables, "paper", "fixed_odds")
    assert len(selected) == 1 and selected.iloc[0].eligible_reported_profit_usd == 12
    assert queries.product_cohort(run.tables, "real_money", "fixed_odds").empty
    frame = queries.chart_cohorts(run.tables, "paper", "fixed_odds", "advertised_ev_band")
    assert frame.eligible_count.sum() == 1
    assert frame.reported_roi.iloc[0] == pytest.approx(1.2)
    for chart in [
        charts.bars(frame, "reported_roi", "Reported ROI", percentage=True),
        charts.scatter(frame),
    ]:
        assert chart is not None
        spec = chart.to_dict()
        assert spec["datasets"]
    text = " ".join(queries.takeaways(run.tables, "paper", "fixed_odds"))
    assert "very small sample" in text and "promotion" in text
    assert_frame_equal(source, run.tables["atlas"]["wager_segments"])


def test_charts_exclude_unavailable_roi_without_zero_imputation():
    frame = pd.DataFrame(
        {
            "label": ["Excluded"],
            "reported_roi": [None],
            "eligible_count": [0],
            "eligible_reported_profit_usd": [None],
            "sample": ["No analyzed wagers"],
        }
    )
    assert charts.bars(frame, "reported_roi", "Reported ROI", percentage=True) is None
    assert charts.scatter(frame) is None


def test_research_queue_orders_breadth_deduplicates_actions_and_separates_sparse_results():
    evidence = pd.DataFrame(
        [
            {
                "candidate_id": "tiny",
                "execution_status": "paper",
                "product": "exchange",
                "eligible_count": 2,
                "event_proxy_groups": 2,
                "reported_roi": 5.0,
            },
            {
                "candidate_id": "broad",
                "execution_status": "paper",
                "product": "exchange",
                "eligible_count": 5856,
                "event_proxy_groups": 4071,
                "reported_roi": 0.022,
            },
            {
                "candidate_id": "loss",
                "execution_status": "paper",
                "product": "exchange",
                "eligible_count": 100,
                "event_proxy_groups": 70,
                "reported_roi": -0.1,
            },
            {
                "candidate_id": "other",
                "execution_status": "real_money",
                "product": "exchange",
                "eligible_count": 10000,
                "event_proxy_groups": 9000,
                "reported_roi": 0.1,
            },
            {
                "candidate_id": "missing",
                "execution_status": "paper",
                "product": "exchange",
                "eligible_count": 0,
                "event_proxy_groups": None,
                "reported_roi": None,
            },
        ]
    )
    proposals = pd.concat([evidence, evidence.iloc[[1]]], ignore_index=True)
    tables = {
        "discovery": {"candidate_results": evidence},
        "recommendations": {"proposals": proposals},
    }
    before = evidence.copy(deep=True)
    queue = queries.research_queue(tables, "paper", "exchange")
    assert queue.candidate_id.tolist() == ["broad", "loss", "tiny", "missing"]
    assert queue.set_index("candidate_id").section.to_dict() == {
        "broad": "Positive historical patterns",
        "loss": "Loss investigations",
        "tiny": "More observations needed",
        "missing": "More observations needed",
    }
    assert_frame_equal(before, evidence)
    assert queries.research_queue(tables, "paper", "fixed_odds").empty


def test_finding_explains_interval_including_zero_and_next_question():
    row = pd.Series(
        {
            "product": "exchange",
            "predicate_json": '{"sportsbook":"Novig"}',
            "eligible_count": 5856,
            "event_proxy_groups": 4071,
            "reported_roi": 0.022,
            "eligible_reported_profit_usd": 16449.22,
            "cluster_roi_ci_low": -0.0131,
            "cluster_roi_ci_high": 0.0562,
        }
    )
    title, result, detail, question = queries.research_finding(row)
    assert "Novig" in title and "5,856" in result
    assert "includes zero" in detail and "4,071" in detail
    assert "prospectively" in question and "available" in question


def test_completion_labels_order_publications_not_replays(dashboard_run):
    import json

    root, run_id = dashboard_run
    attempts = root / "attempts"
    # A later successful replay cannot replace the original completion time.
    original = catalogue_labels(root)[run_id]
    replay = attempts / ("a" * 32)
    replay.mkdir(exist_ok=True)
    (replay / "outcome.json").write_text(
        json.dumps(
            {
                "status": "completed_historical",
                "run_id": run_id,
                "completed_at": "2099-01-01T00:00:00Z",
            }
        )
    )
    assert catalogue_labels(root)[run_id] == original
    for outcome in attempts.glob("*/outcome.json"):
        record = json.loads(outcome.read_text())
        record.pop("completed_at", None)
        outcome.write_text(json.dumps(record))
    assert "Generation time unavailable" in catalogue_labels(root)[run_id]


def test_all_data_browser_preserves_populations_and_metrics(dashboard_run):
    from oddsjam_ev.dashboard.pages.workspace import selected_queue

    run = load_published_run(*dashboard_run)
    source = run.tables["discovery"]["candidate_results"].copy(deep=True)
    all_rows = selected_queue(run, "all", "all")
    for mode in queries.MODES:
        for product in run.tables["canonical"]["economic_summary"]["product"].unique():
            expected = queries.research_queue(run.tables, mode, product)
            selected = all_rows.loc[
                all_rows.execution_status.eq(mode) & all_rows["product"].eq(product)
            ]
            assert sorted(selected.candidate_id) == sorted(expected.candidate_id)
    assert_frame_equal(source, run.tables["discovery"]["candidate_results"])


def test_saved_filter_scope_uses_exact_membership_not_global_candidates(dashboard_run):
    from oddsjam_ev.dashboard.pages.workspace import selected_queue

    run = load_published_run(*dashboard_run)
    diagnostics = run.tables["atlas"]["filter_segment_diagnostics"]
    identities = diagnostics.filter_id.dropna().unique()
    assert len(identities)
    identity = identities[0]
    selected = selected_queue(run, "all", "all", identity)
    expected = diagnostics.loc[
        diagnostics.filter_id.eq(identity) & diagnostics.view.ne("filter_id__attribution_status")
    ]
    assert len(selected) == len(expected)
    assert sorted(selected.reported_roi.dropna()) == sorted(expected.reported_roi.dropna())
    assert selected.cluster_roi_ci_low.isna().all()
    assert selected_queue(run, "all", "all", "nonexistent_filter").empty
