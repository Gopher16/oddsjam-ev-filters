"""Linked context uses exact evidence; unsupported requests never widen scope."""

import pandas as pd
import pytest
from test_analysis_context import context

from oddsjam_ev.dashboard.context_queries import (
    context_chart_rows,
    context_explanations,
    select_context_rows,
)


def fixtures():
    c = context(publications=(("visuals", "sha"), ("explanations", "model-sha")))
    bindings = {
        "source_id": "source",
        "configuration_version": "config",
        "visuals": "sha",
        "explanations": "model-sha",
    }
    rows = pd.DataFrame(
        {
            "bet_id": ["a", "b", "c"],
            "funding": ["paper", "paper", "real_money"],
            "product": ["exchange"] * 3,
            "selection_regime": ["discretionary"] * 3,
            "event_id": ["e1", "e2", "e1"],
            "observed_at": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"],
        }
    )
    return c, bindings, rows


def test_event_drawdown_handoff_to_explanations_and_risk():
    c, pins, rows = fixtures()
    c, _ = c.transition(
        event_ids=("e1",),
        wager_ids=("a",),
        focus="drawdown",
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
        time_column="observed_at",
    )
    expected = select_context_rows(c, rows, pins).rows
    for view in ("charts", "explanations", "risk"):
        new, changes = c.transition(view=view)
        assert set(changes) <= {"view"}
        pd.testing.assert_frame_equal(select_context_rows(new, rows, pins).rows, expected)
    assert list(expected.bet_id) == ["a"]
    result = context_explanations(c, rows, {"predictions": rows.assign(split="test")}, pins)
    assert result.reasons == () and list(result.rows.bet_id) == ["a"]


@pytest.mark.parametrize("field,ids", [("wager_ids", ()), ("event_ids", ()), ("cohort_ids", ())])
def test_explicit_empty_never_means_all(field, ids):
    c, pins, rows = fixtures()
    c, _ = c.transition(**{field: ids})
    result = select_context_rows(c, rows, pins)
    assert result.rows.empty and not result.reasons


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"wager_ids": ("c",)}, "unknown_selected_id:wager_ids"),
        ({"event_ids": ("absent",)}, "unknown_selected_id:event_ids"),
        ({"cohort_ids": ("missing",)}, "unknown_selected_id:cohort_ids"),
    ],
)
def test_selected_id_integrity(changes, reason):
    c, pins, rows = fixtures()
    result = select_context_rows(c.transition(**changes)[0], rows, pins)
    assert result.rows.empty and reason in result.reasons


def test_stale_pins_missing_columns_duplicate_identity():
    c, pins, rows = fixtures()
    for key in pins:
        result = select_context_rows(c, rows, {**pins, key: "stale"})
        assert result.rows.empty and result.reasons
    assert select_context_rows(c, rows.drop(columns="selection_regime"), pins).reasons
    assert select_context_rows(c, pd.concat([rows, rows]), pins).reasons


def test_membership_intersection_and_boundary():
    c, pins, rows = fixtures()
    c, _ = c.transition(cohort_ids=("cohort",), event_ids=("e2",))
    result = select_context_rows(c, rows, pins, memberships={"cohort": ["a"]})
    assert result.rows.empty and not result.reasons
    assert select_context_rows(c, rows, pins, memberships={"cohort": ["c"]}).reasons
    c = context(start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z", time_column="observed_at")
    assert list(select_context_rows(c, rows, pins).rows.bet_id) == ["a"]
    rows["observed_at"] = "2026-01-01"
    assert select_context_rows(c, rows, pins).reasons == ("missing_or_naive_time_evidence",)


def test_explanation_coverage_and_pin_are_required():
    c, pins, rows = fixtures()
    publication = {"predictions": rows.iloc[:1].assign(split="test")}
    assert context_explanations(c, rows, publication, pins).reasons
    c = context(wager_ids=())
    assert "missing_explanations_pin" in context_explanations(c, rows, publication, pins).reasons


def test_aggregate_exact_scope_and_no_raw_selection_fallback():
    c, pins, _ = fixtures()
    frame = pd.DataFrame(
        {
            "cohort_fingerprint": ["x", "y"],
            "funding": ["paper"] * 2,
            "product": ["exchange"] * 2,
            "selection_regime": ["discretionary"] * 2,
            "weighting": ["equal_stake"] * 2,
            "roi": [0.1, -0.1],
        }
    )
    c, _ = c.transition(cohort_ids=("x",))
    assert list(context_chart_rows(c, frame, pins).rows.roi) == [0.1]
    assert context_chart_rows(c.transition(wager_ids=("a",))[0], frame, pins).reasons
    assert context_chart_rows(c.transition(weighting="stake_weighted")[0], frame, pins).reasons
    assert context_chart_rows(c, frame.drop(columns="selection_regime"), pins).reasons
    assert context_chart_rows(c.transition(cohort_ids=())[0], frame, pins).rows.empty


def test_explanation_weighting_is_never_silently_converted():
    c, pins, rows = fixtures()
    c, _ = c.transition(weighting="stake_weighted")
    result = context_explanations(c, rows, {"predictions": rows.assign(split="test")}, pins)
    assert result.rows.empty and "unsupported_explanation_weighting" in result.reasons
