"""Exact intersection selection, missing evidence and non-mutating presentation."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.dashboard.signal_filters import filter_signals, signal_specificity


def test_and_across_fields_or_within_field_and_missing_dimensions():
    source = pd.DataFrame(
        {
            "predicate_json": [
                '{"sportsbook":"Novig"}',
                '{"sportsbook":"Novig","sport":"basketball","market_family":"player_prop"}',
                '{"sportsbook":"Prophet X","sport":"basketball","market_family":"player_prop"}',
                '{"sportsbook":"Novig","sport":"tennis"}',
                '{"sport":"basketball"}',
            ],
            "reported_roi": [0.1, 0.2, -0.1, 0.3, 0.4],
        }
    )
    before = source.copy(deep=True)
    selected = filter_signals(
        source,
        {
            "sportsbook": ["Novig", "Prophet X"],
            "sport": ["basketball"],
            "market_family": ["player_prop"],
        },
    )
    assert selected.index.tolist() == [1, 2]
    assert selected.reported_roi.tolist() == [0.2, -0.1]
    assert filter_signals(source, {"sportsbook": ["Novig"], "league": ["NBA"]}).empty
    assert_frame_equal(source, filter_signals(source, {"sportsbook": []}))
    assert signal_specificity(source).dimension_count.tolist() == [1, 3, 3, 2, 1]
    assert_frame_equal(before, source)
    with pytest.raises(ValueError):
        filter_signals(source, {"invented_dimension": ["value"]})
    assert filter_signals(source.iloc[:0], {"sport": ["tennis"]}).empty


def test_minimum_wagers_boundary_preserves_metrics():
    from oddsjam_ev.dashboard.signal_filters import minimum_support

    frame = pd.DataFrame(
        {"eligible_count": [5, 99, 100, 101], "reported_roi": [-1.0, 0.2, 0.1, -0.1]}
    )
    assert minimum_support(frame, 100).eligible_count.tolist() == [100, 101]
    assert minimum_support(frame, 5).reported_roi.tolist() == [-1.0, 0.2, 0.1, -0.1]
    with pytest.raises(ValueError):
        minimum_support(frame, 0)


def test_minimum_display_configuration():
    from oddsjam_ev.dashboard.artifacts import DashboardConfig

    assert DashboardConfig().min_analyzed_wagers == 100
    assert DashboardConfig(min_analyzed_wagers=25).min_analyzed_wagers == 25
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            DashboardConfig(min_analyzed_wagers=value)
