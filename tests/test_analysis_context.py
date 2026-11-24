"""Synthetic portable scope and transition contracts."""

import json
from dataclasses import replace

import pytest

from oddsjam_ev.analysis.analysis_context import AnalysisContext


def context(**changes):
    return replace(
        AnalysisContext(
            "source",
            "config",
            "investigation",
            (("visuals", "sha"),),
            "paper",
            "exchange",
            "discretionary",
        ),
        **changes,
    )


def test_round_trips_and_empty_scope():
    c = context(wager_ids=(), event_ids=("event",), focus="drawdown")
    assert AnalysisContext.from_json(c.to_json()) == c
    assert AnalysisContext.from_token(c.to_token()) == c
    assert c.wager_ids == ()
    for view in ("explanations", "risk", "charts"):
        new, changes = c.transition(view=view)
        assert new.wager_ids == () and new.event_ids == ("event",)
        assert set(changes) <= {"view"}


def test_reset_is_explicit_visible_and_preserves_pins():
    c = context(wager_ids=("a",), focus="event_cluster")
    new, changes = c.reset_selection()
    assert new.wager_ids is None and new.publications == c.publications
    assert new.investigation_id == c.investigation_id
    assert set(changes) == {"wager_ids", "focus"}


@pytest.mark.parametrize(
    "changes",
    [
        {"funding": "all"},
        {"product": "mixed"},
        {"contract_version": True},
        {"wager_ids": ("a", "a")},
        {"event_ids": ["a"]},
        {"weighting": "unknown"},
        {"publications": (("a", "x"), ("a", "y"))},
        {"view": "other"},
        {"start": "2026-01-01"},
        {"time_column": "created_at"},
        {"start": "2026-01-02Z", "end": "2026-01-01Z", "time_column": "time"},
    ],
)
def test_invalid_context(changes):
    with pytest.raises(ValueError):
        context(**changes)


def test_strict_serialization():
    data = json.loads(context().to_json())
    data["surprise"] = 1
    with pytest.raises(ValueError):
        AnalysisContext.from_json(json.dumps(data))
    with pytest.raises(ValueError):
        AnalysisContext.from_json('{"view":"charts","view":"risk"}')
    for raw in ("!", "x" * 30000):
        with pytest.raises(ValueError):
            AnalysisContext.from_token(raw)
    with pytest.raises(ValueError):
        context(wager_ids=("x" * 17000,)).to_json()


def test_reserved_pins_and_missing_version():
    with pytest.raises(ValueError):
        context(publications=(("source_id", "forged"),))
    data = json.loads(context().to_json())
    del data["contract_version"]
    with pytest.raises(ValueError):
        AnalysisContext.from_json(json.dumps(data))
