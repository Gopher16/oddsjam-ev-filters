"""Exact registry and attribution contracts without canonical fact dependencies."""

import pandas as pd
import pytest

from oddsjam_ev.analysis.filter_attribution import (
    build_filter_attribution,
    resolve_labels,
    validate_registry,
)


def registry():
    return {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [
            {"id": "a", "name": "PROD - A (Slow, Build)", "former_names": ["Old A"]},
            {"id": "b", "name": "TEST - B", "clone_of": "a"},
        ],
    }


def test_literal_comma_and_memberships():
    mapping = validate_registry(registry())
    assert resolve_labels("PROD - A (Slow, Build)", mapping) == (["a"], "exact")
    assert resolve_labels("PROD - A (Slow, Build), TEST - B", mapping) == (
        ["a", "b"],
        "resolved_memberships",
    )
    assert resolve_labels("Old A", mapping) == (["a"], "exact")
    assert resolve_labels("PROD - A (Slow, Build), Unknown", mapping) == ([], "unresolved")
    assert resolve_labels("", mapping) == ([], "missing")


def test_memberships_do_not_multiply_economics():
    lineage = pd.DataFrame(
        {"bet_id": ["bet", "bet"], "saved_filter_names": ["Old A, TEST - B", "Old A"]}
    )
    bridge = build_filter_attribution(lineage, registry())
    assert len(bridge) == 2
    assert not bridge.duplicated(["bet_id", "filter_id"]).any()
    assert {"stake", "profit", "bet_profit", "reported_profit_usd"}.isdisjoint(bridge.columns)
    assert set(bridge.historical_settings_status) == {"configuration_effectivity_unknown"}


@pytest.mark.parametrize("change", ["dangling", "cycle", "alias", "id"])
def test_invalid_registry(change):
    value = registry()
    if change == "dangling":
        value["filters"][1]["clone_of"] = "missing"
    if change == "cycle":
        value["filters"][0]["clone_of"] = "b"
    if change == "alias":
        value["filters"][1]["former_names"] = ["Old A"]
    if change == "id":
        value["filters"][1]["id"] = "a"
    with pytest.raises(ValueError):
        validate_registry(value)


def test_ambiguous_segmentation_stays_unresolved():
    mapping = {"A": "a", "B, C": "bc", "A, B": "ab", "C": "c"}
    assert resolve_labels("A, B, C", mapping) == ([], "unresolved")


def test_unresolved_labels_never_dropped():
    table = build_filter_attribution(
        pd.DataFrame({"bet_id": ["a", "b"], "saved_filter_names": ["", "Unknown source label"]}),
        registry(),
    )
    assert len(table) == 2
    assert set(table.attribution_status) == {"missing", "unresolved"}
    assert table.filter_id.isna().all()
