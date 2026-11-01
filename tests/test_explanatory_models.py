"""Synthetic event-disjoint model and immutable explanation contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.explanatory_models import (
    FEATURES,
    ModelPolicy,
    apply_scope,
    audit_features,
    fit_population,
    prepare_candidates,
)
from oddsjam_ev.io.explanatory_models import select_published_predictions


def candidates(signal=True, n=600):
    rng = np.random.default_rng(8)
    ev = rng.uniform(0.02, 0.1, n)
    returns = ev * 10 - 0.5 if signal else np.zeros(n)
    frame = pd.DataFrame(
        {
            "bet_id": [f"b{i:04}" for i in range(n)],
            "namespace": "verified-source",
            "game_id": [f"e{i // 2}" for i in range(n)],
            "execution_status": "paper",
            "product": "fixed_odds",
            "diagnostic_eligible": True,
            "stake": rng.uniform(10, 500, n),
            "unit_return": returns,
            "odds": -110,
            "sportsbook": np.where(ev > 0.06, "A", "B"),
            "sport": "basketball",
            "league": "NBA",
            "market_family": "spread",
            "advertised_ev_fraction": ev,
            "clv": returns,
            "created_at": "unverified",
        }
    )
    frame["profit"] = frame.stake * returns
    return prepare_candidates(frame)


def test_signal_null_splits_and_additivity():
    policy = ModelPolicy()
    result = fit_population(candidates(), policy)
    assert result["selected_model"] != "constant"
    assert result["test_improves_both_references"]
    records = result["records"]
    assert records.groupby("event_key").split.nunique().max() == 1
    assert set(records.loc[records.background_member, "split"]) == {"train"}
    assert records.background_member.sum() == 16
    assert np.allclose(
        records.predicted_return,
        records.base_return + records[[f"shap__{f}" for f in FEATURES]].sum(axis=1),
        atol=1e-6,
    )
    assert records.reconstruction_error.abs().max() < 1e-6
    null = fit_population(candidates(False), policy)
    assert null["selected_model"] == "constant"
    assert not null["test_improves_both_references"]
    assert "no demonstrated" in null["interpretation"]


def test_determinism_leakage_and_fixed_model_selection():
    data = candidates()
    first = fit_population(data, ModelPolicy())
    data["clv"] = 1234
    data["created_at"] = "tomorrow"
    second = fit_population(data.sample(frac=1, random_state=1), ModelPolicy())
    assert_frame_equal(first["records"], second["records"])
    subset = select_published_predictions({"predictions": first["records"]}, ["b0001", "b0009"])
    assert subset.model_id.nunique() == 1
    assert subset.background_id.nunique() == 1
    assert_frame_equal(subset, first["records"][first["records"].bet_id.isin(["b0001", "b0009"])])
    with pytest.raises(ValueError):
        select_published_predictions({"predictions": first["records"]}, ["missing"])


def test_scope_and_decline_support():
    data = candidates()
    assert (
        apply_scope(
            data, {"advertised_ev_fraction": {"gt": 0.03}, "sportsbook": ["A"]}
        ).advertised_ev_fraction.min()
        > 0.03
    )
    for scope in (
        {"profit": {"gt": 0}},
        {"sportsbook": []},
        {"advertised_ev_fraction": {"approximately": 0.03}},
    ):
        with pytest.raises(ValueError):
            apply_scope(data, scope)
    assert fit_population(data.head(20), ModelPolicy())["status"] == "declined"
    data.loc[0, "event_key"] = None
    assert "incomplete_event_ids" in fit_population(data, ModelPolicy())["reasons"]
    with pytest.raises(ValueError, match="separately"):
        fit_population(data.assign(product=["x", *["y"] * 599]), ModelPolicy())
    with pytest.raises(ValueError, match="reconcile"):
        fit_population(candidates().assign(profit=0), ModelPolicy())
    assert (
        "row_budget_exceeded"
        in fit_population(candidates(), replace(ModelPolicy(), max_rows=100))["reasons"]
    )


def test_inventory_unknowns_and_train_only_categories():
    data = candidates()
    baseline = fit_population(data, ModelPolicy())
    heldout = baseline["records"].query("split != 'train'").bet_id
    data.loc[data.bet_id.isin(heldout), "sportsbook"] = "unseen"
    result = fit_population(data, ModelPolicy())
    assert "unseen" not in str(result["safe_model"]["preprocessing"]["categories"])
    assert result["coverage"].unseen_category.sum() == len(heldout)
    audit = audit_features({"model_candidates": data})
    assert not audit.loc[
        audit.field.isin(["clv", "profit", "created_at", "stake"]), "model_eligible"
    ].any()
    assert set(audit.loc[audit.model_eligible, "field"]) == set(FEATURES)
    assert len(audit) == len(data.columns)


def test_forest_shap_and_unknown_numeric_training_imputation():
    data = candidates()
    # A non-linear signal forces the bounded tree comparator over the linear model.
    data["unit_return"] = np.where(data.advertised_ev_fraction > 0.06, 1.0, -1.0)
    data["profit"] = data.stake * data.unit_return
    data["sportsbook"] = "constant"
    result = fit_population(data, ModelPolicy())
    assert result["selected_model"] == "forest"
    assert result["records"].reconstruction_error.abs().max() < 1e-6
    assert len(result["safe_model"]["trees"]) == 64
    data["advertised_ev_fraction"] = np.nan
    data["offered_decimal_odds"] = np.nan
    assert "no_estimable_feature_variation" in fit_population(data, ModelPolicy())["reasons"]


def test_invalid_policy_and_duplicate_grain():
    with pytest.raises(ValueError):
        ModelPolicy(background_rows=129)
    with pytest.raises(ValueError):
        prepare_candidates(pd.concat([candidates(), candidates()]))


def test_grouped_permutation_matches_exact_shap_for_pairwise_signal():
    import shap

    from oddsjam_ev.analysis.explanatory_models import permutation_shap

    class Pairwise:
        def predict(self, x):
            return x[:, 0] * x[:, 1] + 2 * x[:, 2]

    x = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    background = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    model = Pairwise()
    contributions, sd, reference = permutation_shap(
        model, x, background, {"a": [0], "b": [1], "c": [2]}, 10
    )
    exact = shap.ExactExplainer(model.predict, background)(x)
    assert np.allclose(contributions, exact.values)
    assert np.allclose(reference, exact.base_values)
    assert np.isfinite(sd).all()


def test_forest_explanation_budget_and_learning_curves():
    data = candidates()
    data["unit_return"] = np.where(data.advertised_ev_fraction > 0.06, 1.0, -1.0)
    data["profit"] = data.stake * data.unit_return
    data["sportsbook"] = "constant"
    declined = fit_population(data, replace(ModelPolicy(), max_explanation_work=1))
    assert declined["reasons"] == ["explanation_work_budget_exceeded"]
    result = fit_population(candidates(), ModelPolicy())
    assert len(result["learning_curves"]) == 6
    assert set(result["learning_curves"].training_event_fraction) == {0.5, 1.0}
    assert len(result["permutation"]) == len(FEATURES)
    assert result["training_features"].constant_train.any()
