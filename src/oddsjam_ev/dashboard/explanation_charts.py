"""
explanation_charts.py
===============================================================================
Purpose
-------
Interactive, unit-explicit views of already published model associations.

What this module provides
-------------------------
Grouped importance, distributions, dependence, waterfall and model-quality plots.

Design Notes
------------
Every chart is one population/model/reference; no fitting or causal interpretation.
===============================================================================
"""

import altair as alt
import pandas as pd

from oddsjam_ev.dashboard.decision_charts import data, zero

PALETTE = ["#55C8B6", "#8AB4FF", "#FFC977", "#EC97AD", "#BEABFA", "#8BD3E6"]


def importance(frame, signed=False):
    """Compare full-row selected/reference summaries on one additive-return scale."""
    metric = "mean_signed" if signed else "mean_absolute"
    return (
        alt.Chart(data(frame, ["feature", "cohort", metric, "wagers"]))
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)),
            yOffset="cohort:N",
            x=alt.X(
                metric + ":Q",
                title="Mean signed contribution" if signed else "Mean absolute contribution",
                axis=alt.Axis(format=".1%"),
            ),
            color=alt.Color(
                "cohort:N",
                scale=alt.Scale(range=PALETTE[:2]),
                legend=alt.Legend(orient="bottom"),
                title=None,
            ),
            tooltip=["feature:N", "cohort:N", alt.Tooltip(metric + ":Q", format=".3%"), "wagers:Q"],
        )
        .properties(width="container", height=230)
    )


def distribution(frame):
    """Jittered contribution distributions with selected-row drill-through."""
    pick = alt.selection_point(name="explanation_wager", fields=["bet_id"], empty=True)
    points = (
        alt.Chart(
            data(
                frame,
                [
                    "bet_id",
                    "feature",
                    "feature_value",
                    "split",
                    "contribution",
                    "trajectory_sd",
                    "jitter",
                ],
            )
        )
        .mark_circle(size=20)
        .encode(
            y=alt.Y("feature:N", title=None, axis=alt.Axis(labelLimit=220)),
            yOffset=alt.YOffset("jitter:Q"),
            x=alt.X(
                "contribution:Q",
                title="Contribution to predicted unit return",
                axis=alt.Axis(format=".1%"),
            ),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(range=PALETTE[:3]),
                legend=alt.Legend(orient="bottom"),
                title="Published split",
            ),
            opacity=alt.condition(pick, alt.value(0.65), alt.value(0.12)),
            tooltip=[
                "feature:N",
                "feature_value:N",
                "split:N",
                alt.Tooltip("contribution:Q", format=".3%"),
                alt.Tooltip("trajectory_sd:Q", format=".3%", title="Trajectory SD (not a CI)"),
                "bet_id:N",
            ],
        )
        .add_params(pick)
    )
    return (points + zero()).properties(width="container", height=280).interactive()


def dependence(frame, feature, label, color):
    """Recorded feature versus frozen attribution; no partial-dependence intervention."""
    source = frame.copy()
    source["contribution"] = source["shap__" + feature]
    numeric = feature in {"advertised_ev_fraction", "offered_decimal_odds"}
    pick = alt.selection_point(name="explanation_wager", fields=["bet_id"], empty=True)
    return (
        alt.Chart(
            data(source, list(dict.fromkeys(["bet_id", feature, color, "contribution", "split"])))
        )
        .mark_circle(size=40)
        .encode(
            x=alt.X(
                feature + (":Q" if numeric else ":N"),
                title=label,
                axis=alt.Axis(format=".0%")
                if feature == "advertised_ev_fraction"
                else alt.Axis(labelLimit=160),
            ),
            y=alt.Y(
                "contribution:Q", title="Frozen feature contribution", axis=alt.Axis(format=".1%")
            ),
            color=alt.Color(
                color + ":N",
                title=color.replace("_", " ").title(),
                legend=None if frame[color].nunique() > 12 else alt.Legend(labelLimit=180),
            ),
            opacity=alt.condition(pick, alt.value(0.6), alt.value(0.12)),
            tooltip=[
                feature + (":Q" if numeric else ":N"),
                color + ":N",
                "split:N",
                alt.Tooltip("contribution:Q", format=".3%"),
                "bet_id:N",
            ],
        )
        .add_params(pick)
        .properties(width="container", height=300)
        .interactive()
    )


def waterfall(frame):
    """Reference plus signed steps to the exact published prediction."""
    bars = (
        alt.Chart(data(frame, ["step", "start", "end", "change", "kind", "value", "order"]))
        .mark_bar()
        .encode(
            y=alt.Y(
                "step:N", title=None, sort=alt.SortField("order"), axis=alt.Axis(labelLimit=240)
            ),
            x=alt.X(
                "start:Q",
                title="Predicted unit return · reference + contributions",
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(zero=True),
            ),
            x2="end:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Reference", "Prediction", "Raises prediction", "Lowers prediction"],
                    range=["#8AB4FF", "#BEABFA", "#55C8B6", "#EC97AD"],
                ),
                title=None,
                legend=alt.Legend(orient="bottom", columns=2),
            ),
            tooltip=[
                "step:N",
                "value:N",
                alt.Tooltip("start:Q", format=".3%"),
                alt.Tooltip("end:Q", format=".3%"),
                alt.Tooltip("change:Q", format=".3%"),
            ],
        )
    )
    return (bars + zero()).properties(width="container", height=320)


def utility(frame):
    """Show untouched-test errors for selected models and the simple references."""
    source = frame.loc[frame.split.eq("test")].copy()
    source["label"] = source.model.str.replace("_", " ")
    return (
        alt.Chart(data(source, ["label", "mse", "mae", "wagers", "events", "selected"]))
        .mark_bar()
        .encode(
            y=alt.Y("label:N", title=None, axis=alt.Axis(labelLimit=250)),
            x=alt.X("mse:Q", title="Test MSE (squared unit-return fractions)"),
            color=alt.Color(
                "selected:N",
                scale=alt.Scale(domain=[True, False], range=["#55C8B6", "#8AB4FF"]),
                title="Selected on validation",
            ),
            tooltip=[
                "label:N",
                alt.Tooltip("mse:Q", format=".5f"),
                alt.Tooltip("mae:Q", format=".5f"),
                "wagers:Q",
                "events:Q",
            ],
        )
        .properties(width="container", height=180)
    )


def calibration(rows):
    """Describe selected fixed prediction bands; do not recalibrate or fit a line."""
    source = (
        rows.assign(
            band=pd.cut(
                rows.predicted_return, [-float("inf"), -0.1, 0, 0.025, 0.05, 0.1, float("inf")]
            ).astype(str)
        )
        .groupby("band", observed=True)
        .agg(
            predicted=("predicted_return", "mean"),
            observed=("observed_return", "mean"),
            wagers=("bet_id", "size"),
        )
        .reset_index()
    )
    points = (
        alt.Chart(data(source, ["band", "predicted", "observed", "wagers"]))
        .mark_circle(color=PALETTE[0])
        .encode(
            x=alt.X("predicted:Q", title="Mean model prediction", axis=alt.Axis(format=".0%")),
            y=alt.Y("observed:Q", title="Observed equal-wager return", axis=alt.Axis(format=".0%")),
            size=alt.Size("wagers:Q", title="Wagers", scale=alt.Scale(range=[60, 500])),
            tooltip=[
                "band:N",
                alt.Tooltip("predicted:Q", format=".2%"),
                alt.Tooltip("observed:Q", format=".2%"),
                "wagers:Q",
            ],
        )
    )
    bounds = [
        float(min(source.predicted.min(), source.observed.min(), 0)),
        float(max(source.predicted.max(), source.observed.max(), 0.01)),
    ]
    diagonal = (
        alt.Chart(alt.Data(values=[{"x": b, "y": b} for b in bounds]))
        .mark_line(strokeDash=[4, 4], color="#B8C7D9")
        .encode(x="x:Q", y="y:Q")
    )
    return (points + diagonal).properties(width="container", height=250).interactive()


def residuals(rows, field):
    """Show support-weighted visual summaries without inferential intervals."""
    source = (
        rows.groupby(field, dropna=False)
        .agg(mean_residual=("residual", "mean"), wagers=("bet_id", "size"))
        .reset_index()
    )
    return (
        alt.Chart(data(source, [field, "mean_residual", "wagers"]))
        .mark_circle(size=75, color=PALETTE[1])
        .encode(
            y=alt.Y(field + ":N", title=None, sort="-x", axis=alt.Axis(labelLimit=200)),
            x=alt.X(
                "mean_residual:Q",
                title="Observed − predicted mean return",
                axis=alt.Axis(format=".1%"),
            ),
            tooltip=[field + ":N", alt.Tooltip("mean_residual:Q", format=".2%"), "wagers:Q"],
        )
        .properties(width="container", height=min(500, max(160, 22 * len(source))))
        .interactive()
    )
