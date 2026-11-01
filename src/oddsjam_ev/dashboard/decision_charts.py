"""Interactive notebook-inspired views over exact published rows and matched cohorts."""

import json
import re

import altair as alt
import pandas as pd

from oddsjam_ev.dashboard.decision_queries import TIERS, comparison_frame, title
from oddsjam_ev.dashboard.labels import pretty

COLORS = ["#55C8B6", "#8AB4FF", "#FFC977", "#EC97AD", "#BEABFA", "#8BD3E6"]


def data(frame, columns):
    """Keep only chart fields; explicit inline rows preserve every published outlier."""
    selected = frame.reindex(columns=columns)
    return alt.Data(values=json.loads(selected.to_json(orient="records", date_format="iso")))


def zero():
    return (
        alt.Chart(alt.Data(values=[{"zero": 0}]))
        .mark_rule(color="#A7B4C5", opacity=0.5)
        .encode(x="zero:Q")
    )


def evidence_map(records):
    """Selectable contrast/support map; source intervals never acquire correction labels."""
    rows = []
    for r in records:
        stat = r["comparison_evidence"]
        if (
            r["evidence_tier"] == "insufficient_or_conflicting"
            or stat.get("weighted_difference") is None
        ):
            continue
        rows.append(
            {
                "candidate_id": r["candidate_id"],
                "idea": title(r),
                "contrast": stat["weighted_difference"],
                "low": stat.get("weighted_ci_low"),
                "high": stat.get("weighted_ci_high"),
                "events": r["rank_components"]["minimum_arm_event_groups"],
                "tier": TIERS[r["evidence_tier"]],
                "queue": r["queue"].replace("_", " "),
                "order": r["presentation_order"],
            }
        )
    if not rows:
        return None
    source = data(pd.DataFrame(rows), list(rows[0]))
    pick = alt.selection_point(name="candidate_pick", fields=["candidate_id"], empty=True)
    base = alt.Chart(source).encode(
        y=alt.Y(
            "events:Q",
            title="Minimum arm event support",
            scale=alt.Scale(type="symlog", constant=1),
        ),
        tooltip=[
            "idea:N",
            "tier:N",
            "queue:N",
            alt.Tooltip("contrast:Q", format=".2%"),
            "events:Q",
            "order:Q",
        ],
    )
    points = (
        base.mark_point(filled=True, size=85)
        .encode(
            x=alt.X(
                "contrast:Q",
                title="Subgroup − complement · stake-weighted return",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color("tier:N", scale=alt.Scale(range=COLORS), title="Evidence"),
            shape=alt.Shape("queue:N", title="Review queue"),
            opacity=alt.condition(pick, alt.value(0.9), alt.value(0.2)),
        )
        .add_params(pick)
    )
    interval = base.mark_rule(opacity=0.25).encode(x="low:Q", x2="high:Q")
    return (interval + points + zero()).properties(height=330).interactive()


def comparison(record, *, contrast=False):
    frame = comparison_frame([record])
    frame = frame.loc[frame.role.eq("contrast") if contrast else frame.role.ne("contrast")]
    source = data(frame, ["role", "measure", "return", "low", "high", "eligible_wagers"])
    base = alt.Chart(source).encode(
        y=alt.Y("measure:N" if contrast else "role:N", title=None),
        tooltip=["measure:N", "role:N", alt.Tooltip("return:Q", format=".2%"), "eligible_wagers:Q"],
    )
    dots = base.mark_point(filled=True, size=100).encode(
        x=alt.X(
            "return:Q",
            title="Subgroup − complement" if contrast else "Historical return",
            axis=alt.Axis(format=".1%"),
            scale=alt.Scale(zero=True),
        ),
        color=alt.Color("measure:N", scale=alt.Scale(range=COLORS), title=None),
        shape="measure:N",
    )
    chart = dots + zero()
    if contrast:
        chart = base.mark_rule().encode(x="low:Q", x2="high:Q") + chart
    return chart.properties(height=140).interactive()


def cohorts(frame):
    result = frame.copy()
    result["population"] = result.execution_status.str.replace("_", " ") + " · " + result["product"]
    result["label"] = result.category_json.map(
        lambda v: " · ".join(
            pretty(value, {"ev_bucket": "advertised_ev_band"}.get(key, key))
            for key, value in json.loads(v).items()
        )
        or "Whole scope"
    )
    return result


def profit_groups(frame, metric):
    frame = cohorts(frame)
    if frame.empty:
        return None
    fmt = "$,.0f" if metric == "reported_profit_usd" else ".1%"
    source = data(frame, ["label", "population", metric, "eligible_count"])
    chart = (
        alt.Chart(source)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("label:N", title=None, axis=alt.Axis(labelLimit=180, labelOverlap=False)),
            x=alt.X(
                metric + ":Q", title=metric.replace("_", " ").title(), axis=alt.Axis(format=fmt)
            ),
            color=alt.condition(alt.datum[metric] < 0, alt.value(COLORS[3]), alt.value(COLORS[0])),
            tooltip=[
                "population:N",
                "label:N",
                "eligible_count:Q",
                alt.Tooltip(metric + ":Q", format=fmt),
            ],
        )
        .properties(height=max(150, min(650, frame.label.nunique() * 24)), width=470)
    )
    if frame.population.nunique() == 1:
        return chart.properties(width="container")
    return chart.facet(facet=alt.Facet("population:N", title=None), columns=1).resolve_scale(
        y="independent"
    )


def ev_scatter(frame, *, equal=False):
    frame = cohorts(frame)
    if frame.empty:
        return None
    frame = frame.loc[frame.ev_matched_count.gt(0)].copy()
    if frame.empty:
        return None
    x = "ev_mean_advertised_ev_fraction" if equal else "ev_weighted_advertised_ev_fraction"
    y = "ev_matched_equal_stake_return" if equal else "ev_matched_roi"
    frame.loc[frame.view.eq("overall"), "label"] = frame.loc[
        frame.view.eq("overall"), "filter_label" if "filter_label" in frame else "filter_id"
    ].fillna("Whole scope")
    values = pd.concat([frame[x], frame[y]]).dropna()
    lo, hi = min(0.0, values.min()), max(0.0, values.max())
    base = alt.Chart(
        data(
            frame,
            [
                "label",
                "population",
                x,
                y,
                "ev_matched_count",
                "ev_eligible_coverage_rate",
                "lifecycle",
            ],
        )
    )
    points = base.mark_point(filled=True, opacity=0.75).encode(
        x=alt.X(x + ":Q", title="Matched advertised EV", axis=alt.Axis(format=".0%")),
        y=alt.Y(
            y + ":Q",
            title="Matched equal-stake return" if equal else "Matched stake-weighted ROI",
            axis=alt.Axis(format=".0%"),
        ),
        size=alt.Size(
            "ev_matched_count:Q", title="Matched wagers", scale=alt.Scale(range=[30, 700])
        ),
        color=alt.Color("population:N", scale=alt.Scale(range=COLORS), title="Evidence population"),
        tooltip=[
            "label:N",
            "population:N",
            "ev_matched_count:Q",
            alt.Tooltip("ev_eligible_coverage_rate:Q", format=".1%"),
            alt.Tooltip(x + ":Q", format=".2%"),
            alt.Tooltip(y + ":Q", format=".2%"),
        ],
    )
    if "lifecycle" in frame:
        points = points.encode(shape=alt.Shape("lifecycle:N", title="Confirmed lifecycle"))
    identity = (
        alt.Chart(alt.Data(values=[{"x": lo, "y": lo}, {"x": hi, "y": hi}]))
        .mark_line(strokeDash=[4, 4], color="#A7B4C5")
        .encode(x="x:Q", y="y:Q")
    )
    return (points + identity + zero()).properties(height=340).interactive()


def bucket_panel(frame, *, equal=False):
    frame = cohorts(frame)
    if frame.empty:
        return None

    def low(label):
        matches = re.findall(r"\[(-?\d+(?:\.\d+)?|-inf)", str(label))
        return tuple(float(v) for v in matches) if matches else (float("inf"),)

    order = sorted(frame.label.unique(), key=low)
    series = []
    for row in frame.to_dict("records"):
        # Both EV and return use the same EV-matched cohort. Closing series is
        # separate unless a complete joint cohort can support all three lines.
        joint = row["joint_matched_count"] > 0
        prefix = "joint" if joint else "ev"
        for name, column in (
            (
                "Equal-stake return" if equal else "Stake-weighted ROI",
                prefix + ("_matched_equal_stake_return" if equal else "_matched_roi"),
            ),
            (
                "Advertised EV",
                prefix
                + ("_mean_advertised_ev_fraction" if equal else "_weighted_advertised_ev_fraction"),
            ),
        ):
            series.append(
                {
                    "label": row["label"],
                    "population": row["population"],
                    "series": name,
                    "value": row.get(column),
                    "matched_wagers": row[prefix + "_matched_count"],
                }
            )
        if joint:
            series.append(
                {
                    "label": row["label"],
                    "population": row["population"],
                    "series": "Fair closing-reference EV estimate",
                    "value": row.get(
                        "joint_mean_closing_price_ev_fraction"
                        if equal
                        else "joint_weighted_closing_price_ev_fraction"
                    ),
                    "matched_wagers": row["joint_matched_count"],
                }
            )
    pick = alt.selection_point(name="bucket_pick", fields=["label"], empty=True)
    top = (
        alt.Chart(
            data(pd.DataFrame(series), ["label", "population", "series", "value", "matched_wagers"])
        )
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=140)
            ),
            y=alt.Y("value:Q", title="Matched return / EV", axis=alt.Axis(format=".0%")),
            color=alt.Color("series:N", scale=alt.Scale(range=COLORS), title=None),
            strokeDash="population:N",
            opacity=alt.condition(pick, alt.value(1), alt.value(0.2)),
            tooltip=[
                "label:N",
                "population:N",
                "series:N",
                "matched_wagers:Q",
                alt.Tooltip("value:Q", format=".2%"),
            ],
        )
        .add_params(pick)
        .properties(height=230)
    )
    bottom = (
        alt.Chart(data(frame, ["label", "population", "eligible_count", "reported_profit_usd"]))
        .mark_bar()
        .encode(
            x=alt.X(
                "label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=140)
            ),
            y=alt.Y("eligible_count:Q", title="Eligible wagers"),
            color=alt.Color("population:N", scale=alt.Scale(range=COLORS)),
            opacity=alt.condition(pick, alt.value(0.8), alt.value(0.2)),
            tooltip=[
                "label:N",
                "population:N",
                "eligible_count:Q",
                alt.Tooltip("reported_profit_usd:Q", format="$,.2f"),
            ],
        )
        .properties(height=150)
    )
    return alt.vconcat(top, bottom).resolve_scale(color="independent")


def calendar(frame, metric):
    if frame.empty:
        return None
    frame = frame.copy()
    frame["series"] = (
        frame.execution_status
        + " · "
        + frame["product"]
        + " · "
        + frame.platform.fillna("All platforms")
    )
    legend = alt.selection_point(name="platform_pick", fields=["series"], bind="legend")
    chart = (
        alt.Chart(
            data(
                frame,
                [
                    "period_start",
                    "series",
                    metric,
                    "period_status",
                    "boundary_period",
                    "coverage_status",
                ],
            )
        )
        .mark_line(point=False, invalid="break-paths-show-domains")
        .encode(
            x=alt.X("period_start:T", title="Source calendar date"),
            y=alt.Y(metric + ":Q", title=metric.replace("_", " "), axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(range=COLORS),
                legend=alt.Legend(title="Click to isolate · Shift for multiple", labelLimit=300),
            ),
            opacity=alt.condition(legend, alt.value(1), alt.value(0.1)),
            tooltip=[
                "series:N",
                "period_start:T",
                alt.Tooltip(metric + ":Q", format=",.2f"),
                "period_status:N",
                "boundary_period:N",
                "coverage_status:N",
            ],
        )
        .add_params(legend)
    )
    return chart.properties(height=350).interactive()


def activity(frame, view):
    if frame.empty:
        return None
    frame = frame.loc[frame.view.eq(view)].copy()
    if frame.empty:
        return None
    frame["population"] = frame.execution_status + " · " + frame["product"]
    for key in ("source_weekday", "source_hour"):
        frame[key] = frame.category_json.map(lambda v, k=key: json.loads(v).get(k))
    source = data(
        frame,
        [
            "population",
            "source_weekday",
            "source_hour",
            "mean_candidates_per_active_source_day",
            "logged_candidate_count",
            "active_source_day_denominator",
        ],
    )
    base = alt.Chart(source).encode(
        tooltip=[
            "population:N",
            "source_weekday:O",
            "source_hour:O",
            "logged_candidate_count:Q",
            "active_source_day_denominator:Q",
            alt.Tooltip("mean_candidates_per_active_source_day:Q", format=".2f"),
        ]
    )
    if view == "weekday_hour":
        chart = base.mark_rect().encode(
            x=alt.X("source_hour:O", title="Source hour"),
            y=alt.Y("source_weekday:O", title="Source weekday · Mon=0"),
            color=alt.Color("mean_candidates_per_active_source_day:Q", title="Per active day"),
        )
    else:
        key = "source_weekday" if view == "weekday" else "source_hour"
        chart = base.mark_bar(color=COLORS[0]).encode(
            x=alt.X(key + ":O", title="Mon=0" if view == "weekday" else "Source hour"),
            y=alt.Y("mean_candidates_per_active_source_day:Q", title="Mean per active day"),
        )
    if frame.population.nunique() == 1:
        return chart.properties(width="container", height=240)
    return chart.properties(width=470, height=240).facet(
        facet=alt.Facet("population:N", title=None), columns=1
    )


def liquidity_heatmap(frame):
    frame = cohorts(frame)
    if frame.empty:
        return None
    frame["liquidity"] = frame.category_json.map(lambda v: json.loads(v).get("liquidity_bucket"))
    frame["ev"] = frame.category_json.map(lambda v: json.loads(v).get("ev_bucket"))
    pick = alt.selection_point(name="liquidity_pick", fields=["liquidity", "ev"], empty=True)
    heat = (
        alt.Chart(
            data(frame, ["liquidity", "ev", "population", "ev_matched_roi", "ev_matched_count"])
        )
        .mark_rect()
        .encode(
            x=alt.X("ev:N", title="Advertised EV band"),
            y=alt.Y("liquidity:N", title="Verified USD quote band"),
            color=alt.Color(
                "ev_matched_roi:Q",
                title="Matched ROI",
                scale=alt.Scale(scheme="redblue", domainMid=0),
            ),
            opacity=alt.condition(pick, alt.value(1), alt.value(0.2)),
            tooltip=[
                "population:N",
                "liquidity:N",
                "ev:N",
                "ev_matched_count:Q",
                alt.Tooltip("ev_matched_roi:Q", format=".2%"),
            ],
        )
        .add_params(pick)
        .properties(width=470, height=260)
    )
    if frame.population.nunique() == 1:
        heat = heat.properties(width="container")
    else:
        heat = heat.facet(facet=alt.Facet("population:N", title=None), columns=1)

    selected = (
        alt.Chart(
            data(
                frame,
                ["label", "liquidity", "ev", "population", "ev_matched_roi", "ev_matched_count"],
            )
        )
        .transform_filter(pick)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", title="Selected quote / EV cells", axis=alt.Axis(labelLimit=200)),
            x=alt.X(
                "ev_matched_roi:Q", title="EV-matched ROI", axis=alt.Axis(format=".1%"), stack=None
            ),
            color="population:N",
            yOffset="population:N",
            tooltip=[
                "label:N",
                "population:N",
                "ev_matched_count:Q",
                alt.Tooltip("ev_matched_roi:Q", format=".2%"),
            ],
        )
        .properties(height=200)
    )
    return alt.vconcat(heat, selected).resolve_scale(color="independent")
