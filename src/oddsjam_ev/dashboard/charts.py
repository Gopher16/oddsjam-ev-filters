"""Interactive display charts of persisted cohorts, with no fitted models or new cohorts."""

import altair as alt
import pandas as pd

TEAL = "#51B8B2"
BLUE = "#76A9FA"
LOSS = "#E99494"
MUTED = "#A7B4C5"


def bars(frame, metric, title, *, currency=False, percentage=False, top=12):
    """Volume-ordered cohort bars, with explicit denominators in hover labels."""
    data = (
        frame.loc[frame[metric].notna()]
        .sort_values("eligible_count", ascending=False, kind="stable")
        .head(top)
        .copy()
    )
    if data.empty:
        return None
    data["bar_label"] = data.label + data.eligible_count.map(
        lambda n: f" · n={int(n):,}" + (" *" if n < 30 else "")
    )
    fmt = ".1%" if percentage else "$,.2f" if currency else ",.0f"
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y(
                "bar_label:N",
                sort=data.bar_label.tolist(),
                title=None,
                axis=alt.Axis(labelLimit=280, labelOverlap=False, labelPadding=8),
            ),
            x=alt.X(f"{metric}:Q", title=title, axis=alt.Axis(format=fmt)),
            color=alt.condition(alt.datum[metric] < 0, alt.value(LOSS), alt.value(TEAL)),
            tooltip=[
                alt.Tooltip("label:N", title="Segment"),
                alt.Tooltip(f"{metric}:Q", title=title, format=fmt),
                alt.Tooltip("eligible_count:Q", title="Analyzed wagers", format=",.0f"),
                alt.Tooltip("sample:N", title="Sample context"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color=MUTED, opacity=0.5)
        .encode(x="zero:Q")
    )
    return (chart + zero).properties(height=max(200, len(data) * 29)).configure_view(stroke=None)


def scatter(frame):
    data = frame.loc[frame.reported_roi.notna() & frame.eligible_count.gt(0)].copy()
    if data.empty:
        return None
    data["Sample"] = data.eligible_count.map(
        lambda n: "Fewer than 30 wagers" if n < 30 else "30+ wagers (not a validation rule)"
    )
    return (
        alt.Chart(data)
        .mark_circle(size=110, opacity=0.8)
        .encode(
            x=alt.X(
                "eligible_count:Q",
                title="Analyzed wagers (compressed scale)",
                scale=alt.Scale(type="symlog", constant=1),
                axis=alt.Axis(tickCount=5, grid=False, format="~s"),
            ),
            y=alt.Y(
                "reported_roi:Q",
                title="Reported ROI",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "Sample:N",
                scale=alt.Scale(
                    domain=["Fewer than 30 wagers", "30+ wagers (not a validation rule)"],
                    range=["#F4C878", TEAL],
                ),
                legend=alt.Legend(orient="bottom", title=None, labelLimit=400, columns=1),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Segment"),
                alt.Tooltip("eligible_count:Q", title="Analyzed wagers", format=",.0f"),
                alt.Tooltip("reported_roi:Q", title="Reported ROI", format=".2%"),
                alt.Tooltip(
                    "eligible_reported_profit_usd:Q", title="Reported profit", format="$,.2f"
                ),
            ],
        )
        .properties(height=320)
        .interactive()
        .configure_view(stroke=None)
    )


def counts(frame, *, title="Records", donut=False):
    if frame.empty:
        return None
    chart = alt.Chart(frame).encode(
        tooltip=[
            alt.Tooltip("label:N", title="Category"),
            alt.Tooltip("count:Q", title=title, format=",.0f"),
        ],
    )
    if donut:
        return (
            chart.mark_arc(innerRadius=65)
            .encode(
                theta="count:Q",
                color=alt.Color(
                    "label:N",
                    title=None,
                    scale=alt.Scale(
                        domain=frame.label.tolist(), range=[TEAL, MUTED, BLUE, LOSS][: len(frame)]
                    ),
                    legend=alt.Legend(orient="bottom", labelLimit=400, columns=1),
                ),
            )
            .properties(height=260)
            .configure_view(stroke=None)
        )
    return (
        chart.mark_bar(cornerRadiusEnd=3, color=TEAL)
        .encode(
            y=alt.Y(
                "label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)
            ),
            x=alt.X("count:Q", title=title),
        )
        .properties(height=max(180, len(frame) * 32))
        .configure_view(stroke=None)
    )
