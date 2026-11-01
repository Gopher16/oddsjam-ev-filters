"""Notebook-inspired diagnostics with exact published scope and explicit blockers."""

import streamlit as st

from oddsjam_ev.dashboard import decision_charts as charts
from oddsjam_ev.dashboard.decision_queries import chart_rows
from oddsjam_ev.dashboard.pages.decision_workspace import scope

CHARTS = {
    "Profit by group": "profit_by_group",
    "Matched EV vs return": "ev_roi_scatter",
    "Saved-filter EV vs return": "filter_ev_roi_scatter",
    "EV bands": "ev_bucket_summary",
    "Liquidity": "liquidity_summary",
    "Liquidity × EV": "liquidity_ev_summary",
    "Time to event": "time_to_event_summary",
    "Cumulative profit by platform": "cumulative_profit_by_platform",
    "Bet volume over time": "volume_over_time",
    "Average weekday / hour activity": "average_activity",
}


def population_panels(rows, builder):
    """Responsive panels keep every population visible without pooling economics."""
    if rows.empty:
        return False
    panels = st.columns(2)
    rendered = False
    for i, ((funding, product), part) in enumerate(
        rows.groupby(["execution_status", "product"], sort=True)
    ):
        with panels[i % 2]:
            st.caption(funding.replace("_", " ") + " · " + product)
            chart = builder(part)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
                rendered = True
    return rendered


def render(workspace):
    st.subheader("Explore the recorded history")
    st.caption(
        "Descriptive diagnostics, separate from prioritized research ideas. Each population keeps its own economics. Charts use exact published cohorts; selectors do not construct new intersections."
    )
    if workspace is None:
        st.info("Verified visual diagnostics are not linked to this report.")
        return
    if any(
        st.session_state.get(k)
        for k in ("filter_workspace_tiers", "filter_workspace_setup", "filter_workspace_lifecycle")
    ):
        st.info(
            "Evidence/setup/lifecycle selectors narrow idea records, not population diagnostics. Clear those selectors to inspect exact published historical chart scopes; no pooled idea cohort is invented."
        )
        return
    chosen = st.selectbox("Diagnostic", list(CHARTS), key="diagnostic_chart")
    chart_id = CHARTS[chosen]
    filters = scope()
    readiness = chart_rows(workspace, "visual_readiness", **{**filters, "selections": {}})
    readiness = readiness.loc[readiness.chart_id.eq(chart_id)]
    if chart_id == "filter_ev_roi_scatter" and filters["saved"] == "all":
        readiness = workspace["visuals"]["visual_readiness"].query(
            "scope_kind == 'filter' and chart_id == 'filter_ev_roi_scatter'"
        )
        for field, column in (("mode", "execution_status"), ("product", "product")):
            if filters[field] != "all":
                readiness = readiness.loc[readiness[column].eq(filters[field])]
    if readiness.empty or not readiness.supported_candidates.gt(0).any():
        st.info("Chart blocked: this scope has no supported source data for this diagnostic.")
        if len(readiness):
            for remedy in readiness.remediation.dropna().unique():
                st.write("**To enable:** " + remedy)
        else:
            st.caption(
                "Publish this exact scope and its readiness audit first; no other report is substituted."
            )
        return
    metric_min = int(st.session_state.get("filter_min_wagers", 100))
    table = (
        "visual_activity"
        if chart_id == "average_activity"
        else "visual_periods"
        if chart_id in {"volume_over_time", "cumulative_profit_by_platform"}
        else "visual_cohorts"
    )
    rows = chart_rows(workspace, table, **filters)
    if chart_id == "filter_ev_roi_scatter" and filters["saved"] == "all":
        rows = (
            workspace["visuals"]["visual_cohorts"]
            .query("scope_kind == 'filter' and view == 'overall'")
            .copy()
        )
        if any(filters["selections"].values()):
            rows = rows.iloc[:0]
        for field, column in (("mode", "execution_status"), ("product", "product")):
            if filters[field] != "all":
                rows = rows.loc[rows[column].eq(filters[field])]
    chart = None
    rendered = False
    if table == "visual_cohorts":
        if not rows.empty:
            rows = rows.loc[rows.eligible_count.ge(metric_min)]
        equal = (
            (
                st.radio(
                    "Return weighting",
                    ["Stake-weighted", "Equal-stake"],
                    horizontal=True,
                    key="diagnostic_weighting",
                )
                == "Equal-stake"
            )
            if chart_id != "profit_by_group"
            else False
        )
        if chart_id == "profit_by_group":
            view = st.selectbox(
                "Group by", ["sport", "platform", "league", "market"], key="diagnostic_group"
            )
            rows = rows.loc[rows.view.eq(view)]
            metric = st.selectbox(
                "Metric",
                ["reported_profit_usd", "stake_weighted_roi", "equal_stake_mean_return"],
                format_func=lambda v: v.replace("_", " ").title(),
                key="diagnostic_metric",
            )
            rendered = population_panels(rows, lambda part: charts.profit_groups(part, metric))
        elif chart_id in {"ev_roi_scatter", "filter_ev_roi_scatter"}:
            rows = rows.loc[rows.view.eq("platform" if chart_id == "ev_roi_scatter" else "overall")]
            if chart_id == "filter_ev_roi_scatter":
                names = {
                    f["id"]: f["name"]
                    for f in st.session_state.published_run.manifests["canonical"]["specification"][
                        "filter_registry"
                    ]["filters"]
                }
                rows["filter_label"] = rows.filter_id.map(lambda v: names.get(v, v))
                lifecycle = {}
                for record in workspace["queues"]["records"]:
                    lifecycle.setdefault(record["filter_id"], set()).add(
                        record["configuration_sheet"]["lifecycle"]["status"]
                    )
                rows["lifecycle"] = rows.filter_id.map(
                    lambda v: next(iter(lifecycle[v]))
                    if len(lifecycle.get(v, set())) == 1
                    else "unknown"
                )
            chart = charts.ev_scatter(rows, equal=equal)
        else:
            view = {
                "ev_bucket_summary": "ev_bucket",
                "liquidity_summary": "liquidity_bucket",
                "liquidity_ev_summary": "liquidity_ev",
                "time_to_event_summary": "time_to_event_bucket",
            }[chart_id]
            rows = rows.loc[
                rows.view.eq(view) & ~rows.category_json.str.contains("<unavailable>", regex=False)
            ]
            if chart_id == "liquidity_ev_summary":
                rendered = population_panels(rows, charts.liquidity_heatmap)
            else:
                chart = charts.bucket_panel(rows, equal=equal)
        st.caption(
            "EV/return comparisons use the same EV-matched rows (or joint EV/closing rows where supported). Volume and profit panels retain all eligible rows with their own denominators."
        )
        if not readiness.closing_series_matched_count.gt(0).any():
            st.caption(
                "Closing-reference EV estimate unavailable: fair closing prices, source and observation timing are not verified. No zero line is substituted for missing EV."
            )
    elif table == "visual_periods":
        resolution = st.radio(
            "Calendar resolution",
            ["D", "W-SUN", "M"],
            format_func=lambda v: {"D": "Daily", "W-SUN": "Weekly", "M": "Monthly"}[v],
            horizontal=True,
            key="diagnostic_resolution",
            index=1,
        )
        rows = rows.loc[rows.resolution.eq(resolution)]
        by_platform = st.checkbox("Separate platforms", value=True, key="diagnostic_platforms")
        rows = rows.loc[rows.platform.notna() if by_platform else rows.platform.isna()]
        if chart_id == "volume_over_time":
            metric = "candidate_count"
        else:
            unit = st.radio(
                "Profit basis",
                ["Reported dollars", "Fixed $1 units"],
                horizontal=True,
                key="diagnostic_profit_basis",
            )
            metric = (
                "cumulative_attributed_reported_profit_usd"
                if unit == "Reported dollars"
                else "cumulative_attributed_fixed_unit_notional_profit_usd"
            )
        chart = charts.calendar(rows, metric)
        st.caption(
            "Source-calendar attribution only—not placement/settlement order, bankroll growth or drawdown. Unknown gaps break the line. Boundary periods may be partial; undated economics are excluded from this plot. Activity is not suppressed by the minimum-performance count."
        )
    else:
        view = st.radio(
            "Activity view",
            ["weekday", "hour", "weekday_hour"],
            horizontal=True,
            key="diagnostic_activity",
        )
        rendered = population_panels(rows, lambda part: charts.activity(part, view))
        st.caption(
            "Mean per active source day within each bucket—not all covered days or future opportunity supply. Literal source weekday/hour; timezone and placement semantics are not inferred."
        )
    if chart is None and not rendered:
        st.info(
            "No exact published chart cohort matches the selected dimensions and display threshold. Clear a selector or explicitly publish the missing intersection; broader data is not substituted."
        )
    else:
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.caption(
            "Hover for denominators. Click bucket/heatmap cells to link highlights; use legends to isolate calendar series. Zoomable charts support pan and reset. No tests or ranks are recomputed."
        )
    with st.expander("Accessible chart data & coverage"):
        if not rows.empty:
            st.dataframe(
                rows.drop(columns=[c for c in rows if c.endswith("_id")], errors="ignore"),
                hide_index=True,
            )
            st.download_button(
                "Download displayed chart data",
                rows.to_csv(index=False),
                file_name="published-chart-rows.csv",
                mime="text/csv",
                on_click="ignore",
                key="diagnostic_download",
            )
        st.dataframe(
            readiness[
                [
                    "execution_status",
                    "product",
                    "status",
                    "supported_candidates",
                    "denominator_candidates",
                    "coverage_rate",
                    "remediation",
                ]
            ],
            hide_index=True,
        )
        st.caption(
            "Chart readiness concerns source availability; it is not statistical validation."
        )


def history(run, mode, limit):
    from oddsjam_ev.dashboard.pages.workspace import signals

    st.header("Explore history")
    a, b = st.tabs(["Interactive diagnostics", "Legacy descriptive slices"])
    with a:
        render(st.session_state.get("decision_workspace"))
    with b:
        signals(run, mode, limit)
