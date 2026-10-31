"""Stakeholder workspace over verified published cohorts; no research is rerun."""

import json

import pandas as pd
import streamlit as st

from oddsjam_ev.dashboard import charts, queries
from oddsjam_ev.dashboard.labels import (
    DIMENSIONS,
    PRODUCTS,
    money,
    percent,
    pretty,
    sample_label,
    segment_label,
)
from oddsjam_ev.dashboard.pages.evidence import evidence_page
from oddsjam_ev.dashboard.pages.paper_tests import paper_tests
from oddsjam_ev.dashboard.pages.research import confidence
from oddsjam_ev.dashboard.presentation import plot, table
from oddsjam_ev.dashboard.signal_filters import filter_signals, minimum_support, signal_specificity


def selected_queue(run, mode, product, saved_filter="all"):
    """Browse all populations without pooling their economics or identities."""
    if saved_filter != "all":
        source = run.tables["atlas"]["filter_segment_diagnostics"].copy()
        source = source.loc[source.filter_id.eq(saved_filter)]
        if mode != "all":
            source = source.loc[source.execution_status.eq(mode)]
        if product != "all":
            source = source.loc[source["product"].eq(product)]
        # Only exact persisted membership slices; never intersect global summaries.
        source = source.loc[source.view.ne("filter_id__attribution_status")].copy()
        source["predicate_json"] = (
            source.apply(
                lambda row: json.dumps(
                    {d: row[d] for d in queries.DIMENSIONS if d in row["view"].split("__")},
                    sort_keys=True,
                ),
                axis=1,
            )
            if len(source)
            else pd.Series(dtype=str)
        )
        source["event_proxy_groups"] = source.eligible_event_id_count
        source["cluster_roi_ci_low"] = float("nan")
        source["cluster_roi_ci_high"] = float("nan")
        source["section"] = "More observations needed"
        source["candidate_id"] = (
            source.filter_id
            + ":"
            + source.execution_status
            + ":"
            + source["product"]
            + ":"
            + source.predicate_json
        )
        return source.sort_values(
            ["event_proxy_groups", "eligible_count", "candidate_id"], ascending=[False, False, True]
        )
    frames = []
    for funding in queries.MODES if mode == "all" else [mode]:
        products = (
            run.tables["canonical"]["economic_summary"]["product"].unique()
            if product == "all"
            else [product]
        )
        for item in products:
            frames.append(queries.research_queue(run.tables, funding, item))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["event_proxy_groups", "eligible_count", "candidate_id"],
        ascending=[False, False, True],
        na_position="last",
    )


def _card(row, *, loss=False):
    with st.container(border=True):
        st.caption(
            queries.MODE_LABELS[row.execution_status].upper()
            + " · "
            + PRODUCTS.get(row["product"], row["product"])
        )
        st.subheader(segment_label(row))
        a, b, c = st.columns(3)
        a.metric(
            "Hypothetical ROI" if row.execution_status == "paper" else "Reported ROI",
            percent(row.reported_roi),
        )
        b.metric("Analyzed wagers", f"{int(row.eligible_count):,}")
        c.metric("Reported profit", money(row.eligible_reported_profit_usd))
        st.write(
            "**TEST hypothesis:** "
            + (
                "Compare an unchanged baseline with a paper version excluding this exact subgroup."
                if loss
                else "Track this exact subgroup as a prospective paper TEST, recording quote availability."
            )
        )
        low, high = row.cluster_roi_ci_low, row.cluster_roi_ci_high
        badge = "Interval unavailable"
        if pd.notna(low) and pd.notna(high):
            badge = f"Published interval {percent(low)} to {percent(high)}"
            if low <= 0 <= high:
                badge += " · includes zero"
        st.caption(badge + " · Historical signal, not PROD approval.")
        with st.expander("Historical evidence and subgroup definition"):
            _, _, detail, _ = queries.research_finding(row)
            st.write(detail)
            st.write(json.loads(row.predicate_json))
            st.caption(
                "Analytical subgroup only. Omitted scanner settings are not confirmed defaults or unrestricted controls. A prospective test needs a separate complete baseline and parameter sheet."
            )
            st.caption(
                "Signals overlap. Group breadth determines reading order, not expected future profit. Fees and execution availability are not independently verified."
            )


def _reset_signal_filters():
    for key in list(st.session_state):
        if key.startswith("filter_signal_"):
            del st.session_state[key]


def _signal_sections(frame, *, loss=False):
    specific = frame.loc[frame.dimension_count.ge(2)]
    broad = frame.loc[frame.dimension_count.lt(2)]
    st.subheader("Specific TEST hypotheses" if not loss else "Specific loss investigations")
    st.caption(
        "Two or more published dimensions. Specificity is not proof of a stronger edge. Within each section: event-group breadth, then wager count."
    )
    if specific.empty:
        st.info("No multidimensional evidence meets the display convention for this selection.")
    for _, row in specific.head(5).iterrows():
        _card(row, loss=loss)
    with st.expander(f"Overall performance drivers · {len(broad):,} broad signals"):
        for _, row in broad.head(5).iterrows():
            _card(row, loss=loss)


def sidebar_signal_filters(run, mode, product):
    """Place simultaneous signal controls beside funding/product selections."""
    registry = {
        f["id"]: f["name"]
        for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
    }
    available = set(run.tables["atlas"]["filter_segment_diagnostics"].filter_id)
    st.sidebar.selectbox(
        "Filter",
        ["all", *sorted(k for k in registry if k in available)],
        format_func=lambda v: "All saved filters" if v == "all" else registry[v],
        key="filter_signal_saved",
    )
    queue = selected_queue(run, mode, product)
    selections = {}
    st.sidebar.subheader("Signal filters")
    st.sidebar.caption(
        "Combine fields with AND; multiple choices in one field use OR. Empty means all."
    )
    predicates = queue.predicate_json.map(json.loads)
    for dimension in queries.DIMENSIONS:
        values = sorted({str(p[dimension]) for p in predicates if dimension in p})
        key = "filter_signal_" + dimension
        if key in st.session_state:
            st.session_state[key] = [v for v in st.session_state[key] if v in values]
        selections[dimension] = st.sidebar.multiselect(
            DIMENSIONS[dimension],
            values,
            format_func=lambda v, d=dimension: pretty(v, d),
            key=key,
        )
    st.sidebar.button("Reset signal filters", on_click=_reset_signal_filters)


def signals(run, mode, limit):
    st.header("Signals & leaks")
    st.caption(
        "Descriptive historical cards · corrected comparisons are available separately on Within-filter evidence."
    )
    st.caption(
        "Find TEST ideas and loss pockets across your published history. Optional controls narrow the evidence shown."
    )
    saved_filter = st.session_state.get("filter_signal_saved", "all")
    queue = selected_queue(run, mode, st.session_state.get("filter_product", "all"), saved_filter)
    if saved_filter != "all":
        registry = {
            f["id"]: f["name"]
            for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
        }
        st.caption("Saved-filter scope: " + registry[saved_filter])
        st.info(
            "Showing exact published saved-filter membership slices. These have no published subgroup confidence intervals. Global discovery results are not substituted; combinations absent at this filter grain remain unavailable."
        )
    queue = signal_specificity(queue)
    selections = {d: st.session_state.get("filter_signal_" + d, []) for d in queries.DIMENSIONS}
    active = [
        f"{DIMENSIONS[d]}: {', '.join(pretty(v, d) for v in values)}"
        for d, values in selections.items()
        if values
    ]
    if active:
        st.caption("Active signal filters · " + " | ".join(active))
    queue = filter_signals(queue, selections)
    if queue.empty:
        st.info(
            "No published evidence for this combination. Broader or separate summaries are not substituted. Clear a field or publish this intersection through the research pipeline."
        )
        return
    minimum = int(st.session_state.get("filter_min_wagers", 100))
    hidden = int(queue.eligible_count.lt(minimum).sum())
    queue = minimum_support(queue, minimum)
    st.caption(
        f"Minimum {minimum:,} analyzed wagers per displayed cohort · {hidden:,} smaller cohorts hidden. Change the sidebar threshold to inspect them."
    )
    if queue.empty:
        st.info(
            "No published signal cohorts meet the minimum analyzed-wager setting. Lower it to inspect smaller cohorts."
        )
        return
    positive = queue.loc[queue.reported_roi.gt(0)]
    losses = queue.loc[queue.reported_roi.le(0)]
    a, b, c = st.columns(3)
    a.metric("Positive historical signals", len(positive), border=True)
    b.metric("Loss pockets to investigate", len(losses), border=True)
    c.metric("Smaller cohorts hidden", hidden, border=True)
    st.caption(
        "Counts are overlapping research segments, not independent bets. The minimum-wager setting controls display only, not statistical validity."
    )
    tab1, tab2, tab3 = st.tabs(
        ["Positive signals · TEST ideas", "Potential leaks", "Compare evidence"]
    )
    with tab1:
        st.caption(
            "Largest event-group breadth first, then analyzed wager count. Up to five ideas; no new statistical edge score is inferred."
        )
        if positive.empty:
            st.info(
                "No positive historical signals meet the display convention for this selection."
            )
        _signal_sections(positive)
    with tab2:
        if losses.empty:
            st.success(
                "No negative historical subgroups meet the display convention for this selection."
            )
        _signal_sections(losses, loss=True)
    with tab3:
        chart = queue.copy()
        chart["label"] = (
            chart.apply(segment_label, axis=1)
            + " · "
            + chart.execution_status.map(queries.MODE_LABELS)
        )
        plot(charts.scatter(chart))
        st.caption(
            "Hover for sample size and return. Funding groups remain separate points; no pooled ROI or new subgroup is generated."
        )
        with st.expander("All published subgroup results"):
            table(
                chart[
                    [
                        "label",
                        "eligible_count",
                        "event_proxy_groups",
                        "reported_roi",
                        "eligible_reported_profit_usd",
                    ]
                ],
                limit,
            )
    with st.expander("Tracked prospective tests"):
        if (
            "experiments" not in run.tables
            or run.tables["experiments"]["experiment_protocols"].empty
        ):
            st.info(
                "No prospective tests are tracked in this published report. Historical paper records are not automatically registered experiments."
            )
        else:
            for funding in queries.MODES if mode == "all" else [mode]:
                st.caption(queries.MODE_LABELS[funding])
                table(queries.experiment_metrics(run.tables, funding), limit)


def health(run, mode, limit):
    st.header("Filter health")
    st.caption(
        "Recorded saved-filter membership: strengths and loss pockets within that filter, not global platform results."
    )
    source = run.tables["atlas"]["filter_segment_diagnostics"].copy()
    if mode != "all":
        source = source.loc[source.execution_status.eq(mode)]
    product = st.session_state.get("filter_product", "all")
    if product != "all":
        source = source.loc[source["product"].eq(product)]
    registry = {
        f["id"]: f["name"]
        for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
    }
    ids = sorted(
        identity for identity in source.filter_id.dropna().unique() if identity in registry
    )
    if not ids:
        st.info("No resolved saved-filter evidence for this selection.")
        return
    identity = st.selectbox(
        "Saved filter", ids, format_func=lambda v: registry.get(v, v), key="filter_saved"
    )
    source = source.loc[source.filter_id.eq(identity)]
    views = sorted(source.view.unique())
    view = st.selectbox(
        "Published breakdown",
        views,
        index=next((i for i, v in enumerate(views) if v.endswith("__market_family")), 0),
        format_func=lambda v: "Market"
        if v.endswith("__market_family")
        else "Platform"
        if v.endswith("__sportsbook")
        else "Overall membership",
        key="filter_health_view",
    )
    selected = source.loc[source.view.eq(view)].copy()
    minimum = int(st.session_state.get("filter_min_wagers", 100))
    hidden = int(selected.eligible_count.lt(minimum).sum())
    selected = minimum_support(selected, minimum)
    st.caption(
        f"Minimum {minimum:,} analyzed wagers per displayed cohort · {hidden:,} smaller cohorts hidden."
    )
    if selected.empty:
        st.info(
            "No saved-filter cohorts meet the minimum analyzed-wager setting. Lower it to inspect smaller cohorts."
        )
        return
    st.caption(
        "Each row is a separate funding/product cohort. Overlapping views are not added. A negative result is an observed loss, not proof the subgroup should be removed."
    )
    columns = [
        c
        for c in [
            "execution_status",
            "product",
            *queries.DIMENSIONS,
            "candidate_count",
            "eligible_count",
            "reported_roi",
        ]
        if c in selected and not selected[c].isna().all()
    ]
    dimension = (
        "market_family"
        if view.endswith("__market_family")
        else "sportsbook"
        if view.endswith("__sportsbook")
        else None
    )
    selected["label"] = selected.apply(
        lambda row: (pretty(row[dimension], dimension) + " · " if dimension else "")
        + queries.MODE_LABELS[row.execution_status]
        + " · "
        + PRODUCTS.get(row["product"], row["product"]),
        axis=1,
    )
    selected["sample"] = selected.eligible_count.map(sample_label)
    plot(charts.bars(selected, "reported_roi", "Reported ROI", percentage=True))
    with st.expander("Exact saved-filter cohort results"):
        table(selected[columns], limit)
    if not selected.reported_roi.lt(0).any():
        st.info("No negative reported ROI in this published breakdown.")
    else:
        st.write(
            "**Potential leak investigation:** compare a prospective paper exclusion against the unchanged filter. Existing membership is selective and does not establish a causal improvement."
        )
    st.caption(
        "CLV, realized-versus-advertised edge and widening comparisons are not published at this filter grain. They are not inferred here. Current PROD status is not established by a historical filter name."
    )


PAGES = {
    "Signals & leaks": signals,
    "Filter health": health,
    "Within-filter evidence": evidence_page,
    "Paper-test definitions": paper_tests,
    "Evidence & methodology": confidence,
}
