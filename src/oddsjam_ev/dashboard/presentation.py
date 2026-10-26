"""Progressive disclosure and consistent stakeholder display formatting."""

import pandas as pd
import streamlit as st

from oddsjam_ev.dashboard.labels import money, percent

LIMITATIONS = (
    "Reported returns preserve the export’s profit values; fee inclusion is not independently verified. "
    "Paper outcomes are hypothetical. Funding labels are assumptions, and unclassified records remain separate. "
    "Core Exchange activity was selectively chosen. Taking existing exchange orders does not prove fills. "
    "Overlapping research segments are not independent or additive."
)
COLUMN_LABELS = {
    "label": "Segment",
    "candidate_count": "Logged wagers",
    "eligible_count": "Analyzed wagers",
    "excluded_count": "Excluded wagers",
    "eligible_stake_usd": "Analyzed stake",
    "eligible_reported_profit_usd": "Reported profit",
    "reported_roi": "Reported ROI",
    "eligible_event_id_count": "Event groups",
    "largest_stake_share": "Largest wager’s stake share",
    "uncertainty_status": "Uncertainty",
    "opportunity_proxy_count": "Logged opportunity proxies",
    "selection_proxy_count": "Selection proxies",
    "linked_source_record_count": "Linked records",
    "advertised_ev_coverage_rate": "Advertised EV recorded",
    "product": "Product",
    "execution_status": "Funding group",
    "count": "Records",
    "denominator": "Population size",
    "population": "Population",
    "dimension": "Field",
    "category": "Category",
}


def table(frame, limit=20):
    """Supporting data only, with readable headers and no technical IDs by default."""
    if frame.empty:
        st.info("No published rows for this selection.")
        return
    ordinal = st.session_state.get("table_ordinal", 0)
    st.session_state.table_ordinal = ordinal + 1
    page_size = min(limit, 20)
    start = 0
    if len(frame) > page_size:
        key = f"filter_table_{st.session_state.get('page', 'view')}_{ordinal}"
        if st.session_state.get(key, 1) > len(frame):
            st.session_state[key] = 1
        start = (
            int(
                st.number_input(
                    "First supporting row",
                    min_value=1,
                    max_value=len(frame),
                    value=1,
                    step=1,
                    key=key,
                )
            )
            - 1
        )
    display = frame.iloc[start : start + page_size].copy()
    technical = [
        c
        for c in display
        if c.endswith("_id")
        or c.endswith("_version")
        or c.endswith("_json")
        or c.startswith("additive_")
        or c.startswith("counts_additive_")
    ]
    display = display.drop(columns=technical, errors="ignore")
    for column in display:
        if "roi" in column or column.endswith("_share") or column.endswith("_rate"):
            if pd.api.types.is_numeric_dtype(display[column]) or display[column].isna().all():
                display[column] = display[column].map(percent)
        elif column.endswith("_usd"):
            display[column] = display[column].map(money)
        elif pd.api.types.is_numeric_dtype(display[column]) and not pd.api.types.is_bool_dtype(
            display[column]
        ):
            display[column] = display[column].map(
                lambda v: "Unavailable" if pd.isna(v) else f"{v:,.0f}"
            )
    display = display.rename(
        columns={c: COLUMN_LABELS.get(c, c.replace("_", " ").capitalize()) for c in display}
    )
    st.table(display, hide_index=True)
    st.caption(
        f"Rows {start + 1:,}–{min(start + page_size, len(frame)):,} of {len(frame):,}. Display paging does not change cohort metrics."
    )


def require_mode(mode):
    if mode is None:
        st.info(
            "Choose which bets to review in the sidebar. Real-money labels, paper tracking and unclassified funding stay separate."
        )
        return False
    return True


def plot(chart):
    if chart is None:
        st.info(
            "No analyzed results are available for this chart. Excluded records are not treated as zero returns."
        )
    else:
        st.altair_chart(chart, width="stretch")
