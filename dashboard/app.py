"""Stakeholder research viewer; published artifacts remain read-only."""

import os
from pathlib import Path

import streamlit as st

from oddsjam_ev.dashboard.artifacts import (
    catalogue_labels,
    load_published_run,
    operational_attempts,
    read_dashboard_config,
)
from oddsjam_ev.dashboard.evidence import load_evidence, load_paper_definitions
from oddsjam_ev.dashboard.labels import PRODUCTS, report_dates
from oddsjam_ev.dashboard.pages.workspace import PAGES, sidebar_signal_filters
from oddsjam_ev.dashboard.queries import MODE_LABELS, MODES, validate_display_contract


def main():
    st.set_page_config(page_title="Betting research", page_icon="◈", layout="wide")
    st.markdown(
        """<style>
    .block-container {padding-top:2rem; max-width:1440px;}
    [data-testid="stMetricValue"] {font-size:1.8rem;}
    [data-testid="stMetricLabel"] {color:#B8C7D9;}
    [data-testid="stSidebar"] {border-right:1px solid #263448;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
    h1 {letter-spacing:-0.04em;} h2 {letter-spacing:-0.025em;}
    .stMain {background:radial-gradient(ellipse at top right,#142b38 0%,#0e1621 60%);}
    </style>""",
        unsafe_allow_html=True,
    )
    st.title("Betting research")
    st.caption(
        "Understand the recorded results. Explore the evidence. Keep the limitations in view."
    )
    try:
        cfg = read_dashboard_config(
            Path(os.environ.get("ODDSJAM_DASHBOARD_CONFIG", "configs/dashboard.yaml"))
        )
        root = Path(cfg.artifact_root).resolve()
        labels = catalogue_labels(root, cfg.display_timezone)
    except Exception:
        st.session_state.pop("published_run", None)
        st.error(
            "The local report catalogue is unavailable. Check the dashboard configuration; no older report has been substituted."
        )
        return
    st.sidebar.header("Choose your report")
    st.sidebar.caption(
        "Newest recorded publication opens automatically. Generation time is separate from source-data freshness."
    )
    selected = st.sidebar.selectbox(
        "Research report",
        list(labels),
        index=0 if labels else None,
        placeholder="No published reports",
        format_func=lambda i: labels.get(i, i),
        key="selected_run",
    )
    if st.session_state.get("last_selection") != selected:
        st.session_state.pop("published_run", None)
        for key in list(st.session_state):
            if key.startswith(("filter_", "evidence_term_")) or key in {"mode", "page"}:
                del st.session_state[key]
        st.session_state.last_selection = selected
    if selected is not None:
        st.sidebar.write(labels[selected])
        st.sidebar.caption(
            "Times come from recorded successful processing attempts. A replay does not make source data newer. Missing generation times use a labeled analysis cutoff."
        )
    if selected is not None and "published_run" not in st.session_state:
        st.session_state.pop("published_run", None)
        try:
            with st.spinner("Checking report integrity…"):
                run = load_published_run(root, selected)
                validate_display_contract(run.tables)
                st.session_state.published_run = run
        except Exception:
            st.error(
                "This report could not be verified. Review its source files and integrity before reloading. No fallback was loaded."
            )
            return
    run = st.session_state.get("published_run")
    if run is None:
        st.info(
            "No published results were found. Publish research separately; no data has been substituted."
        )
        return
    try:
        if run.root != root or run.result["run_id"] != selected:
            raise ValueError("Selection mismatch")
        run.verify()
    except Exception:
        st.session_state.pop("published_run", None)
        st.error(
            "The published files changed or are no longer verifiable. Cached results were cleared. Review the files, then explicitly reopen the report."
        )
        return
    st.session_state.pop("verified_evidence", None)
    try:
        st.session_state.verified_evidence = load_evidence(run, cfg)
    except Exception:
        st.error(
            "Linked within-filter evidence could not be verified. It has been cleared; descriptive results remain separate. No fallback evidence was loaded."
        )
    st.session_state.pop("paper_definition_snapshot", None)
    try:
        st.session_state.paper_definition_snapshot = load_paper_definitions(
            run, cfg, st.session_state.get("verified_evidence")
        )
    except Exception:
        st.error(
            "The paper definition snapshot could not be verified. Definition results were cleared; no live register or fallback was loaded."
        )
    dates = report_dates(run, cfg.display_timezone)
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.caption("ANALYSIS CUTOFF")
        a.write(dates["Analysis cutoff"])
        b.caption("DATA LAST RECEIVED")
        b.write(dates["Data last received"])
        c.caption("EXPORT DATE")
        c.write(dates["Export date"])
    st.sidebar.divider()
    mode = st.sidebar.selectbox(
        "Funding",
        ["all", *MODES],
        format_func=lambda v: "All funding groups" if v == "all" else MODE_LABELS[v],
        key="mode",
    )
    products = sorted(run.tables["canonical"]["economic_summary"]["product"].unique())
    st.sidebar.selectbox(
        "Product",
        ["all", *products],
        format_func=lambda p: "All products" if p == "all" else PRODUCTS.get(p, p),
        key="filter_product",
    )
    st.sidebar.caption(
        "All-data browsing keeps each signal’s funding and product label. Paper returns remain hypothetical; no blended ROI is calculated."
    )
    st.sidebar.number_input(
        "Minimum analyzed wagers",
        min_value=1,
        max_value=1000000,
        value=cfg.min_analyzed_wagers,
        step=25,
        key="filter_min_wagers",
        help="Display threshold for signal and saved-filter cohorts, not a statistical validation gate.",
    )
    if st.session_state.get("page", "Signals & leaks") == "Signals & leaks":
        sidebar_signal_filters(run, mode, st.session_state.get("filter_product", "all"))
    page = st.sidebar.radio("Explore", list(PAGES), key="page")
    st.sidebar.caption(
        "Read-only historical research. No filters, tests or trades are changed here."
    )
    try:
        st.session_state.table_ordinal = 0
        PAGES[page](run, mode, min(cfg.max_display_rows, 20))
        if page == "Evidence & methodology":
            with st.expander("Technical details · local processing attempts"):
                st.caption(
                    "Local operational observations, not research results. A failed attempt never loads an older report automatically."
                )
                attempts = operational_attempts(root)
                if attempts:
                    st.table(attempts[: cfg.max_display_rows], hide_index=True)
    except Exception:
        st.session_state.pop("published_run", None)
        st.error(
            "This published view is unavailable or unsupported. Cached results were cleared; no fallback was loaded."
        )


if __name__ == "__main__":
    main()
