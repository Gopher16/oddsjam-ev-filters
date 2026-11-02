"""Visual opportunity and production review over immutable published evidence."""

import json

import pandas as pd
import streamlit as st

from oddsjam_ev.analysis.decision_records import parameter_sheet
from oddsjam_ev.dashboard import decision_charts as charts
from oddsjam_ev.dashboard.decision_queries import (
    FIELDS,
    TIERS,
    select_records,
    study_draft,
    terms,
    title,
    visible_headlines,
)
from oddsjam_ev.dashboard.labels import PRODUCTS, percent, pretty


def reset():
    for key in list(st.session_state):
        if key.startswith(("filter_signal_", "filter_workspace_")) or key == "decision_selected":
            del st.session_state[key]
    st.query_params.pop("candidate", None)
    st.session_state.pop("last_map_event", None)
    for key in list(st.session_state):
        if key.startswith("explanation_") and key not in {
            "explanation_bundle",
            "explanation_publication",
        }:
            del st.session_state[key]
    st.session_state.workspace_chart_revision = (
        st.session_state.get("workspace_chart_revision", 0) + 1
    )


def sidebar(workspace, run):
    """One shared selector set; no data-dependent fitting or hidden intersections."""
    records = workspace["queues"]["records"] if workspace else []
    registry = {
        f["id"]: f["name"]
        for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
    }
    explanation_page = st.session_state.get("page") == "Explain returns"
    source = workspace["visuals"]["visual_candidates"] if workspace and explanation_page else None
    selected_saved = st.session_state.get("filter_signal_saved", "all")
    st.sidebar.selectbox(
        "Saved filter",
        [
            "all",
            *sorted(
                {
                    r["filter_id"]
                    for r in records
                    if r["source_kind"] != "global_descriptive_catalogue"
                }
                | (
                    set(
                        run.tables["canonical"]["filter_attribution"]
                        .loc[lambda f: f.attribution_status.eq("resolved"), "filter_id"]
                        .dropna()
                    )
                    if explanation_page
                    else set()
                )
                | ({selected_saved} if selected_saved != "all" else set())
            ),
        ],
        format_func=lambda v: "All saved filters" if v == "all" else registry.get(v, v),
        key="filter_signal_saved",
    )
    st.sidebar.caption(
        "Recorded-wager selection: AND across fields, OR within. Model and background stay fixed. Empty means all."
        if explanation_page
        else "AND across fields · OR within a field. Only exact published slices qualify; empty means all."
    )
    for field, label in FIELDS.items():
        options = {terms(r)[field] for r in records if field in terms(r)}
        if source is not None:
            if field in {"sportsbook", "sport", "league", "market_family"}:
                options |= set(source[field].fillna("<missing>").astype(str))
            elif field in {"odds_band", "advertised_ev_band"}:
                options |= set(run.tables["atlas"]["wager_dimensions"][field].dropna().astype(str))
                options = {v for v in options if v.startswith("[") and v.endswith(")")}
        options |= set(st.session_state.get("filter_signal_" + field, []))
        options = sorted(options)
        st.sidebar.multiselect(
            label,
            options,
            format_func=lambda v, d=field: pretty(v, d),
            key="filter_signal_" + field,
        )
    with st.sidebar.expander("Evidence & configuration"):
        st.multiselect(
            "Evidence tier", list(TIERS), format_func=TIERS.get, key="filter_workspace_tiers"
        )
        st.multiselect(
            "Setup readiness",
            ["blocked", "ready_for_operator_configuration_review"],
            key="filter_workspace_setup",
            format_func=lambda v: v.replace("_", " "),
        )
        st.multiselect(
            "Confirmed lifecycle",
            ["PROD", "TEST", "retired", "unknown"],
            key="filter_workspace_lifecycle",
        )
    st.sidebar.button("Reset signal filters", on_click=reset)


def scope():
    return {
        "mode": st.session_state.get("mode", "all"),
        "product": st.session_state.get("filter_product", "all"),
        "saved": st.session_state.get("filter_signal_saved", "all"),
        "selections": {k: st.session_state.get("filter_signal_" + k, []) for k in FIELDS},
    }


def selected_records(workspace):
    return select_records(
        workspace["queues"]["records"],
        **scope(),
        minimum=int(st.session_state.get("filter_min_wagers", 100)),
        tiers=st.session_state.get("filter_workspace_tiers", []),
        setup=st.session_state.get("filter_workspace_setup", []),
        lifecycle=st.session_state.get("filter_workspace_lifecycle", []),
    )


def selected_changed():
    ident = st.session_state.get("decision_selected")
    if ident:
        st.query_params["candidate"] = ident
    else:
        st.query_params.pop("candidate", None)


def focus(ident):
    st.session_state.decision_selected = ident
    st.query_params["candidate"] = ident


def explain_idea(candidate_id):
    """Navigate with an explicit recorded-row scope; do not choose/refit a model."""
    st.session_state.explanation_candidate_id = candidate_id
    st.session_state.page = "Explain returns"


def card(record):
    with st.container(border=True):
        st.markdown("**" + title(record) + "**")
        st.caption(
            TIERS[record["evidence_tier"]]
            + " · "
            + ("Setup unresolved" if record["setup_status"] == "blocked" else "Setup review ready")
        )
        sub = record["historical_economics"]["subgroup"]
        a, b = st.columns(2)
        a.metric("Stake-weighted ROI", percent(sub["weighted_roi"]))
        b.metric("Equal-stake return", percent(sub["equal_stake_return"]))
        c = record["rank_components"]
        st.caption(
            f"{sub['eligible_count']:,} wagers · ≥{c['minimum_arm_event_groups']:,} events per arm · published priority {record['presentation_order']}"
        )
        st.button(
            "Inspect setup & plan test",
            key="inspect-" + record["candidate_id"],
            on_click=focus,
            args=(record["candidate_id"],),
            width="stretch",
        )


def details(records, workspace):
    if not records:
        return
    ids = [r["candidate_id"] for r in records]
    lookup = {r["candidate_id"]: r for r in records}
    deep = st.query_params.get("candidate")
    if "decision_selected" not in st.session_state and deep in ids:
        st.session_state.decision_selected = deep
    elif deep and deep not in ids:
        st.info(
            "The linked idea is outside the selected evidence scope. Clear selectors to inspect it; no other idea is substituted."
        )
    if st.session_state.get("decision_selected") not in [None, *ids]:
        del st.session_state["decision_selected"]
    chosen = st.selectbox(
        "Inspect a published idea",
        ids,
        index=None,
        format_func=lambda v: title(lookup[v])
        + " · "
        + lookup[v]["execution_status"].replace("_", " "),
        placeholder="Select a card, chart point or idea",
        key="decision_selected",
        on_change=selected_changed,
    )
    if chosen is None:
        return
    row = lookup[chosen]
    st.subheader(title(row))
    lifecycle = st.session_state.get("lifecycle_report_snapshot")
    if lifecycle:
        decisions = [
            e
            for e in lifecycle["decisions"]
            if e["payload"]["source"]["definition"]["source"]["candidate_id"] == chosen
        ]
        if decisions:
            latest = decisions[-1]
            st.info(
                "Reviewed follow-up: "
                + latest["payload"]["action"]
                + ". Inspect Paper-test definitions for approved changes, confirmed application periods and monitoring. This decision is not production authorization."
            )

    prospective = st.session_state.get("prospective_report_snapshot")
    if prospective and "prospective_reviews" in prospective["tables"]:
        studies = prospective["tables"]["prospective_reviews"]
        studies = studies.loc[studies.candidate_id.eq(chosen)]
        if not studies.empty:
            st.caption(
                "Prospective follow-up for this exact candidate · inspect Paper-test definitions for per-version outcomes"
            )
            st.dataframe(
                studies[
                    [
                        "test_key",
                        "version",
                        "status",
                        "scheduled_review_utc",
                        "review_due",
                        "review_recorded",
                        "readiness",
                    ]
                ],
                hide_index=True,
            )

    st.caption(
        "Evidence source: "
        + row["execution_status"].replace("_", " ")
        + " · "
        + PRODUCTS.get(row["product"], row["product"])
        + " · "
        + TIERS[row["evidence_tier"]]
    )
    st.button(
        "Explain this idea’s recorded wagers",
        key="explain-" + chosen,
        on_click=explain_idea,
        args=(chosen,),
    )
    a, b = st.columns(2)
    a.altair_chart(charts.comparison(row), width="stretch")
    b.altair_chart(charts.comparison(row, contrast=True), width="stretch")
    st.caption(
        "Parent / subgroup / complement use the same economic eligibility. Contrasts show pointwise intervals where published; these are not corrected confidence intervals. Missing intervals remain unavailable."
    )
    tabs = st.tabs(
        ["Full OddsJam setup", "Why this priority", "Overlap & economics", "Plan paper study"]
    )
    with tabs[0]:
        sheet = row["configuration_sheet"]
        st.write("**Setup:** " + row["setup_status"].replace("_", " "))
        st.caption(
            "Inherited, observed, override, unset and unknown are distinct. An omitted historical setting is not a default. Setup readiness does not authorize production."
        )
        if sheet["setup_blockers"]:
            with st.expander(f"Resolve {len(sheet['setup_blockers'])} configuration blockers"):
                st.write([v.replace("_", " ") for v in sheet["setup_blockers"]])
        st.dataframe(pd.DataFrame(parameter_sheet(sheet)), hide_index=True, width="stretch")
        st.download_button(
            "Download full parameter sheet",
            json.dumps(sheet, indent=2),
            file_name="candidate-parameter-sheet.json",
            mime="application/json",
            key="parameter-" + chosen,
            on_click="ignore",
        )
        if row["production_proposal"]:
            st.warning(
                "Proposed paper comparison only. Confirm actual applied settings and an equivalent scanner exclusion before any external change."
            )
            st.json(row["production_proposal"])
    with tabs[1]:
        st.write(row["ranking_explanation"])
        st.json(row["rank_components"])
        st.caption(
            f"Sensitivity without interval objective: order {row['sensitivity_order_without_interval']} (change {row['sensitivity_order_change_without_interval']:+d}). Setup is a separate readiness state; historical dollars do not determine rank."
        )
        st.write(
            "**Published headline status:** "
            + ("Included" if row["headline"] else (row["headline_blocker"] or "Unavailable"))
        )
        if row["original_hypotheses"]:
            st.dataframe(
                pd.DataFrame(row["original_hypotheses"])[
                    ["measure", "direction", "raw_pvalue", "holm_pvalue", "exploratory_holm_reject"]
                ],
                hide_index=True,
            )
        else:
            st.info(
                "Separate descriptive global catalogue: no tests, p-values or corrected claims were published for this idea."
            )
        st.caption(
            "No independently validated or deployable edge is established. Quotes, fees, chronology and event dependence remain limitations."
        )
    with tabs[2]:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Population": k,
                        **{name: value for name, value in v.items() if name != "sizing_regimes"},
                    }
                    for k, v in row["historical_economics"].items()
                ]
            ),
            hide_index=True,
        )
        others = {r["candidate_id"]: title(r) for r in workspace["queues"]["records"]}
        pairs = [
            p
            for p in workspace["queues"]["overlap"]
            if chosen in (p["left_candidate_id"], p["right_candidate_id"])
        ]
        overlap = pd.DataFrame(
            [
                {
                    "Other idea": others[
                        p["right_candidate_id"]
                        if p["left_candidate_id"] == chosen
                        else p["left_candidate_id"]
                    ],
                    "Wager overlap": p["eligible_wager_jaccard"],
                    "Event overlap": p["event_jaccard"],
                    "Shared event proxies": p["shared_event_proxies"],
                }
                for p in pairs
            ]
        )
        if len(overlap):
            st.dataframe(overlap.sort_values("Event overlap", ascending=False), hide_index=True)
        else:
            st.info("No shared recorded wager/event proxies with other published ideas.")
    with tabs[3]:
        global_idea = row["source_kind"] == "global_descriptive_catalogue"
        st.write("**Next step:** " + row["next_action"].replace("_", " "))
        st.download_button(
            "Download global study brief" if global_idea else "Download paper-test draft",
            json.dumps(study_draft(row, workspace), indent=2),
            file_name="global-study-brief.json" if global_idea else "paper-test-draft.json",
            mime="application/json",
            key="draft-" + chosen,
            on_click="ignore",
        )
        st.caption(
            "No registration, approval, launch or enrollment occurs here. Complete baseline, future dates, fixed paper sizing, quote availability and review criteria before saving a test."
        )
        if global_idea:
            st.info(
                "This is a research handoff, not a registerable definition. Global definition integration and a complete new prospective baseline are required; no historical parent is invented."
            )
        with st.expander("Pinned source and exact rule"):
            st.json(
                {
                    "candidate_id": chosen,
                    "predicate": json.loads(row["predicate_json"]),
                    "queue_publication": workspace["queues"]["manifest"]["publication_id"],
                }
            )


def overview(run, mode, limit, *, queue=None):
    workspace = st.session_state.get("decision_workspace")
    st.header(
        "Next tests & production review"
        if queue is None
        else "Ideas to test"
        if queue == "opportunities"
        else "Production concerns"
    )
    if workspace is None:
        st.info(
            "No verified review queue is linked to this report. Publish and attach matching queue/visual artifacts; descriptive history is available in Explore history and is not substituted as a shortlist."
        )
        return
    records = selected_records(workspace)
    total = len(workspace["queues"]["records"])
    if queue == "opportunities":
        records = [r for r in records if r["queue"] == queue]
    elif queue == "production_changes":
        records = [
            r
            for r in records
            if r["queue"] == queue
            or r["direction"] in {"loss_association", "sizing_or_estimand_conflict"}
        ]
    st.caption(
        f"{len(records):,} / {total:,} published ideas in scope · original ranking preserved · no pooled paper/real-money ROI"
    )
    if not records:
        st.info(
            "No published ideas match these selectors. Clear a field or publish the missing intersection; broader summaries are not substituted."
        )
        return
    if queue is None:
        a, b, c = st.columns(3)
        a.metric(
            "Exploratory research ideas",
            sum(r["evidence_tier"] == "support_qualified_exploratory" for r in records),
        )
        b.metric(
            "Corrected historical comparisons",
            sum(r["evidence_tier"] == "corrected_historical_comparison" for r in records),
        )
        c.metric(
            "Confirmed PROD proposals", sum(r["production_proposal"] is not None for r in records)
        )
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Ideas to test")
            heads = visible_headlines(records, "opportunities")
            if not heads:
                st.info(
                    "No published headline ideas in this scope. Insufficient records remain available below; no replacements are invented."
                )
            columns = st.columns(2)
            for i, row in enumerate(heads):
                with columns[i % 2]:
                    card(row)
        with right:
            production(records)
    elif queue == "production_changes":
        production(records)
    else:
        for row in visible_headlines(records, queue):
            card(row)
    st.subheader("Evidence map")
    st.caption(
        "Click a point to inspect its rule and setup. Zoom or pan; unavailable intervals are not drawn. Research priority is not proof of future profit."
    )
    chart = charts.evidence_map(records)
    if chart is not None:
        chart_key = (
            "workspace_map_"
            + str(queue)
            + "_"
            + str(st.session_state.get("workspace_chart_revision", 0))
        )
        event = st.altair_chart(
            chart,
            width="stretch",
            on_select="rerun",
            selection_mode=["candidate_pick"],
            key=chart_key,
        )
        points = event.selection.get("candidate_pick", [])
        current_event = (chart_key, tuple(point.get("candidate_id") for point in points))
        if current_event != st.session_state.get("last_map_event"):
            st.session_state.last_map_event = current_event
            if points:
                chosen = points[0].get("candidate_id")
                if chosen in {r["candidate_id"] for r in records}:
                    focus(chosen)

    else:
        st.info("No support-qualified contrasts are available to plot in this scope.")
    details(records, workspace)
    with st.expander("All published ideas in scope · accessible evidence table"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Idea": title(r),
                        "Queue": r["queue"],
                        "Evidence": TIERS[r["evidence_tier"]],
                        "Published order": r["presentation_order"],
                        "Setup": r["setup_status"],
                        "Historical funding": r["execution_status"],
                        "Product": r["product"],
                    }
                    for r in records
                ]
            ),
            hide_index=True,
        )


def production(records):
    st.subheader("Production concerns")
    rows = visible_headlines(records, "production_changes")
    for r in rows:
        card(r)
    if not rows:
        losses = [r for r in records if r["direction"] == "loss_association"]
        st.info("No confirmed-PROD change proposal is ready for headline review in this scope.")
        if losses:
            st.write(
                f"{len(losses):,} historical loss associations need review. A PROD name or funded wager does not establish the current configuration."
            )
        st.caption(
            "Confirm effective-dated PROD lifecycle and a pinned parent baseline. Then compare baseline against the proposed exclusion on paper."
        )
    conflicts = sum(r["direction"] == "sizing_or_estimand_conflict" for r in records)
    st.caption(
        f"{conflicts:,} sizing/return-measure conflicts remain investigations, not automatic exclusions."
    )


def opportunities(run, mode, limit):
    overview(run, mode, limit, queue="opportunities")


def production_page(run, mode, limit):
    overview(run, mode, limit, queue="production_changes")
