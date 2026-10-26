"""Four stakeholder questions, with details revealed only on request."""

import json

import pandas as pd
import streamlit as st

from oddsjam_ev.dashboard import charts, queries
from oddsjam_ev.dashboard.labels import (
    ACTIONS,
    evidence_step,
    money,
    percent,
    pretty,
    reason_text,
    segment_label,
    view_label,
)
from oddsjam_ev.dashboard.presentation import LIMITATIONS, plot, require_mode, table


def _product():
    return st.session_state.get("filter_product")


def _cohort(run, mode):
    if not require_mode(mode):
        return None
    return queries.product_cohort(run.tables, mode, _product())


def summary(run, mode, limit):
    st.header("Your results at a glance")
    st.caption("What was recorded, where the results came from, and what still needs care.")
    cohort = _cohort(run, mode)
    if cohort is None:
        return
    if cohort.empty:
        st.info("No published wager results exist for this funding group and product.")
        return
    row = cohort.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric(
        "Hypothetical profit" if mode == "paper" else "Reported profit",
        money(row.eligible_reported_profit_usd),
        border=True,
    )
    b.metric(
        "Reported ROI",
        percent(row.reported_roi),
        help="Reported profit divided by stake for the exact published eligible cohort. Fees are not independently verified.",
        border=True,
    )
    c.metric("Analyzed wagers", f"{int(row.eligible_count):,}", border=True)
    d.metric("Analyzed stake", money(row.eligible_stake_usd), border=True)
    st.caption(
        f"{int(row.candidate_count):,} logged wagers in this selection · {int(row.candidate_count - row.eligible_count):,} excluded from reported ROI. Returns describe historical records, not future expectations."
    )
    st.subheader("The findings to act on in research")
    queue = queries.research_queue(run.tables, mode, _product())
    for section in ("Positive historical patterns", "Loss investigations"):
        selected = queue.loc[queue.section.eq(section)].head(1)
        for _, finding in selected.iterrows():
            _finding_card(finding, section)
    with st.expander("Small-sample outliers and other context"):
        for text in queries.takeaways(run.tables, mode, _product()):
            st.write("• " + text)
    st.subheader("Where the results are coming from")
    st.caption(
        "Main charts show cohorts with at least 30 analyzed wagers; smaller cohorts remain in Performance explorer. This is a readability rule, not validation."
    )
    a, b = st.columns(2)
    with a:
        st.markdown("**ROI by platform**")
        plot(
            charts.bars(
                queries.chart_cohorts(run.tables, mode, _product(), "sportsbook").query(
                    "eligible_count >= 30"
                ),
                "reported_roi",
                "Reported ROI",
                percentage=True,
            )
        )
    with b:
        st.markdown("**Profit by market**")
        plot(
            charts.bars(
                queries.chart_cohorts(run.tables, mode, _product(), "market_family").query(
                    "eligible_count >= 30"
                ),
                "eligible_reported_profit_usd",
                "Reported profit",
                currency=True,
            )
        )
    st.caption(
        "Charts show the largest analyzed cohorts by wager count. n is the analyzed sample; * marks fewer than 30 wagers. A high return from a few wagers is not validation."
    )
    with st.expander("View supporting data"):
        table(cohort, limit)


def performance(run, mode, limit):
    st.header("Explore the recorded performance")
    st.caption(
        "Find concentration, loss pockets and sample-size differences within one published view."
    )
    if not require_mode(mode):
        return
    options = sorted(run.tables["atlas"]["wager_segments"].view.unique())
    if not options:
        st.info("No published segment views are available.")
        return
    default = options.index("sportsbook") if "sportsbook" in options else 0
    view = st.selectbox(
        "Compare by", options, index=default, format_func=view_label, key="filter_atlas_view"
    )
    frame = queries.chart_cohorts(run.tables, mode, _product(), view)
    left, right = st.columns([2, 1])
    with left:
        metric = st.radio(
            "Show",
            ["reported_roi", "eligible_reported_profit_usd", "eligible_count"],
            format_func=lambda x: {
                "reported_roi": "Reported ROI",
                "eligible_reported_profit_usd": "Reported profit",
                "eligible_count": "Analyzed wagers",
            }[x],
            horizontal=True,
            key="filter_metric",
        )
        plot(
            charts.bars(
                frame,
                metric,
                {
                    "reported_roi": "Reported ROI",
                    "eligible_reported_profit_usd": "Reported profit",
                    "eligible_count": "Analyzed wagers",
                }[metric],
                percentage=metric == "reported_roi",
                currency=metric.endswith("_usd"),
            )
        )
    with right:
        st.markdown("**Reading this view**")
        st.write(
            "Each bar is one published segment within your selected funding group and product."
        )
        st.write(
            "Teal and rose distinguish positive and negative results. The tooltip always includes the number of analyzed wagers."
        )
        st.write(
            "Missing classifications remain visible rather than being assigned to a market or funding group."
        )
    st.subheader("Is the return backed by volume?")
    plot(charts.scatter(frame))
    st.caption(
        "Fewer than 30 wagers is a display cue for very small samples, not an inference threshold. Larger samples can still be biased or dependent. Pan or zoom the plot and hover for details."
    )
    with st.expander("View supporting data"):
        table(frame.drop(columns=["sample"]), limit)
    with st.expander("Logged opportunity proxies · across all products"):
        proxy = queries.atlas_view(run.tables, mode, view, proxies=True)
        proxy["label"] = (
            proxy.apply(lambda r: " · ".join(pretty(r[d], d) for d in view.split("__")), axis=1)
            if len(proxy)
            else []
        )
        table(proxy[["label", "opportunity_proxy_count", "selection_proxy_count"]], limit)
        st.caption(
            "Counts describe logged proxies across products. They are not executable opportunities and must not be added across overlapping views or funding groups."
        )
    with st.expander("Saved filter membership · overlapping groups"):
        source = run.tables["atlas"]["filter_segment_diagnostics"]
        source = source.loc[
            source.execution_status.eq(mode)
            & source["product"].eq(_product())
            & source.view.eq("filter_id__attribution_status")
        ].copy()
        registry = {
            f["id"]: f["name"]
            for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
        }
        source["Saved filter"] = source.filter_id.map(registry).fillna(
            "Filter not resolved in the registry"
        )
        table(source[["Saved filter", "candidate_count", "eligible_count", "reported_roi"]], limit)
        st.caption(
            "These are recorded saved-filter names, not new segment names. Membership overlaps, so attribution does not establish incremental or causal contribution."
        )


def _bullets(value, *, next_evidence=False):
    parsed = json.loads(value) if isinstance(value, str) else value
    if isinstance(parsed, dict):
        parsed = [f"{k}: {v}" for k, v in parsed.items()]
    if not isinstance(parsed, list):
        parsed = [parsed]
    for item in dict.fromkeys(str(v) for v in parsed):
        st.write("• " + (evidence_step(item) if next_evidence else reason_text(item)))


def _finding_card(row, heading):
    title, result, detail, question = queries.research_finding(row)
    with st.container(border=True):
        st.caption(heading.upper())
        st.subheader(title)
        st.write("**What the records show:** " + result)
        st.write(detail)
        st.write("**Next investigation:** " + question)
        st.caption(
            "Historical research priority, not a validated strategy or instruction to wager. Related segments can overlap."
        )


def followup(run, mode, limit):
    st.header("Which angles deserve investigation next?")
    st.caption("Review questions and evidence gaps, not an automatic trading shortlist.")
    gates = run.tables["discovery"]["temporal_gates"]
    blocked = not gates.empty and gates.status.ne("passed").any()
    if blocked:
        st.info(
            "Historical patterns can be explored, but chronological validation is not established. We cannot yet confirm what was known before each bet."
        )
    with st.expander("What evidence is missing?"):
        reasons = []
        for value in gates.reasons_json:
            reasons += json.loads(value)
        for reason in dict.fromkeys(reason_text(r) for r in reasons):
            st.write("• " + reason)
    if not require_mode(mode):
        return
    candidates = run.tables["recommendations"]["proposals"]
    candidates = candidates.loc[
        candidates.execution_status.eq(mode) & candidates["product"].eq(_product())
    ].copy()
    queue = queries.research_queue(run.tables, mode, _product())
    st.subheader("Start with these research questions")
    st.write(
        "These are existing research segments to investigate or paper-test, not live betting opportunities. Published ‘Review manually’ labels describe workflow, not strength of evidence."
    )
    st.caption(
        "Within each section: more event-proxy groups first, then more analyzed wagers; ties use a stable reference. No profitability score is invented. At least 30 wagers separates the main reading queue from sparse observations; it is not a statistical gate."
    )
    section = st.radio(
        "Research focus",
        ["Positive historical patterns", "Loss investigations", "More observations needed"],
        horizontal=True,
        key="filter_focus",
    )
    shortlist = queue.loc[queue.section.eq(section)]
    st.caption(
        f"Showing up to 5 of {len(shortlist):,} distinct segments in this section. One segment may have multiple published follow-ups; related segments overlap and are not independent discoveries."
    )
    if shortlist.empty:
        st.info("No published candidates belong to this section for your selection.")
    for _, finding in shortlist.head(5).iterrows():
        _finding_card(finding, section)
    st.subheader("Explore the full catalogue and published follow-ups")
    action = st.selectbox(
        "Type of follow-up",
        sorted(candidates.action.unique()),
        index=None,
        placeholder="All follow-up types",
        format_func=lambda a: ACTIONS.get(a, a),
        key="filter_action",
    )
    if action is not None:
        candidates = candidates.loc[candidates.action.eq(action)]
    order = {identity: n for n, identity in enumerate(queue.candidate_id)}
    candidates["reading_order"] = candidates.candidate_id.map(order)
    candidates = candidates.sort_values(["reading_order", "proposal_id"])

    descriptions = {}
    duplicates = {}
    registry = {
        f["id"]: f["name"]
        for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
    }
    for _, r in candidates.iterrows():
        e = queue.loc[queue.candidate_id.eq(r.candidate_id)]
        sample = f"{int(e.iloc[0].eligible_count):,} wagers · " if len(e) == 1 else ""
        label = sample + segment_label(r) + " — " + ACTIONS.get(r.action, r.action)
        if pd.notna(r.target_filter_id):
            label += " · " + registry.get(r.target_filter_id, "Target not resolved")
        duplicates[label] = duplicates.get(label, 0) + 1
        descriptions[r.proposal_id] = label + (
            f" · item {duplicates[label]}" if duplicates[label] > 1 else ""
        )
    identity = st.selectbox(
        "Explore a review item",
        candidates.proposal_id.tolist(),
        index=None,
        placeholder="Search by platform, market, league or EV range",
        format_func=lambda i: descriptions[i],
        key="filter_proposal",
    )
    if identity is not None:
        row = candidates.set_index("proposal_id").loc[identity]
        with st.container(border=True):
            st.subheader(segment_label(row))
            st.write("**Published follow-up:** " + ACTIONS.get(row.action, row.action))
            target = registry.get(row.target_filter_id) if pd.notna(row.target_filter_id) else None
            st.caption(
                "Saved filter target: "
                + (target or "No saved filter assigned. This item describes a research segment.")
            )
            evidence = run.tables["discovery"]["candidate_results"]
            evidence = evidence.loc[evidence.candidate_id.eq(row.candidate_id)]
            if len(evidence) == 1:
                e = evidence.iloc[0]
                x, y, z = st.columns(3)
                x.metric("Analyzed wagers", f"{int(e.eligible_count):,}")
                y.metric("Reported ROI", percent(e.reported_roi))
                z.metric("Reported profit", money(e.eligible_reported_profit_usd))
                if 0 < e.eligible_count < 30:
                    st.warning("Very small sample: this result is not evidence for promotion.")
                if pd.notna(e.cluster_roi_ci_low) and pd.notna(e.cluster_roi_ci_high):
                    st.caption(
                        f"Recorded group interval: {percent(e.cluster_roi_ci_low)} to {percent(e.cluster_roi_ci_high)}. Interpretation depends on the published group assumptions; it does not establish validation."
                    )
            st.markdown("**Why review is needed**")
            blockers = []
            for field in ("evidence_blockers_json", "approval_blockers_json"):
                blockers.extend(json.loads(row[field]))
            _bullets(blockers)
            st.markdown("**Evidence to collect next**")
            _bullets(row.required_next_evidence_json, next_evidence=True)
            st.caption(
                "Published initial status: "
                + str(row.initial_review_status).replace("_", " ").lower()
                + ". This is not the current decision register and cannot be approved here."
            )
            with st.expander("Technical references for this item"):
                st.code(identity)
                st.code(row.candidate_id)
    st.divider()
    st.subheader("Tracked tests")
    if "experiments" not in run.tables or run.tables["experiments"]["experiment_protocols"].empty:
        st.info(
            "No prospective tests are tracked in this report. Historical paper records are still useful for exploration, but are not automatically registered experiments."
        )
        st.caption(
            "Next step: register a test before collecting observations, then publish an updated report separately."
        )
        return
    experiments = run.tables["experiments"]
    names = {}
    for row in experiments["experiment_protocols"].itertuples():
        names[row.experiment_id] = json.loads(row.protocol_json).get(
            "name", "Registered paper test"
        )
    metrics = queries.experiment_metrics(run.tables, mode).copy()
    metrics["Test"] = metrics.experiment_id.map(names)
    with st.expander("Published test outcomes by availability"):
        table(
            metrics[
                [
                    "Test",
                    "availability_status",
                    "enrolled_count",
                    "eligible_count",
                    "hypothetical_reported_roi",
                ]
            ],
            limit,
        )
        st.caption(
            "Unchecked and checked observations remain separate. Paper returns are hypothetical; readiness requires human review."
        )


def confidence(run, mode, limit):
    st.header("Data & methodology")
    st.caption(
        "Use this supporting reference to understand inclusion rules and interpretation. Start with Summary or Research follow-up for findings and next questions."
    )
    summary = run.manifests["canonical"]["summary"]
    total = summary["table_rows"]["bet_facts"]
    eligible = summary["reported_roi_eligible_bets"]
    a, b, c = st.columns(3)
    a.metric(
        "Included in reported ROI",
        f"{eligible / total:.1%}" if total else "Unavailable",
        border=True,
    )
    b.metric("Excluded wager candidates", f"{total - eligible:,}", border=True)
    unknown = int(run.tables["canonical"]["bet_facts"].execution_status.eq("unknown").sum())
    c.metric("Funding not classified", f"{unknown:,}", border=True)
    st.caption(
        "These coverage counts describe all wager candidates in the report. They are not combined monetary performance."
    )
    a, b = st.columns(2)
    with a:
        st.subheader("What can be analyzed?")
        plot(
            charts.counts(
                pd.DataFrame(
                    {
                        "label": ["Included in reported ROI", "Excluded from reported ROI"],
                        "count": [eligible, total - eligible],
                    }
                ),
                title="Wager candidates",
                donut=True,
            )
        )
    with b:
        st.subheader("Why records were excluded")
        rows = []
        for reasons, count in summary["exclusion_combinations"].items():
            codes = json.loads(reasons)
            if codes:
                rows.append({"label": " + ".join(reason_text(c) for c in codes), "count": count})
        exclusions = pd.DataFrame(rows, columns=["label", "count"]).sort_values(
            "count", ascending=False
        )
        shown = exclusions.head(5).copy()
        if len(exclusions) > 5:
            shown = pd.concat(
                [
                    shown,
                    pd.DataFrame(
                        [
                            {
                                "label": "Other exclusion combinations",
                                "count": int(exclusions.iloc[5:]["count"].sum()),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        plot(charts.counts(shown, title="Excluded wager candidates"))
    st.caption(
        "Exclusion combinations are mutually exclusive here. Individual reasons can overlap and must not be summed separately."
    )
    st.subheader("What this means for interpretation")
    items = [
        (
            "Export timing",
            "Not confirmed: a recent import does not prove that the export is current."
            if run.result["specification"]["freshness"]["export_status"] == "unverified"
            else "The recorded export assessment applies at the report cutoff, not the current clock.",
        ),
        (
            "Funding labels",
            "Original snapshot classifications are preserved. Unclassified funding is not silently reassigned to paper or real-money activity.",
        ),
        (
            "Fees and fills",
            "Reported profit is preserved. Fee inclusion and actual order acceptance are not independently verified.",
        ),
        (
            "Historical validation",
            "Missing decision-time and settlement chronology can prevent a valid chronological holdout test.",
        ),
    ]
    for title, text in items:
        with st.container(border=True):
            st.markdown("**" + title + "**")
            st.write(text)
    with st.expander("View supporting quality data"):
        table(exclusions, limit)
        quality = run.tables["atlas"]["atlas_quality"]
        quality = quality.loc[quality.category.isin(["<missing>", "<unmapped>", "<conflict>"])]
        table(quality[["population", "dimension", "category", "count", "denominator"]], limit)
    with st.expander("Technical details · provenance, policies and diagnostics"):
        st.write(LIMITATIONS)
        st.json(
            {
                "run_id": run.result["run_id"],
                "snapshot_id": run.result["specification"]["snapshot_id"],
                "code": run.result["specification"]["code"],
            }
        )
        st.json(run.result["specification"]["configuration"])
        st.json(run.result["specification"]["source_diagnostics"])


PAGES = {
    "Summary": summary,
    "Performance explorer": performance,
    "Research follow-up": followup,
    "Data & methodology": confidence,
}
