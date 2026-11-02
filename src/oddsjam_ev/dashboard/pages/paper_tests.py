"""Read-only prospective intent; no launch or outcome evidence is invented."""

import pandas as pd
import streamlit as st


def paper_tests(run, mode, limit):
    """Display frozen definition versions matching the selected report evidence."""
    st.header("Prospective paper tests")
    st.caption(
        "Frozen plans for future paper tracking. Saving a definition does not approve, launch or enroll an experiment."
    )
    snapshot = st.session_state.get("paper_definition_snapshot")
    if snapshot is None:
        st.info(
            "No paper-test definition snapshot is linked. Download a draft from Within-filter evidence, complete it, then save and publish it with the local definition command."
        )
        return
    st.caption(
        "Definition snapshot knowledge cutoff: " + snapshot["manifest"]["specification"]["as_of"]
    )
    events = snapshot["events"]
    product = st.session_state.get("filter_product", "all")
    events = [
        e
        for e in events
        if (mode == "all" or e["source"]["historical_funding"] == mode)
        and (product == "all" or e["source"]["product"] == product)
    ]
    if not events:
        st.info(
            "No definition versions match this report's exact evidence and display selections. Historical paper wagers are not automatically registered tests."
        )
        return
    st.caption(
        "Funding selectors describe the historical source population. Every prospective definition is paper only. Superseded versions remain visible."
    )
    report = st.session_state.get("prospective_report_snapshot")
    report_tables = report["tables"] if report else {}
    review_rows = report_tables.get("prospective_reviews", pd.DataFrame())
    statuses = (
        {}
        if review_rows.empty
        else dict(zip(review_rows.definition_version_id, review_rows.status, strict=True))
    )
    if report:
        st.caption("Prospective report cutoff: " + report["manifest"]["specification"]["as_of"])
    else:
        st.info(
            "No prospective outcome report attached. Operational status is unknown here; definitions do not establish launch or enrollment."
        )
    rows = pd.DataFrame(
        [
            {
                "Test": e["test_key"],
                "Name": e["protocol"]["name"],
                "Version": e["version"],
                "Current": snapshot["latest"][e["test_key"]] == e["version_id"],
                "Planned start": e["protocol"]["planned_start_utc"],
                "Planned end": e["protocol"]["planned_end_utc"],
                "Status": statuses.get(
                    e["version_id"], "Definition only · no operational evidence attached"
                ),
                "Evaluation": "See per-version prospective evidence"
                if e["version_id"] in statuses
                else "No prospective report attached",
            }
            for e in events
        ]
    )
    st.dataframe(rows, hide_index=True)
    by_id = {e["version_id"]: e for e in events}
    selected = st.selectbox(
        "Paper definition version",
        list(by_id),
        format_func=lambda v: f"{by_id[v]['protocol']['name']} · version {by_id[v]['version']}",
    )
    event = by_id[selected]
    protocol, source = event["protocol"], event["source"]
    if snapshot["latest"][event["test_key"]] != selected:
        st.warning(
            "Superseded definition: do not pool future observations with the replacement version."
        )
    if "configuration_reference" not in protocol:
        st.info(
            "Legacy untyped baseline settings: preserved as recorded, but not verified as a complete scanner setup."
        )
    else:
        st.caption("Configuration-bound definition · pinned parameter sheet reference")
        st.json(protocol["configuration_reference"])
    st.write("**Hypothesis:** " + protocol["hypothesis"])
    st.write("**Trial arm:** " + source["trial_arm"].replace("_", " "))
    st.write("**Baseline:** unchanged parent filter; exact prospective settings recorded below.")
    st.json(
        {
            "exact_candidate_predicate": source["exact_predicate"],
            "baseline": protocol["baseline"],
            "sizing": protocol["sizing"],
        }
    )
    st.write("**Review dates:** " + ", ".join(protocol["review_dates_utc"]))
    st.json(
        {
            k: protocol[k]
            for k in (
                "primary_metric",
                "secondary_metric",
                "min_wagers_per_arm",
                "min_event_groups_per_arm",
                "review_criteria",
                "stop_rules",
                "settlement_followup_days",
                "overlap_policy",
                "quote_availability_policy",
            )
        }
    )
    st.info(
        "Elapsed dates do not establish a launch, observations or completed review. Operational approval, launch, enrollment and outcome linkage use the separate experiment workflow."
    )
    if selected in statuses:
        st.subheader("Version-specific prospective evidence")
        for name, label in (
            ("prospective_reviews", "Scheduled reviews and blocking reasons"),
            ("prospective_metrics", "Parent, subgroup and complement economics"),
            ("prospective_observations", "Captured observations and availability"),
        ):
            table = report_tables[name]
            table = table.loc[table.definition_version_id.eq(selected)]
            st.write("**" + label + "**")
            if name == "prospective_metrics":
                chart = table.loc[table.cohort.eq("all_captured_hypothetical")].set_index("arm")
                st.bar_chart(chart[["stake_weighted_roi", "equal_stake_mean_return"]])
            st.dataframe(table.head(limit), hide_index=True)
        st.caption(
            "Returns are fractions (0.03 = 3%). Fixed paper stakes; voids excluded, pushes included. Parent overlaps subgroup/complement. Failed or unknown quotes remain hypothetical; no fills, pooled versions or promotion are inferred."
        )
    with st.expander("Immutable definition and source provenance"):
        st.json(event)
