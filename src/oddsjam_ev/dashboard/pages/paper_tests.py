"""Read-only prospective intent; no launch or outcome evidence is invented."""

import pandas as pd
import streamlit as st


def paper_tests(run, mode, limit):
    """Display frozen definition versions matching the selected report evidence."""
    st.header("Paper-test definitions")
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
    rows = pd.DataFrame(
        [
            {
                "Test": e["test_key"],
                "Name": e["protocol"]["name"],
                "Version": e["version"],
                "Current": snapshot["latest"][e["test_key"]] == e["version_id"],
                "Planned start": e["protocol"]["planned_start_utc"],
                "Planned end": e["protocol"]["planned_end_utc"],
                "Status": "Definition only · not launched",
                "Evaluation": "No prospective observations linked",
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
    with st.expander("Immutable definition and source provenance"):
        st.json(event)
