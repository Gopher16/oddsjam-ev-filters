"""Read-only reviewed decisions, approved changes, confirmed periods and monitoring."""

import pandas as pd
import streamlit as st


def lifecycle_evidence(version_id):
    """Show version-bound operational evidence without authorizing or applying changes."""
    snapshot = st.session_state.get("lifecycle_report_snapshot")
    if snapshot is None:
        st.caption(
            "No reviewed lifecycle report attached; applied settings and subsequent monitoring are unknown here."
        )
        return
    decisions = [
        e
        for e in snapshot["decisions"]
        if e["payload"]["source"]["definition_version_id"] == version_id
    ]
    if not decisions:
        st.info("No reviewed decision is linked to this exact definition version.")
        return
    st.subheader("Reviewed outcomes and configuration changes")
    st.caption(
        "Production requires the registered prospective policy, a complete fixed-look family assessment and explicit bounded-pilot approval. Approval and external application are separate records."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Decision": e["payload"]["action"],
                    "Recorded": e["recorded_at"],
                    "Reviewer": e["actor"],
                    "Rationale": e["rationale"],
                    "Evidence readiness": e["payload"]["source"]["readiness"],
                    "Production authorized": False,
                }
                for e in decisions
            ]
        ),
        hide_index=True,
    )
    for evaluation in snapshot.get("evaluations", []):
        assessment = evaluation["payload"]["assessment"]
        selected = [r for r in assessment["results"] if r["definition_version_id"] == version_id]
        if selected:
            with st.expander(
                "Fixed-look prospective promotion evidence · "
                + str(assessment["family_size"])
                + " corrected claims"
            ):
                st.caption(
                    "Eligibility at the recorded assessment time permits human review only. Approvals recheck complete-family evidence; marginal bootstrap intervals are not simultaneous confidence intervals."
                )
                st.json(
                    {
                        "selected_results": selected,
                        "complete_family_hypotheses": assessment["hypotheses"],
                    }
                )
    ids = {e["event_id"] for e in decisions}
    changes = [
        v for v in snapshot["changes"].values() if v["event"]["payload"]["decision_event_id"] in ids
    ]
    filters = {v["event"]["payload"]["filter_id"] for v in changes}
    for c in changes:
        p = c["event"]["payload"]
        with st.expander(p["filter_id"] + " · " + p["change_kind"] + " · " + c["status"]):
            st.write(
                "Approval blockers: "
                + (
                    ", ".join(p["approval_blockers"])
                    or "None recorded; explicit human approval still required"
                )
            )
            st.json(
                {
                    "before": p["before_settings"],
                    "after": p["after_settings"],
                    "diff": p["diff"],
                    "scanner_evidence": p.get("ui_evidence", []),
                    "planned_effective_at_utc": p["planned_effective_at_utc"],
                }
            )
    for name, label in (
        ("applications", "Confirmed configuration periods"),
        ("metrics", "Subsequent economics by applied version"),
        ("drift", "Declared drift checks and coverage"),
    ):
        frame = snapshot["tables"][name].loc[lambda f: f.filter_id.isin(filters)]
        st.write("**" + label + "**")
        st.dataframe(frame, hide_index=True)
    st.caption(
        "ROI is a fraction. Paper and operator-reported funded returns remain separate. Alerts and review actions do not deploy, stop or roll back a filter. Missing references remain unknown."
    )
