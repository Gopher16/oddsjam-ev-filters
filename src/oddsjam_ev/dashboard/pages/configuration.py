"""Complete parameter sheets with explicit setup and production-evidence states."""

import json

import pandas as pd
import streamlit as st

from oddsjam_ev.analysis.decision_records import parameter_sheet
from oddsjam_ev.analysis.paper_definitions import draft_definition


def render_sheet(record, publication):
    """Show baseline, logical change and final control state without hiding blockers."""
    st.write("**Setup:** " + record["setup_status"].replace("_", " "))
    lifecycle = record["lifecycle"]
    st.caption(
        "Filter lifecycle: "
        + lifecycle["status"]
        + " · "
        + lifecycle["confirmation_status"]
        + " · effective "
        + lifecycle["effective_at"]
    )
    st.caption(
        "This is prospective setup evidence. Historical performance was not reconstructed for these settings. A ready setup is not production approval."
    )
    if record["setup_blockers"]:
        with st.expander(f"Resolve {len(record['setup_blockers'])} setup blockers"):
            st.write([b.replace("_", " ") for b in record["setup_blockers"]])
    if record["design"] == "exclusion_comparison":
        st.info(
            "Proposed logical change: retain the parent except the entire candidate conjunction. Independent control inversions are not equivalent; scanner implementation remains blocked."
        )
    st.dataframe(pd.DataFrame(parameter_sheet(record)), hide_index=True)
    with st.expander("Exact change and baseline evidence"):
        st.json(record["logical_change"])
        st.json(record["baseline"])
        st.json(lifecycle)
    st.download_button(
        "Download complete parameter sheet",
        json.dumps(record, indent=2),
        file_name="candidate-parameter-sheet.json",
        mime="application/json",
        key="sheet-" + record["decision_id"],
        on_click="ignore",
    )
    if record["setup_status"] == "ready_for_operator_configuration_review":
        # A typed reference is reverified against these bytes when saving, not trusted
        # merely because the client copied IDs into a draft.
        draft = draft_definition(
            {
                "candidate_id": record["candidate_id"],
                "predicate_json": json.dumps(record["logical_change"]["terms"]),
            }
        )
        draft["protocol"]["design"] = record["design"]
        draft["protocol"]["baseline"]["filter_config"] = {
            r["control"]: r["baseline_value"] for r in record["settings"]
        }
        draft["protocol"]["baseline"]["evidence_reference"] = record["baseline"][
            "evidence_reference"
        ]
        draft["protocol"]["configuration_reference"] = {
            "contract_version": 1,
            "publication_id": publication["manifest"]["publication_id"],
            "decision_id": record["decision_id"],
            "record_sha256": record["record_sha256"],
        }
        st.download_button(
            "Download configuration-bound paper draft",
            json.dumps(draft, indent=2),
            file_name="configured-paper-draft.json",
            mime="application/json",
            key="configured-" + record["decision_id"],
            on_click="ignore",
        )


def candidate_sheet(candidate_id):
    """Render sheets for a selected verified candidate; old reports stay usable."""
    publication = st.session_state.get("decision_sheet_snapshot")
    if publication is None:
        st.info(
            "Complete setup sheet unavailable. The analytical predicate does not specify all scanner controls; omitted settings are unknown, not defaults."
        )
        return
    records = [r for r in publication["records"] if r["candidate_id"] == candidate_id]
    if not records:
        st.info("No configuration sheets match this candidate and report.")
        return
    selected = st.selectbox(
        "Configuration change design",
        [r["design"] for r in records],
        format_func=lambda v: "Study subgroup"
        if v == "subgroup_tracking"
        else "Investigate parent exclusion",
        key="config-design-" + candidate_id,
    )
    record = next(r for r in records if r["design"] == selected)
    render_sheet(record, publication)
