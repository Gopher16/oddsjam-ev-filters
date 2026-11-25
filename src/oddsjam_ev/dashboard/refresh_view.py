"""Opt-in local CSV refresh controls, separate from report selection."""

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from oddsjam_ev.dashboard import refresh
from oddsjam_ev.dashboard.artifacts import load_published_run
from oddsjam_ev.dashboard.labels import report_dates
from oddsjam_ev.dashboard.queries import validate_display_contract
from oddsjam_ev.io.ingestion import SourceMetadata
from oddsjam_ev.io.staging import preview_csv, stage_csv
from oddsjam_ev.pipeline.config import configuration_bundle, read_config


def render(config_path: str, artifact_root: Path):
    """Offer bounded staging, explicit build and verified activation in that order."""
    with st.expander("Refresh Bet Tracker CSV"):
        st.caption(
            "Local use only · CSV up to 50 MB / 100,000 rows. Builds preserve your selected report."
        )
        try:
            cfg = read_config(Path(config_path))
            if Path(cfg.output_root).resolve() != artifact_root:
                raise ValueError("Output root mismatch")
            bundle = configuration_bundle(cfg)
            store = Path(bundle["source_config"]["store"])
            private = artifact_root / "dashboard_refresh"
            staging, jobs = private / "staging", private / "jobs"
            upload = st.file_uploader(
                "Drop a Bet Tracker CSV or browse files",
                type=["csv"],
                max_upload_size=50,
                key="refresh_upload",
            )
            settings_path = private / "source_settings.json"
            settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
            default_source = bundle["source_config"]["source"]
            default_namespace = default_source["namespace"]
            if default_namespace == "demo":
                default_namespace = "oddsjam"
            st.caption(
                "Source settings are remembered locally. Coverage cannot be proven from CSV rows; "
                "confirm your usual export once below. Upload time is recorded automatically, "
                "but is not treated as export time."
            )
            with st.expander("Source settings (first upload or overrides)", expanded=not settings):
                namespace = st.text_input(
                    "Source/account alias", value=settings.get("namespace", default_namespace)
                )
                timezone = st.text_input(
                    "Timezone for timestamps without offsets",
                    value=settings.get(
                        "timestamp_timezone", default_source.get("timestamp_timezone", "unknown")
                    ),
                    help="Timestamp offsets in the CSV are preserved. A timezone cannot safely be inferred for naive timestamps.",
                )
                scopes = ["unknown", "full", "partial"]
                scope = st.selectbox(
                    "Usual export coverage",
                    scopes,
                    index=scopes.index(settings.get("scope", "unknown")),
                )
                if st.button("Save source settings", disabled=scope == "unknown"):
                    source = SourceMetadata(namespace, None, timezone, scope)
                    from dataclasses import asdict

                    from oddsjam_ev.io.staging import _private_root

                    _private_root(private)
                    with tempfile.NamedTemporaryFile(mode="w", dir=private, delete=False) as saved:
                        json.dump(asdict(source), saved)
                        saved_path = Path(saved.name)
                    try:
                        os.replace(saved_path, settings_path)
                    finally:
                        saved_path.unlink(missing_ok=True)
                    st.success("Source settings saved for future uploads.")
                exported = st.text_input(
                    "Actual export time (optional override, ISO timestamp with timezone)", value=""
                )
            st.caption(
                f"Using source: {namespace} · coverage: {scope} · naive timestamp timezone: {timezone}"
            )
            if st.button("Preview CSV", disabled=upload is None):
                st.session_state.pop("refresh_preview", None)
                upload.seek(0)
                receipt = stage_csv(
                    upload,
                    staging,
                    filename=upload.name,
                    metadata=SourceMetadata(namespace, exported or None, timezone, scope),
                )
                st.session_state.refresh_preview = preview_csv(staging, receipt["stage_id"], store)
            preview = st.session_state.get("refresh_preview")
            process = st.session_state.get("refresh_process")
            active = process is not None and process.poll() is None
            if preview:
                st.subheader("Review before importing")
                st.caption(
                    "This preview is frozen. Edited upload/metadata controls apply only after Preview CSV again."
                )
                st.write(
                    {
                        "Source": preview["source"],
                        "Coverage": preview["coverage_state"],
                        "Repeated upload": preview["repeated_import"],
                    }
                )
                st.dataframe([preview["candidate_rows"]], hide_index=True)
                st.write(
                    {
                        "Settlement revisions": preview["settlement_revision_rows"],
                        "Source diagnostics": preview["diagnostics"],
                    }
                )
                st.caption(
                    "Unresolved rows remain unresolved. Analytical exclusions are reported after building; candidate counts are not verified fills. Import time is not export freshness."
                )
                confirmed = st.checkbox(
                    "I reviewed this preview and confirm import and report build",
                    key="refresh_confirmed_" + preview["preview_sha256"],
                )
                if st.button(
                    "Import and build report",
                    disabled=active or not confirmed or preview["source"]["scope"] == "unknown",
                ):
                    job = refresh.create_job(jobs, cfg, staging, preview)
                    st.session_state.refresh_job = job
                    st.session_state.refresh_process = refresh.launch(jobs, job)
                    st.rerun()
            job = st.session_state.get("refresh_job")
            if job:
                state = refresh.status(jobs, job)
                if not active and state["status"] in {"running", "queued"}:
                    st.warning(
                        "Worker stopped without a terminal receipt. Review local locks before retrying; no automatic restart occurred."
                    )
                st.write("Build status: " + state["status"])
                if state.get("events"):
                    st.dataframe(state["events"], hide_index=True)
                st.button("Refresh build status")
                if st.button("Cancel build", disabled=not active):
                    refresh.cancel(jobs, job)
                    st.info(
                        "Cancellation requested at the next safe boundary. Committed imports remain; already published reports remain valid."
                    )
                if st.button("Retry frozen build", disabled=active or state["status"] == "ready"):
                    st.session_state.refresh_process = refresh.launch(jobs, job)
                    st.rerun()
                if state["status"] == "ready":
                    run_id = state["outcome"]["run_id"]
                    ready = load_published_run(artifact_root, run_id)
                    st.success("Report built. Your selected report has not changed.")
                    st.write(report_dates(ready))
                    st.write({"Build completed": state["outcome"]["completed_at"]})
                    summary = ready.result["summaries"]["canonical"]["reconciliation"]
                    st.write(
                        {
                            "Reported ROI eligible wagers": summary["reported_roi_eligible_bets"],
                            "Analytical exclusion combinations": summary["exclusion_combinations"],
                        }
                    )
                    st.caption(
                        "This is historical snapshot reanalysis; source/export coverage and chronology retain their original limitations."
                    )
                    if st.button("Activate verified report"):
                        run = load_published_run(artifact_root, run_id)
                        validate_display_contract(run.tables)
                        st.session_state.selected_run = run_id
                        st.session_state.pop("published_run", None)
                        st.rerun()
        except Exception:
            st.error(
                "Refresh could not complete or verify its frozen inputs. Check CSV size (50 MB), row count (100,000), source settings and local receipts. Re-preview if inputs changed. Your selected report was preserved."
            )
