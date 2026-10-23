"""Frozen-input local orchestration with verified checkpoints and private attempts."""

import errno
import hashlib
import json
import tempfile
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.io.atlas import load_atlas, publish_atlas
from oddsjam_ev.io.canonical import build_canonical_bundle, load_canonical_bundle
from oddsjam_ev.io.discovery import load_discovery, publish_discovery
from oddsjam_ev.io.experiment_register import read_experiment_events
from oddsjam_ev.io.experiment_reports import load_experiment_report, publish_experiment_report
from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.manifests import verify_bytes, writer_lock
from oddsjam_ev.io.recommendations import load_recommendations, publish_recommendations
from oddsjam_ev.io.snapshots import load_snapshot, materialize_snapshot
from oddsjam_ev.io.source_csv import fingerprint, read_source
from oddsjam_ev.pipeline.config import PipelineConfig, configuration_bundle
from oddsjam_ev.pipeline.freshness import assess_freshness
from oddsjam_ev.pipeline.manifests import (
    atomic_json,
    code_provenance,
    fingerprint_report,
    load_result,
    read_plan,
)
from oddsjam_ev.pipeline.reporting import render_report

LOADERS = {
    "canonical": load_canonical_bundle,
    "atlas": load_atlas,
    "discovery": load_discovery,
    "recommendations": load_recommendations,
    "experiments": load_experiment_report,
}
TRANSIENT = {errno.EINTR, errno.EAGAIN, errno.ETIMEDOUT, errno.ESTALE}


def utc_now() -> str:
    """System recording time; never derive source chronology from this clock."""
    return datetime.now(UTC).isoformat()


def _summary(stage, path):
    tables = LOADERS[stage](path)
    summary = {"table_rows": {key: len(value) for key, value in tables.items()}}
    if stage == "canonical":
        summary["reconciliation"] = json.loads((path / "manifest.json").read_text())["summary"]
        summary["reported_economics_separate_cohorts"] = json.loads(
            tables["economic_summary"].to_json(orient="records")
        )
    if stage == "discovery":
        summary["test_status"] = tables["candidate_results"].test_status.value_counts().to_dict()
        summary["temporal_gates"] = json.loads(tables["temporal_gates"].to_json(orient="records"))
    if stage == "recommendations":
        summary["research_actions"] = tables["proposals"].action.value_counts().to_dict()
    return summary


def run_pipeline(
    cfg: PipelineConfig,
    *,
    csv: Path | None = None,
    snapshot: Path | None = None,
    as_of: str | None = None,
    resume: str | None = None,
    register: Path | None = None,
    temporal_evidence: Path | None = None,
    decisions: Path | None = None,
) -> dict:
    """Run one explicit input or resume an exact frozen plan.

    Exit codes: 0 completed (possibly descriptive/gated), 3 input notice, 2 failure.
    All error details are stable codes, never raw exception text. Completed stage
    artifacts survive failures; only a fully verified run receives a result bundle.
    """
    root = Path(cfg.output_root)
    attempt_id = uuid.uuid4().hex
    with writer_lock(root):
        attempt = root / "attempts" / attempt_id
        attempt.mkdir(parents=True)
        atomic_json(attempt / "started.json", {"started_at": utc_now(), "attempt_id": attempt_id})
        events = []
        run_id = None
        current_stage = "preflight"

        def event(stage, state, **fields):
            entry = {"stage": stage, "state": state, "recorded_at": utc_now(), **fields}
            atomic_json(attempt / f"{len(events):04d}.json", entry)
            events.append(entry)

        def finish(status, exit_code, error_code=None):
            record = {
                "attempt_id": attempt_id,
                "run_id": run_id,
                "status": status,
                "exit_code": exit_code,
                "error_code": error_code,
                "completed_at": utc_now(),
                "stages": events,
            }
            atomic_json(attempt / "outcome.json", record)
            return record

        def notice(code):
            event("input", "notice", code=code)
            (attempt / "notice.md").write_text(
                "# Research input notice\n\nStatus: "
                + code
                + ". No new analytical report was published. No older input was substituted.\n"
            )
            return finish(code, 3)

        try:
            input_request = None
            configs = configuration_bundle(cfg)
            code = code_provenance()
            if resume:
                if any(
                    v is not None
                    for v in (csv, snapshot, as_of, register, temporal_evidence, decisions)
                ):
                    raise ValueError("Resume cannot change inputs")
                if len(resume) != 64 or any(c not in "0123456789abcdef" for c in resume):
                    raise ValueError("Invalid resume identity")
                plan = read_plan(root / "plans" / f"{resume}.json")
                if plan["run_id"] != resume:
                    raise ValueError("Resume plan identity mismatch")
                spec = plan["specification"]
                if (
                    spec["configuration"] != configs
                    or spec["pipeline_policy"] != asdict(cfg)
                    or spec["code"] != code
                ):
                    raise ValueError("Changed code or configuration cannot resume")
                snapshot = Path(plan["snapshot_path"])
                register = Path(plan["register_path"]) if plan["register_path"] else None
                as_of = spec["as_of"]
            else:
                if (csv is None) == (snapshot is None):
                    if csv is None:
                        return notice("missing_input")
                    raise ValueError("Exactly one input required")
                mode = "historical_reanalysis" if snapshot else "manual_import"
                if csv is not None:
                    if as_of is not None:
                        raise ValueError("CSV cutoff is captured after import; cannot backdate")
                    if not csv.is_file():
                        return notice("missing_input")
                    current_stage = "ingestion"
                    csv_bytes = csv.read_bytes()
                    _, diagnostic = read_source(csv_bytes)
                    if any(diagnostic["invalid_numeric_values"].values()):
                        raise ValueError("Invalid numeric source values")
                    source_cfg = configs["source_config"]
                    metadata = json.loads(decisions.read_text()) if decisions else []
                    source_hash = hashlib.sha256(csv_bytes).hexdigest()
                    request_id = fingerprint([source_hash, configs, code, metadata])
                    input_request = root / "input_requests" / f"{request_id}.json"
                    import_id = fingerprint([source_cfg["source"]["namespace"], source_hash])
                    if not input_request.exists():
                        atomic_json(
                            input_request,
                            {
                                "already_imported": (
                                    Path(source_cfg["store"]) / "imports" / import_id
                                ).exists(),
                            },
                        )
                    request = json.loads(input_request.read_text())
                    imported = import_csv(
                        csv, Path(source_cfg["store"]), SourceMetadata(**source_cfg["source"])
                    )
                    if imported["source_sha256"] != source_hash:
                        raise ValueError("Input changed during ingestion")
                    event(
                        "ingestion",
                        "reused" if imported["repeated_import"] else "completed",
                        import_id=imported["import_id"],
                    )
                    if imported["repeated_import"] and (
                        request["already_imported"] or input_request.with_suffix(".done").exists()
                    ):
                        return notice("no_new_source_content")
                    # Receipt time is stable even if snapshot publication is interrupted.
                    as_of = imported["imported_at"]
                    current_stage = "snapshot"
                    snapshot = materialize_snapshot(
                        Path(source_cfg["store"]), as_of=as_of, decisions=metadata
                    )
                elif decisions is not None:
                    raise ValueError(
                        "Resolutions belong to snapshot construction, not snapshot replay"
                    )
                if as_of is None:
                    raise ValueError("Snapshot mode requires an explicit aware knowledge cutoff")
                as_of = aware(as_of).isoformat()
                if aware(as_of) > aware(utc_now()):
                    raise ValueError("Future knowledge cutoff forbidden")
                if not snapshot.is_dir():
                    return notice("missing_input")
                load_snapshot(snapshot)
                manifest = json.loads((snapshot / "manifest.json").read_text())
                imports = manifest["specification"]["imports"]
                if any(any(i["diagnostics"]["invalid_numeric_values"].values()) for i in imports):
                    raise ValueError("Invalid numeric source values")
                cutoff = manifest["specification"]["as_of"]
                if cutoff and aware(cutoff) > aware(as_of):
                    raise ValueError("Snapshot cutoff exceeds run cutoff")
                freshness = assess_freshness(imports, as_of, cfg.max_export_age_days)
                if mode == "manual_import" and freshness["export_status"] != "recent":
                    return notice(
                        "stale_input"
                        if freshness["export_status"] == "stale"
                        else "freshness_unverified"
                    )
                spec = {
                    "contract_version": 1,
                    "mode": mode,
                    "as_of": as_of,
                    "snapshot_id": manifest["snapshot_id"],
                    "snapshot_manifest_sha256": fingerprint_report(
                        (snapshot / "manifest.json").read_text()
                    ),
                    "configuration": configs,
                    "pipeline_policy": asdict(cfg),
                    "code": code,
                    "freshness": freshness,
                    "source_diagnostics": [
                        {
                            "import_id": i["import_id"],
                            "diagnostics": i["diagnostics"],
                        }
                        for i in imports
                    ],
                    "temporal_evidence": json.loads(temporal_evidence.read_text())
                    if temporal_evidence
                    else None,
                    "experiment_events": read_experiment_events(register, as_of)
                    if register
                    else None,
                }
                run_id = fingerprint(spec)
                plan = {
                    "run_id": run_id,
                    "specification": spec,
                    "snapshot_path": str(snapshot.resolve()),
                    "register_path": str(register.resolve()) if register else None,
                }
                plan_path = root / "plans" / f"{run_id}.json"
                if plan_path.exists():
                    if read_plan(plan_path)["specification"] != spec:
                        raise ValueError("Plan conflict")
                else:
                    atomic_json(plan_path, plan)
                if input_request is not None and not input_request.with_suffix(".done").exists():
                    atomic_json(input_request.with_suffix(".done"), {"run_id": run_id})
            current_stage = "preflight"
            run_id = fingerprint(spec)
            # Recheck frozen sources on every attempt, including resumes.
            verify_bytes(snapshot / "manifest.json", spec["snapshot_manifest_sha256"])
            load_snapshot(snapshot)
            if register and read_experiment_events(register, as_of) != spec["experiment_events"]:
                raise ValueError("Frozen register changed")
            event("preflight", "completed", snapshot_id=spec["snapshot_id"])
            outputs, summaries = {}, {}
            cache = root / "checkpoints" / run_id

            def stage(name, function):
                nonlocal current_stage
                current_stage = name
                event(name, "started")
                checkpoint = cache / f"{name}.json"
                if checkpoint.exists():
                    record = json.loads(checkpoint.read_text())
                    if (
                        fingerprint({k: v for k, v in record.items() if k != "checkpoint_sha256"})
                        != record["checkpoint_sha256"]
                    ):
                        raise ValueError("Checkpoint integrity failure")
                    if record["run_id"] != run_id or record["stage"] != name:
                        raise ValueError("Checkpoint run mismatch")
                    out = Path(record["path"])
                    verify_bytes(out / "manifest.json", record["manifest_sha256"])
                    LOADERS[name](out)
                    state = "reused"
                else:
                    for retry in range(cfg.transient_retries + 1):
                        try:
                            out = function()
                            LOADERS[name](out)
                            break
                        except OSError as exc:
                            if exc.errno not in TRANSIENT or retry == cfg.transient_retries:
                                raise
                            event(name, "retry", retry=retry + 1, code="transient_io")
                    record = {
                        "run_id": run_id,
                        "stage": name,
                        "path": str(out.resolve()),
                        "manifest_sha256": fingerprint_report((out / "manifest.json").read_text()),
                    }
                    record["checkpoint_sha256"] = fingerprint(record)
                    atomic_json(checkpoint, record)
                    state = "completed"
                outputs[name] = record
                summaries[name] = _summary(name, out)
                event(name, state)
                return out

            artifacts = root / "artifacts"
            canonical = stage(
                "canonical",
                lambda: build_canonical_bundle(
                    snapshot,
                    artifacts / "canonical",
                    CanonicalConfig(**configs["canonical_config"]),
                    configs["registry"],
                ),
            )
            atlas = stage(
                "atlas",
                lambda: publish_atlas(
                    canonical, artifacts / "atlas", AtlasConfig(**configs["atlas_config"])
                ),
            )
            discovery = stage(
                "discovery",
                lambda: publish_discovery(
                    atlas,
                    artifacts / "discovery",
                    DiscoveryConfig(**configs["discovery_config"]),
                    spec["temporal_evidence"],
                ),
            )
            stage(
                "recommendations",
                lambda: publish_recommendations(
                    discovery,
                    atlas,
                    canonical,
                    artifacts / "recommendations",
                    RecommendationConfig(**configs["recommendation_config"]),
                    configs["registry"],
                    configs["capabilities"],
                ),
            )
            if register is not None:
                stage(
                    "experiments",
                    lambda: publish_experiment_report(
                        register, canonical, artifacts / "experiments", as_of
                    ),
                )
            current_stage = "publication"
            if (
                code_provenance() != spec["code"]
                or configuration_bundle(cfg) != spec["configuration"]
            ):
                raise ValueError("Code or configuration changed during execution")
            report = render_report(spec, summaries)
            result = {
                "run_id": run_id,
                "specification": spec,
                "outputs": outputs,
                "summaries": summaries,
                "report": report,
                "report_sha256": fingerprint_report(report),
            }
            result["payload_sha256"] = fingerprint(result)
            destination = root / "results" / run_id
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                if load_result(destination) != result:
                    raise ValueError("Result replay mismatch")
            else:
                with tempfile.TemporaryDirectory(
                    prefix=".stage-", dir=destination.parent
                ) as temporary:
                    staged = Path(temporary) / "result"
                    staged.mkdir()
                    atomic_json(staged / "manifest.json", result)
                    (staged / "report.md").write_text(report)
                    load_result(staged)
                    staged.rename(destination)
            event("publication", "completed", run_id=run_id)
            return finish(
                "completed_historical" if spec["mode"] == "historical_reanalysis" else "completed",
                0,
            )
        except Exception as exc:
            # Unexpected failures must not dump source-bearing exception text to logs.
            code = (
                "io_failure"
                if isinstance(exc, OSError)
                else "validation_failure"
                if isinstance(exc, (ValueError, TypeError, KeyError, AssertionError))
                else "unexpected_failure"
            )
            event(current_stage, "failed", code=code)
            return finish("failed", 2, code)
