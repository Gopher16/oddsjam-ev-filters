"""
research_jobs.py
===============================================================================
Purpose
-------
Freeze and dispatch an explicit bounded graph of existing research publishers.

What this module provides
-------------------------
Allowlisted dependency closure, complete policy snapshots and optional-stage gates.

Design Notes
------------
No commands/plugins, automatic training, scheduling, activation or new statistics.
===============================================================================
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.analysis.review_config import ReviewConfig
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig
from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.io.decision_records import load_decisions, publish_decisions
from oddsjam_ev.io.manifests import verify_bytes
from oddsjam_ev.io.review_queues import load_review_queues, publish_review_queues
from oddsjam_ev.io.signal_prioritization import load_prioritization, publish_prioritization
from oddsjam_ev.io.sizing_diagnostics import load_sizing_diagnostics, publish_sizing_diagnostics
from oddsjam_ev.io.sizing_history import load_sizing_history, publish_sizing_history
from oddsjam_ev.io.visual_diagnostics import load_visual_diagnostics, publish_visual_diagnostics
from oddsjam_ev.io.within_filter_discovery import (
    load_within_filter_discovery,
    publish_within_filter_discovery,
)

# Insertion order is a fixed topological order, not configurable executable code.
DEPENDENCIES = {
    "sizing": ("canonical",),
    "sizing_diagnostics": ("canonical", "sizing"),
    "within_filter_discovery": ("canonical", "atlas", "sizing_diagnostics"),
    "prioritization": ("within_filter_discovery", "sizing_diagnostics"),
    "sheets": ("within_filter_discovery", "prioritization"),
    "visuals": ("canonical", "atlas", "sizing_diagnostics"),
    "queues": ("within_filter_discovery", "prioritization", "sheets", "visuals"),
    "models": ("canonical", "visuals"),
}
POLICIES = {
    "sizing_diagnostics": SizingDiagnosticsConfig,
    "within_filter_discovery": MultidimensionalConfig,
    "prioritization": PrioritizationConfig,
    "visuals": VisualConfig,
    "queues": ReviewConfig,
}
LOADERS = {
    "sizing": load_sizing_history,
    "sizing_diagnostics": load_sizing_diagnostics,
    "within_filter_discovery": load_within_filter_discovery,
    "prioritization": load_prioritization,
    "sheets": load_decisions,
    "visuals": load_visual_diagnostics,
    "queues": load_review_queues,
}


def dependency_order(stages: list[str]) -> list[str]:
    """Expand requested stages to a fixed topological allowlisted closure.

    Parameters
    ----------
    stages : list of str
        Nonempty unique supported stage names.

    Returns
    -------
    list of str
        Optional stages only; canonical/atlas remain core-run prerequisites.
    """
    if (
        not isinstance(stages, list)
        or not stages
        or any(not isinstance(s, str) or s not in DEPENDENCIES for s in stages)
        or len(set(stages)) != len(stages)
    ):
        raise ValueError("Unique allowlisted research stages required")
    selected = set(stages)
    for name in reversed(DEPENDENCIES):
        if name in selected:
            selected.update(DEPENDENCIES[name])
    return [name for name in DEPENDENCIES if name in selected]


def _document(path, base):
    file = Path(path)
    if not file.is_absolute():
        file = base / file
    with file.open("rb") as stream:
        raw = stream.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Evidence file exceeds job input bound")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "content": yaml.safe_load(raw)}


def freeze_job(path: Path, *, fit_models: bool = False) -> dict:
    """Read and validate a job once; snapshot policies and evidence contents.

    Parameters
    ----------
    path : pathlib.Path
        Local YAML request. Evidence file paths resolve against its directory.
    fit_models : bool
        Separate explicit training opt-in, false for ordinary refresh.

    Returns
    -------
    dict
        Versioned JSON-compatible frozen request, independent of mutable files.
    """
    if type(fit_models) is not bool:
        raise ValueError("Model fitting requires a boolean opt-in")
    path = Path(path)
    raw = _document(path, Path.cwd())["content"]
    allowed = {
        "contract_version",
        "stages",
        "stage_configs",
        "sizing_history_file",
        "sizing_previous",
        "decision_evidence_file",
        "decision_effective_at",
        "model_policy",
        "model_scope",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) - allowed
        or type(raw.get("contract_version")) is not int
        or raw["contract_version"] != 1
    ):
        raise ValueError("Unsupported research job schema")
    order = dependency_order(raw.get("stages"))
    configs = raw.get("stage_configs", {})
    if not isinstance(configs, dict) or set(configs) - set(POLICIES) or set(configs) - set(order):
        raise ValueError("Policies must belong to selected stages")
    frozen = {
        "contract_version": 1,
        "requested_stages": sorted(raw["stages"]),
        "order": order,
        "dependencies": {s: list(DEPENDENCIES[s]) for s in order},
        "stage_configs": {
            s: asdict(POLICIES[s](**configs.get(s, {}))) for s in order if s in POLICIES
        },
        "fit_models": fit_models,
        "sizing_history": _document(raw["sizing_history_file"], path.parent)
        if raw.get("sizing_history_file")
        else None,
        "decision_evidence": _document(raw["decision_evidence_file"], path.parent)
        if raw.get("decision_evidence_file")
        else None,
        "decision_effective_at": aware(raw["decision_effective_at"]).isoformat()
        if raw.get("decision_effective_at")
        else None,
        "sizing_previous": None,
        "model_policy": None,
        "model_scope": raw.get("model_scope", {}),
        "model_runtime": None,
    }
    if not isinstance(frozen["model_scope"], dict):
        raise ValueError("Model scope must be an object")
    if fit_models and "models" not in order:
        raise ValueError("Model opt-in requires a requested model stage")
    if "models" in order:
        from importlib.metadata import version

        import pandas as pd

        from oddsjam_ev.analysis.explanatory_models import ModelPolicy, apply_scope

        frozen["model_runtime"] = {name: version(name) for name in ("scikit-learn", "shap")}
        frozen["model_policy"] = asdict(ModelPolicy(**raw.get("model_policy", {})))
        # Existing scope validator also rejects unsupported feature/operator names.
        columns = {key: pd.Series(dtype="object") for key in frozen["model_scope"]}
        apply_scope(pd.DataFrame(columns), frozen["model_scope"])
    elif "model_policy" in raw or "model_scope" in raw:
        raise ValueError("Model settings require a requested model stage")
    if raw.get("sizing_previous"):
        if not frozen["sizing_history"]:
            raise ValueError("Sizing predecessor requires explicit history")
        previous = Path(raw["sizing_previous"])
        previous = previous if previous.is_absolute() else path.parent / previous
        load_sizing_history(previous)
        frozen["sizing_previous"] = {
            "path": str(previous.resolve()),
            "manifest_sha256": hashlib.sha256(
                (previous / "manifest.json").read_bytes()
            ).hexdigest(),
        }
    # Fail before any processing on values not representable by the manifest contract.
    return json.loads(json.dumps(frozen, allow_nan=False))


def verify_job_inputs(job, as_of):
    """Reverify frozen external lineage and model runtime on every attempt.

    Parameters
    ----------
    job : dict
        Frozen version-1 request from the core run specification.
    as_of : str
        Aware core knowledge cutoff; evidence cannot come from its future.

    Returns
    -------
    None
        Raises on changed predecessor bytes, future sizing knowledge or runtime.
    """
    if job["sizing_history"] is not None:
        cutoff = job["sizing_history"]["content"].get("knowledge_as_of")
        if cutoff is None or aware(cutoff) > aware(as_of):
            raise ValueError("Sizing history exceeds run knowledge cutoff")
    if job["decision_effective_at"] is not None and aware(job["decision_effective_at"]) > aware(
        as_of
    ):
        raise ValueError("Decision effective time exceeds run cutoff")
    previous = job["sizing_previous"]
    if previous:
        path = Path(previous["path"])
        verify_bytes(path / "manifest.json", previous["manifest_sha256"])
        load_sizing_history(path)
    if job["model_runtime"] is not None:
        from importlib.metadata import version

        if any(version(name) != value for name, value in job["model_runtime"].items()):
            raise ValueError("Frozen model runtime changed")


def load_stage(name, path):
    """Verify one allowlisted publication using its existing immutable reader.

    Parameters
    ----------
    name : str
        Supported optional-stage identity.
    path : pathlib.Path
        Exact immutable publication directory.

    Returns
    -------
    dict or tuple
        Existing reader's verified tables, records and audits.
    """
    if name == "models":
        from oddsjam_ev.io.explanatory_models import load_model_publication

        return load_model_publication(path)
    return LOADERS[name](path)


def stage_blockers(name, job, receipts):
    """Return explicit missing evidence/opt-in/dependency codes before building.

    Parameters
    ----------
    name : str
        Supported optional stage.
    job : dict
        Validated frozen request.
    receipts : dict
        Already evaluated prerequisite stage receipts.

    Returns
    -------
    list of str
        Stable blockers; an empty list permits the existing publisher to run.
    """
    reasons = [
        "blocked_dependency:" + dep
        for dep in DEPENDENCIES[name]
        if dep in receipts and receipts[dep]["state"] == "blocked"
    ]
    if name == "models" and not job["fit_models"]:
        reasons.append("model_build_not_requested")
    if name == "sheets" and job["decision_effective_at"] is None:
        reasons.append("missing_decision_effective_time")
    return reasons


def build_stage(name, paths, output_root, job, as_of, frozen_root):
    """Dispatch fixed publishers without shell commands or dashboard inference.

    Parameters
    ----------
    name : str
        Allowlisted optional stage.
    paths : dict
        Verified prerequisite publication paths.
    output_root, frozen_root : pathlib.Path
        Private artifact and frozen-input destinations.
    job : dict
        Validated frozen request.
    as_of : str
        Core run's explicit knowledge cutoff.

    Returns
    -------
    pathlib.Path
        Existing publisher's verified immutable publication.
    """
    p = paths
    cfg = job["stage_configs"]
    if name == "sizing":
        history = job["sizing_history"]
        previous = job["sizing_previous"]
        if previous:
            verify_bytes(Path(previous["path"]) / "manifest.json", previous["manifest_sha256"])
            load_sizing_history(Path(previous["path"]))
        history_path = None
        if history:
            from oddsjam_ev.pipeline.manifests import atomic_json

            history_path = frozen_root / "sizing_history.json"
            if not history_path.exists():
                atomic_json(history_path, history["content"])
            elif json.loads(history_path.read_text()) != history["content"]:
                raise ValueError("Frozen sizing input corrupted")
        return publish_sizing_history(
            p["canonical"],
            output_root,
            history_path=history_path,
            knowledge_as_of=None if history else as_of,
            previous=Path(previous["path"]) if previous else None,
        )
    if name == "sizing_diagnostics":
        return publish_sizing_diagnostics(
            p["canonical"], p["sizing"], output_root, SizingDiagnosticsConfig(**cfg[name])
        )
    if name == "within_filter_discovery":
        return publish_within_filter_discovery(
            p["canonical"],
            p["atlas"],
            p["sizing_diagnostics"],
            output_root,
            MultidimensionalConfig(**cfg[name]),
        )
    if name == "prioritization":
        return publish_prioritization(
            p["within_filter_discovery"],
            p["sizing_diagnostics"],
            output_root,
            PrioritizationConfig(**cfg[name]),
        )
    if name == "sheets":
        evidence = job["decision_evidence"]
        return publish_decisions(
            p["within_filter_discovery"],
            p["prioritization"],
            evidence["content"]
            if evidence
            else {"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []},
            output_root,
            as_of,
            job["decision_effective_at"],
        )
    if name == "visuals":
        return publish_visual_diagnostics(
            p["canonical"],
            p["atlas"],
            p["sizing_diagnostics"],
            output_root,
            VisualConfig(**cfg[name]),
        )
    if name == "queues":
        return publish_review_queues(
            p["within_filter_discovery"],
            p["prioritization"],
            p["sheets"],
            p["visuals"],
            output_root,
            ReviewConfig(**cfg[name]),
        )
    if name == "models" and job["fit_models"]:
        from oddsjam_ev.analysis.explanatory_models import ModelPolicy
        from oddsjam_ev.io.explanatory_models import publish_explanatory_models

        return publish_explanatory_models(
            p["canonical"],
            p["visuals"],
            output_root,
            ModelPolicy(**job["model_policy"]),
            job["model_scope"],
        )
    raise ValueError("Stage not allowed or model fitting not requested")


def verify_job_manifest(result):
    """Verify final receipt inventory, dependency lineage and linked artifact bytes.

    Parameters
    ----------
    result : dict
        Integrity-checked final core result containing a research-job manifest.

    Returns
    -------
    None
        Raises if any requested stage, lineage pin or artifact no longer reconciles.
    """
    manifest = result["job_manifest"]
    spec, job = result["specification"], result["specification"]["research_job"]
    from oddsjam_ev.io.source_csv import fingerprint

    if (
        manifest["contract_version"] != 1
        or manifest["snapshot_id"] != spec["snapshot_id"]
        or manifest["snapshot_manifest_sha256"] != spec["snapshot_manifest_sha256"]
        or manifest["configuration_sha256"] != fingerprint(spec["configuration"])
        or manifest["code_sha256"] != fingerprint(spec["code"])
        or set(manifest["receipts"]) != set(DEPENDENCIES)
        or job["order"] != dependency_order(job["requested_stages"])
        or job["dependencies"] != {s: list(DEPENDENCIES[s]) for s in job["order"]}
    ):
        raise ValueError("Job manifest scope/inventory mismatch")
    receipts = manifest["receipts"]
    previous = {}
    for name in DEPENDENCIES:
        receipt = receipts[name]
        if name not in job["order"]:
            if receipt != {"state": "not_requested"} or name in result["outputs"]:
                raise ValueError("Unrequested stage output")
            continue
        blockers = stage_blockers(name, job, previous)
        if blockers:
            if receipt != {"state": "blocked", "reasons": blockers} or name in result["outputs"]:
                raise ValueError("Blocked stage receipt mismatch")
        else:
            output = result["outputs"][name]
            expected = {
                dep: result["outputs"][dep]["manifest_sha256"] for dep in DEPENDENCIES[name]
            }
            if (
                receipt != {"state": "completed", "output": output}
                or output["dependencies"] != expected
                or output["run_id"] != result["run_id"]
                or output["stage"] != name
            ):
                raise ValueError("Job dependency receipt mismatch")
            path = Path(output["path"])
            verify_bytes(path / "manifest.json", output["manifest_sha256"])
            load_stage(name, path)
        previous[name] = receipt
