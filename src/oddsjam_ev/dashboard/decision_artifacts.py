"""Verified workspace attachments, confined to the selected report and trusted root."""

import json
from pathlib import Path

from oddsjam_ev.dashboard.artifacts import _digest, _require, within
from oddsjam_ev.dashboard.evidence import fingerprint_manifest
from oddsjam_ev.io.manifests import verify_bytes
from oddsjam_ev.io.review_queues import load_review_queues
from oddsjam_ev.io.visual_diagnostics import load_visual_diagnostics


def load_workspace(run, cfg, evidence, decisions, cached=None):
    """Load explicitly linked queues/charts, rehashing every byte before cached display.

    Parameters
    ----------
    run, cfg : PublishedRun, DashboardConfig
        Selected verified research and trusted publication links.
    evidence, decisions : dict or None
        Matching verified inference and configuration evidence.
    cached : dict or None
        Session-local cache, reused only after its content hashes are verified.

    Returns
    -------
    dict or None
        Complete workspace or explicit unavailable state; no fallback research.
    """
    links = cfg.workspace_publications.get(run.result["run_id"])
    if links is None:
        return None
    _require(evidence is not None and decisions is not None)
    root = Path(cfg.evidence_root).resolve()
    paths = {k: within(root, root / v) for k, v in links.items()}
    key = (
        str(root),
        run.result["run_id"],
        tuple(sorted((k, str(p)) for k, p in paths.items())),
        fingerprint_manifest(decisions["manifest"]),
        tuple(
            fingerprint_manifest(evidence["manifests"][s]) for s in ("discovery", "prioritization")
        ),
    )
    if cached is not None and cached["cache_key"] == key:
        for path, digest in cached["expected_files"].items():
            verify_bytes(within(root, path), digest)
        return cached
    manifests, expected = {}, {}
    for name, path in paths.items():
        mp = within(root, path / "manifest.json")
        manifests[name] = json.loads(mp.read_text())
        expected[mp] = _digest(mp)
        for file, digest in manifests[name]["output_sha256"].items():
            _require(Path(file).name == file)
            expected[within(root, path / file)] = digest
    q, v = (manifests[k]["specification"] for k in ("queues", "visuals"))
    for stage in ("canonical", "atlas"):
        _require(
            v["source_manifest_sha256"][stage] == run.result["outputs"][stage]["manifest_sha256"]
        )
    _require(v["canonical_bundle_id"] == run.manifests["canonical"]["bundle_id"])
    _require(v["atlas_id"] == run.manifests["atlas"]["atlas_id"])
    _require(q["input_publication_ids"]["visuals"] == manifests["visuals"]["publication_id"])
    _require(q["input_manifest_sha256"]["visuals"] == expected[paths["visuals"] / "manifest.json"])
    for stage in ("discovery", "prioritization", "decisions"):
        manifest = decisions["manifest"] if stage == "decisions" else evidence["manifests"][stage]
        _require(q["input_publication_ids"][stage] == manifest["publication_id"])
        _require(q["input_manifest_sha256"][stage] == fingerprint_manifest(manifest))
    tables = load_visual_diagnostics(paths["visuals"])
    queues = load_review_queues(paths["queues"])
    return {
        "queues": queues,
        "visuals": tables,
        "manifests": manifests,
        "expected_files": expected,
        "cache_key": key,
    }
