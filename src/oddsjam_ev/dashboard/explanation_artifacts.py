"""
explanation_artifacts.py
===============================================================================
Purpose
-------
Verify explicit frozen model attachments against the selected report and visuals.

What this module provides
-------------------------
Confined paths, lineage checks and byte-verified session cache reuse.

Design Notes
------------
No model discovery, fitting, prediction or automatic subset-model selection.
===============================================================================
"""

import json
from pathlib import Path

import numpy as np

from oddsjam_ev.dashboard.artifacts import _digest, _require, within
from oddsjam_ev.dashboard.evidence import fingerprint_manifest
from oddsjam_ev.io.explanatory_models import load_model_publication
from oddsjam_ev.io.manifests import verify_bytes

FEATURES = {
    "sportsbook": "Platform",
    "sport": "Sport",
    "league": "League",
    "market_family": "Market",
    "advertised_ev_fraction": "Advertised EV",
    "offered_decimal_odds": "Offered decimal odds",
}


def _validate_rows(publication, candidates):
    rows = publication["predictions"]
    eligibility = publication["eligibility"]
    if eligibility.empty:
        _require(rows.empty and publication["audit"]["scoped_wagers"] == 0)
        return
    _require(not eligibility.bet_id.duplicated().any())
    _require(set(eligibility.bet_id).issubset(set(candidates.bet_id)))
    if rows.empty:
        return
    _require(not rows.bet_id.duplicated().any())
    _require(set(rows.bet_id).issubset(set(eligibility.bet_id)))
    _require(set(rows.model_id).issubset(set(publication["models"])))
    _require(set(rows.split) <= {"train", "validation", "test"})
    contributions = rows[["shap__" + f for f in FEATURES]]
    _require(np.isfinite(contributions.to_numpy()).all())
    _require(np.isfinite(rows[["predicted_return", "observed_return", "base_return"]]).all().all())
    _require(
        np.allclose(
            rows.base_return + contributions.sum(axis=1),
            rows.predicted_return,
            atol=1e-6,
            rtol=1e-6,
        )
    )
    _require(rows.groupby(["funding", "product", "event_key"]).split.nunique().le(1).all())
    for _, part in rows.groupby(["funding", "product"]):
        _require(part.model_id.nunique() == 1 and part.background_id.nunique() == 1)
        _require(set(part.loc[part.background_member, "split"]) <= {"train"})
    indexed = candidates.set_index("bet_id").loc[rows.bet_id]
    _require(indexed.execution_status.tolist() == rows.funding.tolist())
    _require(indexed["product"].tolist() == rows["product"].tolist())
    _require(indexed.diagnostic_eligible.eq(True).all())
    _require(np.allclose(indexed.unit_return, rows.observed_return))


def load_explanations(run, cfg, workspace, cached=None):
    """Load all explicitly attached model publications, preserving their identities.

    Parameters
    ----------
    run, cfg : PublishedRun, DashboardConfig
        Verified report and trusted local attachment configuration.
    workspace : dict or None
        Verified visual sources matching the selected report.
    cached : dict or None
        Previous session bundle, reused only after hashing every source/output byte.

    Returns
    -------
    dict or None
        Global/subset publications and exact identities, or explicit unavailability.
    """
    links = cfg.explanation_publications.get(run.result["run_id"])
    if links is None:
        return None
    _require(workspace is not None)
    root = Path(cfg.evidence_root).resolve()
    paths = [within(root, root / path) for path in [links["global"], *links["subsets"]]]
    _require(len(set(paths)) == len(paths))
    sources = {
        "canonical": run.result["outputs"]["canonical"]["manifest_sha256"],
        "visuals": fingerprint_manifest(workspace["manifests"]["visuals"]),
    }
    key = (
        run.result["run_id"],
        str(root),
        tuple(str(p) for p in paths),
        tuple(sorted(sources.items())),
    )
    if cached is not None and cached["cache_key"] == key:
        for path, digest in cached["expected_files"].items():
            verify_bytes(within(root, path), digest)
        return cached
    publications, expected = {}, {}
    for i, path in enumerate(paths):
        mp = within(root, path / "manifest.json")
        manifest = json.loads(mp.read_text())
        expected[mp] = _digest(mp)
        for file, digest in manifest["output_sha256"].items():
            _require(Path(file).name == file)
            expected[within(root, path / file)] = digest
        publication = load_model_publication(path)
        spec = publication["specification"]
        _require(spec["source_manifest_sha256"] == sources)
        _require(bool(spec["scope"]) == (i != 0))
        _validate_rows(publication, workspace["visuals"]["visual_candidates"])
        publication["publication_id"] = manifest["publication_id"]
        publication["kind"] = "global" if i == 0 else "subset"
        publications[manifest["publication_id"]] = publication
    _require(len(publications) == len(paths))
    return {
        "publications": publications,
        "global_id": next(iter(publications)),
        "expected_files": expected,
        "cache_key": key,
    }
