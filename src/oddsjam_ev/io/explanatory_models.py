"""
explanatory_models.py
===============================================================================
Purpose
-------
Publish immutable, verified offline model evidence without executable pickles.

What this module provides
-------------------------
Pinned lineage, feature inventory, bounded subset builds and fixed-model selection.

Design Notes
------------
No dashboard fitting, queue ranking changes, launches or source mutations.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn

from oddsjam_ev.analysis.explanatory_models import (
    LIMITATIONS,
    ModelPolicy,
    apply_scope,
    audit_features,
    fit_population,
    prepare_candidates,
)
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.visual_diagnostics import load_visual_diagnostics

TABLES = (
    "feature_inventory",
    "predictions",
    "utility",
    "importance",
    "calibration",
    "coverage",
    "residual_groups",
    "permutation",
    "training_features",
    "learning_curves",
    "redundancy",
    "source_periods",
    "eligibility",
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_publication(path: Path) -> dict:
    """Read verified safe tables and metadata from one exact publication.

    Parameters
    ----------
    path : pathlib.Path
        Explicit immutable directory; never resolve a latest model.

    Returns
    -------
    dict
        Verified frames plus audit and specification.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec["contract_version"] != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Model publication specification integrity failure")
    ModelPolicy(**spec["policy"])
    result = {}
    expected_files = {f"{name}.parquet" for name in TABLES} | {"audit.json", "models.json"}
    if set(manifest["output_sha256"]) != expected_files:
        raise ValueError("Model publication output inventory mismatch")
    for file, digest in manifest["output_sha256"].items():
        verify_bytes(path / file, digest)
    for name in TABLES:
        result[name] = pd.read_parquet(path / f"{name}.parquet")
    result["audit"] = json.loads((path / "audit.json").read_text())
    result["models"] = json.loads((path / "models.json").read_text())
    result["specification"] = spec
    for name in TABLES:
        if len(result[name]) != result["audit"]["table_rows"][name]:
            raise ValueError("Model publication row reconciliation failed")
    for model_id, model in result["models"].items():
        if fingerprint(model) != model_id:
            raise ValueError("Frozen model identity mismatch")
    return result


def select_published_predictions(publication: dict, bet_ids: list[str]) -> pd.DataFrame:
    """Select fixed-model rows without fitting or changing SHAP background.

    Parameters
    ----------
    publication : dict
        Verified model publication returned by the loader.
    bet_ids : list of str
        Unique canonical wagers to inspect.

    Returns
    -------
    pandas.DataFrame
        Stored predictions and explanations with unchanged model/reference IDs.
    """
    if len(set(bet_ids)) != len(bet_ids):
        raise ValueError("Duplicate subset row identities")
    rows = publication["predictions"]
    if rows.empty and not bet_ids:
        return rows.copy()
    if not set(bet_ids).issubset(set(rows.get("bet_id", []))):
        raise ValueError("Subset includes wagers without published explanations")
    return rows[rows.bet_id.isin(bet_ids)].copy()


def publish_explanatory_models(
    canonical: Path,
    visuals: Path,
    output_root: Path,
    policy: ModelPolicy,
    scope: dict | None = None,
) -> Path:
    """Build/replay one bounded all-population or explicit subset publication.

    Parameters
    ----------
    canonical, visuals : pathlib.Path
        Verified publications sharing exact canonical manifest lineage.
    output_root : pathlib.Path
        New private artifact root, outside every source directory.
    policy : ModelPolicy
        Explicit approved support/resource policy.
    scope : dict, optional
        Typed offline restrictions. A subset refit gets a separate publication.

    Returns
    -------
    pathlib.Path
        Atomic verified output. Sparse populations retain explicit decline audits.
    """
    canonical, visuals, output_root = map(Path, (canonical, visuals, output_root))
    if any(output_root.resolve().is_relative_to(p.resolve()) for p in (canonical, visuals)):
        raise ValueError("Model outputs must not modify sources")
    source = load_canonical_bundle(canonical)
    visual = load_visual_diagnostics(visuals)
    vm = json.loads((visuals / "manifest.json").read_text())
    if vm["specification"]["source_manifest_sha256"]["canonical"] != _digest(
        canonical / "manifest.json"
    ):
        raise ValueError("Model source canonical provenance mismatch")
    frame = prepare_candidates(visual["visual_candidates"])
    if set(frame.bet_id) != set(source["bet_facts"].bet_id):
        raise ValueError("Visual/canonical wager populations do not reconcile")
    if scope is not None and not isinstance(scope, dict):
        raise ValueError("Scope must be an object")
    scoped = apply_scope(frame, scope or {})
    spec = {
        "contract_version": 1,
        "policy": asdict(policy),
        "scope": scope or {},
        "source_manifest_sha256": {
            "canonical": _digest(canonical / "manifest.json"),
            "visuals": _digest(visuals / "manifest.json"),
        },
        "code_sha256": {
            "io": _digest(Path(__file__)),
            "analysis": _digest(Path(__file__).parents[1] / "analysis" / "explanatory_models.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "target": "reported profit / positive stake; equal-wager weights; unit-return fraction",
        "split": "seeded namespace/event 60/20/20; non-temporal exploratory",
        "explanation": "forest: seeded antithetic grouped permutation SHAP, two orders per background; ridge: analytic interventional SHAP; units=unit-return fractions; strict additivity",
        "policy_approval": "Owner approved exploratory thresholds in conversation, 2026-11-01; no production/forward-validation claims",
        "model_selection": "lowest validation MSE among constant, Ridge alpha=10, RF trees=64 depth=4 min_leaf=25; test never selects",
        "limitations": LIMITATIONS,
    }
    publication_id = fingerprint(spec)
    destination = output_root / publication_id
    with writer_lock(output_root):
        if destination.exists():
            load_model_publication(destination)
            return destination
        try:
            if len(frame) > policy.max_rows or len(scoped) > policy.max_rows:
                raise ValueError("Whole-publication row budget exceeded")
            population_groups = list(
                scoped.groupby(["execution_status", "product"], dropna=False, sort=True)
            )
            if len(population_groups) > policy.max_populations:
                raise ValueError("Population budget exceeded")
            results, models = [], {}
            tables = {name: [] for name in TABLES}
            tables["feature_inventory"].append(
                audit_features({**source, "model_candidates": frame})
            )
            for (funding, product), population in population_groups:
                result = fit_population(population, policy)
                context = {"funding": str(funding), "product": str(product)}
                eligibility = population[["bet_id", "event_key", "diagnostic_eligible"]].copy()
                eligibility["complete_event_identity"] = population.event_key.notna()
                eligibility["population_status"] = result["status"]
                eligibility["population_reasons_json"] = json.dumps(result["reasons"])
                if "diagnostic_exclusions_json" in population:
                    eligibility["economic_exclusions_json"] = population.diagnostic_exclusions_json
                tables["eligibility"].append(eligibility.assign(**context))
                for name, source_name in (
                    ("predictions", "records"),
                    ("utility", "utility"),
                    ("importance", "importance"),
                    ("calibration", "calibration"),
                    ("coverage", "coverage"),
                    ("residual_groups", "residual_groups"),
                    ("permutation", "permutation"),
                    ("training_features", "training_features"),
                    ("learning_curves", "learning_curves"),
                    ("redundancy", "redundancy"),
                    ("source_periods", "source_periods"),
                ):
                    if source_name in result:
                        tables[name].append(result[source_name].assign(**context))
                if "safe_model" in result:
                    models[result["model_id"]] = result["safe_model"]
                results.append(
                    {
                        **context,
                        **{
                            key: value
                            for key, value in result.items()
                            if not isinstance(value, pd.DataFrame) and key != "safe_model"
                        },
                        "source_wagers": len(population),
                        "ineligible_wagers": int(
                            (~population.diagnostic_eligible.fillna(False).eq(True)).sum()
                        ),
                    }
                )
            frames = {
                name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                for name, parts in tables.items()
            }
            audit = {
                "source_wagers": len(frame),
                "scoped_wagers": len(scoped),
                "scope_excluded_wagers": len(frame) - len(scoped),
                "populations": results,
                "scope_empty": scoped.empty,
                "table_rows": {name: len(table) for name, table in frames.items()},
                "interpretation": "No model or SHAP publication authorizes promotion, production changes or causal claims.",
            }
            with tempfile.TemporaryDirectory(dir=output_root, prefix=".staging-") as temporary:
                stage = Path(temporary) / publication_id
                stage.mkdir()
                for name, table in frames.items():
                    table.to_parquet(stage / f"{name}.parquet", index=False)
                write_json(stage / "audit.json", audit)
                write_json(stage / "models.json", models)
                output_hashes = {p.name: _digest(p) for p in stage.iterdir()}
                write_json(
                    stage / "manifest.json",
                    {
                        "publication_id": publication_id,
                        "specification": spec,
                        "output_sha256": output_hashes,
                    },
                )
                load_model_publication(stage)
                stage.rename(destination)
        except Exception as exc:
            failure = output_root / f"{publication_id}.failure.json"
            if not failure.exists():
                write_json(
                    failure,
                    {
                        "publication_id": publication_id,
                        "exception": type(exc).__name__,
                        "reason": str(exc),
                    },
                )
            raise
    return destination
