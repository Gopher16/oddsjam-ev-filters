"""
canonical.py
===============================================================================
Purpose
-------
Publish reproducible canonical facts from one explicitly pinned ingestion snapshot.

What this module provides
-------------------------
Verified input loading, canonical transformations, atomic bundles, and readback.

Design Notes
------------
No acquisition dependency, ingestion mutation, default-snapshot updates, or notebook
execution occurs. Bundles retain source manifest, policy, code, and output hashes.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.bet_facts import build_bet_facts
from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.feature_catalog import build_feature_catalog
from oddsjam_ev.analysis.filter_attribution import build_filter_attribution, validate_registry
from oddsjam_ev.analysis.opportunity_facts import build_opportunity_facts
from oddsjam_ev.analysis.payouts import qualify_economics, summarize_economics
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.snapshots import load_snapshot
from oddsjam_ev.io.source_csv import fingerprint

TABLE_NAMES = (
    "bet_facts",
    "bet_legs",
    "source_audit",
    "source_features",
    "opportunity_facts",
    "bet_opportunity_links",
    "filter_attribution",
    "feature_catalog",
    "economic_summary",
)


def _code_hashes() -> dict[str, str]:
    analysis = Path(__file__).resolve().parents[1] / "analysis"
    names = [
        "canonical_config",
        "bet_facts",
        "payouts",
        "filter_attribution",
        "opportunity_facts",
        "feature_catalog",
    ]
    files = [analysis / f"{name}.py" for name in names] + [
        Path(__file__),
        Path(__file__).with_name("source_csv.py"),
        Path(__file__).resolve().parents[1] / "metrics/odds.py",
    ]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def build_canonical_bundle(
    snapshot: Path, output_root: Path, cfg: CanonicalConfig, registry: dict
) -> Path:
    """Build deterministic private facts without altering source or legacy outputs.

    Parameters
    ----------
    snapshot : pathlib.Path
        Explicit published ingestion snapshot directory; no latest lookup.
    output_root : pathlib.Path
        Private destination, normally data/processed/canonical.
    cfg : CanonicalConfig
        Versioned economic and product policy.
    registry : dict
        Dated historical attribution registry, pinned without retroactive settings.

    Returns
    -------
    pathlib.Path
        Immutable bundle containing Parquet tables and manifest.json.
    """
    current = load_snapshot(snapshot)
    source_manifest = json.loads((snapshot / "manifest.json").read_text())
    observations = pd.read_parquet(snapshot / "observations.parquet")
    selected = observations[observations._ingest_is_current.eq(True)].reset_index(drop=True)
    try:
        assert_frame_equal(current, selected)
    except AssertionError:
        raise ValueError(
            "Ingestion current rows do not reconcile to the observation audit"
        ) from None
    validate_registry(registry)
    built = build_bet_facts(observations, cfg, registry)
    bets = qualify_economics(built["bet_facts"], cfg)
    opportunities, links = build_opportunity_facts(built["lineage"])
    tables = {
        "bet_facts": bets,
        "bet_legs": built["bet_legs"],
        "source_audit": built["source_audit"],
        "source_features": current.merge(
            built["lineage"][["_ingest_observation_id", "bet_id"]],
            on="_ingest_observation_id",
            validate="one_to_one",
            sort=False,
        ),
        "opportunity_facts": opportunities,
        "bet_opportunity_links": links,
        "filter_attribution": build_filter_attribution(built["lineage"], registry),
        "feature_catalog": build_feature_catalog(current),
        "economic_summary": summarize_economics(bets),
    }
    for table in tables.values():
        # Parquet represents missing object values as None on readback. Normalize
        # that logical representation before hashing/publishing/replay comparison.
        for column in table.select_dtypes(include="object"):
            table[column] = table[column].where(table[column].notna(), None)
        table["ingestion_snapshot_id"] = source_manifest["snapshot_id"]
        table["canonical_contract_version"] = 1
    if len(links) != len(current) or len(built["source_audit"]) != len(observations):
        raise ValueError("Canonical source reconciliation failed")
    if bets.bet_id.duplicated().any() or links.source_observation_id.duplicated().any():
        raise ValueError("Canonical identities are not unique")
    specification = {
        "canonical_contract_version": 1,
        "ingestion_manifest": source_manifest,
        "economic_policy": asdict(cfg),
        "filter_registry": registry,
        "code_sha256": _code_hashes(),
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
    }
    bundle_id = fingerprint(specification)
    counts = {name: len(table) for name, table in tables.items()}
    summary = {
        "table_rows": counts,
        "source_dispositions": {
            str(k): int(v)
            for k, v in built["source_audit"].canonical_disposition.value_counts().items()
        },
        "reported_roi_eligible_bets": int(bets.reported_roi_eligible.sum()),
        "fee_verified_net_roi_eligible_bets": 0,
        "candidate_products": {str(k): int(v) for k, v in bets["product"].value_counts().items()},
        "ticket_candidates_with_unresolved_members": int(
            bets.unresolved_leg_observations.gt(0).sum()
        ),
        "unknown_timezone_bets": int(bets.created_at_utc.isna().sum()),
        "exclusion_combinations": {
            str(k): int(v) for k, v in bets.economic_exclusions_json.value_counts().items()
        },
        "currency": "USD",
        "fee_inclusion": "unverified",
        "unique_opportunities_verified": False,
    }
    with writer_lock(output_root):
        destination = output_root / bundle_id
        if destination.exists():
            existing = load_canonical_bundle(destination)
            for name in TABLE_NAMES:
                assert_frame_equal(tables[name], existing[name])
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            hashes = {}
            for name, frame in tables.items():
                path = stage / f"{name}.parquet"
                frame.to_parquet(path, index=False)
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(
                stage / "manifest.json",
                {
                    "bundle_id": bundle_id,
                    "specification": specification,
                    "summary": summary,
                    "output_sha256": hashes,
                },
            )
            stage.rename(destination)
    return destination


def load_canonical_bundle(path: Path) -> dict[str, pd.DataFrame]:
    """Verify and load all tables from an explicitly selected canonical bundle.

    Parameters
    ----------
    path : pathlib.Path
        Published bundle directory.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Verified tables; no implicit migration into the legacy notebook loader.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["bundle_id"]:
        raise ValueError("Canonical specification integrity failure")
    result = {}
    for name in TABLE_NAMES:
        file = f"{name}.parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = pd.read_parquet(path / file)
        if len(result[name]) != manifest["summary"]["table_rows"][name]:
            raise ValueError("Canonical row count integrity failure")
    return result
