"""
attribution_audit.py
===============================================================================
Purpose
-------
Reconcile versioned attribution publications without permitting economic changes.

What this module provides
-------------------------
Verified before/after coverage and exact preservation checks for canonical tables.

Design Notes
------------
Filter memberships overlap. Candidate counts are descriptive, never additive bets.
===============================================================================
"""

import hashlib
import json
import tempfile
from pathlib import Path

from pandas.testing import assert_frame_equal

from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint


def reconcile_tables(before: dict, after: dict) -> dict:
    """Reject any non-attribution changes and summarize distinct candidate coverage.

    Parameters
    ----------
    before, after : dict
        Verified canonical table dictionaries.

    Returns
    -------
    dict
        Coverage, changed memberships and exact preservation evidence.
    """
    if set(before) != set(after) or "filter_attribution" not in before:
        raise ValueError("Canonical audit table set mismatch")
    unchanged = sorted(set(before) - {"filter_attribution"})
    for name in unchanged:
        try:
            assert_frame_equal(before[name], after[name])
        except AssertionError as exc:
            raise ValueError(f"Non-attribution table changed: {name}") from exc
    old, new = before["filter_attribution"], after["filter_attribution"]

    def population(frame, status):
        return set(frame.loc[frame.attribution_status.eq(status), "bet_id"])

    coverage = {
        status: {
            "before_distinct_candidates": len(population(old, status)),
            "after_distinct_candidates": len(population(new, status)),
            "gained_candidates": len(population(new, status) - population(old, status)),
            "lost_candidates": len(population(old, status) - population(new, status)),
        }
        for status in ("resolved", "unresolved", "missing")
    }

    def memberships(frame):
        part = frame.loc[frame.attribution_status.eq("resolved")]
        return set(zip(part.bet_id, part.filter_id, strict=True))

    old_pairs, new_pairs = memberships(old), memberships(new)
    ids = sorted({ident for _, ident in old_pairs | new_pairs})
    return {
        "unchanged_tables": unchanged,
        "coverage": coverage,
        "membership_pairs_added": len(new_pairs - old_pairs),
        "membership_pairs_removed": len(old_pairs - new_pairs),
        "per_filter_distinct_candidates": [
            {
                "filter_id": ident,
                "before": sum(i == ident for _, i in old_pairs),
                "after": sum(i == ident for _, i in new_pairs),
            }
            for ident in ids
        ],
        "limitations": "Membership/status counts overlap and are not additive. Exact non-attribution table equality includes wager identity, eligibility, source economics, product and funding evidence. Historical timing/settings remain unknown.",
    }


def publish_attribution_audit(before: Path, after: Path, output_root: Path) -> Path:
    """Publish an immutable reconciliation for matched source/economic policies.

    Parameters
    ----------
    before, after : pathlib.Path
        Explicit verified canonical publications.
    output_root : pathlib.Path
        Private audit destination outside either source bundle.

    Returns
    -------
    pathlib.Path
        Verified content-addressed audit directory.
    """
    before, after, output_root = map(Path, (before, after, output_root))
    if any(output_root.resolve().is_relative_to(p.resolve()) for p in (before, after)):
        raise ValueError("Audit output must not modify a source bundle")
    tables = [load_canonical_bundle(p) for p in (before, after)]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (before, after)]
    for key in ("ingestion_manifest", "economic_policy"):
        if manifests[0]["specification"][key] != manifests[1]["specification"][key]:
            raise ValueError(f"Attribution audit requires identical {key}")
    payload = reconcile_tables(*tables)
    spec = {
        "audit_contract_version": 1,
        "canonical_bundle_ids": [m["bundle_id"] for m in manifests],
        "canonical_manifest_sha256": [
            hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest() for p in (before, after)
        ],
        "registry_fingerprints": [
            fingerprint(m["specification"]["filter_registry"]) for m in manifests
        ],
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            existing = json.loads((destination / "manifest.json").read_text())
            if (
                existing["specification"] != spec
                or json.loads(verify_bytes(destination / "audit.json", existing["output_sha256"]))
                != payload
            ):
                raise ValueError("Attribution audit replay mismatch")
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            write_json(stage / "audit.json", payload)
            write_json(
                stage / "manifest.json",
                {
                    "audit_id": ident,
                    "specification": spec,
                    "output_sha256": hashlib.sha256(
                        (stage / "audit.json").read_bytes()
                    ).hexdigest(),
                },
            )
            stage.rename(destination)
    return destination
