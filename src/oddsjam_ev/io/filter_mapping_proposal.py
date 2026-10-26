"""
filter_mapping_proposal.py
===============================================================================
Purpose
-------
Publish private mapping proposals tied to verified inventory and registry evidence.

What this module provides
-------------------------
Immutable review artifacts with source hashes, verified replay and review Markdown.

Design Notes
------------
No registry, canonical attribution, funding policy or published history is modified.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from pathlib import Path

from oddsjam_ev.analysis.filter_mapping_proposal import build_mapping_proposal
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.label_inventory import load_label_inventory
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(payload):
    lines = [
        "# Historical identity mapping proposal",
        "",
        "Proposal only: no new aliases or memberships are approved or applied. Existing registered IDs are preserved. Similar names do not prove a rename or clone. Historical settings remain unknown.",
        "",
        "Counts refer to distinct linked canonical wager candidates per exact source string and are non-additive. Raw timestamp bounds are lexical, not verified chronology. See the pinned inventory for import/timezone lineage.",
        "",
        "## Proposed treatment",
        "",
    ]
    for row in payload["proposals"]:
        lines += [
            "### " + row["label_id"],
            "",
            "Exact source string: " + json.dumps(row["raw_label"], ensure_ascii=True),
            "",
            f"- Status: {row['status']}; confidence: {row['confidence']}",
            f"- Candidate IDs: {json.dumps(row['proposed_filter_ids'])}",
            f"- Linked wager candidates: {row['evidence']['inventory_label']['linked_wager_candidate_count']}",
            f"- Reason: {row['reason']}",
        ]
        for candidate in row["relationship_candidates"]:
            lines.append(
                "- Review lead (not an alias): " + json.dumps(candidate, ensure_ascii=True)
            )
        if row["segmentation_candidate_ids"]:
            lines.append(
                "- Possible memberships (unapproved): "
                + json.dumps(row["segmentation_candidate_ids"])
            )
        lines.append("")
    lines += ["## Ambiguity queue", ""]
    lines += [
        f"- {json.dumps(r['raw_label'], ensure_ascii=True)} ({r['label_id']}): {r['question']}"
        for r in payload["ambiguity_queue"]
    ]
    lines += [
        "",
        "Record decisions separately with exact source labels, confirmed IDs, supporting evidence and recorded-at time. Do not edit this immutable publication. Unconfirmed cases remain unresolved; current screenshots establish current names/settings only. Generic/manual tags and deleted filters require no invented historical parameters.",
    ]
    return "\n".join(lines) + "\n"


def load_mapping_proposal(path: Path) -> dict:
    """Verify a published proposal before reading review rows.

    Parameters
    ----------
    path : pathlib.Path
        Explicit proposal directory.

    Returns
    -------
    dict
        Verified proposal payload; not an approved mapping.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec.get("proposal_contract_version") != 1 or fingerprint(spec) != manifest["proposal_id"]:
        raise ValueError("Mapping proposal specification integrity failure")
    payload = json.loads(
        verify_bytes(path / "proposal.json", manifest["output_sha256"]["proposal.json"])
    )
    verify_bytes(path / "review.md", manifest["output_sha256"]["review.md"])
    if payload["summary"] != manifest["summary"]:
        raise ValueError("Mapping proposal summary mismatch")
    return payload


def publish_mapping_proposal(inventory: Path, canonical: Path, output_root: Path) -> Path:
    """Publish a proposal using the inventory's exact pinned canonical registry.

    Parameters
    ----------
    inventory, canonical : pathlib.Path
        Explicit verified inventory and its original canonical bundle.
    output_root : pathlib.Path
        Private destination outside source bundles.

    Returns
    -------
    pathlib.Path
        Content-addressed immutable proposal directory.
    """
    inventory, canonical, output_root = map(Path, (inventory, canonical, output_root))
    for source in (inventory, canonical):
        if output_root.resolve().is_relative_to(source.resolve()):
            raise ValueError("Proposal output must not modify a source bundle")
    data = load_label_inventory(inventory)
    im = json.loads((inventory / "manifest.json").read_text())
    load_canonical_bundle(canonical)
    cm = json.loads((canonical / "manifest.json").read_text())
    if (
        cm["bundle_id"] != im["specification"]["canonical_bundle_id"]
        or _digest(canonical / "manifest.json")
        != im["specification"]["source_manifest_sha256"]["canonical"]
        or fingerprint(cm["specification"]["filter_registry"])
        != im["specification"]["registry_sha256"]
    ):
        raise ValueError("Canonical registry does not match inventory provenance")
    payload = build_mapping_proposal(data, cm["specification"]["filter_registry"])
    files = [
        Path(__file__),
        Path(__file__).parents[1] / "analysis" / "filter_mapping_proposal.py",
        Path(__file__).parents[1] / "analysis" / "filter_attribution.py",
    ]
    spec = {
        "proposal_contract_version": 1,
        "inventory_id": im["inventory_id"],
        "snapshot_id": im["specification"]["snapshot_id"],
        "inventory_manifest_sha256": _digest(inventory / "manifest.json"),
        "inventory_payload_sha256": im["output_sha256"]["inventory.json"],
        "canonical_bundle_id": cm["bundle_id"],
        "canonical_manifest_sha256": _digest(canonical / "manifest.json"),
        "registry_sha256": im["specification"]["registry_sha256"],
        "similarity_policy": "strip_PROD_TEST_prefix_casefold_collapse_whitespace_then_SequenceMatcher_0.82_review_only",
        "short_tag_policy": "unmatched_labels_with_at_most_two_whitespace_tokens_require_tag_review",
        "python_version": platform.python_version(),
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
    }
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            if load_mapping_proposal(destination) != payload:
                raise ValueError("Mapping proposal replay mismatch")
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as temporary:
            stage = Path(temporary)
            write_json(stage / "proposal.json", payload)
            (stage / "review.md").write_text(_report(payload), encoding="utf-8")
            write_json(
                stage / "manifest.json",
                {
                    "proposal_id": ident,
                    "specification": spec,
                    "summary": payload["summary"],
                    "output_sha256": {
                        name: _digest(stage / name) for name in ("proposal.json", "review.md")
                    },
                },
            )
            load_mapping_proposal(stage)
            stage.rename(destination)
        return destination
