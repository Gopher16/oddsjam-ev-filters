"""
filter_mapping_proposal.py
===============================================================================
Purpose
-------
Prepare evidence-linked historical identity proposals without changing attribution.

What this module provides
-------------------------
Exact registry matching, conservative historical IDs and an ambiguity review queue.

Design Notes
------------
Name similarity is a review lead, never identity evidence or historical settings.
Source strings and registered IDs remain unchanged; proposals are not approvals.
===============================================================================
"""

import re
from difflib import SequenceMatcher

from oddsjam_ev.analysis.filter_attribution import resolve_labels, validate_registry
from oddsjam_ev.io.source_csv import fingerprint


def _comparison_name(name):
    return " ".join(re.sub(r"^(PROD|TEST)\s*-\s*", "", name, flags=re.I).casefold().split())


def build_mapping_proposal(inventory: dict, registry: dict) -> dict:
    """Prepare review rows from an exact-label inventory and its pinned registry.

    Parameters
    ----------
    inventory : dict
        Verified inventory payload; strings are never normalized in the output.
    registry : dict
        Registry pinned by the inventory's canonical bundle.

    Returns
    -------
    dict
        Proposals, registered identities and evidence-linked ambiguity questions.
        No mapping in this payload authorizes a registry or attribution update.
    """
    mapping = validate_registry(registry)
    registered = {row["id"]: row for row in registry["filters"]}
    labels = inventory["labels"]
    seen = set()
    for row in labels:
        raw = row["raw_label"]
        if raw is not None and not isinstance(raw, str):
            raise ValueError("Inventory labels must be strings or null")
        if row["label_id"] != fingerprint({"raw_label": raw}) or row["label_id"] in seen:
            raise ValueError("Invalid or duplicate inventory label identity")
        seen.add(row["label_id"])
        ids, status = resolve_labels(raw or "", mapping)
        if ids != row["resolved_filter_ids"] or status != row["registry_resolution_status"]:
            raise ValueError("Inventory resolution does not match pinned registry")
    # Observed whole strings can be segmentation leads, but do not become aliases.
    vocabulary = {name: (ident, "registered") for name, ident in mapping.items()}
    for row in labels:
        raw = row["raw_label"]
        if raw and raw.strip() and raw not in vocabulary and not row["resolved_filter_ids"]:
            vocabulary[raw] = ("historical_label_" + row["label_id"], "proposed")
    proposed_ids = {v[0] for v in vocabulary.values() if v[1] == "proposed"}
    if proposed_ids & set(registered):
        raise ValueError("Proposed historical identity collides with a registered ID")
    proposals, queue = [], []
    for row in sorted(labels, key=lambda r: r["label_id"]):
        raw = row["raw_label"]
        ids, resolution = resolve_labels(raw or "", mapping)
        candidates = []
        if raw and raw.strip() and not ids:
            comparable = _comparison_name(raw)
            for name, (ident, origin) in sorted(vocabulary.items()):
                if name == raw:
                    continue
                other = _comparison_name(name)
                score = SequenceMatcher(None, comparable, other, autojunk=False).ratio()
                if comparable == other or score >= 0.82:
                    candidates.append(
                        {
                            "filter_id": ident,
                            "name": name,
                            "origin": origin,
                            "relationship": "possible_rename_or_separate_filter",
                            "basis": "prefix_case_spacing_equivalent"
                            if comparable == other
                            else "name_similarity_only",
                            "similarity": round(score, 4),
                        }
                    )
        segmentation_ids, segmentation_status = [], "not_needed"
        if raw and raw.strip() and not ids:
            alternatives = {name: value[0] for name, value in vocabulary.items() if name != raw}
            segmentation_ids, segmentation_status = resolve_labels(raw, alternatives)
        if ids:
            state = "existing_mapping"
            proposed = ids
            reason = "Exact pinned name/alias or unambiguous registered membership; preserve IDs."
            confidence = "pinned_registry"
        elif not raw or not raw.strip():
            state, proposed = "unassigned", []
            reason = "Missing or whitespace-only attribution cannot establish an identity."
            confidence = "insufficient_evidence"
        else:
            state, proposed = "proposed_distinct_identity", ["historical_label_" + row["label_id"]]
            reason = "Retain this whole source string separately pending identity review."
            confidence = "observed_label_only"
        result = {
            "label_id": row["label_id"],
            "raw_label": raw,
            "status": state,
            "approval_status": "not_required_existing_mapping" if ids else "not_approved",
            "proposed_filter_ids": proposed,
            "proposed_display_name": registered[ids[0]]["name"]
            if resolution == "exact"
            else raw
            if state == "proposed_distinct_identity"
            else None,
            "proposed_former_names": [],
            "registry_resolution_status": resolution,
            "confidence": confidence,
            "reason": reason,
            "relationship_candidates": candidates,
            "segmentation_candidate_ids": segmentation_ids,
            "segmentation_status": segmentation_status,
            "historical_settings_status": "configuration_effectivity_unknown",
            "evidence": {
                "inventory_label": row,
                "import_lineage_label_id": row["label_id"],
                "timestamp_bounds_label_id": row["label_id"],
                "limitations": "Counts are non-additive; raw lexical dates do not establish a rename timeline. Current settings and PROD/TEST text do not establish historical settings or funding.",
            },
        }
        proposals.append(result)
        short_tag = bool(raw and len(raw.split()) <= 2 and not ids)
        plausible_rename = any(c["basis"] == "prefix_case_spacing_equivalent" for c in candidates)
        if plausible_rename or segmentation_ids or (raw and not raw.strip()) or short_tag:
            queue.append(
                {
                    "label_id": row["label_id"],
                    "raw_label": raw,
                    "question": "Is this a rename, separate filter/clone, multiple memberships, or a non-filter tag? Retain separately if unknown.",
                    "relationship_candidates": candidates,
                    "segmentation_candidate_ids": segmentation_ids,
                    "decision": "pending",
                    "confirmed_filter_ids": [],
                    "evidence_reference": None,
                    "recorded_at": None,
                }
            )
    return {
        "proposal_contract_version": 1,
        "summary": {
            "label_values": len(proposals),
            "existing_mapping_values": sum(r["status"] == "existing_mapping" for r in proposals),
            "proposed_distinct_values": sum(
                r["status"] == "proposed_distinct_identity" for r in proposals
            ),
            "unassigned_values": sum(r["status"] == "unassigned" for r in proposals),
            "ambiguity_queue_values": len(queue),
        },
        "registered_identities": [
            {
                "filter_id": ident,
                "display_name": row["name"],
                "former_names": row.get("former_names", []),
                "registered_clone_of": row.get("clone_of"),
                "relationship_evidence": "pinned_registry_only_not_inferred_from_similarity",
            }
            for ident, row in sorted(registered.items())
        ],
        "proposals": proposals,
        "ambiguity_queue": queue,
    }
