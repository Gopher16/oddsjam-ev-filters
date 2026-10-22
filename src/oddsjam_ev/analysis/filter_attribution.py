"""
filter_attribution.py
===============================================================================
Purpose
-------
Resolve historical label membership without multiplying authoritative bet totals.

What this module provides
-------------------------
Registry validation, exact alias resolution, and a non-economic membership bridge.

Design Notes
------------
Current settings are never applied retroactively. Commas inside names are literal.
Unknown or ambiguously segmented strings remain unresolved rather than guessed.
===============================================================================
"""

from __future__ import annotations

import json

import pandas as pd


def validate_registry(registry: dict) -> dict[str, str]:
    """Validate registry references and return exact name/alias to ID mappings.

    Parameters
    ----------
    registry : dict
        Dated historical registry with filters and stable IDs.

    Returns
    -------
    dict[str, str]
        Unambiguous exact names and aliases. Settings are not execution evidence.
    """
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported filter registry schema")
    rows = registry["filters"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("Registry IDs must be unique nonempty strings")
    mapping = {}
    parents = {}
    for row in rows:
        parent = row.get("clone_of")
        if parent:
            if parent not in ids:
                raise ValueError("Dangling registry clone reference")
            parents[row["id"]] = parent
        for name in [row["name"], *row.get("former_names", [])]:
            if (
                not isinstance(name, str)
                or not name
                or (name in mapping and mapping[name] != row["id"])
            ):
                raise ValueError("Registry names/aliases must be unambiguous nonempty strings")
            mapping[name] = row["id"]
    for ident in ids:
        visited = set()
        while ident in parents:
            if ident in visited:
                raise ValueError("Registry clone cycle")
            visited.add(ident)
            ident = parents[ident]
    return mapping


def resolve_labels(raw: str, mapping: dict[str, str]) -> tuple[list[str], str]:
    """Resolve a label string only when its segmentation is unambiguous.

    Parameters
    ----------
    raw : str
        Original source attribution.
    mapping : dict
        Validated exact names and aliases.

    Returns
    -------
    tuple[list[str], str]
        Sorted distinct IDs and resolution status. Exact names take precedence.
    """
    if not raw:
        return [], "missing"
    if raw in mapping:
        return [mapping[raw]], "exact"
    # Bound work for unexpected source strings; do not parse arbitrary comma tokens.
    if len(raw) > 4096 or sum(raw.count(separator) for separator in (", ", "; ", " | ")) > 128:
        return [], "unresolved"
    memo = {}

    def parse(start):
        if start == len(raw):
            return [()]
        if start in memo:
            return memo[start]
        solutions = set()
        for name, ident in mapping.items():
            if not raw.startswith(name, start):
                continue
            end = start + len(name)
            if end == len(raw):
                solutions.add((ident,))
            else:
                for separator in (", ", "; ", " | "):
                    if raw.startswith(separator, end):
                        for tail in parse(end + len(separator)):
                            solutions.add((ident, *tail))
            if len(solutions) > 1:
                break
        memo[start] = list(solutions)[:2]
        return memo[start]

    solutions = parse(0)
    if len(solutions) == 1:
        return sorted(set(solutions[0])), "resolved_memberships"
    return [], "unresolved"


def build_filter_attribution(lineage: pd.DataFrame, registry: dict) -> pd.DataFrame:
    """Build unique bet/filter memberships plus explicit unresolved-label rows.

    Parameters
    ----------
    lineage : pandas.DataFrame
        Current wager/leg rows with bet_id and original saved_filter_names.
    registry : dict
        Validated dated registry.

    Returns
    -------
    pandas.DataFrame
        Bridge with no monetary fields. A bridge is not an additive portfolio.
    """
    mapping = validate_registry(registry)
    rows = []
    for bet_id, group in lineage.groupby("bet_id", sort=True):
        memberships = set()
        labels = sorted(set(group.saved_filter_names.fillna("")))
        unresolved = []
        for label in labels:
            ids, status = resolve_labels(label, mapping)
            memberships.update(ids)
            if not ids:
                unresolved.append((label, status))
        for ident in sorted(memberships):
            rows.append(
                [
                    bet_id,
                    ident,
                    "resolved",
                    json.dumps(labels),
                    registry["snapshot_date"],
                    "configuration_effectivity_unknown",
                ]
            )
        for label, status in unresolved:
            rows.append(
                [
                    bet_id,
                    None,
                    status,
                    json.dumps([label]),
                    registry["snapshot_date"],
                    "configuration_effectivity_unknown",
                ]
            )
    return pd.DataFrame(
        rows,
        columns=[
            "bet_id",
            "filter_id",
            "attribution_status",
            "source_labels_json",
            "registry_snapshot_date",
            "historical_settings_status",
        ],
    )
