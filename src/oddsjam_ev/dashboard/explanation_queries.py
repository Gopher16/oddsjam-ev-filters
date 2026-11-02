"""
explanation_queries.py
===============================================================================
Purpose
-------
Select frozen explanation rows and summarize within a single model/reference.

What this module provides
-------------------------
Exact row identity, Decimal threshold selection, bounded chart data and offline requests.

Design Notes
------------
No fitting, new inference, economic weighting, memberships duplication or causal effects.
===============================================================================
"""

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

import pandas as pd

from oddsjam_ev.dashboard.explanation_artifacts import FEATURES

CONTEXT = ["funding", "product", "model_id", "background_id"]


def _number(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _match(frame, field, choice):
    if field in ("sportsbook", "sport", "league", "market_family"):
        return frame[field].fillna("<missing>").astype(str).eq(choice)
    if field in ("odds_band", "advertised_ev_band"):
        match = re.fullmatch(r"\[\s*([^,]+),\s*([^\)]+)\)", str(choice))
        if not match:
            raise ValueError("Explanation selectors require explicit half-open numeric bands")
        lo, hi = map(Decimal, match.groups())
        if lo.is_nan() or hi.is_nan() or lo >= hi:
            raise ValueError("Invalid explanation band")
        values = frame["odds" if field == "odds_band" else "advertised_ev_percent_exact"].map(
            _number
        )
        return values.map(
            lambda v: v is not None and lo <= v < hi and (field != "odds_band" or abs(v) >= 100)
        )
    if field == "advertised_ev_percent":
        match = re.fullmatch(r"\s*(>=|>|<=|<)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*", str(choice))
        if not match:
            raise ValueError("Unsupported exact EV comparison")
        op, endpoint = match[1], Decimal(match[2])
        return frame.advertised_ev_percent_exact.map(_number).map(
            lambda v: False
            if v is None
            else {">": v > endpoint, ">=": v >= endpoint, "<": v < endpoint, "<=": v <= endpoint}[
                op
            ]
        )
    raise ValueError(f"Unsupported explanation selector: {field}")


def select_candidates(
    candidates, attribution, *, mode="all", product="all", saved="all", selections=None, idea=None
):
    """Intersect sidebar and optional idea predicates on unique source wager IDs.

    Parameters
    ----------
    candidates, attribution : pandas.DataFrame
        Verified unique visual candidates and potentially overlapping saved memberships.
    mode, product, saved : str
        Shared sidebar selections; all retains unknown/unattributed source rows.
    selections : dict, optional
        OR within a field and AND across fields, including exact EV thresholds.
    idea : dict, optional
        A published candidate drill-through, additionally intersected with selectors.

    Returns
    -------
    pandas.DataFrame
        Unique matching source rows; memberships never duplicate examples.
    """
    if candidates.bet_id.duplicated().any():
        raise ValueError("Explanation source rows must be unique wagers")
    keep = pd.Series(True, index=candidates.index)
    for field, value in (("execution_status", mode), ("product", product)):
        if value != "all":
            keep &= candidates[field].eq(value)
    if saved != "all":
        ids = attribution.loc[
            attribution.filter_id.eq(saved) & attribution.attribution_status.eq("resolved"),
            "bet_id",
        ]
        keep &= candidates.bet_id.isin(set(ids))
    for field, choices in (selections or {}).items():
        if choices:
            masks = [_match(candidates, field, value) for value in choices]
            keep &= pd.concat(masks, axis=1).any(axis=1)
    if idea is not None:
        keep &= candidates.execution_status.eq(idea["execution_status"]) & candidates["product"].eq(
            idea["product"]
        )
        if idea["source_kind"] != "global_descriptive_catalogue":
            ids = attribution.loc[
                attribution.filter_id.eq(idea["filter_id"])
                & attribution.attribution_status.eq("resolved"),
                "bet_id",
            ]
            keep &= candidates.bet_id.isin(set(ids))
        for term in json.loads(idea["predicate_json"]):
            if term["operator"] == "==":
                choice = term["value"]
            elif term["dimension"] == "advertised_ev_percent":
                choice = term["operator"] + " " + str(term["endpoint"])
            else:
                raise ValueError("Unsupported idea explanation predicate")
            keep &= _match(candidates, term["dimension"], choice)
    return candidates.loc[keep].copy()


def explanation_rows(publication, candidates, split="all"):
    """Select stored predictions by identity without changing any model/background.

    Parameters
    ----------
    publication : dict
        Verified frozen model tables.
    candidates : pandas.DataFrame
        Unique selected source rows.
    split : str
        all/train/validation/test; never a new split or refit.

    Returns
    -------
    pandas.DataFrame
        Original predictions and contributions with reference identities intact.
    """
    if split not in ("all", "train", "validation", "test"):
        raise ValueError("Unsupported published split")
    rows = publication["predictions"]
    if rows.empty:
        return rows.copy()
    result = rows.loc[rows.bet_id.isin(set(candidates.bet_id))].copy()
    return result if split == "all" else result.loc[result.split.eq(split)].copy()


def importance(selected, reference):
    """Compare selected/reference mean contributions separately for each model.

    Parameters
    ----------
    selected, reference : pandas.DataFrame
        Stored rows using the same model/background, never pooled across populations.

    Returns
    -------
    pandas.DataFrame
        Signed/magnitude summaries, identity and sample counts.
    """
    rows = []
    if selected.empty:
        return pd.DataFrame()
    for key, cohort in selected.groupby(CONTEXT, sort=True):
        parent = reference
        for field, value in zip(CONTEXT, key, strict=True):
            parent = parent.loc[parent[field].eq(value)]
        if parent.empty:
            raise ValueError("No matching frozen reference")
        for label, group in (("Selected cohort", cohort), ("Published reference", parent)):
            for feature, title in FEATURES.items():
                column = group["shap__" + feature]
                rows.append(
                    {
                        **dict(zip(CONTEXT, key, strict=True)),
                        "population": key[0].replace("_", " ") + " · " + key[1],
                        "feature": title,
                        "feature_key": feature,
                        "cohort": label,
                        "mean_absolute": float(column.abs().mean()),
                        "mean_signed": float(column.mean()),
                        "wagers": len(group),
                    }
                )
    return pd.DataFrame(rows)


def chart_sample(rows, limit=1200):
    """Take a stable display-only sample, keeping every model represented.

    Parameters
    ----------
    rows : pandas.DataFrame
        Full selected frozen rows; summaries always use all of them.
    limit : int
        Maximum displayed wagers, not a model-fit or inferential sample.

    Returns
    -------
    pandas.DataFrame
        Deterministic, bounded row sample with unchanged contribution values.
    """
    if rows.empty:
        return rows.copy()
    groups = list(rows.groupby(CONTEXT, sort=True))
    if limit < len(groups):
        raise ValueError("Display budget cannot represent every model")
    parts = []
    for i, (_, group) in enumerate(groups):
        count = limit // len(groups) + int(i < limit % len(groups))
        hashes = group.bet_id.map(lambda v: hashlib.sha256(str(v).encode()).hexdigest())
        parts.append(
            group.assign(_display_hash=hashes)
            .sort_values("_display_hash")
            .head(count)
            .drop(columns="_display_hash")
        )
    return pd.concat(parts, ignore_index=True)


def distribution(rows):
    """Expand only bounded chart rows into labeled original-feature contributions.

    Parameters
    ----------
    rows : pandas.DataFrame
        Display-sampled stored predictions.

    Returns
    -------
    pandas.DataFrame
        Six original-feature contributions per row; no one-hot feature inflation.
    """
    parts = []
    for feature, label in FEATURES.items():
        part = rows[["bet_id", *CONTEXT, "split"]].copy()
        part["population"] = part.funding.str.replace("_", " ") + " · " + part["product"]
        part["feature"] = label
        part["feature_value"] = rows[feature].astype(str)
        part["contribution"] = rows["shap__" + feature]
        part["trajectory_sd"] = rows["shap_sampling_sd__" + feature]
        part["jitter"] = rows.bet_id.map(
            lambda v, f=feature: int(hashlib.sha256((str(v) + f).encode()).hexdigest()[:8], 16)
            / 0xFFFFFFFF
            - 0.5
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def waterfall(row):
    """Build a local additive path, preserving reference and full contribution sum.

    Parameters
    ----------
    row : pandas.Series
        One published prediction and its grouped contributions.

    Returns
    -------
    pandas.DataFrame
        Reference, six changes and final prediction on one unit-return scale.
    """
    current = float(row.base_return)
    records = [
        {
            "step": "Reference",
            "start": 0.0,
            "end": current,
            "change": current,
            "kind": "Reference",
            "value": "Training background",
        }
    ]
    for feature in sorted(FEATURES, key=lambda f: -abs(row["shap__" + f])):
        change = float(row["shap__" + feature])
        records.append(
            {
                "step": FEATURES[feature],
                "start": current,
                "end": current + change,
                "change": change,
                "kind": "Raises prediction" if change >= 0 else "Lowers prediction",
                "value": str(row[feature]),
            }
        )
        current += change
    records.append(
        {
            "step": "Prediction",
            "start": 0.0,
            "end": float(row.predicted_return),
            "change": float(row.predicted_return),
            "kind": "Prediction",
            "value": "Frozen model output",
        }
    )
    return pd.DataFrame(records).assign(order=range(len(records)))


def offline_scope(selections, mode="all", product="all", saved="all", idea=None, base_scope=None):
    """Export only scopes exactly expressible by the existing offline model builder.

    Parameters
    ----------
    selections : dict
        Shared signal controls.
    mode, product, saved : str
        Source-context selections.
    idea : dict, optional
        Candidate drill-through; requires independent baseline review for refits.
    base_scope : dict, optional
        Current separately fitted scope; requests must not silently broaden it.

    Returns
    -------
    tuple[dict, list[str]]
        Typed scope and blockers; blocked scopes must not be submitted as broader runs.
    """
    scope, blockers = {}, []
    if saved != "all" or idea is not None:
        blockers.append(
            "Saved-filter and candidate membership scopes require a reviewed offline specification; no broader refit request is substituted."
        )
    for field, value in (("execution_status", mode), ("product", product)):
        if value != "all":
            scope[field] = [value]
    for field, choices in selections.items():
        if not choices:
            continue
        if field in ("sportsbook", "sport", "league", "market_family"):
            scope[field] = choices
        elif field == "advertised_ev_band" and len(choices) == 1:
            match = re.fullmatch(r"\[\s*([^,]+),\s*([^\)]+)\)", choices[0])
            if not match:
                blockers.append("EV request requires an explicit numeric band.")
                continue
            lo, hi = map(Decimal, match.groups())
            bounds = {}
            if lo.is_finite():
                bounds["ge"] = float(lo / 100)
            if hi.is_finite():
                bounds["lt"] = float(hi / 100)
            scope.setdefault("advertised_ev_fraction", {}).update(bounds)
        elif field == "advertised_ev_percent" and len(choices) == 1:
            match = re.fullmatch(r"\s*(>=|>|<=|<)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*", choices[0])
            if not match:
                blockers.append("Exact EV request requires a supported numeric comparison.")
                continue
            op = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}[match[1]]
            scope.setdefault("advertised_ev_fraction", {})[op] = float(Decimal(match[2]) / 100)
        else:
            blockers.append(
                f"{field}: this selection cannot be exported exactly by the current typed builder; use a reviewed offline request."
            )
    for field, values in (base_scope or {}).items():
        if field not in scope:
            scope[field] = values
        elif isinstance(values, list):
            scope[field] = sorted(set(scope[field]) & set(values))
            if not scope[field]:
                blockers.append(f"{field}: no overlap with the separately fitted model scope.")
        else:
            for operator, endpoint in values.items():
                current = scope[field].get(operator)
                scope[field][operator] = (
                    endpoint
                    if current is None
                    else (
                        max(current, endpoint)
                        if operator in {"gt", "ge"}
                        else min(current, endpoint)
                    )
                )
    return scope, blockers
