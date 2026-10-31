"""
review_catalogue.py
===============================================================================
Purpose
-------
Enumerate global logged-history slices before looking at economic outcomes.

What this module provides
-------------------------
Bounded full catalogue, population-specific parents and explicit missing-feature audit.

Design Notes
------------
Global ideas are descriptive and separate from the previously corrected family.
===============================================================================
"""

import json

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.visual_features import MISSING, band
from oddsjam_ev.io.source_csv import fingerprint


class ReviewBudgetExceeded(ValueError):
    """A whole review publication must abort rather than truncate evidence."""


def enumerate_global(data, cfg):
    """Freeze global candidates from features only, preserving population parents.

    Parameters
    ----------
    data : pandas.DataFrame
        Unique verified visual candidates, including continuous source EV.
    cfg : ReviewConfig
        Versioned views, cutpoints and complete-catalogue budgets.

    Returns
    -------
    tuple[list, dict, list]
        Candidate records, scope membership sets and excluded-category coverage.
    """
    features = data[
        [
            "bet_id",
            "execution_status",
            "product",
            "sportsbook",
            "sport",
            "league",
            "market_family",
            "advertised_ev_percent_exact",
            "odds",
        ]
    ].copy()
    features["advertised_ev_band"] = features.advertised_ev_percent_exact.map(
        lambda v: band(v, cfg.ev_edges)
    )
    odds = pd.to_numeric(features.odds, errors="coerce")
    features["odds_band"] = odds.where(np.isfinite(odds) & odds.abs().ge(100)).map(
        lambda v: band(v, cfg.odds_edges)
    )
    candidates, members, coverage = [], {}, []
    usage = 0
    for (funding, product), parent in features.groupby(
        ["execution_status", "product"], sort=True, dropna=False
    ):
        scope = {"execution_status": funding, "product": product}
        parent_id = "global_parent:" + fingerprint(scope)
        members[parent_id] = set(parent.bet_id)
        usage += len(parent)
        for view in cfg.global_views:
            for key, group in parent.groupby(list(view), sort=True, dropna=False):
                values = key if isinstance(key, tuple) else (key,)
                if any(
                    pd.isna(v) or str(v).strip() in MISSING | {"<unavailable>", "<unmapped>"}
                    for v in values
                ):
                    coverage.append(
                        {
                            **scope,
                            "view": list(view),
                            "candidate_count": len(group),
                            "reason": "unavailable_categories_retained_in_parent_and_complement",
                        }
                    )
                    continue
                predicate = [
                    {"dimension": k, "operator": "==", "value": str(v)}
                    for k, v in zip(view, values, strict=True)
                ]
                ident = "global_candidate:" + fingerprint({**scope, "predicate": predicate})
                candidates.append(
                    {
                        "candidate_id": ident,
                        "parent_id": parent_id,
                        "filter_id": "prospective_global:"
                        + fingerprint({**scope, "predicate": predicate}),
                        **scope,
                        "predicate_json": json.dumps(predicate, sort_keys=True),
                        "source_kind": "global_descriptive_catalogue",
                    }
                )
                members[ident] = set(group.bet_id)
                usage += len(group)
                if len(candidates) > cfg.max_candidates or usage > cfg.max_memberships:
                    raise ReviewBudgetExceeded(
                        "Global catalogue budget exceeded; no partial publication"
                    )
    return candidates, members, coverage
