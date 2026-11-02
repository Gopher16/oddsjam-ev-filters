"""
promotion.py
===============================================================================
Purpose
-------
Evaluate one predeclared prospective family for bounded human pilot review.

What this module provides
-------------------------
Arm gates, paired event-bootstrap intervals, cluster influence t tests and Holm correction.

Design Notes
------------
Passing is eligibility for explicit bounded-pilot review, not proof of executable
profit or a trading instruction. Bootstrap intervals are marginal, not simultaneous.
===============================================================================
"""

import numpy as np
from scipy.stats import t

from oddsjam_ev.io.source_csv import fingerprint

CAPS = {
    "max_stake_per_wager_usd": 1.0,
    "max_total_deployed_stake_usd": 100.0,
    "max_duration_days": 14,
    "max_active_pilots": 1,
}
GATES = {
    "min_wagers_per_arm": 100,
    "min_event_groups_per_arm": 100,
    "max_wager_stake_share": 0.05,
    "max_event_stake_share": 0.10,
    "bootstrap_resamples": 5000,
    "alpha": 0.05,
}
APPROVAL = "User approved prospective bounded-pilot review policy in Codex conversation 2026-11-02; pilot caps chosen at user request"


def validate_promotion_policy(p):
    """Validate the specifically reviewed fixed-look policy and bounded resource contract.

    Parameters
    ----------
    p : dict
        Production-policy extension to the human review policy.

    Returns
    -------
    None
        Raises if thresholds, cap ceilings or claim accounting differ from approval.
    """
    required = {
        "approval_reference",
        "gates",
        "pilot_caps",
        "family_definitions",
        "seed",
        "max_bootstrap_cells",
        "evaluation_rule",
    }
    if (
        set(p) - {"family"} != required
        or p["approval_reference"] != APPROVAL
        or p["gates"] != GATES
        or p["pilot_caps"] != CAPS
        or p["evaluation_rule"]
        != "one_terminal_look_complete_family_both_measures_both_directions_absolute_and_contrast"
    ):
        raise ValueError("Unsupported or unreviewed prospective promotion policy")
    if (
        type(p["seed"]) is not int
        or not 0 <= p["seed"] < 2**32
        or type(p["max_bootstrap_cells"]) is not int
        or not 1 <= p["max_bootstrap_cells"] <= 50000000
    ):
        raise ValueError("Bounded deterministic promotion computation required")
    if any(type(v) not in (int, float) for v in p["pilot_caps"].values()) or any(
        type(v) not in (int, float) for v in p["gates"].values()
    ):
        raise ValueError("Typed numeric promotion gates/caps required; booleans are not limits")
    family = p["family_definitions"]
    if (
        not isinstance(family, list)
        or not 1 <= len(family) <= 16
        or len({r["version_id"] for r in family}) != len(family)
    ):
        raise ValueError(
            "One complete distinct predeclared family of at most 16 definitions required"
        )
    for row in family:
        if set(row) != {"publication", "version_id", "trial_arm"} or row["trial_arm"] not in {
            "subgroup",
            "complement",
        }:
            raise ValueError("Exact definition and predeclared trial arm required")


def evaluate_family(policy: dict, populations: dict, readiness: dict) -> dict:
    """Evaluate a complete prospective family once, retaining failed correction slots.

    Parameters
    ----------
    policy : dict
        Reviewed promotion extension with frozen family and gates.
    populations : dict
        Version ID to unique prospective observation DataFrame, no historical pooling.
    readiness : dict
        Per-version source blockers, including chronology, pending outcomes and deviations.

    Returns
    -------
    dict
        Eight directional hypotheses per version and eligible human-review statuses.
    """
    expected = {r["version_id"] for r in policy["family_definitions"]}
    if set(populations) != expected or set(readiness) != expected:
        raise ValueError("Complete declared prospective family required")
    results, claims = [], []
    work = 0
    for row in policy["family_definitions"]:
        ident, trial = row["version_id"], row["trial_arm"]
        data = populations[ident].copy()
        blockers = list(readiness[ident])
        if not data.empty and (
            data.observation_key.duplicated().any() or data.enrollment_id.duplicated().any()
        ):
            raise ValueError("Unique prospective observations required")
        clusters = []
        for arm in (trial, "complement" if trial == "subgroup" else "subgroup"):
            a = data.loc[
                data.arm.eq(arm)
                & data.original_quote_available.eq(True)
                & ~data.result.isin(["pending", "void"])
            ].copy()  # noqa: E712
            if len(a) < 100:
                blockers.append(arm + "_insufficient_wagers")
            known = a.event_namespace.fillna("").astype(str).str.strip().ne("") & a.event_id.fillna(
                ""
            ).astype(str).str.strip().ne("")
            if not known.all():
                blockers.append(arm + "_incomplete_event_ids")
            a = a.loc[known]
            if not a.empty and (
                not np.isfinite(
                    a[["paper_stake_usd", "paper_profit_usd", "unit_return"]].to_numpy(float)
                ).all()
                or not a.paper_stake_usd.gt(0).all()
            ):
                blockers.append(arm + "_invalid_economics")
            grouped = a.groupby(["event_namespace", "event_id"], sort=True).agg(
                stake=("paper_stake_usd", "sum"),
                profit=("paper_profit_usd", "sum"),
                count=("enrollment_id", "size"),
                unit_sum=("unit_return", "sum"),
            )
            if len(grouped) < 100:
                blockers.append(arm + "_insufficient_event_groups")
            stake = a.paper_stake_usd.sum()
            if (
                stake <= 0
                or (len(a) and a.paper_stake_usd.max() / stake > 0.05)
                or (len(grouped) and grouped.stake.max() / stake > 0.10)
            ):
                blockers.append(arm + "_stake_concentration_or_invalid_stake")
            clusters.append(grouped)
        slots = {
            (m, contrast, direction): None
            for m in ("weighted", "equal_stake")
            for contrast in ("absolute", "contrast")
            for direction in ("positive", "negative")
        }
        intervals = {}
        if not blockers:
            joint = (
                clusters[0]
                .join(clusters[1], how="outer", lsuffix="_a", rsuffix="_b")
                .fillna(0)
                .sort_index()
            )
            g = len(joint)
            work += g * 5000
            if work > policy["max_bootstrap_cells"]:
                raise ValueError(
                    "Promotion bootstrap budget exceeded; no partial family evaluation"
                )
            values = joint.to_numpy(float)
            cols = {c: i for i, c in enumerate(joint.columns)}
            rng = np.random.default_rng(int(fingerprint([policy["seed"], ident])[:8], 16))
            draws = []
            for offset in range(0, 5000, 64):
                ix = rng.integers(0, g, size=(min(64, 5000 - offset), g))
                sums = values[ix].sum(axis=1)
                outputs = []
                for den, num in (("stake", "profit"), ("count", "unit_sum")):
                    da, db = sums[:, cols[den + "_a"]], sums[:, cols[den + "_b"]]
                    if np.any(da <= 0) or np.any(db <= 0):
                        blockers.append("degenerate_bootstrap_denominator")
                        break
                    ra, rb = sums[:, cols[num + "_a"]] / da, sums[:, cols[num + "_b"]] / db
                    outputs.extend([ra, ra - rb])
                if blockers:
                    break
                draws.append(np.column_stack(outputs))
            if not blockers:
                bounds = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
                for mi, (measure, den, num) in enumerate(
                    (("weighted", "stake", "profit"), ("equal_stake", "count", "unit_sum"))
                ):
                    da, db = joint[den + "_a"].to_numpy(float), joint[den + "_b"].to_numpy(float)
                    na, nb = joint[num + "_a"].to_numpy(float), joint[num + "_b"].to_numpy(float)
                    ra, rb = na.sum() / da.sum(), nb.sum() / db.sum()
                    sa, sb = (na - ra * da) / da.sum(), (nb - rb * db) / db.sum()
                    for ci, (contrast, estimate, score) in enumerate(
                        (("absolute", ra, sa), ("contrast", ra - rb, sa - sb))
                    ):
                        se = np.sqrt(g / (g - 1) * np.sum(score**2))
                        intervals[measure + "_" + contrast] = {
                            "estimate": float(estimate),
                            "low": float(bounds[0, mi * 2 + ci]),
                            "high": float(bounds[1, mi * 2 + ci]),
                            "interval_method": "marginal_paired_event_bootstrap_percentile",
                            "test_method": "event_cluster_influence_t_approximation",
                        }
                        if not np.isfinite(se) or se <= 0:
                            blockers.append("degenerate_cluster_standard_error")
                        else:
                            slots[(measure, contrast, "positive")] = float(
                                t.sf(estimate / se, g - 1)
                            )
                            slots[(measure, contrast, "negative")] = float(
                                t.cdf(estimate / se, g - 1)
                            )
        if blockers:
            slots = dict.fromkeys(slots)
        for (measure, contrast, direction), raw in slots.items():
            claims.append(
                {
                    "definition_version_id": ident,
                    "measure": measure,
                    "contrast": contrast,
                    "direction": direction,
                    "raw_p": raw,
                    "holm_p": None,
                }
            )
        results.append(
            {
                "definition_version_id": ident,
                "trial_arm": trial,
                "blockers": sorted(set(blockers)),
                "intervals": intervals,
                "eligible_for_human_pilot_review": False,
            }
        )
    # Missing tests retain their full-family penalty via a p=1 placeholder.
    ordered = sorted(
        range(len(claims)),
        key=lambda i: (claims[i]["raw_p"] if claims[i]["raw_p"] is not None else 1.0, i),
    )
    previous = 0.0
    for rank, index in enumerate(ordered):
        raw = claims[index]["raw_p"]
        previous = min(1.0, max(previous, (len(claims) - rank) * (raw if raw is not None else 1.0)))
        claims[index]["holm_p"] = previous if raw is not None else None
    for result in results:
        positive = [
            c
            for c in claims
            if c["definition_version_id"] == result["definition_version_id"]
            and c["direction"] == "positive"
        ]
        result["eligible_for_human_pilot_review"] = bool(
            not result["blockers"]
            and all(c["holm_p"] is not None and c["holm_p"] <= 0.05 for c in positive)
            and all(v["estimate"] > 0 and v["low"] > 0 for v in result["intervals"].values())
        )
    return {
        "results": results,
        "hypotheses": claims,
        "family_size": len(claims),
        "bootstrap_cells": work,
        "interpretation": "bounded_human_pilot_review_only_not_fill_or_profitability_proof",
    }
