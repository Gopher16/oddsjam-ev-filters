"""
review_ranking.py
===============================================================================
Purpose
-------
Order research evidence through transparent nondominated fronts and diversity.

What this module provides
-------------------------
Direction agreement, pointwise-bound components, sensitivity and overlap explanations.

Design Notes
------------
No probability-of-edge score, new test, pooling or production authority is created.
===============================================================================
"""

from itertools import combinations

import numpy as np

from oddsjam_ev.analysis.review_catalogue import ReviewBudgetExceeded


def classify(stat, corrected):
    """Require agreement in absolute returns and subgroup-minus-complement contrasts."""
    w, u = stat["subgroup_weighted_roi"], stat["subgroup_equal_stake_return"]
    dw = w - stat["complement_weighted_roi"]
    du = u - stat["complement_equal_stake_return"]
    if not np.isfinite([w, u, dw, du]).all():
        direction = "unavailable"
    elif w > 0 and u > 0 and dw > 0 and du > 0:
        direction = "opportunity"
    elif w < 0 and u < 0 and dw < 0 and du < 0:
        direction = "loss_association"
    elif w < 0 <= u or u < 0 <= w or dw < 0 <= du or du < 0 <= dw:
        direction = "sizing_or_estimand_conflict"
    else:
        direction = "relative_or_neutral_association"
    qualified = stat["test_status"] in {
        "approximate_conditional_exploratory",
        "gates_passed_preflight",
    }
    if not qualified or direction not in {"opportunity", "loss_association"}:
        tier = "insufficient_or_conflicting"
    elif corrected:
        tier = "corrected_historical_comparison"
    else:
        tier = "support_qualified_exploratory"
    return direction, tier


def components(stat, direction):
    """Return Pareto objectives with unavailable interval evidence ranked last."""
    if direction == "opportunity":
        bounds = [stat.get("weighted_ci_low", np.nan), stat.get("equal_stake_ci_low", np.nan)]
    else:
        bounds = [-stat.get("weighted_ci_high", np.nan), -stat.get("equal_stake_ci_high", np.nan)]
    bound = float(min(bounds)) if np.isfinite(bounds).all() else None
    events = min(stat["subgroup_groups"], stat["complement_groups"])
    shares = [stat["subgroup_largest_event_share"], stat["complement_largest_event_share"]]
    share = float(max(shares)) if np.isfinite(shares).all() else None
    return {
        "conservative_pointwise_contrast": bound,
        "minimum_arm_event_groups": int(events),
        "worst_arm_event_stake_share": share,
    }


def _fronts(rows, objectives):
    if not rows:
        return {}
    matrix = np.asarray([objectives(row) for row in rows], dtype=float)
    dominates = np.ones((len(rows), len(rows)), dtype=bool)
    strict = np.zeros_like(dominates)
    for i in range(matrix.shape[1]):
        dominates &= matrix[:, i, None] >= matrix[None, :, i]
        strict |= matrix[:, i, None] > matrix[None, :, i]
    dominates &= strict
    remaining = np.ones(len(rows), dtype=bool)
    result, front = {}, 1
    while remaining.any():
        indexes = np.flatnonzero(remaining & ~dominates[remaining].any(axis=0))
        for index in indexes:
            result[rows[index]["candidate_id"]] = front
        remaining[indexes] = False
        front += 1
    return result


def rank_records(records, members, data, cfg):
    """Assign deterministic queue fronts, semantic groups and overlap-aware headlines.

    Parameters
    ----------
    records : list
        Complete mutable presentation records, already classified without sorting.
    members : dict
        Raw subgroup and parent sets after the original full-family correction.
    data : pandas.DataFrame
        Unique eligible economic rows indexed by canonical candidate identity.
    cfg : ReviewConfig
        Frozen display policy and pair budget.

    Returns
    -------
    tuple[list, list, list]
        All sorted records, complete nonzero overlap pairs and distinct idea groups.
    """
    if len(records) * (len(records) - 1) // 2 > cfg.max_pairs:
        raise ReviewBudgetExceeded("Review overlap-pair budget exceeded")
    tier_order = {
        "corrected_historical_comparison": 2,
        "support_qualified_exploratory": 3,
        "insufficient_or_conflicting": 4,
    }
    for row in records:
        row["tier_order"] = tier_order[row["evidence_tier"]]
        row["pareto_front"] = None
        row["sensitivity_front_without_interval"] = None
    for queue in sorted({r["queue"] for r in records}):
        for tier in sorted(tier_order):
            part = [r for r in records if r["queue"] == queue and r["evidence_tier"] == tier]

            def objective(row):
                c = row["rank_components"]
                return (
                    c["conservative_pointwise_contrast"]
                    if c["conservative_pointwise_contrast"] is not None
                    else -np.inf,
                    c["minimum_arm_event_groups"],
                    -(
                        c["worst_arm_event_stake_share"]
                        if c["worst_arm_event_stake_share"] is not None
                        else np.inf
                    ),
                )

            fronts = _fronts(part, objective)
            sensitivity = _fronts(part, lambda row: objective(row)[1:])
            for row in part:
                row["pareto_front"] = fronts[row["candidate_id"]]
                row["sensitivity_front_without_interval"] = sensitivity[row["candidate_id"]]

    def ordering(row):
        c = row["rank_components"]
        return (
            row["queue"],
            row["tier_order"],
            row["pareto_front"],
            -c["minimum_arm_event_groups"],
            c["worst_arm_event_stake_share"]
            if c["worst_arm_event_stake_share"] is not None
            else np.inf,
            row["candidate_id"],
        )

    sensitivity_ordered = sorted(
        records,
        key=lambda row: (
            row["queue"],
            row["tier_order"],
            row["sensitivity_front_without_interval"],
            -row["rank_components"]["minimum_arm_event_groups"],
            row["rank_components"]["worst_arm_event_stake_share"]
            if row["rank_components"]["worst_arm_event_stake_share"] is not None
            else np.inf,
            row["candidate_id"],
        ),
    )
    sensitivity_counts = {}
    for row in sensitivity_ordered:
        queue = row["queue"]
        sensitivity_counts[queue] = sensitivity_counts.get(queue, 0) + 1
        row["sensitivity_order_without_interval"] = sensitivity_counts[queue]
    ordered = sorted(records, key=ordering)
    eligible, events = {}, {}
    for row in ordered:
        key = row["candidate_id"]
        group = data.loc[sorted(members[key])]
        group = group.loc[group.diagnostic_eligible]
        eligible[key] = set(group.bet_id)
        known = group.loc[~group.game_id.fillna("").isin(["", "<missing>", "<conflict>"])]
        events[key] = set(zip(known.namespace, known.game_id, strict=True))
    overlap, lookup = [], {}
    for left, right in combinations(ordered, 2):
        a, b = left["candidate_id"], right["candidate_id"]
        raw_a, raw_b = members[a], members[b]
        wager_union, event_union = eligible[a] | eligible[b], events[a] | events[b]
        shared, shared_events = eligible[a] & eligible[b], events[a] & events[b]
        if not (raw_a & raw_b or shared_events):
            continue
        pair = {
            "left_candidate_id": a,
            "right_candidate_id": b,
            "shared_raw_candidates": len(raw_a & raw_b),
            "shared_eligible_wagers": len(shared),
            "eligible_wager_jaccard": len(shared) / len(wager_union) if wager_union else 0.0,
            "shared_event_proxies": len(shared_events),
            "event_jaccard": len(shared_events) / len(event_union) if event_union else 0.0,
            "equivalent_parent_and_subgroup": raw_a == raw_b
            and members[left["parent_id"]] == members[right["parent_id"]],
            "same_economic_population": left["execution_status"] == right["execution_status"]
            and left["product"] == right["product"],
            "correction_unchanged": True,
        }
        overlap.append(pair)
        lookup[frozenset((a, b))] = pair
    representatives, heads, idea_groups, equivalences = {}, {}, {}, {}
    queue_counts = {}
    for row in ordered:
        ident, queue = row["candidate_id"], row["queue"]
        group_key = (queue, row["hypothesis_group_id"])
        row["representative_id"] = representatives.setdefault(group_key, ident)
        row["suppression_reason"] = (
            None
            if row["representative_id"] == ident
            else "same_exact_predicate_other_parent_or_population_evidence_not_pooled"
        )
        eq_key = (
            queue,
            row["direction"],
            row["execution_status"],
            row["product"],
            frozenset(members[row["parent_id"]]),
            frozenset(members[ident]),
        )
        if eq_key in equivalences and members[ident]:
            row["representative_id"] = equivalences[eq_key]
            row["suppression_reason"] = "exact_parent_subgroup_equivalence_presentation_only"
        else:
            equivalences[eq_key] = ident
        row["headline"] = False
        selected = heads.setdefault(queue, [])
        if (
            queue in {"opportunities", "production_changes"}
            and row["evidence_tier"] != "insufficient_or_conflicting"
            and row["suppression_reason"] is None
        ):
            similar = next(
                (
                    other
                    for other in selected
                    if (pair := lookup.get(frozenset((ident, other))))
                    and pair["same_economic_population"]
                    and max(pair["eligible_wager_jaccard"], pair["event_jaccard"])
                    >= cfg.overlap_limit
                ),
                None,
            )
            row["headline_blocker"] = (
                "overlaps_headline:" + similar
                if similar
                else "display_cap"
                if len(selected) >= cfg.headline_limit
                else None
            )
            if row["headline_blocker"] is None:
                selected.append(ident)
                row["headline"] = True
        else:
            row["headline_blocker"] = (
                row["suppression_reason"] or "investigation_or_insufficient_evidence"
            )
        queue_counts[queue] = queue_counts.get(queue, 0) + 1
        row["presentation_order"] = queue_counts[queue]
        c = row["rank_components"]
        row["ranking_explanation"] = (
            f"{row['evidence_tier']}; front {row['pareto_front']}; minimum arm events {c['minimum_arm_event_groups']}; worst event stake share {c['worst_arm_event_stake_share']}; pointwise contrast {c['conservative_pointwise_contrast']}; setup readiness ranked separately; stable identity breaks ties"
        )
        idea_groups.setdefault(row["hypothesis_group_id"], []).append(ident)
    by_id = {row["candidate_id"]: row for row in ordered}
    for row in ordered:
        while by_id[row["representative_id"]]["representative_id"] != row["representative_id"]:
            row["representative_id"] = by_id[row["representative_id"]]["representative_id"]
        row["sensitivity_order_change_without_interval"] = (
            row["sensitivity_order_without_interval"] - row["presentation_order"]
        )
    groups = [
        {
            "hypothesis_group_id": key,
            "candidate_ids": sorted(ids),
            "constituent_evidence_not_pooled": True,
        }
        for key, ids in sorted(idea_groups.items())
    ]
    return ordered, overlap, groups
