"""
paired_comparison.py
===============================================================================
Purpose
-------
Compute exploratory paired subgroup/complement contrasts at event-proxy grain.

What this module provides
-------------------------
Approved arm gates, joint cluster percentile intervals and delta-method t tests.

Design Notes
------------
The two arms share event draws; no execution-level IID bootstrap is used. Approximate
marginal tests condition on unverified event proxies and are not holdout validation.
===============================================================================
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import t

from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig


class PrioritizationBudgetExceeded(ValueError):
    """Fail-closed computation/pair budget overflow with private audit context."""

    def __init__(self, audit):
        self.audit = audit
        super().__init__("Prioritization budget exceeded; no partial family published")


def paired_comparison(
    subgroup: pd.DataFrame,
    complement: pd.DataFrame,
    cfg: PrioritizationConfig,
    seed: int,
    evaluate: bool = True,
) -> tuple[dict, dict]:
    """Estimate differences on the same eligible parent partition under arm gates.

    Parameters
    ----------
    subgroup, complement : pandas.DataFrame
        Unique disjoint prepared sizing-diagnostic candidates.
    cfg : PrioritizationConfig
        Declared arm support/concentration policy and resampling budget.
    seed : int
        Stable partition-specific random seed shared by both return estimands.
    evaluate : bool
        False returns preflight gates/work only; no resampling or tests.

    Returns
    -------
    tuple[dict, dict]
        Coverage/interval evidence and directional raw p-values per estimand.
        Failed gates or degenerate inputs leave tests unavailable, not zero.
    """
    if set(subgroup.bet_id) & set(complement.bet_id):
        raise ValueError("Paired arms are not disjoint")
    arms = [
        g.loc[g.diagnostic_eligible].sort_values("bet_id").copy() for g in (subgroup, complement)
    ]
    rows = []
    gates = []
    clusters = []
    for label, arm in zip(("subgroup", "complement"), arms, strict=True):
        known = arm.namespace.fillna("").astype(str).str.strip().ne("") & ~arm.game_id.fillna(
            ""
        ).astype(str).str.strip().isin(["", "<missing>", "<conflict>"])
        grouped = (
            arm.loc[known]
            .groupby(["namespace", "game_id"], sort=True)
            .agg(
                stake=("stake", "sum"),
                profit=("profit", "sum"),
                count=("bet_id", "size"),
                unit_sum=("unit_return", "sum"),
            )
        )
        stake = arm.stake.sum()
        coverage = float(known.mean()) if len(arm) else 0.0
        largest = float(arm.stake.max() / stake) if stake > 0 else np.nan
        event_share = float(grouped.stake.max() / stake) if stake > 0 and len(grouped) else np.nan
        if len(arm) < cfg.min_eligible_wagers:
            gates.append(label + "_insufficient_wagers")
        if len(grouped) < cfg.min_event_groups:
            gates.append(label + "_insufficient_event_groups")
        if coverage < cfg.min_event_coverage:
            gates.append(label + "_incomplete_event_coverage")
        if not np.isfinite(largest) or largest > cfg.max_wager_stake_share:
            gates.append(label + "_wager_concentration")
        if not np.isfinite(event_share) or event_share > cfg.max_event_stake_share:
            gates.append(label + "_event_concentration")
        if stake <= 0 or not np.isfinite(stake):
            gates.append(label + "_invalid_stake")
        rows.append(
            {
                "label": label,
                "eligible_count": len(arm),
                "stake_usd": float(stake),
                "reported_profit_usd": float(arm.profit.sum()) if len(arm) else np.nan,
                "weighted_roi": float(arm.profit.sum() / stake) if stake > 0 else np.nan,
                "equal_stake_return": float(arm.unit_return.mean()) if len(arm) else np.nan,
                "unknown_sizing_regime_count": int(arm.sizing_regime.eq("<unknown>").sum()),
                "groups": len(grouped),
                "coverage": coverage,
                "largest_wager_share": largest,
                "largest_event_share": event_share,
            }
        )
        clusters.append(grouped)
    result = {
        f"{r['label']}_{key}": value for r in rows for key, value in r.items() if key != "label"
    }
    result.update(
        gate_reasons_json=json.dumps(gates),
        test_status="blocked_policy_gates",
        weighted_difference=np.nan,
        equal_stake_difference=np.nan,
        weighted_ci_low=np.nan,
        weighted_ci_high=np.nan,
        equal_stake_ci_low=np.nan,
        equal_stake_ci_high=np.nan,
        weighted_standard_error=np.nan,
        equal_stake_standard_error=np.nan,
        cluster_degrees_of_freedom=0,
        bootstrap_seed=seed,
        joint_event_count=0,
        bootstrap_cells=0,
        interval_method="paired_event_proxy_percentile",
        marginal_method="delta_method_cluster_influence_t_approximation",
        event_independence_verified=False,
        temporal_validation_status="blocked_unverified_chronology",
    )
    tests = {
        measure: {"positive": np.nan, "negative": np.nan} for measure in ("weighted", "equal_stake")
    }
    if gates:
        return result, tests
    joint = (
        clusters[0]
        .join(clusters[1], how="outer", lsuffix="_a", rsuffix="_b")
        .fillna(0)
        .sort_index()
    )
    g = len(joint)
    cells = g * cfg.bootstrap_resamples
    if cells > cfg.max_bootstrap_cells:
        raise PrioritizationBudgetExceeded(
            {"reason": "bootstrap_cells", "requested": cells, "maximum": cfg.max_bootstrap_cells}
        )
    arrays = {c: joint[c].to_numpy(float) for c in joint}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite event comparison totals")
    estimates = []
    ses = []
    for denom, numer in (("stake", "profit"), ("count", "unit_sum")):
        da, db = arrays[denom + "_a"], arrays[denom + "_b"]
        na, nb = arrays[numer + "_a"], arrays[numer + "_b"]
        ra, rb = na.sum() / da.sum(), nb.sum() / db.sum()
        score = (na - ra * da) / da.sum() - (nb - rb * db) / db.sum()
        estimates.append(float(ra - rb))
        ses.append(float(np.sqrt(g / (g - 1) * np.sum(score**2))))
    result.update(
        weighted_difference=estimates[0],
        equal_stake_difference=estimates[1],
        joint_event_count=g,
        weighted_standard_error=ses[0],
        equal_stake_standard_error=ses[1],
        cluster_degrees_of_freedom=g - 1,
        bootstrap_cells=cells,
    )
    if not evaluate:
        result["test_status"] = "gates_passed_preflight"
        return result, tests
    rng = np.random.default_rng(seed)
    draws = []
    batch = max(1, min(128, 1000000 // g))
    for start in range(0, cfg.bootstrap_resamples, batch):
        indexes = rng.integers(0, g, size=(min(batch, cfg.bootstrap_resamples - start), g))
        outputs = []
        for denom, numer in (("stake", "profit"), ("count", "unit_sum")):
            da = arrays[denom + "_a"][indexes].sum(axis=1)
            db = arrays[denom + "_b"][indexes].sum(axis=1)
            if (da <= 0).any() or (db <= 0).any():
                result["test_status"] = "blocked_invalid_resample_denominator"
                return result, tests
            outputs.append(
                arrays[numer + "_a"][indexes].sum(axis=1) / da
                - arrays[numer + "_b"][indexes].sum(axis=1) / db
            )
        draws.append(np.column_stack(outputs))
    draws = np.concatenate(draws)
    tail = (1 - cfg.confidence_level) / 2
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
    for i in range(2):
        if not np.isfinite(ses[i]) or ses[i] <= 0:
            result["test_status"] = "blocked_degenerate_comparison"
            return result, {m: {"positive": np.nan, "negative": np.nan} for m in tests}
    result.update(
        weighted_ci_low=float(bounds[0, 0]),
        weighted_ci_high=float(bounds[1, 0]),
        equal_stake_ci_low=float(bounds[0, 1]),
        equal_stake_ci_high=float(bounds[1, 1]),
        test_status="approximate_conditional_exploratory",
    )
    for i, measure in enumerate(("weighted", "equal_stake")):
        statistic = estimates[i] / ses[i]
        tests[measure] = {
            "positive": float(t.sf(statistic, g - 1)),
            "negative": float(t.sf(-statistic, g - 1)),
        }
    return result, tests
