"""
sizing_diagnostics.py
===============================================================================
Purpose
-------
Compare stake-weighted portfolio returns with hypothetical equal-stake diagnostics.

What this module provides
-------------------------
Paired-cohort economics, concentration, applied-regime splits and conditional
namespace/event-proxy intervals, without changing canonical metric definitions.

Design Notes
------------
Normalized returns assume proportional profit scaling; they are not historical
replays. Event proxies are not proven independent. Filter views are non-additive.
===============================================================================
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.robustness import cluster_bootstrap
from oddsjam_ev.io.source_csv import fingerprint


@dataclass(frozen=True)
class SizingDiagnosticsConfig:
    """Explicit hypothetical unit, event support and bounded bootstrap policy.

    Attributes
    ----------
    fixed_unit_usd : float
        Notional comparison unit, not a recommended stake.
    min_event_groups, bootstrap_resamples, seed : int
        Support, Monte Carlo budget and deterministic seed.
    confidence_level : float
        Conditional percentile interval level, not a significance decision.
    contract_version : int
        Versioned descriptive metric contract.
    """

    fixed_unit_usd: float = 1.0
    min_event_groups: int = 100
    bootstrap_resamples: int = 1000
    seed: int = 20261030
    confidence_level: float = 0.95
    contract_version: int = 1

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported sizing diagnostic contract")
        if (
            isinstance(self.fixed_unit_usd, bool)
            or not isinstance(self.fixed_unit_usd, (int, float))
            or not np.isfinite(self.fixed_unit_usd)
            or self.fixed_unit_usd <= 0
        ):
            raise ValueError("Positive finite fixed unit required")
        for name, lo, hi in [
            ("min_event_groups", 2, 1000000),
            ("bootstrap_resamples", 100, 10000),
            ("seed", 0, 2**32 - 1),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError("Invalid sizing diagnostic bootstrap policy")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0 < self.confidence_level < 1
        ):
            raise ValueError("Invalid sizing diagnostic confidence level")


def _prepare(bets, linkage):
    required = {
        "bet_id",
        "product",
        "execution_status",
        "reported_roi_eligible",
        "reported_stake_usd",
        "reported_profit_usd",
        "economic_exclusions_json",
        "namespace",
        "game_id",
    }
    linked = {
        "bet_id",
        "product",
        "execution_status",
        "source_reported_stake_usd",
        "linkage_status",
        "applied_policy_id",
        "bankroll_amount",
    }
    if required - set(bets) or linked - set(linkage):
        raise ValueError("Missing sizing diagnostic provenance")
    for frame in (bets, linkage):
        if frame.bet_id.isna().any() or frame.bet_id.duplicated().any():
            raise ValueError("Unique sizing wager identities required")
    if set(bets.bet_id) != set(linkage.bet_id):
        raise ValueError("Sizing population does not reconcile")
    if not bets.reported_roi_eligible.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Boolean canonical eligibility required")
    data = bets.copy().merge(
        linkage[
            [
                "bet_id",
                "product",
                "execution_status",
                "source_reported_stake_usd",
                "linkage_status",
                "applied_policy_id",
                "bankroll_amount",
            ]
        ],
        on="bet_id",
        validate="one_to_one",
        suffixes=("", "_sizing"),
    )
    for key in ("product", "execution_status"):
        if not data[key].equals(data[key + "_sizing"]):
            raise ValueError("Sizing scope disagrees with canonical facts")
    if not np.allclose(
        pd.to_numeric(data.reported_stake_usd, errors="coerce"),
        pd.to_numeric(data.source_reported_stake_usd, errors="coerce"),
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise ValueError("Sizing source stake changed")
    data["stake"] = pd.to_numeric(data.reported_stake_usd, errors="coerce")
    data["profit"] = pd.to_numeric(data.reported_profit_usd, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        data["unit_return"] = data.profit / data.stake
    reasons = []
    for row in data.itertuples():
        reason = []
        if not row.reported_roi_eligible:
            reason.append("canonical_ineligible")
        if not np.isfinite(row.stake) or row.stake <= 0:
            reason.append("nonpositive_or_nonfinite_stake")
        if not np.isfinite(row.profit):
            reason.append("nonfinite_profit")
        if not np.isfinite(row.unit_return):
            reason.append("nonfinite_unit_return")
        if row.product not in {"fixed_odds", "exchange"}:
            reason.append("unsupported_product")
        reasons.append(json.dumps(reason))
    data["diagnostic_exclusions_json"] = reasons
    data["diagnostic_eligible"] = data.diagnostic_exclusions_json.eq("[]")
    applied = data.applied_policy_id.fillna("").astype(str).ne("")
    if (applied & data.linkage_status.ne("application_verified")).any():
        raise ValueError("Sizing application is not verified")
    data["sizing_regime"] = data.applied_policy_id.where(applied, "<unknown>")
    bank = pd.to_numeric(data.bankroll_amount, errors="coerce")
    valid_bank = applied & np.isfinite(bank) & bank.gt(0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = data.stake / bank
    valid_bank &= np.isfinite(ratio) & data.diagnostic_eligible
    data["bankroll_relative_stake"] = ratio.where(valid_bank)
    data["bankroll_relative_status"] = np.where(
        valid_bank,
        "available",
        np.where(~applied, "application_unknown", "bankroll_or_eligible_stake_unavailable"),
    )
    return data.sort_values("bet_id").reset_index(drop=True)


def _summary(group, cfg, cache, include_intervals=True):
    e = group.loc[group.diagnostic_eligible].sort_values("bet_id")
    n = len(e)
    stake = float(e.stake.sum()) if n else np.nan
    profit = float(e.profit.sum()) if n else np.nan
    unit = float(e.unit_return.mean()) if n else np.nan
    if n and not np.isfinite([stake, profit, unit, e.unit_return.sum() * cfg.fixed_unit_usd]).all():
        raise ValueError("Sizing diagnostic aggregate exceeds numeric range")
    known = e.namespace.fillna("").astype(str).str.strip().ne("") & ~e.game_id.fillna("").astype(
        str
    ).str.strip().isin(["", "<conflict>", "<missing>"])
    covered = e.loc[known]
    groups = (
        covered.groupby(["namespace", "game_id"], sort=True)
        .agg(
            stake=("stake", "sum"),
            profit=("profit", "sum"),
            wager_count=("bet_id", "size"),
            unit_sum=("unit_return", "sum"),
        )
        .reset_index()
    )
    cohort = fingerprint(sorted(e.bet_id.tolist()))
    interval_cohort = fingerprint(sorted(covered.bet_id.tolist()))
    key = (interval_cohort, include_intervals)
    if key not in cache:
        bounds = (np.nan,) * 4
        status = (
            "insufficient_event_proxy_groups"
            if include_intervals
            else "comparison_inference_deferred"
        )
        if include_intervals and len(groups) >= cfg.min_event_groups:
            policy = DiscoveryConfig(
                min_event_groups=cfg.min_event_groups,
                bootstrap_resamples=cfg.bootstrap_resamples,
                confidence_level=cfg.confidence_level,
                seed=cfg.seed,
            )
            seed = (cfg.seed + int(interval_cohort[:8], 16)) % (2**32)
            w = cluster_bootstrap(groups, policy, seed)
            equal = groups.assign(stake=groups.wager_count.astype(float), profit=groups.unit_sum)
            u = cluster_bootstrap(equal, policy, seed)
            bounds = (*w, *u)
            status = (
                "conditional_event_proxy_intervals"
                if all(np.isfinite(bounds))
                else "conditional_event_proxy_degenerate_interval"
            )
        cache[key] = (bounds, status)
    bounds, status = cache[key]
    absolute = float(e.profit.abs().sum()) if n else np.nan
    shares = e.stake / stake if n else pd.Series(dtype=float)
    hhi = float((shares**2).sum()) if n else np.nan
    canonical = group.loc[group.reported_roi_eligible]
    cs = canonical.stake.sum(min_count=1)
    cp = canonical.profit.sum(min_count=1)
    gs = groups.stake.sum(min_count=1)
    gp = groups.profit.sum(min_count=1)
    relative = e.bankroll_relative_stake.dropna()
    weighted = profit / stake if n else np.nan
    return {
        "candidate_count": len(group),
        "canonical_eligible_count": len(canonical),
        "diagnostic_eligible_count": n,
        "excluded_count": len(group) - n,
        "additional_excluded_count": int(
            (group.reported_roi_eligible & ~group.diagnostic_eligible).sum()
        ),
        "cohort_fingerprint": cohort,
        "canonical_reported_roi": float(cp / cs)
        if np.isfinite(cs)
        and cs > 0
        and np.isfinite(cp)
        and np.isfinite(canonical.stake).all()
        and np.isfinite(canonical.profit).all()
        else np.nan,
        "eligible_stake_usd": stake,
        "reported_profit_usd": profit,
        "stake_weighted_roi": weighted,
        "equal_stake_mean_return": unit,
        "fixed_unit_usd": cfg.fixed_unit_usd,
        "fixed_unit_notional_profit_usd": float(e.unit_return.sum() * cfg.fixed_unit_usd)
        if n
        else np.nan,
        "weighted_equal_sign_disagreement": bool(n and np.sign(weighted) != np.sign(unit)),
        "largest_stake_share": float(shares.max()) if n else np.nan,
        "top_five_stake_share": float(shares.nlargest(5).sum()) if n else np.nan,
        "stake_hhi": hhi,
        "stake_weight_equivalent_count": 1 / hhi if n else np.nan,
        "largest_absolute_profit_share": float(e.profit.abs().max() / absolute)
        if absolute > 0
        else np.nan,
        "event_proxy_count": len(groups),
        "event_covered_count": len(covered),
        "event_missing_count": n - len(covered),
        "interval_cohort_fingerprint": interval_cohort,
        "event_covered_stake_usd": gs,
        "event_covered_stake_weighted_roi": float(gp / gs) if len(groups) else np.nan,
        "event_covered_equal_stake_mean_return": float(covered.unit_return.mean())
        if len(covered)
        else np.nan,
        "largest_event_stake_share_of_covered_stake": float(groups.stake.max() / gs)
        if len(groups)
        else np.nan,
        "weighted_roi_ci_low": bounds[0],
        "weighted_roi_ci_high": bounds[1],
        "equal_stake_ci_low": bounds[2],
        "equal_stake_ci_high": bounds[3],
        "uncertainty_status": status,
        "bankroll_relative_available_count": len(relative),
        "bankroll_relative_stake_mean": float(relative.mean()) if len(relative) else np.nan,
        "currency": "USD",
        "fee_inclusion": "unverified",
        "normalization_status": "hypothetical_proportional_scaling_not_replay",
    }


def build_sizing_diagnostics(
    tables: dict, linkage: pd.DataFrame, cfg: SizingDiagnosticsConfig
) -> dict:
    """Compare paired estimands by funding/product, verified regime and saved filter.

    Parameters
    ----------
    tables : dict
        Verified canonical tables including distinct facts and attribution bridge.
    linkage : pandas.DataFrame
        Verified matching sizing publication; unknown application remains separate.
    cfg : SizingDiagnosticsConfig
        Declared diagnostic unit and event-proxy uncertainty policy.

    Returns
    -------
    dict
        Auditable candidate rows and descriptive cohort metrics. Overall and regime
        cohorts partition candidates; filter views overlap and must not be summed.
    """
    data = _prepare(tables["bet_facts"], linkage)
    bridge = tables["filter_attribution"]
    if not {"bet_id", "filter_id", "attribution_status"} <= set(bridge) or not set(
        bridge.bet_id
    ) <= set(data.bet_id):
        raise ValueError("Invalid sizing attribution bridge")
    resolved = bridge.loc[bridge.attribution_status.eq("resolved"), ["bet_id", "filter_id"]]
    if resolved.filter_id.isna().any() or resolved.duplicated().any():
        raise ValueError("Duplicate or missing sizing filter identity")
    metrics = []
    cache = {}
    filtered = data.merge(resolved, on="bet_id", validate="one_to_many")
    views = [
        ("overall", data, []),
        ("regime", data, ["sizing_regime"]),
        ("filter", filtered, ["filter_id"]),
        ("filter_regime", filtered, ["filter_id", "sizing_regime"]),
    ]
    for view, frame, extra in views:
        keys = ["execution_status", "product", *extra]
        for values, group in frame.groupby(keys, sort=True, dropna=False):
            metrics.append(
                {
                    "view": view,
                    "filter_id": None,
                    "sizing_regime": None,
                    **dict(zip(keys, values, strict=True)),
                    **_summary(group, cfg, cache),
                }
            )
    columns = [
        "view",
        "execution_status",
        "product",
        "filter_id",
        "sizing_regime",
        *_summary(data.iloc[:0], cfg, {}).keys(),
    ]
    return {"candidate_diagnostics": data, "cohort_metrics": pd.DataFrame(metrics, columns=columns)}


def summarize_sizing_cohort(group: pd.DataFrame, cfg: SizingDiagnosticsConfig) -> dict:
    """Reuse paired descriptive economics without performing comparison inference.

    Parameters
    ----------
    group : pandas.DataFrame
        Unique prepared candidate rows from a verified sizing diagnostics table.
    cfg : SizingDiagnosticsConfig
        Original diagnostic policy; preserves the declared notional unit.

    Returns
    -------
    dict
        Identical point estimands and coverage fields with intervals unavailable.
    """
    if group.bet_id.isna().any() or group.bet_id.duplicated().any():
        raise ValueError("Unique cohort wager identities required")
    return _summary(group, cfg, {}, include_intervals=False)
