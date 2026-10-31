"""
visual_diagnostics.py
===============================================================================
Purpose
-------
Build reusable notebook-inspired visual datasets from unique verified wager facts.

What this module provides
-------------------------
Matched economics, overlapping filter scopes, calendar/activity panels and chart gates.

Design Notes
------------
No fitting or new hypothesis tests. Scope overlap is not additive; dates are source
calendar attribution unless stronger source evidence is explicitly declared.
===============================================================================
"""

import json

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.visual_features import MISSING, prepare_visuals
from oddsjam_ev.io.source_csv import fingerprint

CHARTS = {
    "profit_by_group": ("visual_cohorts", "overall/platform/sport/league/market"),
    "ev_roi_scatter": ("visual_cohorts", "overall/platform"),
    "filter_ev_roi_scatter": ("visual_cohorts", "filter overall"),
    "ev_bucket_summary": ("visual_cohorts", "ev_bucket"),
    "liquidity_summary": ("visual_cohorts", "liquidity_bucket"),
    "liquidity_ev_summary": ("visual_cohorts", "liquidity_ev"),
    "time_to_event_summary": ("visual_cohorts", "time_to_event_bucket"),
    "cumulative_profit_by_platform": ("visual_periods", "D/W-SUN/M platform"),
    "volume_over_time": ("visual_periods", "D/W-SUN/M overall/platform"),
    "average_activity": ("visual_activity", "weekday/hour/weekday_hour active-day mean"),
}


class VisualBudgetExceeded(ValueError):
    """Abort a whole publication, never return truncated visual evidence."""


def _summary(group, cfg):
    e = group.loc[group.diagnostic_eligible]
    n = len(e)
    stake = float(e.stake.sum()) if n else np.nan
    profit = float(e.profit.sum()) if n else np.nan
    unit = float(e.unit_return.mean()) if n else np.nan
    result = {
        "candidate_count": len(group),
        "eligible_count": n,
        "excluded_count": len(group) - n,
        "eligible_stake_usd": stake,
        "reported_profit_usd": profit,
        "stake_weighted_roi": profit / stake if n else np.nan,
        "equal_stake_mean_return": unit,
        "fixed_unit_notional_profit_usd": float(e.unit_return.sum() * cfg.fixed_unit_usd)
        if n
        else np.nan,
        "fixed_unit_usd": cfg.fixed_unit_usd,
        "largest_wager_stake_share": float(e.stake.max() / stake) if n else np.nan,
        "unknown_sizing_regime_count": int(e.sizing_regime.eq("<unknown>").sum()),
        "performance_display_eligible": n >= cfg.min_performance_wagers,
        "source_date_count": int(group.source_date.notna().sum()),
        "interval_status": "not_estimated_descriptive_visual_stage",
        "fee_status": "source_reported_profit_not_independently_net_verified",
    }
    known = ~e.game_id.fillna("").astype(str).isin(MISSING) & ~e.namespace.fillna("").astype(
        str
    ).isin(MISSING)
    result["event_proxy_groups"] = len(e.loc[known, ["namespace", "game_id"]].drop_duplicates())
    result["event_id_coverage_rate"] = float(known.mean()) if n else np.nan
    for label, mask in (
        ("ev", e.ev_status.eq("valid")),
        ("closing", e.closing_price_ev_fraction.notna()),
        ("joint", e.ev_status.eq("valid") & e.closing_price_ev_fraction.notna()),
    ):
        matched = e.loc[mask]
        count = len(matched)
        ms = float(matched.stake.sum()) if count else np.nan
        result[label + "_matched_count"] = count
        result[label + "_eligible_coverage_rate"] = count / n if n else np.nan
        result[label + "_matched_stake_usd"] = ms
        result[label + "_matched_roi"] = float(matched.profit.sum() / ms) if count else np.nan
        result[label + "_matched_equal_stake_return"] = (
            float(matched.unit_return.mean()) if count else np.nan
        )
        if label != "closing":
            result[label + "_mean_advertised_ev_fraction"] = (
                float(matched.advertised_ev_fraction.mean()) if count else np.nan
            )
            result[label + "_weighted_advertised_ev_fraction"] = (
                float((matched.stake * matched.advertised_ev_fraction).sum() / ms)
                if count
                else np.nan
            )
        if label != "ev":
            result[label + "_mean_closing_price_ev_fraction"] = (
                float(matched.closing_price_ev_fraction.mean()) if count else np.nan
            )
            result[label + "_weighted_closing_price_ev_fraction"] = (
                float((matched.stake * matched.closing_price_ev_fraction).sum() / ms)
                if count
                else np.nan
            )
    monetary = [v for k, v in result.items() if isinstance(v, float) and not np.isnan(v)]
    if not np.isfinite(monetary).all():
        raise ValueError("Visual summary exceeds finite numeric range")
    return result


def _periods(group, scope, cfg):
    rows = []
    for platform, source in [
        (None, group),
        *list(group.groupby("sportsbook", sort=True, dropna=False)),
    ]:
        dated = source.loc[source.source_date.notna()].copy()
        if dated.empty:
            continue
        dated["date"] = pd.to_datetime(dated.source_date)
        lo, hi = dated.date.min(), dated.date.max()
        if (hi - lo).days > cfg.max_calendar_days:
            raise VisualBudgetExceeded("Source calendar span exceeds declared visual budget")
        for resolution in ("D", "W-SUN", "M"):
            dated["period"] = dated.date.dt.to_period(resolution)
            first, last = dated.period.min(), dated.period.max()
            partitions = {key: part for key, part in dated.groupby("period", sort=True)}
            for period in pd.period_range(first, last, freq=resolution):
                part = partitions.get(period)
                if part is None:
                    # Unobserved dates/periods do not establish zero betting activity.
                    result = {
                        "candidate_count": None,
                        "eligible_count": None,
                        "reported_profit_usd": np.nan,
                        "fixed_unit_notional_profit_usd": np.nan,
                    }
                else:
                    result = _summary(part, cfg)
                rows.append(
                    {
                        **scope,
                        "platform": platform,
                        "resolution": resolution,
                        "period_start": period.start_time.isoformat(),
                        "period_end_exclusive": (period + 1).start_time.isoformat(),
                        "period_status": "observed"
                        if part is not None
                        else "unobserved_coverage_unknown",
                        "boundary_period": period in (first, last),
                        "coverage_status": "export_calendar_coverage_unverified",
                        "time_basis": "source_wall_calendar_retrospective_not_settlement_or_bankroll",
                        **result,
                    }
                )
        # Cumulative amounts are attributed to source date, not a cash-balance series.
    frame = pd.DataFrame(rows)
    if len(frame):
        keys = ["scope_id", "resolution", "platform"]
        for metric in ("reported_profit_usd", "fixed_unit_notional_profit_usd"):
            frame["cumulative_attributed_" + metric] = frame.groupby(keys, dropna=False)[
                metric
            ].cumsum()
        frame["cumulative_interpretation"] = (
            "known_eligible_source_date_totals_not_settlement_order_or_bankroll"
        )
    return frame


def _activity(group, scope):
    rows = []
    dated = group.loc[group.source_date.notna()]
    for view, keys in (
        ("weekday", ["source_weekday"]),
        ("hour", ["source_hour"]),
        ("weekday_hour", ["source_weekday", "source_hour"]),
    ):
        supported = dated.dropna(subset=keys)
        for key, part in supported.groupby(keys, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            dates = part.source_date.nunique()
            rows.append(
                {
                    **scope,
                    "view": view,
                    "category_json": json.dumps(
                        dict(zip(keys, [int(v) for v in values], strict=True)), sort_keys=True
                    ),
                    "logged_candidate_count": len(part),
                    "active_source_day_denominator": dates,
                    "mean_candidates_per_active_source_day": len(part) / dates,
                    "denominator_semantics": "days_with_observations_in_this_bucket_not_all_covered_days",
                    "time_basis": "source_wall_calendar_timezone_not_inferred",
                    "coverage_status": "unknown_calendar_coverage",
                }
            )
    return pd.DataFrame(rows)


def build_visual_diagnostics(canonical, dimensions, diagnostics, cfg):
    """Build scope-specific visual tables and readiness for every requested chart.

    Parameters
    ----------
    canonical : dict
        Verified source facts and resolved attribution bridge.
    dimensions, diagnostics : pandas.DataFrame
        Exact matching atlas dimensions and prepared sizing diagnostics.
    cfg : VisualConfig
        Declared display/source policy and budgets.

    Returns
    -------
    tuple[dict, dict]
        Persistable tables and audit; budget violations abort without partial output.
    """
    work = prepare_visuals(canonical, dimensions, diagnostics, cfg)
    scopes, memberships = [], []
    for (funding, product), group in work.groupby(
        ["execution_status", "product"], sort=True, dropna=False
    ):
        scope = {
            "scope_kind": "population",
            "filter_id": None,
            "execution_status": funding,
            "product": product,
        }
        scopes.append((scope, group.bet_id.tolist()))
    attribution = canonical["filter_attribution"]
    if not set(attribution.bet_id) <= set(work.bet_id):
        raise ValueError("Unknown visual attribution candidate")
    bridge = attribution.loc[
        attribution.attribution_status.eq("resolved"), ["bet_id", "filter_id"]
    ].drop_duplicates()
    if bridge.filter_id.isna().any():
        raise ValueError("Resolved visual filter identity missing")
    joined = work[["bet_id", "execution_status", "product"]].merge(
        bridge, on="bet_id", validate="one_to_many"
    )
    for (ident, funding, product), part in joined.groupby(
        ["filter_id", "execution_status", "product"], sort=True
    ):
        scopes.append(
            (
                {
                    "scope_kind": "filter",
                    "filter_id": ident,
                    "execution_status": funding,
                    "product": product,
                },
                part.bet_id.tolist(),
            )
        )
    if sum(len(ids) for _, ids in scopes) > cfg.max_scope_links:
        raise VisualBudgetExceeded("Visual membership budget exceeded; no scopes truncated")
    cohorts, periods, activities, readiness, coverage = [], [], [], [], []
    work = work.set_index("bet_id", drop=False).rename_axis(None)
    count = 0
    for source_scope, ids in scopes:
        scope = {
            "scope_id": "visual_scope:" + fingerprint(source_scope),
            **source_scope,
            "overlapping_scope_totals_not_additive": source_scope["scope_kind"] == "filter",
        }
        group = work.loc[sorted(ids)].copy()
        memberships.extend(
            {"scope_id": scope["scope_id"], "bet_id": ident} for ident in sorted(ids)
        )
        cohorts.append(
            {
                **scope,
                "view": "overall",
                "category_json": "{}",
                "cohort_fingerprint": fingerprint(sorted(ids)),
                **_summary(group, cfg),
            }
        )
        views = {
            "platform": ["sportsbook"],
            "sport": ["sport"],
            "league": ["league"],
            "market": ["market_family"],
            "ev_bucket": ["ev_bucket"],
            "liquidity_bucket": ["liquidity_bucket"],
            "liquidity_ev": ["liquidity_bucket", "ev_bucket"],
            "time_to_event_bucket": ["time_to_event_bucket"],
        }
        for view, columns in views.items():
            for column in columns:
                group[column] = group[column].fillna("<missing>").replace("", "<missing>")
            for key, part in group.groupby(columns, sort=True, dropna=False):
                key = key if isinstance(key, tuple) else (key,)
                cohorts.append(
                    {
                        **scope,
                        "view": view,
                        "category_json": json.dumps(
                            dict(zip(columns, map(str, key), strict=True)), sort_keys=True
                        ),
                        "cohort_fingerprint": fingerprint(sorted(part.bet_id)),
                        **_summary(part, cfg),
                    }
                )
        period = _periods(group, scope, cfg)
        activity = _activity(group, scope)
        if len(period):
            periods.append(period)
        if len(activity):
            activities.append(activity)
        for feature, column in (
            ("advertised_ev", "ev_status"),
            ("source_calendar", "source_date_status"),
            ("source_clock", "source_clock_status"),
            ("liquidity", "liquidity_status"),
            ("time_to_event", "timing_status"),
            ("closing_reference", "closing_status"),
        ):
            for state, n in group[column].value_counts(dropna=False).items():
                coverage.append(
                    {
                        **scope,
                        "feature": feature,
                        "status": str(state),
                        "candidate_count": int(n),
                        "denominator_candidates": len(group),
                        "coverage_rate": n / len(group),
                    }
                )
        e = group.loc[group.diagnostic_eligible]
        supports = {
            "profit_by_group": len(e),
            "ev_roi_scatter": int(e.ev_status.eq("valid").sum()),
            "filter_ev_roi_scatter": int(e.ev_status.eq("valid").sum())
            if scope["scope_kind"] == "filter"
            else 0,
            "ev_bucket_summary": int(e.ev_status.eq("valid").sum()),
            "liquidity_summary": int(e.liquidity_usd.notna().sum()),
            "liquidity_ev_summary": int((e.liquidity_usd.notna() & e.ev_status.eq("valid")).sum()),
            "time_to_event_summary": int(e.time_to_event_hours.notna().sum()),
            "cumulative_profit_by_platform": int(e.source_date.notna().sum()),
            "volume_over_time": int(group.source_date.notna().sum()),
            "average_activity": int(group.source_date.notna().sum()),
        }
        remedies = {
            "liquidity_summary": "Obtain decision-time USD liquidity evidence; raw source/proxy tags are not verified capacity.",
            "liquidity_ev_summary": "Verify liquidity units/timing and matched EV coverage.",
            "time_to_event_summary": "Verify created_at as placement time, event_start_date semantics and timezone/offset; resolve DST conflicts.",
            "filter_ev_roi_scatter": "Select a resolved filter scope with matched EV/economic eligibility.",
            "ev_roi_scatter": "Obtain valid numeric advertised EV on eligible economic rows.",
            "ev_bucket_summary": "Obtain valid numeric source EV; preserve exact endpoints.",
            "cumulative_profit_by_platform": "Obtain valid source calendar dates and eligible economics; settlement timestamps are needed for settlement-time curves.",
            "volume_over_time": "Resolve missing/conflicting source calendar dates without timezone guessing.",
            "average_activity": "Resolve source calendar dates; verify full calendar coverage for covered-day averages.",
        }
        for chart, (table, view) in CHARTS.items():
            n = supports[chart]
            denominator = (
                len(group) if chart in {"volume_over_time", "average_activity"} else len(e)
            )
            status = "blocked" if n == 0 else "partial" if n < denominator else "supported"
            readiness.append(
                {
                    **scope,
                    "chart_id": chart,
                    "status": status,
                    "table": table,
                    "views": view,
                    "supported_candidates": n,
                    "denominator_candidates": denominator,
                    "coverage_rate": n / denominator if denominator else np.nan,
                    "blocker": None
                    if status == "supported"
                    else "required_feature_or_economics_unavailable",
                    "remediation": None
                    if status == "supported"
                    else remedies.get(
                        chart,
                        "Resolve diagnostic economic exclusions; unsupported products remain excluded.",
                    ),
                    "performance_threshold_applies": chart
                    not in {"volume_over_time", "average_activity"},
                    "scope_meets_performance_display_threshold": n >= cfg.min_performance_wagers,
                    "interpretation": "descriptive_not_validation_or_forecast",
                    "closing_series_matched_count": int(e.closing_price_ev_fraction.notna().sum()),
                    "closing_series_status": "available_on_matched_rows"
                    if e.closing_price_ev_fraction.notna().any()
                    else "blocked",
                    "closing_series_remediation": None
                    if e.closing_price_ev_fraction.notna().any()
                    else "Verify fair closing American-price semantics, reference source and timing before enabling the closing EV series.",
                    "source_clock_candidate_count": int(group.source_hour.notna().sum()),
                    "calendar_interpretation": "source_calendar_only_placement_and_coverage_unverified",
                }
            )
        count = (
            len(cohorts)
            + len(coverage)
            + len(readiness)
            + len(memberships)
            + sum(map(len, periods))
            + sum(map(len, activities))
            + len(work)
        )
        if count > cfg.max_rows:
            raise VisualBudgetExceeded("Visual output budget exceeded; no partial charts published")
    tables = {
        "visual_candidates": work.reset_index(drop=True),
        "visual_memberships": pd.DataFrame(memberships, columns=["scope_id", "bet_id"]),
        "visual_cohorts": pd.DataFrame(cohorts),
        "visual_periods": pd.concat(periods, ignore_index=True)
        if periods
        else pd.DataFrame(
            columns=["scope_id", "resolution", "platform", "period_start", "period_status"]
        ),
        "visual_activity": pd.concat(activities, ignore_index=True)
        if activities
        else pd.DataFrame(columns=["scope_id", "view", "category_json"]),
        "visual_coverage": pd.DataFrame(coverage),
        "visual_readiness": pd.DataFrame(readiness),
    }
    overall = tables["visual_cohorts"].loc[
        lambda f: f.scope_kind.eq("population") & f.view.eq("overall")
    ]
    if int(overall.candidate_count.sum()) != len(work) or int(overall.eligible_count.sum()) != int(
        work.diagnostic_eligible.sum()
    ):
        raise ValueError("Visual population economics do not reconcile")
    audit = {
        "candidate_count": len(work),
        "eligible_count": int(work.diagnostic_eligible.sum()),
        "scope_count": len(scopes),
        "scope_membership_count": len(memberships),
        "unresolved_or_missing_attribution_candidates": len(set(work.bet_id) - set(bridge.bet_id)),
        "table_rows": {key: len(value) for key, value in tables.items()},
        "chart_catalog": CHARTS,
        "time_basis": "source_wall_calendar_retrospective_not_settlement_or_bankroll",
        "activity_denominator": "active_source_days_in_bucket",
        "notional_unit_usd": cfg.fixed_unit_usd,
        "minimum_performance_wagers": cfg.min_performance_wagers,
        "intervals": "not_estimated",
        "hypothesis_tests_performed": False,
        "liquidity_source_verified": cfg.liquidity_usd_verified,
        "closing_reference_verified": cfg.closing_fair_american_verified,
        "placement_semantics_verified": cfg.placement_time_verified,
        "event_semantics_verified": cfg.event_time_verified,
    }
    # JSON roundtrip ensures tuples in manifest/audit replay use the same representation.
    return tables, json.loads(json.dumps(audit))
