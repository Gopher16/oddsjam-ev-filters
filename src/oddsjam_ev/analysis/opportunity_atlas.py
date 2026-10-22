"""
opportunity_atlas.py
===============================================================================
Purpose
-------
Describe logged wager candidates and opportunity proxies at separate grains.

What this module provides
-------------------------
Dimension tables, segment economics, non-additive filter diagnostics, and audits.

Design Notes
------------
No confidence or promotion labels are produced. Concentration is descriptive;
proxy groups do not establish independent observations. Missing liquidity and
unknown chronology cannot be repaired by aggregation.
===============================================================================
"""

import numpy as np
import pandas as pd

DIMENSIONS = ("sportsbook", "sport", "league", "market_family", "odds_band", "advertised_ev_band")
VIEWS = [(x,) for x in DIMENSIONS] + [
    ("sportsbook", "market_family"),
    ("league", "market_family"),
    ("odds_band", "advertised_ev_band"),
]


def numeric_band(values, edges, *, odds=False):
    """Assign left-closed bands, preserving missing and invalid input categories.

    Parameters
    ----------
    values : pandas.Series
        Source numeric strings.
    edges : sequence
        Strictly increasing finite boundaries.
    odds : bool
        Reject zero and American odds between -100 and +100.

    Returns
    -------
    pandas.Series
        Stable interval labels with explicit missing and invalid categories.
    """
    raw = values.fillna("").astype(str).str.strip()
    numbers = pd.to_numeric(raw, errors="coerce")
    result = pd.cut(numbers, [-np.inf, *edges, np.inf], right=False).astype(str)
    invalid = ~np.isfinite(numbers)
    if odds:
        invalid |= numbers.abs().lt(100)
    result = (
        result.mask(invalid, "<invalid>")
        .mask(raw.eq(""), "<missing>")
        .mask(raw.eq("<conflict>"), "<conflict>")
    )
    return result.mask(raw.eq("<conflict>"), "<conflict>")


def _dimensions(frame, cfg):
    result = frame.copy()
    for c in ("sportsbook", "sport", "league", "market_name"):
        result[c] = result[c].fillna("").astype(str).str.strip().replace("", "<missing>")
    result["market_family"] = result.market_name.map(cfg.market_families).fillna("<unmapped>")
    result.loc[result.market_name.eq("<missing>"), "market_family"] = "<missing>"
    result.loc[result.market_name.eq("<conflict>"), "market_family"] = "<conflict>"
    result["odds_band"] = numeric_band(result.odds, cfg.odds_edges, odds=True)
    result["advertised_ev_band"] = numeric_band(result.percentage, cfg.ev_edges)
    return result


def _consensus(values):
    unique = values.fillna("").astype(str).unique()
    return unique[0] if len(unique) == 1 else "<conflict>"


def summarize_wagers(bets, dimensions):
    """Aggregate the identical eligible stake/profit cohort within each slice.

    Parameters
    ----------
    bets : pandas.DataFrame
        Unique canonical wagers with derived dimensions.
    dimensions : sequence[str]
        Explicit slice columns; execution and product are always separated.

    Returns
    -------
    pandas.DataFrame
        Descriptive economics and stake/profit concentration, without inference.
    """
    if set(dimensions) - set((*DIMENSIONS, "filter_id", "attribution_status")):
        raise ValueError("Unsupported atlas dimensions")
    if bets.bet_id.duplicated().any():
        # Expanded filter memberships are valid only when filter is a slice key.
        if (
            "filter_id" not in dimensions
            or bets.duplicated(["bet_id", "filter_id", "attribution_status"]).any()
        ):
            raise ValueError("Duplicate wager identities in slice")
    keys = list(dict.fromkeys(["execution_status", "product", *dimensions]))
    rows = []
    for key, g in bets.groupby(keys, dropna=False, sort=True):
        e = g[g.reported_roi_eligible]
        stake = e.reported_stake_usd.sum(min_count=1)
        profit = e.reported_profit_usd.sum(min_count=1)
        absolute = e.reported_profit_usd.abs().sum(min_count=1)
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "candidate_count": len(g),
                "eligible_count": len(e),
                "excluded_count": len(g) - len(e),
                "ticket_candidate_count": int(g.fact_grain.eq("ticket_candidate").sum()),
                "eligible_stake_usd": stake,
                "eligible_reported_profit_usd": profit,
                "reported_roi": profit / stake if pd.notna(stake) and stake > 0 else np.nan,
                "largest_stake_share": e.reported_stake_usd.max() / stake
                if pd.notna(stake) and stake > 0
                else np.nan,
                "largest_absolute_profit_share": e.reported_profit_usd.abs().max() / absolute
                if pd.notna(absolute) and absolute > 0
                else np.nan,
                "eligible_event_id_count": e.game_id.replace("", pd.NA).nunique(),
                "eligible_selection_proxy_count": e.selection_proxy_id.nunique(),
                "eligible_advertised_ev_present_count": int(
                    (~e.advertised_ev_band.isin(["<missing>", "<invalid>", "<conflict>"])).sum()
                ),
                "eligible_missing_event_id_count": int(e.game_id.fillna("").eq("").sum()),
                "currency": "USD",
                "fee_inclusion": "unverified",
                "uncertainty_status": "not_estimated_dependence_unverified",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            *keys,
            "candidate_count",
            "eligible_count",
            "excluded_count",
            "ticket_candidate_count",
            "eligible_stake_usd",
            "eligible_reported_profit_usd",
            "reported_roi",
            "largest_stake_share",
            "largest_absolute_profit_share",
            "eligible_event_id_count",
            "eligible_selection_proxy_count",
            "eligible_advertised_ev_present_count",
            "eligible_missing_event_id_count",
            "currency",
            "fee_inclusion",
            "uncertainty_status",
        ],
    )


def build_atlas(tables, cfg):
    """Build separate descriptive tables from verified canonical inputs.

    Parameters
    ----------
    tables : dict[str, pandas.DataFrame]
        Canonical bundle tables, loaded through its integrity-checking reader.
    cfg : AtlasConfig
        Explicit descriptive policy.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dimensions, segments, attribution diagnostics, and quality audit.
    """
    bets = tables["bet_facts"].sort_values("bet_id").reset_index(drop=True).copy()
    links = tables["bet_opportunity_links"].copy()
    sources = tables["source_features"]
    opportunities = (
        tables["opportunity_facts"].sort_values("opportunity_proxy_id").reset_index(drop=True)
    )
    if not set(links.opportunity_proxy_id) == set(opportunities.opportunity_proxy_id):
        raise ValueError("Atlas opportunity coverage does not reconcile")
    if not set(links.source_observation_id) == set(sources._ingest_observation_id):
        raise ValueError("Atlas source links do not reconcile")
    if (
        bets.bet_id.duplicated().any()
        or opportunities.opportunity_proxy_id.duplicated().any()
        or links.source_observation_id.duplicated().any()
    ):
        raise ValueError("Atlas input identities must be unique")
    if not set(links.bet_id) <= set(bets.bet_id) or not set(links.opportunity_proxy_id) <= set(
        opportunities.opportunity_proxy_id
    ):
        raise ValueError("Atlas links contain orphan identities")
    wagers = _dimensions(bets, cfg)
    selection_links = links.merge(
        opportunities[["opportunity_proxy_id", "selection_proxy_id"]],
        on="opportunity_proxy_id",
        validate="many_to_one",
    )
    selections = selection_links.groupby("bet_id")["selection_proxy_id"].agg(_consensus)
    wagers["selection_proxy_id"] = wagers.bet_id.map(selections).replace("<conflict>", None)
    if not set(wagers.execution_status) <= {"real_money", "paper", "unknown"}:
        raise ValueError("Unknown execution status")
    # Original source metadata is joined one-to-one before consensus; never money.
    source_cols = ["sportsbook", "sport", "league", "market_name", "odds", "percentage"]
    observed = links.merge(
        sources.reindex(columns=["_ingest_observation_id", *source_cols], fill_value=""),
        left_on="source_observation_id",
        right_on="_ingest_observation_id",
        validate="one_to_one",
    )
    if len(observed) != len(links):
        raise ValueError("Atlas source links do not reconcile")
    metadata = (
        observed.groupby("opportunity_proxy_id", sort=True)[source_cols]
        .agg(_consensus)
        .reset_index()
    )
    proxy = _dimensions(
        opportunities[["opportunity_proxy_id", "selection_proxy_id", "identity_quality"]].merge(
            metadata, on="opportunity_proxy_id", validate="one_to_one"
        ),
        cfg,
    )
    # One proxy per execution mode. Counts across modes are explicitly non-additive.
    modes = links.merge(bets[["bet_id", "execution_status"]], on="bet_id", validate="many_to_one")[
        ["opportunity_proxy_id", "execution_status"]
    ].drop_duplicates()
    proxy_modes = proxy.merge(modes, on="opportunity_proxy_id", validate="one_to_many")
    mode_links = links.merge(
        bets[["bet_id", "execution_status"]], on="bet_id", validate="many_to_one"
    )
    link_counts = (
        mode_links.groupby(["opportunity_proxy_id", "execution_status"], sort=True)
        .size()
        .rename("linked_source_record_count")
        .reset_index()
    )
    proxy_modes = proxy_modes.merge(
        link_counts, on=["opportunity_proxy_id", "execution_status"], validate="one_to_one"
    )
    proxy_modes["ev_present"] = ~proxy_modes.advertised_ev_band.isin(
        ["<missing>", "<invalid>", "<conflict>"]
    )
    proxy["linked_source_record_count"] = proxy.opportunity_proxy_id.map(
        links.groupby("opportunity_proxy_id").size()
    )
    wager_views = []
    proxy_views = []
    for dims in [(), *VIEWS]:
        name = "overall" if not dims else "__".join(dims)
        w = summarize_wagers(wagers, dims)
        w["view"] = name
        wager_views.append(w)
        keys = ["execution_status", *dims]
        p = (
            proxy_modes.groupby(keys, dropna=False, sort=True)
            .agg(
                opportunity_proxy_count=("opportunity_proxy_id", "nunique"),
                selection_proxy_count=("selection_proxy_id", "nunique"),
                linked_source_record_count=("linked_source_record_count", "sum"),
                advertised_ev_present_proxy_count=("ev_present", "sum"),
            )
            .reset_index()
        )
        p["advertised_ev_coverage_rate"] = (
            p.advertised_ev_present_proxy_count / p.opportunity_proxy_count
        )
        p["view"] = name
        p["counts_additive_across_modes"] = False
        proxy_views.append(p)
    attribution = tables["filter_attribution"][
        ["bet_id", "filter_id", "attribution_status"]
    ].drop_duplicates()
    attribution["filter_id"] = attribution.filter_id.fillna("").replace(
        "", "<unresolved_or_missing>"
    )
    member = wagers.merge(attribution, on="bet_id", validate="one_to_many")
    filters = []
    for dims in [
        ("filter_id", "attribution_status"),
        ("filter_id", "attribution_status", "market_family"),
        ("filter_id", "attribution_status", "sportsbook"),
    ]:
        f = summarize_wagers(member, dims)
        f["view"] = "__".join(dims)
        f["additive_across_filters"] = False
        filters.append(f)
    quality = []
    for population, frame in [("wager", wagers), ("opportunity_proxy", proxy)]:
        for col in (*DIMENSIONS, "market_name"):
            for value, count in frame[col].value_counts(dropna=False).sort_index().items():
                quality.append(
                    {
                        "population": population,
                        "dimension": col,
                        "category": str(value),
                        "count": int(count),
                        "denominator": len(frame),
                    }
                )
    if "source_audit" in tables:
        for value, count in (
            tables["source_audit"].canonical_disposition.value_counts().sort_index().items()
        ):
            quality.append(
                {
                    "population": "source_observation",
                    "dimension": "canonical_disposition",
                    "category": value,
                    "count": int(count),
                    "denominator": len(tables["source_audit"]),
                }
            )
    for value, count in (
        bets.economic_exclusions_json.value_counts().sort_index().items()
        if "economic_exclusions_json" in bets
        else []
    ):
        quality.append(
            {
                "population": "wager",
                "dimension": "economic_exclusions_json",
                "category": value,
                "count": int(count),
                "denominator": len(bets),
            }
        )
    result = {
        "wager_dimensions": wagers,
        "opportunity_dimensions": proxy,
        "wager_segments": pd.concat(wager_views, ignore_index=True),
        "opportunity_segments": pd.concat(proxy_views, ignore_index=True),
        "filter_segment_diagnostics": pd.concat(filters, ignore_index=True),
        "atlas_quality": pd.DataFrame(
            quality, columns=["population", "dimension", "category", "count", "denominator"]
        ),
    }
    return result
