"""
preprocess_bet_tracker.py
===============================================================================

Purpose
-------
Preprocess an OddsJam Bet Tracker CSV export into analytics-ready parquet files
and QA artifacts.

This script:
  1) Loads the raw CSV (OddsJam export).
  2) Standardizes timestamps to America/New_York, preserving the fact that the
     source CSV is already in ET (tz-naive values are localized to ET).
  3) Applies optional filters:
       - cutoff_date
       - filter-name normalization (fill missing saved_filter_names)
  4) Adds feature engineering:
       - liquidity_bucket (optionally condensed by config)
       - time_to_event_bucket
       - EV-related fields (probabilities, multipliers, EV, EV ROI)
  5) Writes QA artifacts:
       - field validation JSON
       - duplicate audit CSV
       - liquidity metadata JSON
       - status summary CSV + settled-status inference JSON (optional)
  6) Writes outputs:
       - processed parquet (canonical dataset)
       - settled-only parquet (optional convenience output)

Status / Settlement Design
--------------------------
The script supports two independent behaviors:

A) Inferring settled statuses (to build df_settled)
   - Controlled by: status_settlement.exclude_statuses_for_settled_inference

B) Excluding unresolved statuses from the processed output too
   - Controlled by: status_settlement.exclude_statuses_from_processed (bool)

Usage
-----
python scripts/preprocess_bet_tracker.py --config configs/preprocess.yaml
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from oddsjam_ev.dedupe import dedupe_artifacts
from oddsjam_ev.io.load import load_bet_tracker_csv, standardize_timestamp_to_et
from oddsjam_ev.liquidity import add_liquidity_bucket
from oddsjam_ev.metrics.odds import (
    american_to_multiplier,
    american_to_prob,
    compute_ev,
    compute_ev_roi,
)
from oddsjam_ev.qa import duplicate_audit, validate_core_fields


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_lower(x: Any) -> str:
    return str(x).strip().lower()


def _get_inference_exclude_statuses(status_cfg: dict[str, Any]) -> set[str]:
    """
    Resolve the list of statuses that should be excluded from SETTLED inference.

    Priority:
      1) exclude_statuses_for_settled_inference (new)
      2) default {"pending","open","unsettled"}
    """
    raw = status_cfg.get("exclude_statuses_for_settled_inference")
    if raw is None:
        raw = ["pending", "open", "unsettled"]
    return {str(x).strip().lower() for x in raw}


def _infer_settled_statuses(
    df: pd.DataFrame,
    *,
    status_col: str = "status",
    stake_col: str = "stake",
    profit_col: str = "bet_profit",
    profit_known_threshold: float = 0.95,
    exclude_statuses: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build status_summary and infer settled statuses by profit-known rate.

    Returns
    -------
    status_summary_df, settled_statuses
    """
    if exclude_statuses is None:
        exclude_statuses = {"pending", "open", "unsettled"}

    if status_col not in df.columns:
        status_summary = pd.DataFrame(
            columns=["status", "n_bets", "profit_known_rate", "avg_profit", "avg_stake"]
        )
        return status_summary, []

    tmp = df.copy()
    tmp["_profit_known"] = pd.to_numeric(tmp.get(profit_col), errors="coerce").notna()

    status_summary = (
        tmp.groupby(status_col, dropna=False)
        .agg(
            n_bets=(stake_col, "count"),
            profit_known_rate=("_profit_known", "mean"),
            avg_profit=(profit_col, "mean"),
            avg_stake=(stake_col, "mean"),
        )
        .sort_values("n_bets", ascending=False)
        .reset_index()
        .rename(columns={status_col: "status"})
    )

    settled: list[str] = []
    for _, row in status_summary.iterrows():
        s = row["status"]
        if pd.isna(s):
            continue
        s_lower = _safe_lower(s)
        if s_lower in exclude_statuses:
            continue
        if float(row["profit_known_rate"]) > profit_known_threshold:
            settled.append(str(s))

    return status_summary, settled


def _build_settled_mask(
    df: pd.DataFrame, *, status_col: str, settled_statuses: list[str]
) -> pd.Series:
    """
    Robust settled mask:
    - handles casing/whitespace
    - treats NaN as NOT settled
    """
    if status_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    status_norm = df[status_col].astype("string").str.strip().str.lower()
    settled_norm = {str(s).strip().lower() for s in settled_statuses}
    return status_norm.isin(settled_norm).fillna(False)


def _add_time_to_event_bucket(
    df: pd.DataFrame,
    *,
    created_col: str,
    event_start_col: str,
    out_col: str = "time_to_event_bucket",
    tz: str = "America/New_York",
    label_after_start: str = "after_start",
) -> pd.DataFrame:
    """
    Add out_col that buckets time-to-event, defined as:
        event_start_datetime - created_at_datetime

    Handles DST ambiguous/nonexistent timestamps safely.
    """
    out = df.copy()

    if created_col not in out.columns:
        raise KeyError(f"Missing created timestamp column: {created_col}")
    if event_start_col not in out.columns:
        out[out_col] = np.nan
        return out

    created = pd.to_datetime(out[created_col], errors="coerce")
    event_start = pd.to_datetime(out[event_start_col], errors="coerce")

    def _to_et(ts: pd.Series) -> pd.Series:
        if isinstance(ts.dtype, pd.DatetimeTZDtype):
            return ts.dt.tz_convert(tz)
        try:
            return ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            return ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")

    created = _to_et(created)
    event_start = _to_et(event_start)

    delta_hours = (event_start - created).dt.total_seconds() / 3600.0

    bins = [-np.inf, 1, 4, 12, 24, 48, 72, 168, 336, np.inf]
    labels = [
        "<= 1 hour",
        "2 - 4 hours",
        "4 - 12 hours",
        "12 - 24 hours",
        "1 - 2 days",
        "2 - 3 days",
        "3 - 7 days",
        "7 - 14 days",
        "> 14 days",
    ]

    bucketed = pd.cut(delta_hours, bins=bins, labels=labels, right=True, include_lowest=True)

    after_mask = delta_hours < 0
    bucketed = bucketed.astype("object")
    bucketed[after_mask] = label_after_start

    out[out_col] = bucketed
    return out


# -----------------------------------------------------------------------------
# Liquidity condensation helpers
# -----------------------------------------------------------------------------
def _parse_bucket_bounds(label: Any) -> tuple[float | None, float | None]:
    """
    Parse labels like:
      "<=500", "<= 500"
      "500-1k", "500 - 1K"
      "10k-25k"
      "> 1M"
    Returns (lower, upper) in dollars where possible.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None, None

    t = str(label).strip().lower().replace(" ", "")

    m = re.match(r"^<=([\d\.]+)(k|m)?$", t)
    if m:
        v = float(m.group(1))
        mult = m.group(2)
        if mult == "k":
            v *= 1000.0
        elif mult == "m":
            v *= 1_000_000.0
        return None, v

    m = re.match(r"^>([\d\.]+)(k|m)?$", t)
    if m:
        v = float(m.group(1))
        mult = m.group(2)
        if mult == "k":
            v *= 1000.0
        elif mult == "m":
            v *= 1_000_000.0
        return v, None

    m = re.match(r"^([\d\.]+)(k|m)?-([\d\.]+)(k|m)?$", t)
    if m:
        a = float(m.group(1))
        am = m.group(2)
        b = float(m.group(3))
        bm = m.group(4)

        if am == "k":
            a *= 1000.0
        elif am == "m":
            a *= 1_000_000.0

        if bm == "k":
            b *= 1000.0
        elif bm == "m":
            b *= 1_000_000.0

        return a, b

    return None, None


def _condense_liquidity(
    df: pd.DataFrame,
    *,
    liquidity_col: str,
    in_bucket_col: str,
    out_bucket_col: str,
    edges: list[float],
    labels: list[str],
) -> pd.DataFrame:
    """
    Create condensed liquidity buckets.

    Strategy
    --------
    1) If numeric liquidity is available, bucket by that.
    2) Else, attempt to parse the existing bucket label bounds and bucket by mid/lower bound.
    3) Else, leave as NaN.

    Notes
    -----
    - edges/labels define pd.cut bins. Must satisfy len(edges) == len(labels) + 1.
    - edges should be increasing (can include -inf/inf).
    """
    if len(edges) != len(labels) + 1:
        raise ValueError(
            f"Invalid condensed liquidity config: len(edges)={len(edges)} must equal len(labels)+1={len(labels) + 1}"
        )

    out = df.copy()

    # numeric liquidity
    liq_num = pd.to_numeric(out.get(liquidity_col), errors="coerce")

    # fallback numeric from existing bucket label
    if in_bucket_col in out.columns:
        bounds = out[in_bucket_col].apply(_parse_bucket_bounds)
        lower = bounds.apply(lambda x: x[0] if x is not None else None)
        upper = bounds.apply(lambda x: x[1] if x is not None else None)

        lower_num = pd.to_numeric(lower, errors="coerce")
        upper_num = pd.to_numeric(upper, errors="coerce")

        # choose representative value:
        # - if we have both bounds: midpoint
        # - if only upper: use upper
        # - if only lower: use lower
        rep = np.where(
            lower_num.notna() & upper_num.notna(),
            (lower_num + upper_num) / 2.0,
            np.where(upper_num.notna(), upper_num, np.where(lower_num.notna(), lower_num, np.nan)),
        )
        rep = pd.to_numeric(pd.Series(rep, index=out.index), errors="coerce")
    else:
        rep = pd.Series([np.nan] * len(out), index=out.index)

    base_val = liq_num.where(liq_num.notna(), rep)

    condensed = pd.cut(
        base_val,
        bins=edges,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    out[out_bucket_col] = condensed.astype("object")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess OddsJam Bet Tracker export -> parquet + QA artifacts"
    )
    ap.add_argument(
        "--config", required=True, help="Path to YAML config (e.g. configs/preprocess.yaml)"
    )
    args = ap.parse_args()

    cfg = _read_yaml(Path(args.config))

    raw_csv = Path(cfg["raw_csv_path"])
    out_dir = _ensure_dir(Path(cfg.get("output_dir", "data/processed")))
    artifacts_dir = _ensure_dir(Path(cfg.get("artifacts_dir", out_dir / "artifacts")))

    run_id = cfg.get("run_id") or _stamp()

    as_of_date = cfg.get("as_of_date")
    if not as_of_date:
        raise ValueError("configs/preprocess.yaml must set as_of_date (e.g., '02-06-2026').")

    # --------------------------
    # Load raw
    # --------------------------
    df = load_bet_tracker_csv(raw_csv)

    # --------------------------
    # Timestamp standardization
    # --------------------------
    ts_cfg = cfg.get("timestamps", {})
    ts_src = ts_cfg.get("src_col", "created_at")
    ts_out = ts_cfg.get("out_col", "created_at_et")

    df = standardize_timestamp_to_et(
        df,
        src_col=ts_src,
        out_col=ts_out,
        assume_tz_for_naive=ts_cfg.get("assume_tz_for_naive", "America/New_York"),
    )

    if ts_cfg.get("also_write_created_at_est", True):
        df["created_at_est"] = df[ts_out]

    # --------------------------
    # Optional date filter
    # --------------------------
    cutoff = cfg.get("cutoff_date")
    if cutoff:
        tz = "America/New_York"
        cutoff_ts = pd.to_datetime(cutoff, errors="raise")
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize(tz)
        else:
            cutoff_ts = cutoff_ts.tz_convert(tz)
        df = df[df[ts_out] >= cutoff_ts].copy()

    # --------------------------
    # Standardize filter names
    # --------------------------
    filter_cfg = cfg.get("filters", {})
    filter_col = filter_cfg.get("filter_col", "saved_filter_names")
    null_label = filter_cfg.get("null_label", "No Filter")

    if filter_col in df.columns:
        df[filter_col] = df[filter_col].fillna(null_label)
    else:
        df[filter_col] = null_label

    fantasy = filter_cfg.get("fantasy_optimizer_fill", {})
    if fantasy.get("enabled", False):
        stake_col = fantasy.get("stake_col", "stake")
        max_stake = float(fantasy.get("max_stake", 20))
        label = fantasy.get("label", "TEST - Fantasy Optimizer")

        stake_num = pd.to_numeric(df.get(stake_col), errors="coerce")
        mask = df[filter_col].isna() & stake_num.le(max_stake)
        df.loc[mask, filter_col] = label

    # --------------------------
    # Liquidity bucket (base) + optional condensation
    # --------------------------
    liq_cfg = cfg.get("liquidity", {})
    liquidity_col = liq_cfg.get("liquidity_col", "liquidity")
    tags_col = liq_cfg.get("tags_col", "tags")
    out_bucket_col = liq_cfg.get("out_bucket_col", "liquidity_bucket")

    df, liq_meta = add_liquidity_bucket(
        df,
        liquidity_col=liquidity_col,
        tags_col=tags_col,
        out_bucket_col=out_bucket_col,
    )

    condensed_cfg = (liq_cfg.get("condensed") or {}) if isinstance(liq_cfg, dict) else {}
    if condensed_cfg.get("enabled", False):
        labels = list(
            condensed_cfg.get("labels", ["<=500", "500-1k", "1k-2k", "2k-5k", "5k-10k", "> 10k"])
        )
        edges_raw = list(
            condensed_cfg.get("edges", [-np.inf, 500, 1000, 2000, 5000, 10000, np.inf])
        )

        # normalize YAML inf/-inf tokens
        def _edge(x: Any) -> float:
            if isinstance(x, str):
                t = x.strip().lower()
                if t in {"inf", "+inf", "infinity", "+infinity"}:
                    return float("inf")
                if t in {"-inf", "-infinity"}:
                    return float("-inf")
            return float(x)

        edges = [_edge(x) for x in edges_raw]

        df = _condense_liquidity(
            df,
            liquidity_col=liquidity_col,
            in_bucket_col=out_bucket_col,
            out_bucket_col=out_bucket_col,
            edges=edges,
            labels=labels,
        )

        liq_meta = {
            **(liq_meta or {}),
            "condensed": {
                "enabled": True,
                "labels": labels,
                "edges": edges,
            },
        }

    # --------------------------
    # Time-to-event bucket
    # --------------------------
    tte_cfg = cfg.get("time_to_event", {})
    if tte_cfg.get("enabled", True):
        df = _add_time_to_event_bucket(
            df,
            created_col=tte_cfg.get("created_col", ts_out),
            event_start_col=tte_cfg.get("event_start_col", "event_start_date"),
            out_col=tte_cfg.get("out_col", "time_to_event_bucket"),
            tz=tte_cfg.get("tz", "America/New_York"),
            label_after_start=tte_cfg.get("label_after_start", "after_start"),
        )

    # --------------------------
    # QA artifacts
    # --------------------------
    qa_cfg = cfg.get("qa", {})
    qa_stake = qa_cfg.get("stake_col", "stake")
    qa_odds = qa_cfg.get("odds_col", "odds")
    qa_profit = qa_cfg.get("profit_col", "bet_profit")

    field_val = validate_core_fields(df, stake_col=qa_stake, odds_col=qa_odds, profit_col=qa_profit)

    dupe_keys = qa_cfg.get("duplicate_keys", [])
    dupe_report = duplicate_audit(df, keys=dupe_keys) if dupe_keys else pd.DataFrame()

    # --------------------------
    # Optional artifact-dedupe
    # --------------------------
    dedupe_cfg = cfg.get("dedupe_artifacts", {})
    if dedupe_cfg.get("enabled", False):
        df = dedupe_artifacts(
            df,
            identity_cols=dedupe_cfg.get("identity_cols", []),
            time_col=dedupe_cfg.get("time_col", ts_out),
            stake_col=dedupe_cfg.get("stake_col", qa_stake),
            filter_col=dedupe_cfg.get("filter_col", filter_col),
            strategy=dedupe_cfg.get("strategy", "earliest"),
            scope=dedupe_cfg.get("scope", "within_filter"),
        )

    # --------------------------
    # EV feature engineering
    # --------------------------
    ev_cfg = cfg.get("ev_features", {})
    if ev_cfg.get("enabled", True):
        odds_col = ev_cfg.get("odds_col", "odds")
        clv_col = ev_cfg.get("clv_col", "clv")
        stake_col = ev_cfg.get("stake_col", "stake")

        if odds_col in df.columns:
            df["prob_odds"] = df[odds_col].apply(american_to_prob)
            df["odds_multiplier"] = df[odds_col].apply(american_to_multiplier)
        else:
            df["prob_odds"] = np.nan
            df["odds_multiplier"] = np.nan

        if clv_col in df.columns:
            df["prob_clv"] = df[clv_col].apply(american_to_prob)
        else:
            df["prob_clv"] = np.nan

        df["ev"] = compute_ev(
            stake=pd.to_numeric(df.get(stake_col), errors="coerce"),
            prob=pd.to_numeric(df.get("prob_clv"), errors="coerce"),
            odds_multiplier=pd.to_numeric(df.get("odds_multiplier"), errors="coerce"),
        )
        df["ev_roi"] = compute_ev_roi(
            ev=pd.to_numeric(df.get("ev"), errors="coerce"),
            stake=pd.to_numeric(df.get(stake_col), errors="coerce"),
        )

    # --------------------------
    # Status -> infer settled -> df_settled
    # --------------------------
    status_cfg = cfg.get("status_settlement", {})
    status_enabled = status_cfg.get("enabled", True)
    status_col = status_cfg.get("status_col", "status")

    status_summary = pd.DataFrame()
    settled_statuses: list[str] = []
    df_settled = pd.DataFrame()
    settled_mask = pd.Series([False] * len(df), index=df.index)

    if status_enabled:
        threshold = float(status_cfg.get("profit_known_threshold", 0.95))
        exclude_for_inference = _get_inference_exclude_statuses(status_cfg)

        status_summary, settled_statuses = _infer_settled_statuses(
            df,
            status_col=status_col,
            stake_col=status_cfg.get("stake_col", "stake"),
            profit_col=status_cfg.get("profit_col", "bet_profit"),
            profit_known_threshold=threshold,
            exclude_statuses=exclude_for_inference,
        )

        if settled_statuses and status_col in df.columns:
            settled_mask = _build_settled_mask(
                df, status_col=status_col, settled_statuses=settled_statuses
            )
            df_settled = df[settled_mask].copy()

    # --------------------------
    # OPTIONAL: exclude unresolved statuses from *processed* output too
    # --------------------------
    exclude_from_processed = bool(status_cfg.get("exclude_statuses_from_processed", False))
    if exclude_from_processed and status_col in df.columns:
        exclude = _get_inference_exclude_statuses(status_cfg)
        if exclude:
            status_norm = df[status_col].astype("string").str.strip().str.lower()
            df = df[~status_norm.isin(exclude)].copy()

            if isinstance(df_settled, pd.DataFrame) and not df_settled.empty:
                df_settled = df_settled.loc[df_settled.index.intersection(df.index)].copy()
                settled_mask = settled_mask.loc[settled_mask.index.intersection(df.index)]

    # --------------------------
    # Write outputs (parquet)
    # --------------------------
    out_cfg = cfg.get("output", {})
    name_template = out_cfg.get(
        "name_template", "oddsjam-bet-tracker-processed-{as_of_date}.parquet"
    )
    out_parquet = out_dir / name_template.format(as_of_date=as_of_date, run_id=run_id)
    df.to_parquet(out_parquet, index=False)

    stable_name = out_cfg.get("stable_name")
    stable_path = None
    if stable_name:
        stable_path = out_dir / stable_name
        df.to_parquet(stable_path, index=False)

    settled_template = out_cfg.get("settled_name_template")
    settled_path = None
    wrote_settled = False

    n_processed = int(len(df))
    n_settled = int(len(df_settled)) if isinstance(df_settled, pd.DataFrame) else 0

    processed_all_settled = (
        bool(settled_mask.all())
        if (status_enabled and status_col in df.columns and len(df) > 0)
        else False
    )

    if (
        status_enabled
        and status_col in df.columns
        and settled_template
        and n_settled > 0
        and (n_settled != n_processed)
    ):
        settled_path = out_dir / settled_template.format(as_of_date=as_of_date, run_id=run_id)
        df_settled.to_parquet(settled_path, index=False)
        wrote_settled = True

    # --------------------------
    # Write artifacts
    # --------------------------
    (artifacts_dir / f"field-validation-{as_of_date}.json").write_text(
        json.dumps(asdict(field_val), indent=2)
    )
    if not dupe_report.empty:
        dupe_report.to_csv(artifacts_dir / f"duplicate-audit-{as_of_date}.csv", index=False)
    (artifacts_dir / f"liquidity-meta-{as_of_date}.json").write_text(json.dumps(liq_meta, indent=2))

    if status_enabled:
        if not status_summary.empty:
            status_summary.to_csv(artifacts_dir / f"status-summary-{as_of_date}.csv", index=False)

        (artifacts_dir / f"settled-statuses-{as_of_date}.json").write_text(
            json.dumps(
                {
                    "as_of_date": as_of_date,
                    "run_id": run_id,
                    "profit_known_threshold": status_cfg.get("profit_known_threshold", 0.95),
                    "exclude_statuses_for_settled_inference": sorted(
                        list(_get_inference_exclude_statuses(status_cfg))
                    ),
                    "exclude_statuses_from_processed": exclude_from_processed,
                    "settled_statuses": settled_statuses,
                    "n_rows_processed": n_processed,
                    "n_rows_settled": n_settled,
                    "processed_all_settled": processed_all_settled,
                    "settled_written": wrote_settled,
                },
                indent=2,
            )
        )

    # --------------------------
    # Console output
    # --------------------------
    print("✅ Preprocess complete")
    print(f"  rows (processed): {n_processed:,}")
    print(f"  rows (settled):   {n_settled:,}")
    print(f"  parquet:          {out_parquet.resolve()}")
    if stable_path is not None:
        print(f"  stable:           {stable_path.resolve()}")

    if wrote_settled and settled_path is not None:
        print(f"  settled parquet:  {settled_path.resolve()}")
    else:
        if (
            status_enabled
            and status_col in df.columns
            and n_settled == n_processed
            and n_processed > 0
        ):
            print("  settled parquet:  (skipped; processed and settled are identical)")
        elif status_enabled and n_settled == 0:
            print("  settled parquet:  (skipped; no settled rows inferred)")
        else:
            print("  settled parquet:  (skipped)")

    print(f"  artifacts_dir:    {artifacts_dir.resolve()}")
    print("  liquidity_meta:", liq_meta)
    if status_enabled:
        print(f"  settled_statuses ({len(settled_statuses)}): {settled_statuses}")
        print(f"  exclude_from_processed: {exclude_from_processed}")
    print(f"  artifacts suffix: {as_of_date}")


if __name__ == "__main__":
    main()
