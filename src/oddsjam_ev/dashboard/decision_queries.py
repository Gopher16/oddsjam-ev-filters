"""Display-only queue and chart selection; never fit, pool or invent intersections."""

import json
import re
from decimal import Decimal

import pandas as pd

from oddsjam_ev.analysis.paper_definitions import draft_definition
from oddsjam_ev.dashboard.labels import DIMENSIONS, pretty

TIERS = {
    "corrected_historical_comparison": "Corrected historical comparison",
    "support_qualified_exploratory": "Exploratory · support-qualified",
    "insufficient_or_conflicting": "Insufficient or conflicting",
}
FIELDS = {**DIMENSIONS, "advertised_ev_percent": "Exact EV threshold"}


def terms(record):
    """Return exact published constraints; omitted fields remain absent."""
    return {
        p["dimension"]: str(
            p.get("value", p.get("operator", "") + " " + str(p.get("endpoint", "")))
        )
        for p in json.loads(record["predicate_json"])
    }


def normalized(value):
    """Canonicalize numeric display endpoints without changing interval operators."""
    text = str(value)
    if not text.startswith(("[", "(", ">", "<")):
        return text
    return re.sub(r"-?\d+(?:\.\d+)?", lambda m: str(Decimal(m[0]).normalize()), text).replace(
        ", ", ","
    )


def title(record):
    """Describe the analytical slice without implying a complete scanner setup."""
    return " · ".join(
        FIELDS.get(k, k) + ": " + (v + "%" if k == "advertised_ev_percent" else pretty(v, k))
        for k, v in terms(record).items()
    )


def select_records(
    records,
    *,
    mode="all",
    product="all",
    saved="all",
    minimum=100,
    selections=None,
    tiers=(),
    setup=(),
    lifecycle=(),
):
    """Apply AND across explicit fields, OR within, preserving original queue order."""
    selections = selections or {}
    if set(selections) - set(FIELDS):
        raise ValueError("Unsupported workspace field")
    result = []
    for r in records:
        if (
            mode != "all"
            and r["execution_status"] != mode
            or product != "all"
            and r["product"] != product
            or saved != "all"
            and r["filter_id"] != saved
        ):
            continue
        if r["historical_economics"]["subgroup"]["eligible_count"] < minimum:
            continue
        if tiers and r["evidence_tier"] not in tiers or setup and r["setup_status"] not in setup:
            continue
        if lifecycle and r["configuration_sheet"]["lifecycle"]["status"] not in lifecycle:
            continue
        values = terms(r)
        if all(
            not chosen or any(normalized(values.get(d)) == normalized(v) for v in chosen)
            for d, chosen in selections.items()
        ):
            result.append(r)
    return result


def chart_rows(workspace, table, *, mode="all", product="all", saved="all", selections=None):
    """Select exact persisted cohort rows; arbitrary raw-wager aggregation is forbidden."""
    frame = workspace["visuals"][table]
    if frame.empty:
        return frame.copy()
    keep = frame.scope_kind.eq("population" if saved == "all" else "filter")
    if saved != "all":
        keep &= frame.filter_id.eq(saved)
    if mode != "all":
        keep &= frame.execution_status.eq(mode)
    if product != "all":
        keep &= frame["product"].eq(product)
    frame = frame.loc[keep].copy()
    selections = {d: v for d, v in (selections or {}).items() if v}
    if selections:
        if table == "visual_periods" and set(selections) <= {"sportsbook"}:
            # These platform rows are already published; never sum into a new cohort.
            return frame.loc[frame.platform.isin(selections["sportsbook"])].copy()
        if "category_json" not in frame:
            return frame.iloc[:0]
        aliases = {"ev_bucket": "advertised_ev_band"}

        def match(raw):
            values = {aliases.get(k, k): str(v) for k, v in json.loads(raw).items()}
            return all(
                any(normalized(values.get(k)) == normalized(v) for v in chosen)
                for k, chosen in selections.items()
            )

        frame = frame.loc[frame.category_json.map(match)]
    return frame


def visible_headlines(records, queue, limit=4):
    """Keep published representatives; filtering does not silently rerank or refill."""
    return [r for r in records if r["queue"] == queue and r["headline"]][:limit]


def study_draft(record, workspace):
    """Create a source-pinned incomplete draft or explicit global-catalogue handoff."""
    if record["source_kind"] == "global_descriptive_catalogue":
        return {
            "kind": "global_study_brief_not_registerable_definition",
            "candidate_id": record["candidate_id"],
            "predicate_json": record["predicate_json"],
            "queue_publication_id": workspace["queues"]["manifest"]["publication_id"],
            "configuration_sheet": record["configuration_sheet"],
            "planned_start_utc": None,
            "funding": "paper_only",
            "max_funded_stake_usd": 0,
            "required_next_step": "Resolve new prospective baseline and publish supported definition integration; no launch or registration.",
        }
    draft = draft_definition(record)
    sheet = record["configuration_sheet"]
    draft["protocol"]["design"] = sheet["design"]
    if sheet["baseline"]:
        draft["protocol"]["baseline"] = {
            "kind": "unchanged_parent",
            "filter_config": {r["control"]: r["baseline_value"] for r in sheet["settings"]},
            "evidence_reference": sheet["baseline"]["evidence_reference"],
        }
    if (
        sheet.get("decision_id")
        and sheet["setup_status"] == "ready_for_operator_configuration_review"
    ):
        draft["protocol"]["configuration_reference"] = {
            "contract_version": 1,
            "publication_id": workspace["queues"]["manifest"]["specification"][
                "input_publication_ids"
            ]["decisions"],
            "decision_id": sheet["decision_id"],
            "record_sha256": sheet["record_sha256"],
        }
    return draft


def comparison_frame(records):
    """Paired level/contrast display rows, with distinct source population and units."""
    rows = []
    for r in records:
        e, stat = r["historical_economics"], r["comparison_evidence"]
        for measure, key in (
            ("Stake-weighted", "weighted_roi"),
            ("Equal-stake", "equal_stake_return"),
        ):
            for role in ("parent", "subgroup", "complement"):
                rows.append(
                    {
                        "candidate_id": r["candidate_id"],
                        "label": title(r),
                        "measure": measure,
                        "role": role,
                        "return": e[role][key],
                        "eligible_wagers": e[role]["eligible_count"],
                        "tier": TIERS[r["evidence_tier"]],
                    }
                )
        for measure, prefix in (("Stake-weighted", "weighted"), ("Equal-stake", "equal_stake")):
            rows.append(
                {
                    "candidate_id": r["candidate_id"],
                    "label": title(r),
                    "measure": measure,
                    "role": "contrast",
                    "return": stat.get(prefix + "_difference"),
                    "low": stat.get(prefix + "_ci_low"),
                    "high": stat.get(prefix + "_ci_high"),
                    "tier": TIERS[r["evidence_tier"]],
                }
            )
    return pd.DataFrame(rows)
