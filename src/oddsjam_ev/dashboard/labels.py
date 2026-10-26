"""Human-readable labels derived from pinned evidence, never inferred source dates."""

import json
import math
from zoneinfo import ZoneInfo

import pandas as pd

PRODUCTS = {
    "exchange": "Exchanges",
    "fixed_odds": "Sportsbooks",
    "pickem": "Pick’em",
    "unknown": "Product not classified",
}
DIMENSIONS = {
    "sportsbook": "Platform",
    "sport": "Sport",
    "league": "League",
    "market_family": "Market",
    "odds_band": "Odds range",
    "advertised_ev_band": "Advertised EV",
}
ACTIONS = {
    "COLLECT_MORE_DATA": "Collect more evidence",
    "REVIEW_MANUALLY": "Review manually",
    "CREATE_TEST": "Draft a paper test",
    "NO_CHANGE": "No change proposed",
}
MARKETS = {
    "moneyline": "Moneyline",
    "spread": "Point spreads",
    "total": "Game totals",
    "player_prop": "Player props",
}
REASONS = {
    "fees_and_fillability_unverified": "Fees and actual order availability have not been verified.",
    "historical_results_exploratory": "These historical results are exploratory, not a validated strategy.",
    "insufficient_event_proxy_groups": "There are too few event groups for the configured evidence requirement.",
    "stake_threshold_unset": "The minimum analyzed stake required for formal testing has not been chosen.",
    "temporal_validation_blocked": "A valid chronological evaluation is not yet available.",
    "pickem_candidate_leg": "Unlinked Pick’em leg",
    "ticket_payout_and_completeness_unverified": "Ticket payout not verified",
    "unresolved_ticket_members": "Ticket membership unresolved",
    "unknown_product": "Product not classified",
    "unresolved_or_excluded_settlement": "Settlement not eligible",
    "unsupported_bet_type": "Unsupported bet type",
    "decision_features_unverified": "Information available before the bet has not been verified.",
    "embargo_unset": "The separation between evaluation periods has not been specified.",
    "temporal_windows_unset": "Development and holdout periods have not been specified.",
    "no_verified_placement_times": "Verified bet placement times are missing.",
    "placement_time_unverified": "Verified bet placement times are missing.",
    "missing_features_available_at_utc": "We cannot verify when the decision information was available.",
    "missing_information_end_at_utc": "We cannot verify the end of each observation’s information window.",
    "missing_outcome_available_at_utc": "We cannot verify when results became available.",
    "observed_as_of_unverified": "Observation timing has not been verified.",
    "outcome_availability_unverified": "Result availability has not been verified.",
    "outcome_values_as_of_unverified": "Historical result versions have not been verified.",
    "evidence_source_reference_missing": "Supporting chronology evidence has not been supplied.",
}


def pretty(value, dimension=None):
    text = str(value)
    if text in {"<missing>", "None", "nan", ""}:
        return f"{DIMENSIONS.get(dimension, 'Value')} not recorded"
    if text == "<unmapped>":
        return "Market not mapped" if dimension == "market_family" else "Not mapped"
    if text == "<invalid>":
        return "Invalid recorded value"
    if text == "<conflict>":
        return "Conflicting source values"
    if (
        dimension in {"advertised_ev_band", "odds_band"}
        and text.startswith("[")
        and text.endswith(")")
    ):
        lo, hi = (float(v.strip()) for v in text[1:-1].split(","))

        def number(n):
            return f"{n:g}%" if dimension == "advertised_ev_band" else f"{n:+g}"

        if math.isinf(lo):
            return f"Below {number(hi)}"
        if math.isinf(hi):
            return f"{number(lo)} and above"
        return f"{number(lo)} to under {number(hi)}"
    if dimension == "market_family":
        return MARKETS.get(text, text.replace("_", " ").capitalize())
    return text


def view_label(view):
    return " + ".join(DIMENSIONS.get(d, d.replace("_", " ").title()) for d in view.split("__"))


def segment_label(row):
    predicate = json.loads(row["predicate_json"])
    parts = [PRODUCTS.get(row["product"], row["product"])]
    parts += [f"{DIMENSIONS[d]}: {pretty(predicate[d], d)}" for d in DIMENSIONS if d in predicate]
    return " · ".join(parts)


def row_label(row, view):
    return " · ".join(pretty(row[d], d) for d in view.split("__"))


def reason_text(code):
    if code.endswith(":exact_ui_semantics_unverified"):
        dimension = code.split(":", 1)[0]
        return f"The tool’s {DIMENSIONS.get(dimension, dimension)} control has not been checked against this research definition."
    return REASONS.get(code, code.replace("_", " ").strip().capitalize())


def evidence_step(code):
    actions = {
        "fees_and_fillability_unverified": "For executable-return claims, collect evidence of offered prices, available amounts, accepted orders and fees.",
        "historical_results_exploratory": "Keep historical patterns descriptive and register a prospective paper test for follow-up.",
        "insufficient_event_proxy_groups": "Collect more event-group observations and check whether related outcomes are grouped appropriately.",
        "stake_threshold_unset": "Choose and document the minimum analyzed stake for formal testing; no threshold is inferred here.",
        "temporal_validation_blocked": "Record when information and outcomes became available, then define development and holdout periods.",
    }
    if code.endswith(":exact_ui_semantics_unverified"):
        dimension = code.split(":", 1)[0]
        return f"Confirm that the tool’s {DIMENSIONS.get(dimension, dimension)} setting matches this segment’s definition."
    return actions.get(code, reason_text(code))


def local_time(value, zone="America/Chicago", *, short=False):
    time = pd.Timestamp(value)
    if pd.isna(time) or time.tzinfo is None:
        raise ValueError("A confirmed aware timestamp is required")
    return time.tz_convert(ZoneInfo(zone)).strftime(
        "%b %d, %Y" if short else "%b %d, %Y · %I:%M %p %Z"
    )


def report_label(spec, run_id, zone="America/Chicago"):
    mode = (
        "Historical review" if spec["mode"] == "historical_reanalysis" else "Manual export review"
    )
    return f"{local_time(spec['as_of'], zone)} · {mode} · {run_id[:8]}"


def report_dates(run, zone="America/Chicago"):
    imports = run.manifests["canonical"]["specification"]["ingestion_manifest"]["specification"][
        "imports"
    ]
    receipt = max((i["imported_at"] for i in imports), key=lambda v: pd.Timestamp(v))
    exports = [i["source"].get("exported_at") for i in imports]
    return {
        "Analysis cutoff": local_time(run.result["specification"]["as_of"], zone),
        "Data last received": local_time(receipt, zone),
        "Export date": "Not confirmed for all sources"
        if not all(exports)
        else " / ".join(sorted({local_time(v, zone, short=True) for v in exports})),
    }


def money(value):
    return "Unavailable" if pd.isna(value) else f"{'−' if value < 0 else ''}${abs(value):,.2f}"


def percent(value):
    return "Unavailable" if pd.isna(value) else f"{value:.2%}"


def sample_label(count):
    return f"{int(count):,} analyzed wagers" + (" · very small sample" if count < 30 else "")
