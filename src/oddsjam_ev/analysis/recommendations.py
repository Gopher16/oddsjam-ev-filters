"""
recommendations.py
===============================================================================
Purpose
-------
Turn research evidence into reviewable proposals without authorizing deployment.

What this module provides
-------------------------
Deterministic research actions, evidence links, blockers and paper-only protocols.

Design Notes
------------
No profitability rank, automatic filter creation or production action is generated.
===============================================================================
"""

import json
import math

import pandas as pd

from oddsjam_ev.analysis.filter_actions import coverage_impact, resolve_target, translate_predicate
from oddsjam_ev.analysis.recommendation_config import RecommendationConfig, validate_capabilities
from oddsjam_ev.io.source_csv import fingerprint

NEXT_STEPS = {
    "fees_and_fillability_unverified": "Obtain fee treatment and executable-fill evidence; do not subtract guessed fees.",
    "historical_results_exploratory": "Specify a prospective hypothesis and evaluation protocol; do not treat historical results as validation.",
    "temporal_validation_blocked": "Establish placement, feature and outcome availability evidence before declaring temporal windows and embargo.",
    "stake_threshold_unset": "Choose and record the minimum event-covered eligible USD stake required for testing.",
    "insufficient_event_proxy_groups": "Collect additional supported event groups; nominal wager counts do not establish independence.",
    "event_coverage_incomplete": "Resolve missing event identities with source evidence or disclose inference on the covered subset only.",
    "historical_attribution_incomplete": "Resolve missing historical filter labels without inferring settings from current names.",
    "target_not_confirmed_existing": "Confirm the target filter exists and review its current settings before proposing a modification.",
}


def next_step(code: str) -> str:
    """Explain an evidence code as a concrete research follow-up.

    Parameters
    ----------
    code : str
        Persisted blocker or review reason.

    Returns
    -------
    str
        Human-readable next step; unrecognized codes remain explicit.
    """
    if code in NEXT_STEPS:
        return NEXT_STEPS[code]
    if ":exact_ui_semantics_unverified" in code:
        return (
            "Review exact UI labels, units and inclusive/exclusive boundaries for "
            + code.split(":")[0]
            + "; record supporting evidence."
        )
    if ":diagnostic_category" in code:
        return (
            "Keep "
            + code.split(":")[0]
            + " as a diagnostic category; do not translate missing, invalid or conflicting values into a live filter."
        )
    return "Review and resolve the explicit requirement: " + code


def protocol_blockers(protocol: dict) -> list[str]:
    """Check a prospective paper-only protocol without registering an experiment.

    Parameters
    ----------
    protocol : dict
        Hypothesis, duration_days, min_event_groups, primary_metric and funding.

    Returns
    -------
    list[str]
        Missing or unsupported protocol details.
    """
    reasons = []
    if not isinstance(protocol.get("hypothesis"), str) or not protocol["hypothesis"].strip():
        reasons.append("hypothesis_required")
    for key in ["duration_days", "min_event_groups"]:
        if type(protocol.get(key)) is not int or protocol[key] <= 0:
            reasons.append(key + "_required")
    if protocol.get("primary_metric") != "reported_roi":
        reasons.append("primary_metric_must_be_reported_roi")
    if (
        protocol.get("funding") != "paper_only"
        or type(protocol.get("max_funded_stake_usd")) not in (int, float)
        or protocol.get("max_funded_stake_usd") != 0
    ):
        reasons.append("zero_funded_risk_required")
    if (
        not isinstance(protocol.get("review_criteria"), str)
        or not protocol["review_criteria"].strip()
    ):
        reasons.append("review_criteria_required")
    return reasons


def build_recommendations(
    discovery: dict,
    wagers: pd.DataFrame,
    attribution: pd.DataFrame,
    registry: dict,
    capabilities: dict,
    cfg: RecommendationConfig,
    source_report_id: str,
) -> dict[str, pd.DataFrame]:
    """Generate proposals with their original evidence and non-additive coverage.

    Parameters
    ----------
    discovery : dict
        Verified discovery report tables.
    wagers : pandas.DataFrame
        Verified atlas wager dimensions.
    attribution : pandas.DataFrame
        Verified canonical memberships.
    registry, capabilities : dict
        Historical targets and independently reviewed UI semantics.
    cfg : RecommendationConfig
        Research action policy.
    source_report_id : str
        Immutable discovery report identity.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Proposals, exact candidate evidence and per-proposal coverage audits.
    """
    validate_capabilities(capabilities)
    results = discovery["candidate_results"].sort_values("candidate_id")
    if results.candidate_id.duplicated().any():
        raise ValueError("Candidate evidence identities must be unique")
    if set(cfg.candidate_targets) - set(results.candidate_id) or set(cfg.paper_protocols) - set(
        results.candidate_id
    ):
        raise ValueError("Policy references unknown candidate IDs")
    # Validation occurs even when there are no candidates.
    resolve_target(None, registry)
    if wagers.bet_id.duplicated().any():
        raise ValueError("Proposal coverage requires unique wagers")
    member_table = discovery["candidate_memberships"]
    if (
        member_table.duplicated(["candidate_id", "bet_id"]).any()
        or not set(member_table.bet_id) <= set(wagers.bet_id)
        or not set(member_table.candidate_id) <= set(results.candidate_id)
    ):
        raise ValueError("Invalid recommendation memberships")
    members = {k: set(g.bet_id) for k, g in member_table.groupby("candidate_id", sort=True)}
    temporal = discovery["temporal_gates"]
    temporal_blocked = (
        not len(temporal) or not temporal.status.eq("available_conditional_on_evidence").all()
    )
    proposals = []
    audits = []
    for candidate in results.to_dict("records"):
        ident = candidate["candidate_id"]
        predicate = json.loads(candidate["predicate_json"])
        target = resolve_target(cfg.candidate_targets.get(ident), registry)
        translation = translate_predicate(predicate, capabilities)
        blockers = ["fees_and_fillability_unverified", "historical_results_exploratory"]
        if temporal_blocked:
            blockers.append("temporal_validation_blocked")
        if candidate["test_status"].startswith("blocked:"):
            blockers.extend(candidate["test_status"].removeprefix("blocked:").split(","))
        if (
            candidate["event_coverage_rate"] is None
            or pd.isna(candidate["event_coverage_rate"])
            or candidate["event_coverage_rate"] < 1
        ):
            blockers.append("event_coverage_incomplete")
        cohort = wagers.loc[
            wagers.execution_status.eq(candidate["execution_status"])
            & wagers["product"].eq(candidate["product"])
        ]
        impact = coverage_impact(
            cohort, attribution, members.get(ident, set()), target["target_filter_id"]
        )
        actions = []
        if blockers:
            actions.append(("COLLECT_MORE_DATA", []))
        review = json.loads(translation["translation_blockers_json"])
        if target["target_filter_id"] and not target["target_existing"]:
            review.append("target_not_confirmed_existing")
        if impact["cohort_unresolved_attribution_wagers"]:
            review.append("historical_attribution_incomplete")
        if review:
            actions.append(("REVIEW_MANUALLY", []))
        if (
            cfg.draft_paper_hypotheses
            and predicate
            and translation["translation_status"] == "exact_reviewed"
            and candidate["eligible_count"] > 0
            and (
                target["target_filter_id"] is None
                or (target["target_status"] == "TEST" and target["target_existing"])
            )
        ):
            protocol = cfg.paper_protocols.get(ident, {})
            actions.append(("CREATE_TEST", protocol_blockers(protocol)))
        if not actions:
            actions.append(("NO_CHANGE", []))
        actions = [
            (action, requirements)
            for action, requirements in actions
            if action in cfg.allowed_actions
        ]
        if not actions and "NO_CHANGE" in cfg.allowed_actions:
            actions = [("NO_CHANGE", [])]
        for action, approval_blockers in actions:
            payload = {
                "source_report_id": source_report_id,
                "candidate_id": ident,
                "action": action,
                "target_filter_id": target["target_filter_id"],
                "predicate": predicate,
                "policy": cfg.__dict__,
                "capabilities": capabilities,
                "registry": registry,
                "candidate_evidence": candidate,
            }
            # JSON-compatible scalar nulls for stable identity.
            payload["candidate_evidence"] = {
                k: None if isinstance(v, float) and not math.isfinite(v) else v
                for k, v in candidate.items()
            }
            proposal_id = "proposal:" + fingerprint(payload)
            proposal = {
                "proposal_id": proposal_id,
                "source_report_id": source_report_id,
                "candidate_id": ident,
                "action": action,
                "predicate_json": candidate["predicate_json"],
                "execution_status": candidate["execution_status"],
                "product": candidate["product"],
                **target,
                **translation,
                "evidence_blockers_json": json.dumps(sorted(set(blockers))),
                "review_reasons_json": json.dumps(sorted(set(review))),
                "approval_blockers_json": json.dumps(sorted(approval_blockers)),
                "approval_allowed": not approval_blockers,
                "initial_review_status": "PROPOSED",
                "approval_scope": "draft_paper_protocol_only"
                if action == "CREATE_TEST"
                else "research_review_only",
                "funding": "paper_only" if action == "CREATE_TEST" else "not_applicable",
                "live_execution_allowed": False,
                "production_change_allowed": False,
                "prospective_protocol_json": json.dumps(
                    cfg.paper_protocols.get(ident, {}), sort_keys=True
                )
                if action == "CREATE_TEST"
                else "{}",
                "required_next_evidence_json": json.dumps(
                    sorted(set(blockers + review + approval_blockers))
                ),
                "next_steps_json": json.dumps(
                    [next_step(code) for code in sorted(set(blockers + review + approval_blockers))]
                ),
                "interpretation": "unproven_paper_hypothesis"
                if action == "CREATE_TEST"
                else "research_followup_not_strategy_instruction",
            }
            proposals.append(proposal)
            audits.append({"proposal_id": proposal_id, "candidate_id": ident, **impact})
    columns = [
        "proposal_id",
        "source_report_id",
        "candidate_id",
        "action",
        "predicate_json",
        "execution_status",
        "product",
        "target_filter_id",
        "target_status",
        "target_lifecycle",
        "target_existing",
        "translation_status",
        "ui_controls_json",
        "translation_blockers_json",
        "evidence_blockers_json",
        "review_reasons_json",
        "approval_blockers_json",
        "approval_allowed",
        "initial_review_status",
        "approval_scope",
        "funding",
        "live_execution_allowed",
        "production_change_allowed",
        "prospective_protocol_json",
        "required_next_evidence_json",
        "next_steps_json",
        "interpretation",
    ]
    return {
        "proposals": pd.DataFrame(proposals, columns=columns)
        .sort_values("proposal_id")
        .reset_index(drop=True),
        "candidate_evidence": results.reset_index(drop=True),
        "coverage_audit": pd.DataFrame(
            audits,
            columns=[
                "proposal_id",
                "candidate_id",
                "coverage_status",
                "target_observed_wagers",
                "matching_target_wagers",
                "nonmatching_target_wagers",
                "target_eligible_wagers",
                "matching_target_eligible_wagers",
                "nonmatching_target_eligible_wagers",
                "target_eligible_stake_usd",
                "matching_target_stake_share",
                "nonmatching_target_stake_share",
                "other_filter_overlap_wagers",
                "cohort_unresolved_attribution_wagers",
                "future_scanner_coverage_verified",
            ],
        ),
    }
