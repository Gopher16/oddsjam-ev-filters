"""Display-only selection of persisted cohorts; never regenerate research metrics."""

import pandas as pd

MODES = ("real_money", "paper", "unknown")
MODE_LABELS = {
    "real_money": "Real-money labels",
    "paper": "Paper tracking",
    "unknown": "Funding not classified",
}
DIMENSIONS = ("sportsbook", "sport", "league", "market_family", "odds_band", "advertised_ev_band")
ECONOMICS = [
    "product",
    "execution_status",
    "candidate_count",
    "eligible_count",
    "eligible_stake_usd",
    "eligible_reported_profit_usd",
    "reported_roi",
    "currency",
    "fee_inclusion",
]
WAGER_COLUMNS = [
    "product",
    *DIMENSIONS,
    "candidate_count",
    "eligible_count",
    "excluded_count",
    "eligible_stake_usd",
    "eligible_reported_profit_usd",
    "reported_roi",
    "eligible_event_id_count",
    "largest_stake_share",
    "uncertainty_status",
]
PROXY_COLUMNS = [
    *DIMENSIONS,
    "opportunity_proxy_count",
    "selection_proxy_count",
    "linked_source_record_count",
    "advertised_ev_coverage_rate",
    "counts_additive_across_modes",
]
DISCOVERY_COLUMNS = [
    "candidate_id",
    "product",
    "predicate_json",
    "eligible_count",
    "eligible_stake_usd",
    "eligible_reported_profit_usd",
    "reported_roi",
    "event_proxy_groups",
    "event_coverage_rate",
    "cluster_roi_ci_low",
    "cluster_roi_ci_high",
    "interval_status",
    "largest_event_stake_share",
    "raw_pvalue",
    "holm_pvalue",
    "test_status",
    "result_interpretation",
]
PROPOSAL_COLUMNS = [
    "proposal_id",
    "candidate_id",
    "action",
    "product",
    "predicate_json",
    "target_filter_id",
    "translation_status",
    "approval_allowed",
    "approval_scope",
    "initial_review_status",
]
METRIC_COLUMNS = [
    "experiment_id",
    "execution_status",
    "availability_status",
    "enrolled_count",
    "eligible_count",
    "eligible_stake_usd",
    "eligible_reported_profit_usd",
    "hypothetical_reported_roi",
    "not_checked_count",
    "funded_execution_authorized",
    "fee_inclusion",
]


def _mode(frame, mode):
    if mode not in MODES:
        raise ValueError("Select one execution population")
    return frame.loc[frame.execution_status.eq(mode)].copy()


def economics(tables, mode):
    """One published product cohort per row; no combined population or new ROI."""
    return _mode(tables["canonical"]["economic_summary"], mode)[ECONOMICS].copy()


def atlas_view(tables, mode, view, *, proxies=False):
    """Choose one persisted view; no arbitrary multi-filter cohort construction."""
    name = "opportunity_segments" if proxies else "wager_segments"
    columns = PROXY_COLUMNS if proxies else WAGER_COLUMNS
    selected = _mode(tables["atlas"][name], mode)
    if view not in tables["atlas"]["wager_segments"].view.unique():
        raise ValueError("Unknown published view")
    selected = selected.loc[selected.view.eq(view)]
    active_dimensions = view.split("__")
    return selected[[c for c in columns if c not in DIMENSIONS or c in active_dimensions]].copy()


def filter_diagnostics(tables, mode):
    """Membership diagnostics are non-additive across filters and predicates."""
    selected = _mode(tables["atlas"]["filter_segment_diagnostics"], mode)
    return selected[
        [
            "filter_id",
            "attribution_status",
            "view",
            "product",
            "candidate_count",
            "eligible_count",
            "reported_roi",
            "additive_across_filters",
        ]
    ].copy()


def discovery_view(tables, mode, view):
    selected = _mode(tables["discovery"]["candidate_results"], mode)
    return selected.loc[selected.view.eq(view), DISCOVERY_COLUMNS].copy()


def proposals(tables, mode, action=None):
    selected = _mode(tables["recommendations"]["proposals"], mode)
    if action is not None:
        selected = selected.loc[selected.action.eq(action)]
    return selected.sort_values("proposal_id")[PROPOSAL_COLUMNS].copy()


def experiment_metrics(tables, mode):
    if "experiments" not in tables:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return _mode(tables["experiments"]["experiment_metrics"], mode)[METRIC_COLUMNS].copy()


def validate_display_contract(tables):
    """Reject missing display fields before any page exposes partial results."""
    requirements = {
        "canonical": {
            "economic_summary": ECONOMICS,
            "source_audit": ["canonical_disposition"],
            "feature_catalog": ["ingestion_snapshot_id"],
        },
        "atlas": {
            "wager_segments": [*WAGER_COLUMNS, "view", "execution_status"],
            "opportunity_segments": [*PROXY_COLUMNS, "view", "execution_status"],
            "filter_segment_diagnostics": [
                "filter_id",
                "attribution_status",
                "view",
                "product",
                "candidate_count",
                "eligible_count",
                "reported_roi",
                "additive_across_filters",
                "execution_status",
            ],
            "atlas_quality": ["population", "dimension", "category", "count", "denominator"],
        },
        "discovery": {
            "candidate_results": [*DISCOVERY_COLUMNS, "execution_status", "view"],
            "temporal_gates": ["status", "reasons_json"],
        },
        "recommendations": {
            "proposals": [
                *PROPOSAL_COLUMNS,
                "execution_status",
                "translation_blockers_json",
                "evidence_blockers_json",
                "approval_blockers_json",
                "required_next_evidence_json",
                "next_steps_json",
            ]
        },
    }
    if "experiments" in tables:
        requirements["experiments"] = {
            "experiment_metrics": METRIC_COLUMNS,
            "experiment_audit": [
                "experiment_id",
                "readiness",
                "pending_count",
                "unresolved_count",
                "excluded_count",
                "deviation_count",
                "promotion_allowed",
            ],
            "experiment_outcomes": ["source_execution_status", "assumed_execution_status"],
            "experiment_protocols": ["experiment_id", "status", "experiment_type", "registered_at"],
        }
    for stage, names in requirements.items():
        for name, required in names.items():
            frame = tables[stage][name]
            if set(required) - set(frame):
                raise ValueError("Unsupported dashboard table schema")
            if "execution_status" in frame and not frame.execution_status.isin(MODES).all():
                raise ValueError("Unknown execution population")


def funding_provenance(tables, mode):
    """Count published source/current classification pairs; no money or inferred fills."""
    frame = tables["experiments"]["experiment_outcomes"]
    selected = frame.loc[frame.assumed_execution_status.eq(mode)]
    columns = ["source_execution_status", "assumed_execution_status"]
    return selected.groupby(columns, dropna=False).size().rename("observation_count").reset_index()


def product_cohort(tables, mode, product):
    """Select a single published economic row, never pool products or funding modes."""
    selected = economics(tables, mode)
    selected = selected.loc[selected["product"].eq(product)]
    if len(selected) > 1:
        raise ValueError("Product cohort is not unique")
    return selected


def chart_cohorts(tables, mode, product, view):
    """Human-readable annotations on one published wager view; metrics are unchanged."""
    from oddsjam_ev.dashboard.labels import row_label, sample_label

    selected = atlas_view(tables, mode, view)
    selected = selected.loc[selected["product"].eq(product)].copy()
    selected["label"] = (
        selected.apply(lambda r: row_label(r, view), axis=1) if len(selected) else []
    )
    selected["sample"] = selected.eligible_count.map(sample_label)
    return selected


def takeaways(tables, mode, product):
    """Deterministic descriptive statements with counts; never promotion advice."""
    from oddsjam_ev.dashboard.labels import money, percent

    messages = []
    venues = chart_cohorts(tables, mode, product, "sportsbook")
    venues = venues.loc[venues.eligible_count.gt(0)]
    if not venues.empty:
        row = venues.sort_values("eligible_count", ascending=False, kind="stable").iloc[0]
        messages.append(
            f"The largest analyzed platform cohort is {row['label']}: {int(row.eligible_count):,} wagers. This describes logged activity, not the full market."
        )
    markets = chart_cohorts(tables, mode, product, "market_family")
    losses = markets.loc[markets.eligible_reported_profit_usd.lt(0)]
    if not losses.empty:
        row = losses.sort_values("eligible_count", ascending=False, kind="stable").iloc[0]
        messages.append(
            f"{row['label']} recorded {money(row.eligible_reported_profit_usd)} across {int(row.eligible_count):,} analyzed wagers. Review this loss pocket alongside its coverage and exclusions."
        )
    ev = chart_cohorts(tables, mode, product, "advertised_ev_band")
    small = ev.loc[ev.eligible_count.between(1, 29) & ev.reported_roi.notna()].copy()
    if not small.empty:
        row = small.loc[small.reported_roi.abs().idxmax()]
        messages.append(
            f"The {row['label']} EV band reports {percent(row.reported_roi)} ROI from only {int(row.eligible_count):,} wagers. This is a very small sample, not evidence for promotion."
        )
    if not messages:
        messages.append(
            "This selection has no eligible published results to interpret. Review its exclusions in Data confidence."
        )
    return messages


def research_queue(tables, mode, product):
    """Order existing candidates for reading, without scoring edges or changing actions.

    One row per candidate. Event-group breadth then wager count determine reading
    order within each section. The 30-wager boundary is a display convenience,
    never an approval or significance threshold. Missing evidence sorts last.
    """
    evidence = _mode(tables["discovery"]["candidate_results"], mode)
    evidence = evidence.loc[evidence["product"].eq(product)].copy()
    if evidence.candidate_id.duplicated().any():
        raise ValueError("Candidate evidence must be unique")
    proposals = _mode(tables["recommendations"]["proposals"], mode)
    proposals = proposals.loc[proposals["product"].eq(product)]
    evidence = evidence.loc[evidence.candidate_id.isin(proposals.candidate_id)].copy()
    if "predicate_json" in evidence:
        import json

        # Product-wide totals are context, not a new research angle.
        evidence = evidence.loc[
            evidence.predicate_json.map(lambda value: bool(json.loads(value)))
        ].copy()
    evidence["section"] = "More observations needed"
    substantial = evidence.eligible_count.ge(30) & evidence.reported_roi.notna()
    evidence.loc[substantial & evidence.reported_roi.gt(0), "section"] = (
        "Positive historical patterns"
    )
    evidence.loc[substantial & evidence.reported_roi.le(0), "section"] = "Loss investigations"
    return evidence.sort_values(
        ["event_proxy_groups", "eligible_count", "candidate_id"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    )


def research_finding(row):
    """Explain a persisted result and a concrete next research question."""
    from oddsjam_ev.dashboard.labels import money, percent, segment_label

    title = segment_label(row)
    result = f"{percent(row.reported_roi)} reported ROI · {money(row.eligible_reported_profit_usd)} reported profit · {int(row.eligible_count):,} analyzed wagers"
    groups = (
        "Not recorded" if pd.isna(row.event_proxy_groups) else f"{int(row.event_proxy_groups):,}"
    )
    detail = f"Event-proxy groups: {groups}. "
    if pd.notna(row.cluster_roi_ci_low) and pd.notna(row.cluster_roi_ci_high):
        detail += f"Published group interval: {percent(row.cluster_roi_ci_low)} to {percent(row.cluster_roi_ci_high)}. "
        if row.cluster_roi_ci_low <= 0 <= row.cluster_roi_ci_high:
            detail += "The interval includes zero, so these results also allow no positive return. "
    else:
        detail += "No published uncertainty interval is available. "
    if row.eligible_count < 30:
        question = "Collect more observations under a fixed definition before treating this as a testing priority."
    elif pd.notna(row.reported_roi) and row.reported_roi > 0:
        question = "Paper-test this exact segment prospectively: does the positive pattern persist at recorded quotes, and how often are those quotes available?"
    else:
        question = "Investigate whether losses concentrate in a platform, market or EV range already present in the published views; fix a hypothesis before collecting a new paper sample."
    return title, result, detail, question
