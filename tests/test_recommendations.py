"""Evidence-first proposals, unproven paper protocols and version-bound identity."""

from pandas.testing import assert_frame_equal
from recommendation_fixtures import inputs

from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.analysis.recommendations import build_recommendations


def build(data, cfg=None, source="report"):
    return build_recommendations(*data, cfg or RecommendationConfig(), source)


def test_positive_results_cannot_authorize_production_changes():
    data = inputs()
    results = data[0]["candidate_results"]
    results["reported_roi"] = 1000.0
    results["exploratory_holm_reject"] = True
    results["promotion_allowed"] = True
    tables = build(data)
    p = tables["proposals"]
    assert not p.action.isin(["TIGHTEN_EXISTING", "RETIRE_REDUNDANT"]).any()
    assert not p.production_change_allowed.any() and not p.live_execution_allowed.any()
    assert p.evidence_blockers_json.str.contains("temporal_validation_blocked").all()
    assert p.initial_review_status.eq("PROPOSED").all()
    tests = p[p.action.eq("CREATE_TEST")]
    assert len(tests) == 1 and not tests.approval_allowed.any()
    assert tests.funding.eq("paper_only").all()


def test_complete_paper_protocol_and_report_identity():
    data = inputs()
    candidate = data[0]["candidate_results"].query("view == 'sportsbook'").iloc[0].candidate_id
    protocol = {
        "hypothesis": "Prospectively investigate the source predicate",
        "duration_days": 30,
        "min_event_groups": 100,
        "primary_metric": "reported_roi",
        "funding": "paper_only",
        "max_funded_stake_usd": 0,
        "review_criteria": "Review at the declared end; no automatic promotion",
    }
    cfg = RecommendationConfig(
        candidate_targets={candidate: "stable_test_id"}, paper_protocols={candidate: protocol}
    )
    first = build(data, cfg)
    p = first["proposals"].query("action == 'CREATE_TEST'").iloc[0]
    assert p.approval_allowed and p.approval_scope == "draft_paper_protocol_only"
    second = build(data, cfg, source="different_report")
    assert set(first["proposals"].proposal_id).isdisjoint(second["proposals"].proposal_id)
    reordered = (
        dict(data[0], candidate_results=data[0]["candidate_results"].iloc[::-1]),
        data[1].iloc[::-1],
        data[2].iloc[::-1],
        data[3],
        data[4],
    )
    for name, frame in first.items():
        assert_frame_equal(frame, build(reordered, cfg)[name])


def test_unknown_ui_and_planned_target_cannot_create_ready_test():
    data = list(inputs())
    data[4] = {"schema_version": 1, "exact_values": {}}
    p = build(data)["proposals"]
    assert "CREATE_TEST" not in set(p.action)
    assert "REVIEW_MANUALLY" in set(p.action)
    data = list(inputs())
    data[3]["filters"][0]["lifecycle"] = "planned_not_yet_confirmed_created"
    candidate = data[0]["candidate_results"].query("view == 'sportsbook'").iloc[0].candidate_id
    p = build(data, RecommendationConfig(candidate_targets={candidate: "stable_test_id"}))[
        "proposals"
    ]
    assert "CREATE_TEST" not in set(p.action)
    assert p.review_reasons_json.str.contains("target_not_confirmed_existing").any()


def test_empty_candidate_population():
    data = list(inputs())
    data[0] = dict(
        data[0],
        candidate_results=data[0]["candidate_results"].iloc[:0],
        candidate_memberships=data[0]["candidate_memberships"].iloc[:0],
    )
    result = build(data)
    assert all(frame.empty for frame in result.values())


def test_no_change_is_abstention_and_steps_are_concrete():
    data = inputs()
    result = build(data, RecommendationConfig(allowed_actions=("NO_CHANGE",)))
    assert result["proposals"].action.eq("NO_CHANGE").all()
    assert result["proposals"].next_steps_json.str.contains("Choose and record").all()
    assert not result["proposals"].production_change_allowed.any()


def test_funded_protocol_and_unmapped_candidate_policy_are_rejected():
    import pytest

    data = inputs()
    candidate = data[0]["candidate_results"].query("view == 'sportsbook'").iloc[0].candidate_id
    protocol = {
        "hypothesis": "Example",
        "duration_days": 30,
        "min_event_groups": 100,
        "primary_metric": "reported_roi",
        "funding": "real_money",
        "max_funded_stake_usd": 100,
        "review_criteria": "Example review",
    }
    proposals = build(data, RecommendationConfig(paper_protocols={candidate: protocol}))[
        "proposals"
    ]
    draft = proposals.query("action == 'CREATE_TEST'").iloc[0]
    assert not draft.approval_allowed
    assert "zero_funded_risk_required" in draft.approval_blockers_json
    with pytest.raises(ValueError, match="unknown candidate"):
        build(data, RecommendationConfig(candidate_targets={"not-a-candidate": "stable_test_id"}))
