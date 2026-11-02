"""Complete fixed-look prospective claim families and bounded human pilot gates."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from oddsjam_ev.analysis.promotion import evaluate_family, validate_promotion_policy


def policy(*versions):
    p = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/promotion_policy.example.json").read_text()
    )["promotion"]
    p["family_definitions"] = [
        {"publication": "synthetic", "version_id": v, "trial_arm": "subgroup"}
        for v in (versions or ("version-1",))
    ]
    return p


def observations(sub_wins=90, comp_wins=50, count=100):
    return pd.DataFrame(
        [
            {
                "observation_key": f"{arm}-{i}",
                "enrollment_id": f"{arm}-{i}",
                "arm": arm,
                "original_quote_available": True,
                "result": "won" if i < wins else "lost",
                "paper_stake_usd": 1,
                "paper_profit_usd": 1 if i < wins else -1,
                "unit_return": 1 if i < wins else -1,
                "event_namespace": "synthetic",
                "event_id": f"{arm}-event-{i}",
            }
            for arm, wins in (("subgroup", sub_wins), ("complement", comp_wins))
            for i in range(count)
        ]
    )


def test_positive_absolute_and_comparative_results_require_complete_correction():
    p = policy("a", "b")
    validate_promotion_policy(p)
    result = evaluate_family(
        p, {"a": observations(), "b": observations(50, 50)}, {"a": [], "b": []}
    )
    assert result["family_size"] == 16
    assert result["results"][0]["eligible_for_human_pilot_review"]
    assert not result["results"][1]["eligible_for_human_pilot_review"]
    assert all(c["holm_p"] >= c["raw_p"] for c in result["hypotheses"] if c["raw_p"] is not None)
    assert (
        evaluate_family(p, {"a": observations(), "b": observations(50, 50)}, {"a": [], "b": []})
        == result
    )


@pytest.mark.parametrize(
    "data,reasons",
    [
        (observations(count=99), []),
        (observations(90, 95), []),
        (observations(30, 10), []),
        (observations(), ["pending_outcomes"]),
        (observations(), ["protocol_deviation"]),
    ],
)
def test_small_losing_or_unresolved_evidence_never_qualifies(data, reasons):
    result = evaluate_family(policy(), {"version-1": data}, {"version-1": reasons})
    assert not result["results"][0]["eligible_for_human_pilot_review"]
    assert len(result["hypotheses"]) == 8


def test_missing_family_members_unknown_ids_concentration_and_budget_fail_closed():
    with pytest.raises(ValueError, match="Complete"):
        evaluate_family(policy("a", "b"), {"a": observations()}, {"a": []})
    for mutate in (
        lambda f: f.assign(event_id="one-event"),
        lambda f: f.assign(original_quote_available=False),
        lambda f: f.assign(event_namespace=""),
    ):
        result = evaluate_family(policy(), {"version-1": mutate(observations())}, {"version-1": []})
        assert not result["results"][0]["eligible_for_human_pilot_review"]
    p = policy()
    p["max_bootstrap_cells"] = 1
    with pytest.raises(ValueError, match="budget"):
        evaluate_family(p, {"version-1": observations()}, {"version-1": []})
    p = copy.deepcopy(policy())
    p["pilot_caps"]["max_stake_per_wager_usd"] = 2
    with pytest.raises(ValueError, match="unreviewed"):
        validate_promotion_policy(p)
