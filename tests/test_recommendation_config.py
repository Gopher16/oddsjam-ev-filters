"""Research policies cannot enable production actions or fabricate exact UI evidence."""

import pytest

from oddsjam_ev.analysis.recommendation_config import RecommendationConfig, validate_capabilities


@pytest.mark.parametrize("action", ["TIGHTEN_EXISTING", "RETIRE_REDUNDANT", "PLACE_BET"])
def test_production_or_unknown_actions_rejected(action):
    with pytest.raises(ValueError):
        RecommendationConfig(allowed_actions=(action,))


def test_unknown_capabilities_and_unverified_semantics_rejected():
    with pytest.raises(ValueError):
        validate_capabilities({"schema_version": 1, "exact_values": {"profit": {}}})
    with pytest.raises(ValueError):
        validate_capabilities(
            {
                "schema_version": 1,
                "exact_values": {
                    "sportsbook": {
                        "Book": {
                            "ui_control": "book",
                            "ui_value": "Book",
                            "semantics_verified": False,
                            "evidence_reference": "unknown",
                        }
                    }
                },
            }
        )
    with pytest.raises(ValueError):
        RecommendationConfig(candidate_targets={"candidate": None})
