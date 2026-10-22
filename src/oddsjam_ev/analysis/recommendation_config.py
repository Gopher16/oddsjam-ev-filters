"""
recommendation_config.py
===============================================================================
Purpose
-------
Validate research actions without permitting production deployment.

What this module provides
-------------------------
Versioned recommendation policy and reviewed exact UI capability contracts.

Design Notes
------------
Human approval never authorizes live execution through this software.
===============================================================================
"""

from dataclasses import dataclass, field

ACTIONS = frozenset(
    {
        "TIGHTEN_EXISTING",
        "CREATE_TEST",
        "COLLECT_MORE_DATA",
        "REVIEW_MANUALLY",
        "NO_CHANGE",
        "RETIRE_REDUNDANT",
    }
)
RESEARCH_ACTIONS = ACTIONS - {"TIGHTEN_EXISTING", "RETIRE_REDUNDANT"}


@dataclass(frozen=True)
class RecommendationConfig:
    """Research-only policy.

    Attributes
    ----------
    allowed_actions : tuple[str, ...]
        Research action subset. Production changes are unsupported in this contract.
    draft_paper_hypotheses : bool
        Permit explicitly unproven draft paper-test proposals.
    candidate_targets : dict[str, str]
        Explicit candidate to historical registry ID mappings.
    paper_protocols : dict[str, dict]
        User-supplied prospective protocol details, pinned into proposal identity.
    contract_version : int
        Contract version.
    """

    allowed_actions: tuple = ("COLLECT_MORE_DATA", "REVIEW_MANUALLY", "CREATE_TEST", "NO_CHANGE")
    draft_paper_hypotheses: bool = True
    candidate_targets: dict = field(default_factory=dict)
    paper_protocols: dict = field(default_factory=dict)
    contract_version: int = 1

    def __post_init__(self):
        actions = tuple(self.allowed_actions)
        if (
            self.contract_version != 1
            or not actions
            or len(set(actions)) != len(actions)
            or set(actions) - RESEARCH_ACTIONS
        ):
            raise ValueError("Only research actions are allowed; production actions remain blocked")
        if type(self.draft_paper_hypotheses) is not bool:
            raise ValueError("draft_paper_hypotheses must be boolean")
        if not isinstance(self.candidate_targets, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in self.candidate_targets.items()
        ):
            raise ValueError("Targets require nonempty candidate and stable registry IDs")
        if not isinstance(self.paper_protocols, dict) or any(
            not isinstance(k, str) or not isinstance(v, dict)
            for k, v in self.paper_protocols.items()
        ):
            raise ValueError("Paper protocols require candidate-keyed objects")
        object.__setattr__(self, "allowed_actions", actions)


def validate_capabilities(capabilities: dict) -> None:
    """Validate exact reviewed translations; no inferred UI support.

    Parameters
    ----------
    capabilities : dict
        Source reference and dimension/source-value translation objects.

    Returns
    -------
    None
    """
    dimensions = {
        "sportsbook",
        "sport",
        "league",
        "market_family",
        "odds_band",
        "advertised_ev_band",
    }
    if not isinstance(capabilities, dict) or capabilities.get("schema_version") != 1:
        raise ValueError("Unsupported capability schema")
    mappings = capabilities.get("exact_values")
    if not isinstance(mappings, dict) or set(mappings) - dimensions:
        raise ValueError("Unknown capability dimensions")
    for values in mappings.values():
        if not isinstance(values, dict):
            raise ValueError("Capability mappings must be objects")
        for source, translation in values.items():
            if not isinstance(source, str) or not source or not isinstance(translation, dict):
                raise ValueError("Invalid capability mapping")
            if set(translation) != {
                "ui_control",
                "ui_value",
                "semantics_verified",
                "evidence_reference",
            }:
                raise ValueError("Capabilities require control, value and evidence")
            if (
                translation["semantics_verified"] is not True
                or not isinstance(translation["evidence_reference"], str)
                or not translation["evidence_reference"].strip()
            ):
                raise ValueError("Exact capability semantics require explicit supporting evidence")
            if (
                not isinstance(translation["ui_control"], str)
                or not translation["ui_control"].strip()
                or not isinstance(translation["ui_value"], (str, list, dict))
                or not translation["ui_value"]
            ):
                raise ValueError("UI control and value must be explicit")
