"""Synthetic discovery contracts; no private data or network required."""

import pytest

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig


@pytest.mark.parametrize(
    "kw",
    [
        {"min_event_groups": 1},
        {"min_eligible_stake_usd": -1},
        {"alpha": 1},
        {"bootstrap_resamples": 0},
        {"views": (("status",),)},
        {"execution_modes": ["real_money", "real_money"]},
        {"embargo_days": float("nan")},
        {"temporal_windows": [{"fold_id": "bad"}]},
    ],
)
def test_invalid_policy(kw):
    with pytest.raises(ValueError):
        DiscoveryConfig(**kw)
