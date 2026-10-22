"""Small reusable synthetic discovery fixtures."""

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig


def wagers(n=120):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "bet_id": [f"b{i:04d}" for i in range(n)],
            "namespace": "test",
            "game_id": [f"g{i // 2}" for i in range(n)],
            "sportsbook": "Book",
            "sport": "sport",
            "league": "L",
            "market_family": "moneyline",
            "odds_band": "[100,150)",
            "advertised_ev_band": "[2.5,5)",
            "execution_status": "real_money",
            "product": "fixed_odds",
            "reported_roi_eligible": True,
            "reported_stake_usd": 10.0,
            "reported_profit_usd": rng.normal(0, 8, n),
            "created_at_utc": pd.NaT,
        }
    )


def policy(**kw):
    return DiscoveryConfig(
        views=(("sportsbook",),), min_event_groups=20, bootstrap_resamples=100, **kw
    )
