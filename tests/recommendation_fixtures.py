"""Synthetic proposal inputs with no private source records."""

import pandas as pd
from discovery_fixtures import policy, wagers

from oddsjam_ev.io.discovery import build_discovery


def inputs():
    x = wagers(120)
    report = build_discovery(x, policy())
    attribution = pd.DataFrame(
        {"bet_id": x.bet_id, "filter_id": "stable_test_id", "attribution_status": "resolved"}
    )
    registry = {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [
            {
                "id": "stable_test_id",
                "name": "TEST - Example",
                "status": "TEST",
                "lifecycle": "existing",
            }
        ],
    }
    capabilities = {
        "schema_version": 1,
        "exact_values": {
            "sportsbook": {
                "Book": {
                    "ui_control": "sportsbooks",
                    "ui_value": ["Book"],
                    "semantics_verified": True,
                    "evidence_reference": "synthetic_only",
                }
            }
        },
    }
    return report, x, attribution, registry, capabilities
