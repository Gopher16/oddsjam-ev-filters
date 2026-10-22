"""Synthetic discovery contracts; no private data or network required."""

import numpy as np
import pandas as pd
import pytest

from oddsjam_ev.analysis.multiple_testing import adjust_family
from oddsjam_ev.analysis.robustness import (
    event_profit_pvalue,
)


def test_holm_retains_blocked_family_members():
    x = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "raw_pvalue": [0.01, 0.04, 0.03, np.nan],
            "test_status": ["approximate_conditional_exploratory"] * 3 + ["blocked"],
        }
    )
    y = adjust_family(x, 0.05)
    assert y.holm_pvalue.iloc[:3].tolist() == pytest.approx([0.04, 0.09, 0.09])
    assert pd.isna(y.holm_pvalue.iloc[3])
    assert y.family_size.eq(4).all()
    assert y.exploratory_holm_reject.tolist() == [True, False, False, False]


def test_planted_signal_and_null_family_calibration():
    # Fixed synthetic distributions only; this does not verify historical independence.
    rng = np.random.default_rng(818)
    planted = pd.DataFrame({"profit": rng.normal(1, 1, 300)})
    assert event_profit_pvalue(planted) < 1e-8
    for distribution in ["normal", "skewed"]:
        any_rejections = 0
        for _repeat in range(200):
            values = (
                rng.normal(size=(5, 200))
                if distribution == "normal"
                else rng.gamma(4, 1, size=(5, 200)) - 4
            )
            p = [event_profit_pvalue(pd.DataFrame({"profit": row})) for row in values]
            family = pd.DataFrame(
                {
                    "candidate_id": list("abcde"),
                    "raw_pvalue": p,
                    "test_status": "approximate_conditional_exploratory",
                }
            )
            any_rejections += int(adjust_family(family, 0.05).exploratory_holm_reject.any())
        assert any_rejections / 200 <= 0.12
