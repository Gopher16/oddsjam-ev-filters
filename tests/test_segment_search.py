"""Outcome-independent candidate construction and duplicate safeguards."""

import pandas as pd
import pytest
from discovery_fixtures import policy, wagers
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.segment_search import enumerate_segments


def test_duplicate_inputs_and_views_rejected():
    x = wagers()
    with pytest.raises(ValueError, match="identities"):
        enumerate_segments(pd.concat([x, x]), policy())
    with pytest.raises(ValueError, match="views"):
        DiscoveryConfig(views=(("sportsbook", "league"), ("league", "sportsbook")))


def test_ledger_and_memberships_do_not_depend_on_profit_or_order():
    x = wagers()
    a, links = enumerate_segments(x, policy())
    x = x.iloc[::-1].copy()
    x["reported_profit_usd"] = 100000
    x["reported_roi_eligible"] = False
    b, other = enumerate_segments(x, policy())
    assert_frame_equal(a, b)
    assert_frame_equal(links, other)
    assert len(a) == 2
    assert links.groupby("candidate_id").bet_id.nunique().eq(len(x)).all()
