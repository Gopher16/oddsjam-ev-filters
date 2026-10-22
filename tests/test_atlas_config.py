"""Synthetic display-policy boundaries and invalid input checks."""

import pandas as pd
import pytest

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.opportunity_atlas import numeric_band


@pytest.mark.parametrize("edges", [[], [1, 1], [2, 1], [float("inf")]])
def test_invalid_edges(edges):
    with pytest.raises(ValueError, match="edges"):
        AtlasConfig(ev_edges=edges)


def test_mapping_and_version():
    with pytest.raises(ValueError):
        AtlasConfig(market_families={"": ""})
    with pytest.raises(ValueError):
        AtlasConfig(contract_version=2)


def test_bands_boundaries_and_missing_extremes():
    result = numeric_band(
        pd.Series(["", None, "bad", "inf", "-1", "0", "2.5", "1841.76"]), [0, 2.5, 20]
    )
    assert result.iloc[:4].tolist() == ["<missing>", "<missing>", "<invalid>", "<invalid>"]
    assert result.iloc[4:].tolist() == ["[-inf, 0.0)", "[0.0, 2.5)", "[2.5, 20.0)", "[20.0, inf)"]
    assert (
        numeric_band(pd.Series(["0", "99", "-99", "100", "-111.5"]), [-110, 100], odds=True)
        .iloc[:3]
        .eq("<invalid>")
        .all()
    )
