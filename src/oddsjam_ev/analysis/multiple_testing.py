"""
multiple_testing.py
===============================================================================
Purpose
-------
Account for every declared candidate when adjusting approximate tests.

What this module provides
-------------------------
Holm familywise adjustment with blocked tests retained as p=1 placeholders.

Design Notes
------------
Adjustment assumes valid marginal tests; it does not cure selection bias.
===============================================================================
"""

import numpy as np
import pandas as pd


def adjust_family(results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply Holm to the complete run family, retaining unavailable tests.

    Parameters
    ----------
    results : pandas.DataFrame
        One row per declared candidate, including raw_pvalue and test_status.
    alpha : float
        Familywise significance threshold.

    Returns
    -------
    pandas.DataFrame
        Adjusted values and exploratory flags; blocked rows remain null.
    """
    out = results.copy()
    if out.candidate_id.duplicated().any():
        raise ValueError("Duplicate hypotheses in testing family")
    p = pd.to_numeric(out.raw_pvalue, errors="coerce")
    if ((p.dropna() < 0) | (p.dropna() > 1)).any() or not 0 < alpha < 1:
        raise ValueError("Invalid p-values or alpha")
    n = len(out)
    order = np.argsort(p.fillna(1).to_numpy(), kind="stable")
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(p.fillna(1).to_numpy()[order] * (n - np.arange(n)))
    )
    out["holm_pvalue"] = pd.Series(adjusted, index=out.index).where(p.notna())
    out["family_size"] = n
    out["family_id"] = "all_declared_candidates_in_run"
    out["exploratory_holm_reject"] = out.holm_pvalue.le(alpha) & out.test_status.eq(
        "approximate_conditional_exploratory"
    )
    out["result_interpretation"] = "exploratory_not_holdout_validated"
    return out
