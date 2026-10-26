"""Display selection over exact published predicates; never construct new cohorts."""

import json

import pandas as pd

from oddsjam_ev.dashboard.queries import DIMENSIONS


def filter_signals(frame, selections):
    """Use OR within each field and AND across fields, requiring explicit evidence.

    Missing predicate dimensions cannot satisfy a selected constraint. This prevents
    a broad platform summary from masquerading as a platform/sport intersection.
    Values are exact published categories or bands, not inferred numeric thresholds.
    """
    unknown = set(selections) - set(DIMENSIONS)
    if unknown:
        raise ValueError("Unsupported signal filter")
    predicates = frame.predicate_json.map(json.loads)
    keep = pd.Series(True, index=frame.index)
    for dimension, values in selections.items():
        if values:
            keep &= predicates.map(
                lambda predicate, d=dimension, allowed=values: d in predicate
                and str(predicate[d]) in allowed
            )
    return frame.loc[keep].copy()


def signal_specificity(frame):
    """Annotate dimensional depth without changing metrics or evidence ranking."""
    result = frame.copy()
    result["dimension_count"] = result.predicate_json.map(
        lambda value: sum(d in json.loads(value) for d in DIMENSIONS)
    )
    return result


def minimum_support(frame, minimum):
    """Hide small display cohorts without pooling or recomputing returns."""
    if type(minimum) is not int or minimum < 1:
        raise ValueError("Minimum analyzed wagers must be positive")
    return frame.loc[frame.eligible_count.ge(minimum)].copy()
