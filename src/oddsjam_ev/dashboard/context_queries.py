"""
context_queries.py
===============================================================================
Purpose
-------
Bind optional shared context to exact stored dashboard evidence.

What this module provides
-------------------------
Reasoned empty results, exact row selection and frozen explanation/chart adapters.

Design Notes
------------
Bindings and membership maps must come from verified publications, not URL payloads.
No fitting, aggregation, broader fallback or source resolution occurs here.
===============================================================================
"""

from dataclasses import dataclass

import pandas as pd

from oddsjam_ev.analysis.analysis_context import AnalysisContext
from oddsjam_ev.dashboard.explanation_queries import explanation_rows


@dataclass(frozen=True)
class ContextResult:
    """Selection result; blocked reasons remain visible even with zero rows.

    Parameters
    ----------
    rows : pandas.DataFrame
        Copy of stored evidence, never recomputed metrics.
    reasons : tuple of str
        Empty when supported; nonempty means blocked, not a wider selection.
    context : AnalysisContext
        Exact requested context retained for display and subsequent handoff.
    """

    rows: pd.DataFrame
    reasons: tuple[str, ...]
    context: AnalysisContext


def _bindings(context, bindings):
    expected = {
        "source_id": context.source_id,
        "configuration_version": context.configuration_version,
        **dict(context.publications),
    }
    return tuple(
        f"stale_or_missing_reference:{key}"
        for key, value in expected.items()
        if bindings.get(key) != value
    )


def select_context_rows(context, rows, bindings, *, memberships=None):
    """Intersect exact source wager rows using a verified scope binding.

    Parameters
    ----------
    context : AnalysisContext
        Requested exact population and optional cluster/drawdown selections.
    rows : pandas.DataFrame
        Verified unique wager universe. Required bet_id/funding/product/
        selection_regime; optional event_id and explicit aware time column.
    bindings : dict
        Trusted source/configuration and named publication identities. Never take
        these from the unverified token itself.
    memberships : dict, optional
        Verified cohort ID to wager-ID collections, from the pinned publication.

    Returns
    -------
    ContextResult
        Empty on unsupported/missing/unknown evidence, with stable reason codes.
    """

    def blocked(*reasons):
        return ContextResult(rows.iloc[:0].copy(), tuple(reasons), context)

    reasons = _bindings(context, bindings)
    if reasons:
        return blocked(*reasons)
    required = {"bet_id", "funding", "product", "selection_regime"}
    if required - set(rows):
        return blocked("missing_population_or_identity_columns")
    if rows.bet_id.isna().any() or rows.bet_id.duplicated().any():
        return blocked("invalid_wager_identity")
    selected = rows.loc[
        rows.funding.eq(context.funding)
        & rows["product"].eq(context.product)
        & rows.selection_regime.eq(context.selection_regime)
    ].copy()
    # Validate IDs in the scoped universe, before intersection: never silently drop
    # stale or cross-population selected IDs.
    universe = set(selected.bet_id)
    for field, column in (("wager_ids", "bet_id"), ("event_ids", "event_id")):
        ids = getattr(context, field)
        if ids is not None:
            if column not in selected:
                return blocked(f"unsupported_selection:{field}")
            if set(ids) - set(selected[column].dropna()):
                return blocked(f"unknown_selected_id:{field}")
    if context.cohort_ids is not None:
        memberships = memberships or {}
        if set(context.cohort_ids) - set(memberships):
            return blocked("unknown_selected_id:cohort_ids")
        ids = set().union(*(set(memberships[c]) for c in context.cohort_ids))
        if ids - universe:
            return blocked("cohort_population_mismatch")
        selected = selected.loc[selected.bet_id.isin(ids)]
    for field, column in (("wager_ids", "bet_id"), ("event_ids", "event_id")):
        ids = getattr(context, field)
        if ids is not None:
            selected = selected.loc[selected[column].isin(ids)]
    if context.start is not None:
        if context.time_column not in rows:
            return blocked("unsupported_time_basis")
        # Naive values are not converted into invented UTC chronology.
        try:
            values = rows[context.time_column].map(lambda x: pd.Timestamp(x))
            if values.map(lambda x: pd.isna(x) or x.tzinfo is None).any():
                return blocked("missing_or_naive_time_evidence")
            times = pd.to_datetime(selected[context.time_column], utc=True)
            selected = selected.loc[
                (times >= pd.Timestamp(context.start)) & (times < pd.Timestamp(context.end))
            ]
        except (ValueError, TypeError, OverflowError):
            return blocked("invalid_time_evidence")
    return ContextResult(selected.copy(), (), context)


def context_explanations(
    context, candidates, publication, bindings, *, memberships=None, split="all"
):
    """Select frozen predictions with the exact linked wager scope.

    Parameters
    ----------
    context, candidates, publication, bindings : object
        Typed scope, verified wager universe, frozen prediction tables and trusted
        pins. An explanations publication pin is required.
    memberships : dict, optional
        Verified cohort memberships.
    split : str
        Explicit existing published split, passed unchanged to explanation_rows.

    Returns
    -------
    ContextResult
        Stored predictions; unsupported coverage is an explicit blocked result.
    """
    selected = select_context_rows(context, candidates, bindings, memberships=memberships)
    predictions = publication["predictions"]
    reasons = selected.reasons
    if context.weighting != "equal_stake":
        reasons += ("unsupported_explanation_weighting",)
    if "explanations" not in dict(context.publications):
        reasons += ("missing_explanations_pin",)
    if not reasons:
        result = explanation_rows(publication, selected.rows, split)
        if set(selected.rows.bet_id) - set(result.bet_id):
            reasons = ("incomplete_frozen_explanation_coverage",)
        else:
            return ContextResult(result, (), context)
    return ContextResult(predictions.iloc[:0].copy(), reasons, context)


def context_chart_rows(context, frame, bindings):
    """Select exact published cohort fingerprints without building intersections.

    Parameters
    ----------
    context : AnalysisContext
        Exact cohort IDs, population and weighting.
    frame : pandas.DataFrame
        Published cohort table with cohort_fingerprint and explicit population,
        selection_regime and weighting columns. Legacy tables lacking these
        semantics remain unsupported until an explicit downstream adapter exists.
    bindings : dict
        Verified source/configuration/publication pins.

    Returns
    -------
    ContextResult
        Exact stored rows or an explicit unsupported reason. Event/window/wager
        restrictions cannot be applied to aggregate values by this adapter.
    """
    reasons = _bindings(context, bindings)
    if "visuals" not in dict(context.publications):
        reasons += ("missing_visuals_pin",)
    if context.cohort_ids is None or any(
        x is not None for x in (context.wager_ids, context.event_ids, context.start)
    ):
        reasons += ("unsupported_aggregate_intersection",)
    required = {"cohort_fingerprint", "funding", "product", "selection_regime", "weighting"}
    if required - set(frame):
        reasons += ("missing_aggregate_scope_evidence",)
    if reasons:
        return ContextResult(frame.iloc[:0].copy(), reasons, context)
    rows = frame.loc[
        frame.funding.eq(context.funding)
        & frame["product"].eq(context.product)
        & frame.selection_regime.eq(context.selection_regime)
        & frame.weighting.eq(context.weighting)
    ]
    if set(context.cohort_ids) - set(rows.cohort_fingerprint):
        return ContextResult(frame.iloc[:0].copy(), ("unknown_selected_id:cohort_ids",), context)
    return ContextResult(
        rows.loc[rows.cohort_fingerprint.isin(context.cohort_ids)].copy(), (), context
    )
