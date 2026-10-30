"""Interactive drill-through of verified optional within-filter publications."""

import json

import pandas as pd
import streamlit as st

from oddsjam_ev.dashboard.evidence import candidate_rows
from oddsjam_ev.dashboard.labels import PRODUCTS, percent, pretty
from oddsjam_ev.dashboard.queries import MODE_LABELS


def definition(text):
    """Render exact persisted predicates, retaining numeric operators and units."""
    return " · ".join(
        f"{term['dimension'].replace('_', ' ').title()} {term['operator']} "
        + (
            f"{term['endpoint']}%"
            if "endpoint" in term
            else pretty(term["value"], term["dimension"])
        )
        for term in json.loads(text)
    )


def evidence_page(run, mode, limit):
    """Browse published comparisons; all selections affect display only."""
    st.header("Within-filter evidence")
    st.caption(
        "Corrected exploratory comparisons from a frozen candidate family. "
        "The descriptive Signals & leaks cards use a separate publication and are not corrected shortlist entries."
    )
    evidence = st.session_state.get("verified_evidence")
    if evidence is None:
        st.info(
            "No within-filter evidence is linked to this report. Older descriptive reports remain usable."
        )
        return
    audit = evidence["manifests"]["prioritization"]["audit"]
    a, b, c = st.columns(3)
    a.metric("Frozen candidates", audit["candidate_count"])
    b.metric("Corrected hypotheses", audit["hypothesis_count"])
    c.metric("Published shortlist entries", audit["shortlist_count"])
    st.caption(
        "Full publication counts, before display filters. Overlapping candidates are not independent or additive."
    )
    if audit["shortlist_count"] == 0:
        st.info(
            "No candidate qualified for the published corrected shortlist. Other rows remain exploratory observations, not recommended filters."
        )
    st.warning(
        "Chronology is unverified. No candidate is validated or production authorized. Historical exclusions do not establish causal improvements."
    )
    rows = candidate_rows(
        evidence,
        mode,
        st.session_state.get("filter_product", "all"),
        int(st.session_state.get("filter_min_wagers", 100)),
    )
    registry = {
        f["id"]: f["name"]
        for f in run.manifests["canonical"]["specification"]["filter_registry"]["filters"]
    }
    filters = sorted(rows.filter_id.unique())
    saved = st.selectbox(
        "Evidence saved filter",
        ["all", *filters],
        format_func=lambda v: "All saved filters" if v == "all" else registry.get(v, v),
    )
    if saved != "all":
        rows = rows.loc[rows.filter_id.eq(saved)]
    categories = st.multiselect("Evidence categories", sorted(rows.category.unique()))
    if categories:
        rows = rows.loc[rows.category.isin(categories)]
    # Match persisted predicate terms, never intersect aggregate summaries.
    dimensions = sorted({t["dimension"] for p in rows.predicate_json for t in json.loads(p)})
    predicate_rows = rows
    with st.expander("Narrow exact published definitions"):
        for dimension in dimensions:

            def term_label(term):
                return f"{term['operator']} {term.get('value', term.get('endpoint'))}" + (
                    "%" if "endpoint" in term else ""
                )

            values = sorted(
                {
                    term_label(t)
                    for p in predicate_rows.predicate_json
                    for t in json.loads(p)
                    if t["dimension"] == dimension
                }
            )
            key = "evidence_term_" + dimension
            if key in st.session_state:
                st.session_state[key] = [v for v in st.session_state[key] if v in values]
            selected = st.multiselect(
                dimension.replace("_", " ").title(), values, key="evidence_term_" + dimension
            )
            if selected:
                rows = rows.loc[
                    rows.predicate_json.map(
                        lambda p, dimension=dimension, selected=selected: any(
                            t["dimension"] == dimension and term_label(t) in selected
                            for t in json.loads(p)
                        )
                    )
                ]
    st.caption(
        f"{len(rows):,} candidates displayed. Minimum wagers hides subgroup rows only; it never changes approved arm gates, published categories, p-values or the correction family."
    )
    if rows.empty:
        st.info(
            "No published candidates match these display selections. Broader evidence is not substituted."
        )
        return
    rows = rows.copy()
    rows["Definition"] = rows.predicate_json.map(definition)
    rows["Saved filter"] = rows.filter_id.map(lambda v: registry.get(v, v))
    rows["Funding"] = rows.execution_status.map(MODE_LABELS)
    rows["Product"] = rows["product"].map(lambda v: PRODUCTS.get(v, v))
    st.dataframe(
        rows[
            [
                "Saved filter",
                "Funding",
                "Product",
                "Definition",
                "category",
                "test_status",
                "subgroup_eligible_count",
                "complement_eligible_count",
                "included_shortlist",
            ]
        ],
        hide_index=True,
    )
    selected_id = st.selectbox(
        "Candidate evidence",
        rows.candidate_id.tolist(),
        format_func=lambda v: rows.set_index("candidate_id").loc[v, "Saved filter"]
        + " · "
        + rows.set_index("candidate_id").loc[v, "Definition"]
        + " · "
        + rows.set_index("candidate_id").loc[v, "Funding"]
        + " · "
        + rows.set_index("candidate_id").loc[v, "Product"],
    )
    row = rows.set_index("candidate_id").loc[selected_id]
    st.subheader(row["Definition"])
    st.write("Published category: **" + row.category.replace("_", " ") + "**")
    reasons = json.loads(row.gate_reasons_json)
    if reasons:
        st.write("**Blocked gates:** " + "; ".join(r.replace("_", " ") for r in reasons))
    st.caption("Test status: " + row.test_status.replace("_", " "))
    if pd.notna(row.suppression_reason):
        st.info(
            "Equivalent membership suppressed in shortlist presentation only. Its hypotheses remain in correction."
        )
    family = evidence["family"]
    comparison = family["signal_comparisons"].set_index("candidate_id").loc[selected_id]
    economics = family["signal_evidence"]
    economics = economics.loc[
        economics.cohort_id.isin([row.parent_id, selected_id, comparison.complement_id])
        & economics.sizing_regime.isna()
    ]
    display = economics[
        [
            "role",
            "diagnostic_eligible_count",
            "eligible_stake_usd",
            "reported_profit_usd",
            "stake_weighted_roi",
            "equal_stake_mean_return",
            "largest_stake_share",
        ]
    ].copy()
    for col in ("stake_weighted_roi", "equal_stake_mean_return", "largest_stake_share"):
        display[col] = display[col].map(percent)
    st.markdown("**Parent, subgroup and complement economics**")
    st.dataframe(display, hide_index=True)
    st.caption(
        "Weighted ROI = reported profit / eligible stake; equal-stake return = mean wager profit/stake. Paper economics are hypothetical. Complement is this parent minus this subgroup."
    )
    contrasts = pd.DataFrame(
        [
            {
                "Measure": name,
                "Subgroup minus complement": percent(row[prefix + "_difference"]),
                "Pointwise interval": percent(row[prefix + "_ci_low"])
                + " to "
                + percent(row[prefix + "_ci_high"]),
            }
            for name, prefix in (
                ("Stake-weighted ROI", "weighted"),
                ("Equal-stake return", "equal_stake"),
            )
        ]
    )
    st.dataframe(contrasts, hide_index=True)
    st.caption(
        "Published confidence level: "
        + percent(
            evidence["manifests"]["prioritization"]["specification"]["policy"]["confidence_level"]
        )
    )
    st.caption(
        "Paired event-proxy percentile intervals are pointwise, not simultaneous. Marginal tests use the separate approximate cluster-influence t method."
    )
    concentration = pd.DataFrame(
        [
            {
                "Arm": arm,
                "Event groups": row[arm + "_groups"],
                "Event-ID coverage": percent(row[arm + "_coverage"]),
                "Largest wager share": percent(row[arm + "_largest_wager_share"]),
                "Largest event share": percent(row[arm + "_largest_event_share"]),
                "Unknown sizing regime wagers": row[arm + "_unknown_sizing_regime_count"],
            }
            for arm in ("subgroup", "complement")
        ]
    )
    st.dataframe(concentration, hide_index=True)
    hypotheses = evidence["priority"]["signal_hypotheses"]
    st.markdown("**Corrected directional results**")
    directional = hypotheses.loc[
        hypotheses.signal_candidate_id.eq(selected_id),
        [
            "measure",
            "direction",
            "raw_pvalue",
            "holm_pvalue",
            "exploratory_holm_reject",
            "family_size",
            "placeholder_in_correction",
        ],
    ].copy()
    for column in ("raw_pvalue", "holm_pvalue"):
        directional[column] = directional[column].map(
            lambda v: "Unavailable" if pd.isna(v) else f"{v:.6g}"
        )
    st.dataframe(directional, hide_index=True)
    st.caption(
        "Missing p-values are unavailable, not zero. Blocked tests retain conservative correction slots. Display filters do not reduce family size."
    )
    overlap = evidence["priority"]["signal_overlap"]
    overlap = overlap.loc[
        overlap.left_candidate_id.eq(selected_id) | overlap.right_candidate_id.eq(selected_id)
    ].copy()
    st.markdown("**Overlap with the complete published family**")
    if overlap.empty:
        st.info("No nonempty membership overlap was recorded for this candidate.")
    else:
        definitions = family["signal_candidates"].set_index("candidate_id")
        overlap["Other definition"] = overlap.apply(
            lambda r: definition(
                definitions.loc[
                    r.right_candidate_id
                    if r.left_candidate_id == selected_id
                    else r.left_candidate_id,
                    "predicate_json",
                ]
            ),
            axis=1,
        )
        other_ids = overlap.apply(
            lambda r: r.right_candidate_id
            if r.left_candidate_id == selected_id
            else r.left_candidate_id,
            axis=1,
        )
        overlap["Other saved filter"] = other_ids.map(
            lambda v: registry.get(definitions.loc[v, "filter_id"], definitions.loc[v, "filter_id"])
        )
        overlap["Other funding"] = other_ids.map(
            lambda v: MODE_LABELS.get(
                definitions.loc[v, "execution_status"], definitions.loc[v, "execution_status"]
            )
        )
        overlap["Other product"] = other_ids.map(lambda v: definitions.loc[v, "product"])
        st.dataframe(
            overlap[
                [
                    "Other saved filter",
                    "Other funding",
                    "Other product",
                    "Other definition",
                    "relationship",
                    "same_parent",
                    "shared_wagers",
                    "shared_eligible_wagers",
                    "jaccard",
                ]
            ],
            hide_index=True,
        )
    from oddsjam_ev.analysis.paper_definitions import draft_definition

    draft = draft_definition(
        family["signal_candidates"].set_index("candidate_id").loc[selected_id].to_dict()
        | {"candidate_id": selected_id}
    )
    st.download_button(
        "Download paper-test definition draft",
        json.dumps(draft, indent=2),
        file_name="paper-test-definition.json",
        mime="application/json",
        on_click="ignore",
    )
    st.caption(
        "Draft only: complete future dates, baseline settings, paper sizing and review rules before saving. Source rules are pinned; this download does not register or launch a test."
    )
    with st.expander("Publication policy and provenance"):
        st.json(evidence["manifests"]["prioritization"]["specification"])
        st.json(
            {
                "candidate_id": selected_id,
                "parent_id": row.parent_id,
                "exact_predicate": json.loads(row.predicate_json),
                "partition_reconciles": bool(comparison.partition_reconciles),
            }
        )
