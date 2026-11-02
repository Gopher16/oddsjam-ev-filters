"""
explanations.py
===============================================================================
Purpose
-------
Explore global and filtered rows from explicit frozen model publications.

What this module provides
-------------------------
Linked explanation charts, quality gates, source coverage and offline scope handoff.

Design Notes
------------
Sidebar/chart interaction never fits a model, predicts new rows or changes a reference.
===============================================================================
"""

import json

import pandas as pd
import streamlit as st

from oddsjam_ev.dashboard import explanation_charts as charts
from oddsjam_ev.dashboard import explanation_queries as queries
from oddsjam_ev.dashboard.decision_queries import title
from oddsjam_ev.dashboard.explanation_artifacts import FEATURES
from oddsjam_ev.dashboard.labels import percent
from oddsjam_ev.dashboard.pages.decision_workspace import scope


def _label(publication):
    if publication["kind"] == "global":
        return "All-history frozen models · " + publication["publication_id"][:10]
    return (
        "Separately fitted subset · "
        + json.dumps(publication["specification"]["scope"], sort_keys=True)
        + " · "
        + publication["publication_id"][:10]
    )


def _reset_local():
    st.session_state.explanation_chart_revision = (
        st.session_state.get("explanation_chart_revision", 0) + 1
    )
    for key in (
        "explanation_local",
        "explanation_chart_pick",
        "explanation_search",
        "explanation_events",
        "explanation_population",
    ):
        st.session_state.pop(key, None)


def _clear_idea():
    st.session_state.pop("explanation_candidate_id", None)
    _reset_local()


def _population(frame, funding, product):
    if frame.empty:
        return frame.copy()
    return frame.loc[frame.funding.eq(funding) & frame["product"].eq(product)].copy()


def _linked(chart, key, rows):
    event = st.altair_chart(
        chart, width="stretch", on_select="rerun", selection_mode=["explanation_wager"], key=key
    )
    points = event.selection.get("explanation_wager", [])
    identity = (key, tuple(p.get("bet_id") for p in points))
    events = st.session_state.setdefault("explanation_events", {})
    if identity != events.get(key):
        events[key] = identity
        if points and points[0].get("bet_id") in set(rows.bet_id):
            st.session_state.explanation_chart_pick = points[0]["bet_id"]


def _local(rows, sampled, run):
    st.subheader("Explain one recorded wager")
    if rows.empty:
        return
    ids = sampled.bet_id.tolist()
    if st.session_state.get("explanation_local") not in [None, *ids]:
        st.session_state.pop("explanation_local", None)
        st.info(
            "The previous local wager is outside this chart sample/selection. Choose a visible point or paste an in-scope published ID."
        )
    picked = st.session_state.pop("explanation_chart_pick", None)
    if picked in ids:
        st.session_state.explanation_local = picked
        st.session_state.explanation_search = ""
    chosen = st.selectbox(
        "Local wager · chart sample",
        ids,
        index=None,
        key="explanation_local",
        placeholder="Click a distribution/dependence point or select a recorded wager",
    )
    search = st.text_input("Or paste an exact published wager ID", key="explanation_search")
    if search:
        if search not in set(rows.bet_id):
            st.info(
                "This wager has no published explanation in the current selection; no prediction is generated."
            )
            return
        chosen = search
    if chosen is None:
        return
    row = rows.set_index("bet_id").loc[chosen]
    facts = run.tables["canonical"]["bet_facts"].set_index("bet_id")
    if chosen in facts.index:
        fact = facts.loc[chosen]
        st.write(
            " · ".join(
                str(fact.get(f, ""))
                for f in ("sportsbook", "event_name", "market_name", "bet_name")
            )
        )
    a, b, c = st.columns(3)
    a.metric("Reference", percent(row.base_return))
    b.metric("Frozen prediction", percent(row.predicted_return))
    c.metric("Observed return", percent(row.observed_return))
    st.caption(
        f"Published split: {row.split} · Source population: {row.funding} / {row['product']}. Observed outcome is not a predictor."
    )
    st.altair_chart(charts.waterfall(queries.waterfall(row)), width="stretch")
    st.caption(
        "Each change raises or lowers this model's prediction, not the probability of a causal improvement. All steps use unit-return fractions; axes display percentages."
    )
    with st.expander("Exact local contributions & reference identities"):
        st.dataframe(queries.waterfall(row), hide_index=True, width="stretch")
        st.json(
            {
                "bet_id": chosen,
                "model_id": row.model_id,
                "background_id": row.background_id,
                "split": row.split,
                "reconstruction_error": row.reconstruction_error,
            }
        )


def render(run, mode, limit):
    st.header("Explain returns")
    st.caption(
        "Explore recorded feature associations with frozen predictions. Sidebar filters select wagers; they never train or change the background."
    )
    attachment = st.session_state.get("explanation_bundle")
    workspace = st.session_state.get("decision_workspace")
    if attachment is None or workspace is None:
        st.info(
            "No verified explanatory publication is attached to this report. Build and explicitly attach matching offline model evidence; this page never trains or loads a different report."
        )
        return
    publications = attachment["publications"]
    if st.session_state.get("explanation_publication") not in [None, *publications]:
        st.session_state.pop("explanation_publication", None)
        _reset_local()
    ident = st.selectbox(
        "Explanation publication",
        list(publications),
        format_func=lambda v: _label(publications[v]),
        key="explanation_publication",
        on_change=_reset_local,
    )
    publication = publications[ident]
    if publication["kind"] == "subset":
        st.warning(
            "Separately fitted subset model. Its model and training background differ from the all-history publication; SHAP magnitudes are not a common-baseline comparison. Sidebar filtering never selects this model automatically."
        )
        st.json(publication["specification"]["scope"], expanded=False)
    st.caption(
        "Frozen references · equal-wager unit returns · exploratory event-grouped evaluation, not forward validation or causal evidence."
    )
    filters = scope()
    idea_id = st.session_state.get("explanation_candidate_id")
    idea = next((r for r in workspace["queues"]["records"] if r["candidate_id"] == idea_id), None)
    if idea_id and idea is None:
        st.info(
            "The linked idea is unavailable in this report. Clear it to explore other rows; no candidate is substituted."
        )
        st.button("Clear idea drill-through", on_click=_clear_idea)
        return
    if idea:
        st.info(
            "Idea drill-through: "
            + title(idea)
            + ". Intersected with sidebar selections and resolved saved-filter membership where applicable."
        )
        st.button("Clear idea drill-through", on_click=_clear_idea)
    candidates = queries.select_candidates(
        workspace["visuals"]["visual_candidates"],
        run.tables["canonical"]["filter_attribution"],
        **filters,
        idea=idea,
    )
    matching = queries.explanation_rows(publication, candidates)
    a, b, c = st.columns(3)
    a.metric("Selected wagers", f"{len(candidates):,}")
    b.metric("Explained wagers", f"{len(matching):,}")
    c.metric("Unexplained", f"{len(candidates) - len(matching):,}")
    st.caption(
        "Economic exclusions, blocked populations and out-of-scope rows remain unexplained. Expand support below for reasons."
    )
    if any(
        st.session_state.get(k)
        for k in ("filter_workspace_tiers", "filter_workspace_setup", "filter_workspace_lifecycle")
    ):
        st.info(
            "Evidence tier, setup and lifecycle controls apply to idea queues, not these recorded-wager explanations. Their values do not alter a model cohort."
        )
    with st.expander("Population support & blocked models", expanded=matching.empty):
        audit = publication["audit"]["populations"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        k: r.get(k)
                        for k in (
                            "funding",
                            "product",
                            "status",
                            "source_wagers",
                            "eligible_wagers",
                            "event_groups",
                            "selected_model",
                            "test_improves_both_references",
                            "reasons",
                        )
                    }
                    for r in audit
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Model support gates are separate from the sidebar minimum-wager display control. Filtering never retrains, reranks or creates independent validation."
        )
    split_labels = {
        "all": "All published rows (includes training)",
        "test": "Untouched test rows",
        "validation": "Validation rows (used for selection)",
        "train": "Training rows (in-sample)",
    }
    row_controls, population_control = st.columns([3, 2])
    split = row_controls.radio(
        "Published evaluation rows",
        list(split_labels),
        format_func=split_labels.get,
        horizontal=True,
        key="explanation_split",
    )
    selected = queries.explanation_rows(publication, candidates, split)
    reference = publication["predictions"]
    if not reference.empty and split != "all":
        reference = reference.loc[reference.split.eq(split)]
    populations = {(r["funding"], r["product"]): r for r in publication["audit"]["populations"]}
    choices = ["all", *[json.dumps(k) for k in populations]]
    if st.session_state.get("explanation_population") not in [None, *choices]:
        st.session_state.pop("explanation_population", None)
    chosen = population_control.selectbox(
        "Model population",
        choices,
        format_func=lambda v: "All model populations · separate views"
        if v == "all"
        else " · ".join(json.loads(v)).replace("_", " "),
        key="explanation_population",
    )
    if chosen != "all":
        funding, product = json.loads(chosen)
        selected = _population(selected, funding, product)
        if selected.empty:
            st.info(
                "No explanations for this population in the selected rows. Published status: "
                + populations[(funding, product)]["status"]
                + "; "
                + ", ".join(populations[(funding, product)]["reasons"])
            )
    tabs = st.tabs(
        [
            "Drivers & direction",
            "Feature patterns",
            "Local explanation",
            "Model quality & coverage",
            "Offline subset build",
        ]
    )
    with tabs[4]:
        st.subheader("Request a separate offline model")
        st.write(
            "Ordinary chart interaction never trains. A subset build is a new exploratory analysis with new support checks, model selection and background identity."
        )
        st.caption(
            "Export intersects the shared sidebar scope with the selected publication’s fitted scope and idea drill-through. Evaluation-row and model-population display controls do not define an offline training scope. Use Funding/Product in the sidebar to restrict a build."
        )
        request, blockers = queries.offline_scope(
            filters["selections"],
            filters["mode"],
            filters["product"],
            filters["saved"],
            idea,
            base_scope=publication["specification"]["scope"],
        )
        if blockers:
            for reason in blockers:
                st.info(reason)
        else:
            st.json(request, expanded=False)
            st.download_button(
                "Download offline scope request",
                json.dumps(request, indent=2),
                file_name="explanation-scope.json",
                mime="application/json",
                on_click="ignore",
            )
        with st.expander("Offline build instructions & pinned source"):
            st.code(
                "poetry run python scripts/build_explanatory_models.py \\\n  --canonical /path/to/matching-canonical-publication \\\n  --visuals /path/to/matching-visual-publication \\\n  --scope explanation-scope.json",
                language="bash",
            )
            st.json(
                {
                    "publication_id": ident,
                    "source_manifest_sha256": publication["specification"][
                        "source_manifest_sha256"
                    ],
                    "current_model_scope": publication["specification"]["scope"],
                }
            )
            st.caption(
                "An operator runs the build and explicitly attaches its immutable output in dashboard configuration. Repeated subset looks are not independent confirmation. No approval, launch or production change occurs here."
            )
    if selected.empty:
        with tabs[0]:
            st.info(
                "No published explanation rows match this selection. Clear a selector or review population blockers; no fitting or fallback occurs."
            )
        return
    if len(selected) < int(st.session_state.get("filter_min_wagers", 100)):
        st.info(
            "This is a small selected cohort below the signal display threshold. Frozen explanations remain inspectable; the selection has not passed a new support or statistical gate."
        )
    chart_revision = (
        str(st.session_state.get("workspace_chart_revision", 0))
        + "_"
        + str(st.session_state.get("explanation_chart_revision", 0))
    )
    sampled = queries.chart_sample(selected)
    summary = queries.importance(selected, reference)
    groups = list(selected.groupby(queries.CONTEXT, sort=True))
    with tabs[0]:
        signed = (
            st.radio(
                "Attribution summary",
                ["Magnitude", "Signed contribution"],
                horizontal=True,
                key="explanation_summary",
            )
            == "Signed contribution"
        )
        st.caption(
            "All selected rows determine summaries. Absolute contribution measures model reliance; a positive signed value raises the model prediction relative to its training background, not proven profitability."
        )
        for (funding, product, model_id, background_id), rows in groups:
            metadata = populations[(funding, product)]
            with st.container(border=True):
                st.subheader(funding.replace("_", " ").title() + " · " + product.replace("_", " "))
                st.write("**Model utility:** " + metadata["interpretation"])
                a, b, c = st.columns(3)
                a.metric("Selected wagers", f"{len(rows):,}")
                b.metric("Observed return", percent(rows.observed_return.mean()))
                c.metric("Model prediction", percent(rows.predicted_return.mean()))
                st.altair_chart(
                    charts.importance(
                        summary.loc[
                            summary.model_id.eq(model_id) & summary.background_id.eq(background_id)
                        ],
                        signed,
                    ),
                    width="stretch",
                )
                st.caption(
                    f"{metadata['selected_model']} · Model {model_id[:12]} · Background {background_id[:12]} · reference uses the same published evaluation-row selection."
                )
                with st.expander("Full model/reference IDs and explanation method"):
                    st.json(
                        {
                            k: metadata[k]
                            for k in ("model_id", "background_id", "explanation_method")
                        }
                    )
        with st.expander("Exact importance summaries"):
            st.dataframe(summary, hide_index=True, width="stretch")
    with tabs[1]:
        st.caption(
            f"Showing a stable display sample of {len(sampled):,} / {len(selected):,} wagers; summary bars and metrics use every selected row. Click a point, then open Local explanation. Plot sampling does not alter the model/background."
        )
        feature = st.selectbox(
            "Dependence feature",
            list(FEATURES),
            format_func=FEATURES.get,
            key="explanation_feature",
        )
        color = st.selectbox(
            "Color recorded context",
            ["sportsbook", "sport", "league", "market_family", "split"],
            key="explanation_color",
        )
        expanded = queries.distribution(sampled)
        for (funding, product, model_id, background_id), rows in sampled.groupby(
            queries.CONTEXT, sort=True
        ):
            st.subheader(funding.replace("_", " ").title() + " · " + product.replace("_", " "))
            _linked(
                charts.distribution(
                    expanded.loc[
                        expanded.model_id.eq(model_id) & expanded.background_id.eq(background_id)
                    ]
                ),
                "explanation_distribution_" + ident + model_id + background_id + chart_revision,
                rows,
            )
            recorded = rows[feature].notna() & ~rows[feature].astype(str).isin(
                ["", "__missing__", "<missing>"]
            )
            st.caption(
                f"Dependence support: {int(recorded.sum()):,} / {len(rows):,} sampled wagers have a recorded {FEATURES[feature].lower()}. Missing values remain in importance and coverage."
            )
            _linked(
                charts.dependence(rows, feature, FEATURES[feature], color),
                "explanation_dependence_" + ident + model_id + background_id + chart_revision,
                rows,
            )
        st.caption(
            "Dependence views plot recorded feature values against stored contributions—not counterfactual effects. Sparse support and correlated features limit interpretation. Trajectory SD describes permutation variability, not a confidence interval."
        )
    with tabs[2]:
        _local(selected, sampled, run)
    with tabs[3]:
        st.caption(
            "Utility is frozen whole-publication test evidence, not a new test of the sidebar-selected cohort. Calibration/residual plots describe selected rows and retain the displayed evaluation split."
        )
        for (funding, product, _model_id, _), rows in groups:
            st.subheader(funding.replace("_", " ").title() + " · " + product.replace("_", " "))
            utility = _population(publication["utility"], funding, product)
            left, right = st.columns(2)
            left.altair_chart(charts.utility(utility), width="stretch")
            right.altair_chart(charts.calibration(rows), width="stretch")
            st.altair_chart(charts.residuals(rows, "sportsbook"), width="stretch")
            fixed = [
                {
                    "Feature": label,
                    "Selected distinct values": rows[f].nunique(dropna=False),
                    "Fixed in selected cohort": rows[f].nunique(dropna=False) <= 1,
                }
                for f, label in FEATURES.items()
            ]
            st.dataframe(pd.DataFrame(fixed), hide_index=True, width="stretch")
            st.caption(
                "A fixed selected feature can still contribute relative to the original training background. It does not demonstrate a within-subset effect."
            )
        with st.expander("Published utility, coverage, support and sensitivity tables"):
            for table in (
                "utility",
                "coverage",
                "training_features",
                "learning_curves",
                "permutation",
                "redundancy",
                "source_periods",
            ):
                st.write(
                    "**" + table.replace("_", " ").title() + " · entire selected publication**"
                )
                st.dataframe(publication[table], hide_index=True, width="stretch")
            st.caption(
                "Source-period errors use unverified literal calendar labels; they are not forward validation. Learning curves, permutation repeats and correlated-feature checks are descriptive diagnostics, not significance tests."
            )
        with st.expander("Feature eligibility & excluded inputs"):
            st.caption(
                "Timing, liquidity, CLV and historical scanner settings remain excluded pending verified decision-time evidence."
            )
            st.dataframe(publication["feature_inventory"], hide_index=True, width="stretch")
        with st.expander("Interpretation limitations"):
            for text in publication["specification"]["limitations"]:
                st.write("• " + text)
        st.download_button(
            "Download explanation provenance",
            json.dumps(
                {
                    "publication_id": ident,
                    "specification": publication["specification"],
                    "display_split": split,
                    "selection": filters,
                    "candidate_id": idea_id,
                    "models_and_backgrounds": selected[queries.CONTEXT]
                    .drop_duplicates()
                    .to_dict("records"),
                },
                indent=2,
            ),
            file_name="explanation-provenance.json",
            mime="application/json",
            on_click="ignore",
        )
        st.download_button(
            "Download selected frozen explanations",
            selected.to_csv(index=False),
            file_name="selected-frozen-explanations.csv",
            mime="text/csv",
            on_click="ignore",
        )
