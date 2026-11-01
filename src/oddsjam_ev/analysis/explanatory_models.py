"""
explanatory_models.py
===============================================================================
Purpose
-------
Bound exploratory return models and training-reference feature explanations.

What this module provides
-------------------------
Decision-time feature audits, event-disjoint evaluation and additive explanations.

Design Notes
------------
Equal-wager associations are not causal effects, forward validation or scanner rules.
===============================================================================
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oddsjam_ev.io.source_csv import fingerprint

CATEGORICAL = ("sportsbook", "sport", "league", "market_family")
NUMERIC = ("advertised_ev_fraction", "offered_decimal_odds")
FEATURES = (*CATEGORICAL, *NUMERIC)
LIMITATIONS = [
    "Exploratory event-grouped random split; chronology is unverified, not forward validation.",
    "Tracked wagers are a selected sample, not the full scanner opportunity population.",
    "SHAP explains this model relative to its training background, not causes or deployable edges.",
    "Funding/product labels are source attribution, not independent proof of real execution.",
    "Correlated predictors can share or exchange importance; subset refits change the reference.",
    "Fees, liquidity, availability, historical devig and placement-time settings are unverified.",
    "Repeated subset searches are exploratory and do not create independent confirmation.",
]


@dataclass(frozen=True)
class ModelPolicy:
    """Approved exploratory support and resource limits.

    Attributes
    ----------
    min_wagers, min_events : int
        Population support gates, not power guarantees.
    max_rows, max_encoded_features : int
        Hard fit/explanation resource limits; no silent sampling.
    background_rows : int
        Deterministic training-only SHAP reference size, at most 128.
    """

    min_wagers: int = 500
    min_events: int = 200
    min_train_events: int = 100
    min_validation_events: int = 40
    min_test_events: int = 40
    background_rows: int = 16
    max_rows: int = 100000
    max_encoded_features: int = 256
    max_populations: int = 12
    max_explanation_work: int = 650000000
    seed: int = 20261101

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Model policy values must be positive integers")
        if self.background_rows > 128:
            raise ValueError("Background exceeds approved 128-row bound")


def apply_scope(frame: pd.DataFrame, scope: dict) -> pd.DataFrame:
    """Select an explicit offline subset using decision-time fields only.

    Parameters
    ----------
    frame : pandas.DataFrame
        Candidate rows.
    scope : dict
        Field to nonempty list of exact categorical values. Numeric restrictions
        use a dictionary of gt/ge/lt/le boundaries in native feature units.

    Returns
    -------
    pandas.DataFrame
        Exact selected rows; unknown fields/operators fail closed.
    """
    result = frame.copy()
    for field, values in scope.items():
        if field in (*CATEGORICAL, "execution_status", "product"):
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(v, str) for v in values)
            ):
                raise ValueError("Categorical scopes require nonempty string lists")
            result = result[result[field].isin(values)]
        elif field in NUMERIC:
            if not isinstance(values, dict) or not values:
                raise ValueError("Numeric scopes require explicit bounds")
            for op, value in values.items():
                if (
                    op not in ("gt", "ge", "lt", "le")
                    or not isinstance(value, (float, int))
                    or not np.isfinite(value)
                ):
                    raise ValueError("Unsupported scope boundary")
                result = result[getattr(result[field], op)(value)]
        else:
            raise ValueError(f"Unsupported subset feature: {field}")
    return result.copy()


def prepare_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive finite decimal odds without using outcome or closing-price features.

    Parameters
    ----------
    frame : pandas.DataFrame
        Verified visual candidates, one per canonical wager.

    Returns
    -------
    pandas.DataFrame
        Stable row ordering and explicit decision-time transforms.
    """
    required = {
        "bet_id",
        "namespace",
        "game_id",
        "execution_status",
        "product",
        "diagnostic_eligible",
        "stake",
        "profit",
        "unit_return",
        "odds",
        *CATEGORICAL,
        "advertised_ev_fraction",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing model columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    if result.bet_id.isna().any() or result.bet_id.duplicated().any():
        raise ValueError("Model grain must be unique canonical wagers")
    odds = pd.to_numeric(result.odds, errors="coerce")
    result["offered_decimal_odds"] = np.where(
        odds > 0, 1 + odds / 100, np.where(odds <= -100, 1 + 100 / odds.abs(), np.nan)
    )
    result.loc[odds.between(0, 99.999, inclusive="right"), "offered_decimal_odds"] = np.nan
    result["event_key"] = [
        fingerprint([str(n), str(g)])
        if pd.notna(n) and pd.notna(g) and str(n).strip() and str(g).strip()
        else None
        for n, g in zip(result.namespace, result.game_id, strict=True)
    ]
    for feature in NUMERIC:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    for feature in CATEGORICAL:
        result[feature] = result[feature].fillna("__missing__").astype(str)
    return result.sort_values("bet_id").reset_index(drop=True)


def audit_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inventory every supplied source/derived column with explicit eligibility.

    Parameters
    ----------
    tables : dict
        Named verified canonical and derived tables.

    Returns
    -------
    pandas.DataFrame
        Column-level coverage, grain, units and exclusion reasons. Raw observation
        fields are inventoried without interpreting their count as wager support.
    """
    rows = []
    for table, frame in tables.items():
        for field in frame.columns:
            eligible = table == "model_candidates" and field in FEATURES
            reason = (
                "Allowlisted pre-outcome observation; placement semantics not independently verified"
                if eligible
                else "Not allowlisted: linkage, outcome, duplicate representation, unsupported decision-time provenance or fixed population context"
            )
            if field in (
                "liquidity",
                "liquidity_usd",
                "market_width",
                "created_at",
                "source_weekday",
                "time_to_event_hours",
                "event_start_date",
                "clv",
                "clv_source",
                "saved_filter_names",
            ):
                reason = "Excluded: timing/settings/closing/liquidity provenance unsupported; requires verified decision-time evidence"
            rows.append(
                {
                    "table": table,
                    "field": field,
                    "dtype": str(frame[field].dtype),
                    "rows": len(frame),
                    "nonmissing": int(frame[field].notna().sum()),
                    "distinct": int(frame[field].astype(str).nunique()),
                    "model_eligible": eligible,
                    "availability": "source pre-outcome, chronology unverified"
                    if eligible
                    else "not established for modeling",
                    "units": "unit-return fraction"
                    if field == "advertised_ev_fraction"
                    else "decimal total payout per stake"
                    if field == "offered_decimal_odds"
                    else "source native; see canonical feature_catalog",
                    "grain": "unique wager"
                    if table in ("bet_facts", "model_candidates")
                    else "source table grain; not independent wagers",
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def _preprocessor():
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", max_categories=24, sparse_output=False
                ),
                list(CATEGORICAL),
            ),
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                    StandardScaler(),
                ),
                list(NUMERIC),
            ),
        ],
        sparse_threshold=0,
    )


def _safe_model(model, name, preprocessor):
    payload = {"kind": name, "encoded_names": list(preprocessor.get_feature_names_out())}
    if name == "ridge":
        payload.update(coef=model.coef_.tolist(), intercept=float(model.intercept_))
    elif name == "forest":
        payload["trees"] = [
            {
                "left": tree.tree_.children_left.tolist(),
                "right": tree.tree_.children_right.tolist(),
                "feature": tree.tree_.feature.tolist(),
                "threshold": tree.tree_.threshold.tolist(),
                "value": tree.tree_.value.ravel().tolist(),
            }
            for tree in model.estimators_
        ]
    else:
        payload["constant"] = float(model)
    encoder = preprocessor.named_transformers_["categorical"]
    numeric = preprocessor.named_transformers_["numeric"]
    payload["preprocessing"] = {
        "categories": [a.tolist() for a in encoder.categories_],
        "infrequent_categories": [
            None if a is None else a.tolist() for a in encoder.infrequent_categories_
        ],
        "imputation": numeric[0].statistics_.tolist(),
        "indicator_features": numeric[0].indicator_.features_.tolist(),
        "scale_mean": numeric[1].mean_.tolist(),
        "scale": numeric[1].scale_.tolist(),
    }
    return payload


def _predict(model, x):
    # einsum avoids spurious platform BLAS matmul floating-point status warnings.
    if isinstance(model, Ridge):
        return np.einsum("ij,j->i", x, model.coef_) + model.intercept_
    return model.predict(x)


def permutation_shap(model, x, background, groups, seed):
    """Estimate grouped interventional SHAP with bounded antithetic permutations.

    Parameters
    ----------
    model : estimator
        Frozen fitted forest with predict support.
    x, background : numpy.ndarray
        Encoded rows and exclusively training-derived reference rows.
    groups : dict
        Original-feature names to disjoint encoded column indexes.
    seed : int
        Fixed feature-order seed, separate from any evaluation outcome.

    Returns
    -------
    tuple
        Mean grouped contributions, trajectory sampling SD and reference output.
        Every trajectory telescopes exactly. Sampling SD is not a confidence bound.
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros((len(x), len(groups)))
    m2 = np.zeros_like(mean)
    count = 0
    columns = list(groups.values())
    reference = float(model.predict(background).mean())
    for reference_row in background:
        order = rng.permutation(len(groups))
        for traversal in (order, order[::-1]):
            masked = np.broadcast_to(reference_row, x.shape).copy()
            previous = model.predict(masked)
            trajectory = np.zeros_like(mean)
            for feature in traversal:
                masked[:, columns[feature]] = x[:, columns[feature]]
                prediction = model.predict(masked)
                trajectory[:, feature] = prediction - previous
                previous = prediction
            count += 1
            difference = trajectory - mean
            mean += difference / count
            m2 += difference * (trajectory - mean)
    return mean, np.sqrt(m2 / max(count - 1, 1)), reference


def fit_population(frame: pd.DataFrame, policy: ModelPolicy) -> dict:
    """Fit one population, select on validation and explain the frozen train model.

    Parameters
    ----------
    frame : pandas.DataFrame
        Prepared rows from exactly one funding/product population.
    policy : ModelPolicy
        Approved bounded policy; tests may explicitly lower synthetic support gates.

    Returns
    -------
    dict
        Decline reasons or safe model metadata, row predictions, grouped SHAP,
        held-out utility, calibration, and training-background membership.
    """
    if frame[["execution_status", "product"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Fit populations separately by funding/product")
    eligible = frame.diagnostic_eligible.fillna(False).eq(True)
    economic = (
        (frame.stake > 0)
        & np.isfinite(frame.unit_return)
        & np.isfinite(frame.profit)
        & np.isfinite(frame.stake)
    )
    if (eligible & ~economic).any():
        raise ValueError("Eligible economics are not finite positive-stake returns")
    data = frame[eligible].copy().sort_values("bet_id").reset_index(drop=True)
    if not np.allclose(data.unit_return, data.profit / data.stake):
        raise ValueError("Target does not reconcile to reported profit / stake")
    reasons = []
    if len(data) < policy.min_wagers:
        reasons.append("insufficient_wagers")
    if data.event_key.isna().any():
        reasons.append("incomplete_event_ids")
    events = sorted(data.event_key.dropna().unique())
    if len(events) < policy.min_events:
        reasons.append("insufficient_event_groups")
    if len(data) > policy.max_rows:
        reasons.append("row_budget_exceeded")
    if reasons:
        return {
            "status": "declined",
            "reasons": reasons,
            "eligible_wagers": len(data),
            "event_groups": len(events),
        }
    events = np.random.default_rng(policy.seed).permutation(events)
    train_end, validation_end = int(len(events) * 0.6), int(len(events) * 0.8)
    event_splits = {
        e: "train" if i < train_end else "validation" if i < validation_end else "test"
        for i, e in enumerate(events)
    }
    data["split"] = data.event_key.map(event_splits)
    for split, minimum in (
        ("train", policy.min_train_events),
        ("validation", policy.min_validation_events),
        ("test", policy.min_test_events),
    ):
        if data.loc[data.split == split, "event_key"].nunique() < minimum:
            reasons.append(f"insufficient_{split}_events")
    if reasons:
        return {
            "status": "declined",
            "reasons": reasons,
            "eligible_wagers": len(data),
            "event_groups": len(events),
        }
    train, validation = data.split.eq("train").to_numpy(), data.split.eq("validation").to_numpy()
    if not any(data.loc[train, f].nunique(dropna=False) > 1 for f in FEATURES):
        return {
            "status": "declined",
            "reasons": ["no_estimable_feature_variation"],
            "eligible_wagers": len(data),
            "event_groups": len(events),
        }
    preprocessor = _preprocessor()
    preprocessor.fit(data.loc[train, list(FEATURES)])
    encoded_width = len(preprocessor.get_feature_names_out())
    if encoded_width > policy.max_encoded_features:
        return {
            "status": "declined",
            "reasons": ["encoded_feature_budget_exceeded"],
            "eligible_wagers": len(data),
            "event_groups": len(events),
        }
    x = preprocessor.transform(data[list(FEATURES)])
    y = data.unit_return.to_numpy(float)
    constant = float(y[train].mean())
    models = {
        "constant": constant,
        "ridge": Ridge(alpha=10),
        "forest": RandomForestRegressor(
            n_estimators=64, max_depth=4, min_samples_leaf=25, random_state=policy.seed, n_jobs=1
        ),
    }
    predictions = {"constant": np.full(len(data), constant)}
    for name in ("ridge", "forest"):
        models[name].fit(x[train], y[train])
        predictions[name] = _predict(models[name], x)
    predictions["advertised_ev_reference"] = data.advertised_ev_fraction.fillna(constant).to_numpy(
        float
    )
    errors = {
        name: float(np.mean((p[validation] - y[validation]) ** 2))
        for name, p in predictions.items()
        if name in models
    }
    selected = min(errors, key=lambda name: (errors[name], list(models).index(name)))
    p = predictions[selected]
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Nonfinite model/reference predictions")
    utility = []
    for name, prediction in predictions.items():
        for split in ("train", "validation", "test"):
            mask = data.split.eq(split).to_numpy()
            utility.append(
                {
                    "model": name,
                    "split": split,
                    "wagers": int(mask.sum()),
                    "events": int(data.loc[mask, "event_key"].nunique()),
                    "mse": float(np.mean((prediction[mask] - y[mask]) ** 2)),
                    "mae": float(np.mean(abs(prediction[mask] - y[mask]))),
                    "selected": selected == name,
                }
            )
    background_indices = np.flatnonzero(train)
    background_indices = np.sort(
        np.random.default_rng(policy.seed).choice(
            background_indices, min(len(background_indices), policy.background_rows), replace=False
        )
    )
    background = x[background_indices]
    # Linear interventional SHAP is analytic. Forest explanations use a bounded
    # antithetic permutation estimator over original feature groups, preserving
    # valid one-hot blocks rather than intervening on individual encoded levels.
    if selected == "ridge":
        values = (x - background.mean(axis=0)) * models[selected].coef_
        base = float(
            models[selected].intercept_
            + np.einsum("i,i->", background.mean(axis=0), models[selected].coef_)
        )
    elif selected == "constant":
        values, base = np.zeros_like(x), constant
    else:
        values, base = None, None
    encoded_names = preprocessor.get_feature_names_out()
    groups = {f: [] for f in FEATURES}
    for i, name in enumerate(encoded_names):
        matches = [
            f
            for f in FEATURES
            if name == f"numeric__{f}"
            or name == f"numeric__missingindicator_{f}"
            or name.startswith(f"categorical__{f}_")
        ]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous encoded feature: {name}")
        groups[matches[0]].append(i)
    approximation_sd = np.zeros((len(data), len(FEATURES)))
    if selected == "forest":
        work = len(data) * len(background) * 2 * len(FEATURES) * 64
        if work > policy.max_explanation_work:
            return {
                "status": "declined",
                "reasons": ["explanation_work_budget_exceeded"],
                "eligible_wagers": len(data),
                "event_groups": len(events),
            }
        grouped_values, approximation_sd, base = permutation_shap(
            models[selected], x, background, groups, policy.seed
        )
    else:
        grouped_values = np.column_stack(
            [values[:, indexes].sum(axis=1) for indexes in groups.values()]
        )
    if not np.allclose(base + grouped_values.sum(axis=1), p, atol=1e-6, rtol=1e-6):
        raise ValueError("SHAP reconstruction failed")
    safe = _safe_model(models[selected], selected, preprocessor)
    model_id = fingerprint(safe)
    background_ids = data.iloc[background_indices].bet_id.tolist()
    background_id = fingerprint(
        {"model_id": model_id, "bet_ids": background_ids, "encoded_rows": background.tolist()}
    )
    context_columns = [
        c for c in ("source_date", "source_weekday", "source_date_status") if c in data
    ]
    records = data[["bet_id", "event_key", "split", *FEATURES, *context_columns]].copy()
    records["observed_return"] = y
    records["predicted_return"] = p
    records["base_return"] = base
    records["residual"] = y - p
    records["background_member"] = records.bet_id.isin(background_ids)
    records["model_id"], records["background_id"] = model_id, background_id
    for i, f in enumerate(groups):
        records[f"shap__{f}"] = grouped_values[:, i]
        records[f"shap_sampling_sd__{f}"] = approximation_sd[:, i]
    records["reconstruction_error"] = (
        p - base - records[[f"shap__{f}" for f in FEATURES]].sum(axis=1)
    )
    summaries = [
        {
            "split": split,
            "feature": f,
            "mean_abs_shap": float(records.loc[records.split == split, f"shap__{f}"].abs().mean()),
            "mean_shap": float(records.loc[records.split == split, f"shap__{f}"].mean()),
        }
        for split in ("train", "validation", "test")
        for f in FEATURES
    ]
    test_errors = {r["model"]: r["mse"] for r in utility if r["split"] == "test"}
    useful = selected != "constant" and test_errors[selected] < min(
        test_errors["constant"], test_errors["advertised_ev_reference"]
    )
    calibration = (
        records.assign(
            predicted_band=pd.cut(
                records.predicted_return, [-np.inf, -0.1, 0, 0.025, 0.05, 0.1, np.inf]
            ).astype(str)
        )
        .groupby(["split", "predicted_band"], observed=True)
        .agg(
            wagers=("bet_id", "size"),
            observed_return=("observed_return", "mean"),
            predicted_return=("predicted_return", "mean"),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
    )
    coverage = []
    for split in ("validation", "test"):
        for f in FEATURES:
            subset = data[data.split == split]
            coverage.append(
                {
                    "split": split,
                    "feature": f,
                    "missing": int((subset[f].isna() | subset[f].eq("__missing__")).sum()),
                    "unseen_category": int((~subset[f].isin(data.loc[train, f])).sum())
                    if f in CATEGORICAL
                    else 0,
                }
            )
    # Descriptive held-out diagnostics, never selection inputs or inferential tests.
    residual_groups = []
    for feature in CATEGORICAL:
        for (split, value), cohort in records.groupby(["split", feature], observed=True):
            residual_groups.append(
                {
                    "split": split,
                    "feature": feature,
                    "value": value,
                    "wagers": len(cohort),
                    "events": cohort.event_key.nunique(),
                    "mean_residual": float(cohort.residual.mean()),
                    "mae": float(cohort.residual.abs().mean()),
                }
            )
    permutation = []
    test = data.split.eq("test").to_numpy()
    baseline_mse = float(np.mean((p[test] - y[test]) ** 2))
    for feature in FEATURES:
        changes = []
        for repeat in range(3):
            permuted = data.loc[test, list(FEATURES)].copy()
            permuted[feature] = np.random.default_rng(policy.seed + repeat).permutation(
                permuted[feature].to_numpy()
            )
            transformed = preprocessor.transform(permuted)
            predicted = (
                np.full(len(permuted), constant)
                if selected == "constant"
                else _predict(models[selected], transformed)
            )
            changes.append(float(np.mean((predicted - y[test]) ** 2)) - baseline_mse)
        permutation.append(
            {
                "feature": feature,
                "test_mse_increase_mean": float(np.mean(changes)),
                "test_mse_increase_min": min(changes),
                "test_mse_increase_max": max(changes),
                "repeats": 3,
                "interpretation": "marginal permutation sensitivity, not causal or inferential; correlations may distort importance",
            }
        )
    training_features = []
    for feature in FEATURES:
        train_values = data.loc[train, feature]
        training_features.append(
            {
                "feature": feature,
                "distinct_train": int(train_values.nunique(dropna=False)),
                "missing_train": int(train_values.isna().sum()),
                "constant_train": train_values.nunique(dropna=False) <= 1,
            }
        )
    learning = []
    for fraction in (0.5, 1.0):
        selected_events = set(events[: max(1, int(train_end * fraction))])
        learning_train = data.event_key.isin(selected_events).to_numpy()
        learning_preprocessor = _preprocessor()
        learning_preprocessor.fit(data.loc[learning_train, list(FEATURES)])
        learning_x = learning_preprocessor.transform(data[list(FEATURES)])
        for name in models:
            if name == "constant":
                learning_predictions = np.full(len(data), y[learning_train].mean())
            else:
                learning_model = clone(models[name])
                learning_model.fit(learning_x[learning_train], y[learning_train])
                learning_predictions = _predict(learning_model, learning_x)
            learning.append(
                {
                    "model": name,
                    "training_event_fraction": fraction,
                    "train_events": len(selected_events),
                    "train_wagers": int(learning_train.sum()),
                    "validation_mse": float(
                        np.mean((learning_predictions[validation] - y[validation]) ** 2)
                    ),
                    "role": "descriptive support diagnostic; no test access or additional model selection",
                }
            )
    redundancy = []
    from sklearn.metrics import normalized_mutual_info_score

    for i, left in enumerate(CATEGORICAL):
        for right in CATEGORICAL[i + 1 :]:
            redundancy.append(
                {
                    "left": left,
                    "right": right,
                    "measure": "training normalized mutual information",
                    "value": float(
                        normalized_mutual_info_score(data.loc[train, left], data.loc[train, right])
                    ),
                }
            )
    numeric_correlation = data.loc[train, list(NUMERIC)].corr().iloc[0, 1]
    redundancy.append(
        {
            "left": NUMERIC[0],
            "right": NUMERIC[1],
            "measure": "training Pearson correlation; null for constant/missing",
            "value": float(numeric_correlation) if pd.notna(numeric_correlation) else None,
        }
    )
    periods = []
    if "source_date" in records:
        for (split, period), cohort in records.assign(
            source_period=records.source_date.astype(str).str[:7]
        ).groupby(["split", "source_period"]):
            periods.append(
                {
                    "split": split,
                    "source_period": period,
                    "wagers": len(cohort),
                    "mae": float(cohort.residual.abs().mean()),
                    "mean_residual": float(cohort.residual.mean()),
                    "time_basis": "unverified source literal calendar; descriptive only, not chronological validation",
                }
            )
    return {
        "status": "published",
        "reasons": [],
        "eligible_wagers": len(data),
        "event_groups": len(events),
        "selected_model": selected,
        "test_improves_both_references": useful,
        "interpretation": "exploratory predictive association only"
        if useful
        else "no demonstrated held-out predictive utility; explanations describe the model only",
        "model_id": model_id,
        "background_id": background_id,
        "explanation_method": "antithetic grouped permutation SHAP; 2 orders per background; sampling SD is descriptive not a confidence interval"
        if selected == "forest"
        else "analytic interventional linear SHAP"
        if selected == "ridge"
        else "constant zero contributions",
        "safe_model": safe,
        "records": records,
        "utility": pd.DataFrame(utility),
        "importance": pd.DataFrame(summaries),
        "calibration": calibration,
        "coverage": pd.DataFrame(coverage),
        "residual_groups": pd.DataFrame(residual_groups),
        "permutation": pd.DataFrame(permutation),
        "training_features": pd.DataFrame(training_features),
        "learning_curves": pd.DataFrame(learning),
        "redundancy": pd.DataFrame(redundancy),
        "source_periods": pd.DataFrame(periods),
        "limitations": LIMITATIONS,
    }
