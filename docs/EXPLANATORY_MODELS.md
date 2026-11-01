# Offline explanatory model publications

These models describe recorded unit returns. Their SHAP values explain a fitted
prediction relative to a training background. They do not prove why a bet wins,
identify a causal intervention, supply missing OddsJam settings or authorize a
production change. Dashboard explanation views follow in PR F.

## Approved exploratory contract

The owner approved this initial policy on November 1, 2026:

- Separate each source `execution_status` / `product` population. All-history means
  reviewing every population, not blending hypothetical and funded economics.
- One canonical wager per example. Use the existing diagnostic-eligible economics:
  finite reported profit divided by positive stake, with equal-wager training and
  evaluation weights. Reconcile the target before fitting; no stake predictor.
- At least 500 eligible wagers and 200 namespace/event groups, complete event IDs,
  and at least 100/40/40 train/validation/test events. These are operational support
  gates, not guarantees of power or out-of-sample usefulness.
- Stable, seeded event-grouped 60/20/20 splits. Event keys cannot cross splits in a
  population. Chronology is unverified, so this is **exploratory non-temporal
  evaluation**, never forward validation. Source calendar plots do not fix this.
- Constant training-mean and advertised-EV references; Ridge alpha 10 and one random
  forest with 64 trees, depth 4, minimum leaf 25 and one worker. Select constant,
  Ridge or forest by validation MSE only, with no hyperparameter search. The EV
  reference is a comparison, not an outcome-derived prediction. Missing EV falls
  back to the training mean and is disclosed in coverage.
- Keep the selected training model frozen; do not refit on the held-out test set.
  Every predicted row retains train/validation/test identity. Training predictions
  are not held-out evidence. Test error must beat **both** references to receive the
  limited `test_improves_both_references` utility flag; this is not a significance
  test, corrected discovery result, or production approval.
- Use deterministic, training-only background selection (16 rows by default,
  maximum 128), with model and background identities. Report output and SHAP in
  **unit-return fractions**, not percentages, dollars, probabilities or log odds.

## Features and interpretation

`feature_inventory.parquet` reviews every column in the verified canonical tables
and derived model candidates, with source grain, dtype, coverage, units and explicit
exclusion reasons. The original canonical feature catalogue remains the authority
for raw field semantics. Counts from `source_features` are observations, not extra
independent wagers. Source hashes pin both the canonical and visual publications.

Initial predictors are platform, sport, league, market family, original continuous
advertised EV as a fraction, and offered American odds converted to decimal payout.
Categories are encoded using training data only, with at most 24 categories per
field and explicit unseen-category audits. Numeric missing values use training
medians plus indicators; standardization is fit only on training rows. Missing
categories have a visible sentinel, not a fabricated platform or league.

Forest explanations use seeded antithetic **grouped permutation SHAP**, two feature
orders per training-background row. The original six features are the coalitions;
one-hot blocks stay together. This is an approximation, not exact Tree SHAP. Each
row includes trajectory sampling SD (not a confidence interval), and contributions
must reconstruct its prediction to 1e-6. Ridge explanations are analytic
interventional linear SHAP with encoded contributions summed to original features.
The initial historical check exposed a TreeExplainer additivity mismatch; the build
rejected it rather than disabling the check. The bounded permutation estimator uses
the frozen estimator's predictions directly and preserves the additivity guard.

Outcomes, settlement, payouts, profit, CLV, actual stake, IDs and free-text labels
cannot enter predictors. Exact market names remain linkage/context in this initial
bounded feature specification; market family is the supported representation.
Timing, weekdays, liquidity, market width, historical devig/formula and historical
configuration features require better decision-time evidence before inclusion.
Current settings or saved-filter names cannot establish past settings.

One-hot contributions and missing indicators are summed to their original feature.
Sport and league remain related predictors; grouping encoded levels does not remove
correlation or make feature effects causal. Test-set marginal permutation sensitivity
provides a complementary check, with three deterministic repeats, not confidence
intervals or significance. Residual/cohort and calibration tables expose limitations.

## Build and inspect

Use explicit, mutually pinned existing publications; no acquisition or scheduling:

```sh
poetry run python scripts/build_explanatory_models.py \
  --canonical /absolute/path/to/canonical-publication \
  --visuals /absolute/path/to/visual-publication \
  --output-root data/processed/explanatory_models
```

To refit a specific scope, save a JSON request and pass `--scope request.json`:

```json
{
  "execution_status": ["real_money"],
  "product": ["exchange"],
  "sportsbook": ["Novig"],
  "advertised_ev_fraction": {"gt": 0.03}
}
```

Categorical scopes accept nonempty exact-value lists for platform, sport, league,
market family, execution status and product. Numeric scopes accept `gt`, `ge`, `lt`
and `le` for EV fraction and decimal offered odds. Fields combine with AND; category
values combine with OR. Unsupported fields/operators fail closed. A sparse scope
publishes a decline reason. A subset refit has a new model/reference; comparing its
SHAP bars with global bars does not hold the baseline constant. Repeated scoped
looks reuse historical information and are not independent confirmation.

For future sidebar filtering without refitting, load an exact publication using
`load_model_publication` and call `select_published_predictions` with canonical wager
IDs. It returns stored rows with unchanged model/background identities. Unknown or
duplicate row IDs fail rather than triggering training.

## Artifact and resource contract

Content-addressed output pins source manifests, policy, scope, modeling code and
runtime versions. Atomic locked writes publish only verified complete artifacts;
source publications remain unchanged. Replay verifies hashes and rows rather than
training again. Corrupt files fail closed. Private data/model outputs remain ignored.
No executable pickle, joblib or UI estimator loading is used.

- `feature_inventory`: all reviewed columns and exclusions.
- `predictions`: canonical/event IDs, split, feature values, observed/predicted unit
  returns, residual, SHAP reference, grouped contributions, background membership,
  reconstruction error and model/background IDs.
- `utility`: train/validation/test MSE and MAE for each model and reference.
- `importance`: signed and absolute grouped attribution, separately by split.
- `calibration`: observed versus predicted returns in fixed prediction bands.
- `coverage`, `training_features`: unseen/missing features and constant-field support.
- `residual_groups`, `source_periods`, `permutation`: cohort/calendar diagnostics
  and held-out sensitivity; source calendar is explicitly unverified.
- `learning_curves`, `redundancy`: fixed half/full training-event support diagnostics
  and training-only category dependence/numeric correlation.
- `eligibility`: scoped canonical row identities, economic eligibility, event-ID
  completeness, population status and decline reasons.
- `models.json`: safe numeric tree/linear representation, preprocessing and fingerprints.
- `audit.json`: eligible/excluded counts, population declines, utility and interpretation.

Bounds: 100,000 source/scope rows, 12 populations, 256 encoded columns, 128 background
rows and 650 million row × background × order × feature × tree explanation operations per
population. Trees/depth,
category vocabulary and permutation repeats are fixed; no uncontrolled tuning,
interactions, subset expansion or silent row subsampling. Bounds are hard operational
limits rather than a wall-clock or operating-system memory guarantee. Failures leave
only a failure audit, never a partial consumable publication. A population that
cannot be supported declines with reasons instead of a fabricated model.

The model does not alter Holm-corrected comparisons, recommendation queues, saved
filters, prospective launches or production lifecycle evidence. Future chronological
validation, timing provenance, independent prospective confirmation and interactive
SHAP presentation remain separate work.

Method references: [SHAP permutation explanations](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html),
[TreeExplainer output and additivity](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html),
and [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
The forest estimator here implements a bounded original-feature permutation estimator;
it does not claim to reproduce exact TreeExplainer values.
