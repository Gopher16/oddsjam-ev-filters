# Offline explanatory model publications

These models describe recorded unit returns. Their SHAP values explain a fitted
prediction relative to a training background. They do not prove why a bet wins,
identify a causal intervention, supply missing OddsJam settings or authorize a
production change. Current interactive views are documented in [dashboard explanations](DASHBOARD_EXPLANATIONS.md).

## Approved exploratory contract

Use the [approved exploratory model policy](reference/policies/EXPLORATORY.md#explanatory-models)
for support gates, splits, frozen-model selection, references and background bounds.
No model-utility flag is prospective or production authorization.

## Features and interpretation

The [model/explanation reference](reference/MODELS_EXPLANATIONS.md) is authoritative
for feature exclusions, preprocessing, approximate grouped forest SHAP, analytic
linear SHAP, output units, additivity and interpretation. Build and artifact details
remain below; the dashboard consumes these outputs through
[interactive explanations](DASHBOARD_EXPLANATIONS.md).

## Build and inspect

Use explicit, mutually pinned existing publications; no acquisition or scheduling:

```bash
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
validation, timing provenance and independent prospective confirmation remain
separate work. Interactive presentation is implemented in
[dashboard explanations](DASHBOARD_EXPLANATIONS.md).

Method references: [SHAP permutation explanations](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html),
[TreeExplainer output and additivity](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html),
and [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
The forest estimator here implements a bounded original-feature permutation estimator;
it does not claim to reproduce exact TreeExplainer values.
