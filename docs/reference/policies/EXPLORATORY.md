# Approved exploratory policies

These historical-research policies are separate from the
[prospective bounded-pilot policy](PROMOTION.md). Neither retrospective corrected
associations nor model utility qualifies a definition for production approval.
This reference relocates the approved policy text without changing configurations.

## Core discovery is a separate policy

[Core discovery configuration](../../../configs/discovery_policy.yaml) retains its
100 event-group gate, 5,000 resamples, 95% marginal intervals and Holm alpha 0.05.
Its `min_eligible_stake_usd: null` deliberately blocks formal tests; its empty
windows and null embargo do not authorize temporal validation. The later arm
policy below does not fill this unset core stake floor. The dashboard's default
100-wager display control is independent of both policies.

## Within-filter prioritization

### Approval and gates

The user approved this policy on October 30, 2026. Settings are versioned in
[configs/signal_prioritization.yaml](../../../configs/signal_prioritization.yaml) and included in publication identity.

| Gate or setting | Initial value |
| --- | --- |
| Eligible wagers in each arm | At least 100 |
| Namespace/event groups in each arm | At least 100 |
| Event-ID coverage of eligible wagers | 100% |
| Largest wager / arm stake | At most 5% |
| Largest event / arm stake | At most 10% |
| Stakes | Positive finite values; no additional dollar floor |
| Paired event-bootstrap draws | 5,000 |
| Percentile interval | 95%, pointwise |
| Multiple-comparison correction | Holm, alpha 0.05 |
| Family | Every frozen candidate × two return measures × two directions |

These are operational screening gates, not a power guarantee. Unknown historical
sizing regimes remain unknown and are counted. Eligibility inherits the sizing
diagnostics contract; event coverage does not establish event independence.

## Explanatory models

### Approval and model bounds

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

## Evidence and changes

[Statistical evidence](../STATISTICAL_EVIDENCE.md) defines complete families and
interval interpretation. [Model explanations](../MODELS_EXPLANATIONS.md) defines
feature and SHAP semantics. Support gates are operational, not a power guarantee.
No configuration, research result or approval was changed in this consolidation.
Policy revisions require explicit review, a versioned policy and new publications;
editing prose does not retrospectively authorize a different analysis.

Implementation: [core parser](../../../src/oddsjam_ev/analysis/discovery_config.py),
[arm-policy parser](../../../src/oddsjam_ev/analysis/prioritization_config.py),
[model policy and fits](../../../src/oddsjam_ev/analysis/explanatory_models.py).
Verification: [core configuration tests](../../../tests/test_discovery_config.py),
[arm comparisons](../../../tests/test_statistical_prioritization.py),
[models](../../../tests/test_explanatory_models.py).

[Contract reference](../README.md)
