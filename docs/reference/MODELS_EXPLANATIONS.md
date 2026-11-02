# Models and explanations

## Target, population and utility

Use the [approved exploratory model policy](policies/EXPLORATORY.md#explanatory-models)
for the exact support gates, grouped 60/20/20 split, Ridge/forest bounds, constant and
advertised-EV references, validation-only selection and untouched-test utility flag.
Models predict equal-wager unit return (reported profit / positive stake), not dollar
profit, stake-weighted ROI, win probability or executable future return. Separate
funding/product populations; all-history browsing is not pooled training.

Training preprocessing, vocabulary, medians and backgrounds never use test outcomes.
Random event-grouped splits prevent within-population event leakage but are explicitly
non-temporal while chronology remains unverified. Training predictions are in-sample.
A constant selected model with zero contributions is a valid negative result;
a declined population has no invented predictions or explanations.

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

## Output scale and filtered views

Stored predictions, SHAP references and contributions are unit-return fractions:
0.01 corresponds to one percentage point of predicted return. They are not dollars,
probabilities or log odds. Reference plus grouped contributions must reconstruct
the frozen prediction to 1e-6. Mean absolute contribution measures magnitude; signed
mean contribution measures direction relative to that background. Neither estimates
the causal benefit of editing a scanner setting. Correlation can redistribute credit.

Ordinary sidebar selection retrieves stored canonical row IDs from the fixed model;
it preserves the estimator/background, split IDs and full-publication utility results.
It does not train, recompute corrected tests or evaluate a newly clicked cohort as an
independent test. A feature fixed by selection can still contribute relative to the
original training background. UI display sampling does not change summary rows or
predictions; view-specific controls are in [interactive explanations](../DASHBOARD_EXPLANATIONS.md).

A separately fitted subset requires an explicit offline typed-scope build and explicit
attachment. It receives a new scope/model/background/publication identity. Its SHAP
bars cannot be compared to global bars as if their reference were constant. Repeated
subset searches are exploratory. Unsupported requests fail rather than silently
broadening scope. No dashboard interaction runs a fit.

Implementation: [fit and grouped SHAP](../../src/oddsjam_ev/analysis/explanatory_models.py),
[publication integrity](../../src/oddsjam_ev/io/explanatory_models.py),
[fixed row selection](../../src/oddsjam_ev/dashboard/explanation_queries.py).
Verification: [fit/split/additivity](../../tests/test_explanatory_models.py),
[publication/declines](../../tests/test_explanatory_publication.py),
[interactive scope and no training](../../tests/test_dashboard_explanations.py).

[Contract reference](README.md) · [Documentation home](../README.md)
