# Explore frozen explanations

Shared semantics: [authoritative contract reference](reference/MODELS_EXPLANATIONS.md).
This component guide retains its schema, commands and stage-specific requirements.

Open **Explore → Explain returns**. The default publication reviews all recorded
history with a separate frozen model for each supported funding/product population.
It does not pool paper and funded returns. Unsupported populations remain in the
coverage counts and support table; their predictions are not invented.

The current historical publication has 70,635 recorded wagers and 29,975 published
explanations across three supported populations. The funded exchange model selected
the constant baseline: all feature contributions are zero, and this is an honest
negative result. A visually interesting attribution is not evidence of a useful
predictive model. Read the model-utility status beside each importance chart.

## Global and filtered exploration

1. Start with **All-history frozen models**, **All published rows**, and **All model
   populations**. Every supported population gets a separate chart and reference.
2. Use the shared sidebar to select platform, sport, league, market, odds bands,
   advertised EV, funding, product or a resolved saved-filter membership. Fields
   combine with AND, choices within a field with OR. Repeated memberships never
   duplicate a wager. All includes unknown/unattributed source rows.
3. Compare **Selected cohort** with **Published reference** in Drivers & direction.
   Both use the same model, background and displayed train/validation/test selection.
   The reference retains its full published population, unaffected by sidebar filters.
4. Change **Published evaluation rows** to test-only when inspecting held-out rows.
   The default includes training predictions, which are in-sample. The test-error
   panel always shows the original whole-publication test results; it is not a new
   statistical evaluation of a clicked cohort.
5. On an opportunity's detail view, use **Explain this idea’s recorded wagers**.
   Its exact predicate and resolved parent membership intersect the sidebar scope.
   Clear the drill-through explicitly to return to broader exploration. A conflicting
   selector produces an empty view; another idea or model is not substituted.

This page selects canonical row identities, unlike the idea queue's exact published
slice selector. It does not recompute inference, corrections, ranks or scanner rules.
The sidebar minimum-wager control is a signal display threshold, not a fit gate;
small selected explanation cohorts are labeled and remain inspectable. Idea-only
evidence/setup/lifecycle controls do not change model rows, which is stated onscreen.
Exact EV thresholds use original Decimal source values, preserving a strict `>3%`
boundary. Unknown values do not satisfy a numeric predicate.

## Charts and interpretation

- **Drivers & direction:** grouped original-feature mean absolute contribution and
  signed mean contribution, with full selected-cohort/reference counts. Signed values
  raise/lower a model prediction relative to its background; they do not prove that
  changing a setting will improve returns.
- **Feature patterns:** jittered contribution distributions and feature-dependence
  scatterplots, with pan/zoom, full-value tooltips, recorded context coloring and
  point selection. Plots use a deterministic display sample of at most 1,200 wagers,
  preserving representation of every model. Importance summaries use all selected
  rows. Sampling does not alter models, references or prediction values.
- **Local explanation:** click a plotted wager or choose a sampled ID; an exact-ID
  lookup also accepts any explained wager in the current selection. The waterfall
  shows training reference + six feature contributions = frozen prediction. Recorded
  outcome and source context are shown separately, never treated as predictors.
- **Model quality & coverage:** untouched-test errors against constant/advertised-EV
  references, selected-row observed/predicted calibration, platform residuals,
  fixed selected features, feature eligibility and excluded inputs. Published learning
  curves, missing/unseen-category audits, permutation sensitivity, redundancy and
  literal source-calendar diagnostics remain available with their original scope.

Each plot/panel keeps its model/background and population separate. Features fixed
by a sidebar selection can still contribute relative to the original training
background; that does not establish a within-subset effect. Correlated features can
share importance. Forest trajectory SD is approximation variability, not a confidence
interval. Raw CSV values remain unit-return fractions; chart axes show percentages.
Observed returns are equal-wager returns, not stake-weighted ROI or dollar profit.

Timing, liquidity, CLV and historical scanner settings remain excluded until their
decision-time provenance is verified. These explanations cannot fill unknown OddsJam
configuration settings, approve a prospective test or authorize production changes.

## Separately fitted subset models

The **Explanation publication** control is the only way to select an attached subset
refit. These entries are labeled **Separately fitted subset**, show their typed
training scope and display a reference-change warning. Sidebar filtering never
selects or fits them automatically. References from different publications must not
be compared as though their SHAP values shared the same baseline.

**Offline subset build** exports an exact typed request for the existing offline
builder; it does not run it. The export intersects the sidebar with the currently
selected publication's original fitted scope. Display-only population/split controls
are not a training request; use the sidebar Funding/Product controls for that scope.
The source/model provenance can be downloaded separately. Run the request using the
[offline model guide](EXPLANATORY_MODELS.md), then explicitly attach its output.

Saved-filter/candidate membership, unsupported numeric operators, disjoint EV bands
and American-odds scopes that cannot be represented exactly by the existing typed
builder block export. No broader request is silently substituted. Sparse offline
builds may decline; repeated subset looks are not independent confirmation.

## Attach explicit publications

Configure only trusted, existing artifacts relative to `evidence_root`:

```yaml
explanation_publications:
  <exact-research-run-id>:
    global: explanatory_models/<all-history-publication-id>
    subsets:
      - explanatory_models/<separately-fitted-publication-id>
```

The global entry must have empty fitted scope; subset entries must have nonempty
scope. Model source hashes must match the exact selected canonical and visual
publications. Paths and symlinks are confined to the trusted root. Cached views
rehash every manifest/output byte before reuse. Corruption clears the model views;
other verified research remains available. Missing attachments never trigger latest
model discovery, retraining or fallback to another report.

The viewer consumes verified prediction/SHAP tables and safe metadata only. No model
inference or training occurs on ordinary interaction. The explanatory publication
retains target, feature/policy/code/runtime versions, row splits, source lineage,
model and background IDs; provenance and selected-row exports preserve those links.
