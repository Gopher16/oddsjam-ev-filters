# Continuous advertised EV contract

The exported `percentage` field is retained as continuous source data. Immutable
CSV ingestion preserves decoded field text (including whitespace and decimal
precision); the archived CSV preserves original bytes. Canonical source features,
source legs and single-candidate facts retain this field. The atlas adds
`advertised_ev_band` without replacing `percentage`. A published band alone cannot
answer an exact threshold question, even though its underlying source can.

## Units, meaning and timing

The [EV/settings reference](reference/EV_SETTINGS.md) defines percentage points,
fractions, exact decimal fields, quality states, source preservation and timing.
The stage-specific publication interface follows.

## Exact values and quality

See the [exact numeric fields and quality contract](reference/EV_SETTINGS.md#exact-values-and-quality)
for raw lexemes, Decimal strings, Float64 previews and missing/invalid/conflict states.

## Optional publication

```sh
poetry run python scripts/build_continuous_ev.py \
  --canonical <explicit-canonical-bundle-directory>
```

The private destination is `data/processed/continuous_ev/<ev-id>/`. It contains
`source_ev.parquet`, `wager_ev.parquet` and a manifest with coverage/status counts,
contract, canonical manifest hash, source-table hashes, snapshot identity, code
hashes and runtime versions. Hash-verified loading and identical replay are supported.
Input bundles and historical canonical/atlas/report schemas remain unchanged.
This stage is explicitly invoked and is not added to dashboard loading or default
pipeline execution.

The source view has one row per **current source observation**, including legs and
unlinked records; it is not an independent-wager population. Superseded observations
remain in immutable ingestion history but are not included in this current view.
The candidate view has one row per canonical wager candidate and preserves existing
product, funding and reported-ROI eligibility. Ticket EV remains unavailable even
when source legs have identical EV: a ticket requires a justified joint model.
Source and candidate counts must not be added. Eligibility is never inferred from EV.

## Exact thresholds

Use the [exact endpoint contract](reference/EV_SETTINGS.md#exact-endpoints-and-display-bands)
and `analysis.continuous_ev.ev_threshold` on `advertised_ev_percent_exact`. This
helper is not itself a research grid. Delivered bounded families and diagnostics
are documented in [within-filter discovery](WITHIN_FILTER_DISCOVERY.md) and
[sizing diagnostics](SIZING_DIAGNOSTICS.md); calibration extensions remain separate.
Arbitrary sliders never generate validated findings on dashboard load.

Tests cover source-to-canonical-to-Parquet precision, both preprocessing routes,
unit scaling, endpoints, float collisions, missingness, conflicts, ticket exclusions,
source preservation, corrupt artifacts and deterministic replay. No notebooks are
executed or modified by this stage.
