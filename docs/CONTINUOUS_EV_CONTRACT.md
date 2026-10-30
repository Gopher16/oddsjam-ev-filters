# Continuous advertised EV contract

The exported `percentage` field is retained as continuous source data. Immutable
CSV ingestion preserves decoded field text (including whitespace and decimal
precision); the archived CSV preserves original bytes. Canonical source features,
source legs and single-candidate facts retain this field. The atlas adds
`advertised_ev_band` without replacing `percentage`. A published band alone cannot
answer an exact threshold question, even though its underlying source can.

## Units, meaning and timing

The declared interpretation is **percentage points**: `3.125` means 3.125%, or
`0.03125` as a fraction. Values are never rescaled based on magnitude. This is the
source's advertised/theoretical EV estimate. Export semantics have not been
independently verified; presence does not prove availability at decision time.
It is not realized ROI, a verified observed edge, true probability or guaranteed
profit. Source timing remains `source_current_unverified`.

The legacy preprocessing fields `ev` and `ev_roi` are separate CLV-derived modeled
quantities: `ev` uses stake, closing-reference implied probability and placed odds;
`ev_roi` divides that modeled amount by stake. This change does not substitute
advertised EV for those fields or change their formulas.

## Exact values and quality

`analysis/continuous_ev.py` defines the versioned contract and parsing:

- `source_ev_raw`: exact decoded field text where text input is available.
- `source_representation`: text versus previously typed input whose original
  lexical precision cannot be recovered.
- `advertised_ev_percent_exact`: decimal text for exact comparisons.
- `advertised_ev_fraction_exact`: exact decimal scaling by 100, without context
  rounding, even beyond the default Decimal precision.
- `advertised_ev_percent`: nullable Float64 preview; approximate and never the
  authority for exact threshold membership.
- `ev_status`: valid, missing, invalid, conflict, unavailable source column, or
  unavailable ticket model. Null and blank values remain missing, not zero.

Accepted numbers use signed decimal/scientific notation. Percent signs, separators,
underscores, nonfinite values and nonnumeric strings are invalid. Finite values
outside nonzero Float64 representability are unsupported/invalid in this contract;
source strings over 4096 characters are also rejected as numeric inputs. Raw text
remains available. No economically meaningful EV range is invented from magnitude.

New legacy preprocessing outputs additionally retain `percentage_source_raw`
before numeric coercion, through both direct-CSV and pinned-ingestion routes.
The existing numeric `percentage` preview remains; old processed files are not
rewritten. If a caller supplies already typed numbers, the original CSV lexeme
cannot be reconstructed. Use verified canonical input for authoritative provenance.

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

```python
from oddsjam_ev.analysis.continuous_ev import ev_threshold

# A declared decimal endpoint, in percentage points; no epsilon or rounding.
mask = ev_threshold(wager_ev.advertised_ev_percent_exact, ">", "3")
eligible_mask = wager_ev.reported_roi_eligible & mask
```

`>`, `>=`, `<`, `<=` and `==` are explicit operators. `2.999`, `3`, `3.001`, and
`3.000000000000000000000000000001` remain distinguishable. Invalid, conflicting,
missing and unavailable values never match. Bands are never parsed as measurements.
An eligible threshold slice still needs separate product/funding and membership
scope; paper outcomes remain hypothetical and memberships can overlap.

This helper defines predicate semantics, not an approved discovery grid. Finite
predeclared search families, multiple-testing control, calibration, sizing-aware
returns and within-filter comparisons remain follow-up work. Arbitrary sliders
must not generate new validated findings on dashboard load.

Tests cover source-to-canonical-to-Parquet precision, both preprocessing routes,
unit scaling, endpoints, float collisions, missingness, conflicts, ticket exclusions,
source preservation, corrupt artifacts and deterministic replay. No notebooks are
executed or modified by this stage.
