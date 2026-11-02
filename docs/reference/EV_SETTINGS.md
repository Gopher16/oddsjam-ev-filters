# Advertised EV and scanner settings

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

## Exact endpoints and display bands

Exact threshold membership uses `advertised_ev_percent_exact` decimal text through
`ev_threshold`, not rounded Float64 previews or display bands. `>3` excludes 3,
while `>=3` includes it; 2.999, 3 and 3.001 remain distinct. The helper accepts
`>`, `>=`, `<`, `<=`, `==`; individual discovery/model/prospective adapters accept
only their own declared operator set. Unsupported operators fail closed.

Bands have versioned edges and inclusive/exclusive semantics; an aggregate 2.5–5%
band cannot answer an exact >3% question. New cutpoints or arbitrary intersections
need a declared offline build and complete family accounting. Sidebar selection
cannot create an evaluated hypothesis. Models use advertised EV fractions explicitly:
`advertised_ev_fraction > 0.03` is the model-scope equivalent unit to >3 percentage
points, but its scope parser is separate from the exact historical EV helper.

## States and provenance

`analysis/decision_config.py` defines the complete control catalog: Recommended
Filters, formula builder, devig books/weights/method/model version, platform,
sport/league/market types/exact markets/games, odds and EV endpoints, liquidity,
minimum recommended bet, market width, date range, pregame/live/time-to-event,
timezone, warning behavior, paper sizing and bankroll/Kelly context.

Each input setting has an explicit state:

| State | Interpretation |
| --- | --- |
| `observed` | A sourced value, not a guessed default. |
| `unrestricted` | A deliberate lack of an additional restriction within the prospective universe. |
| `unset` | Confirmed inactive control; value must be null, not zero. |
| `platform_default` | Verified default with platform/version evidence. |
| `not_applicable` | Explicit reason/evidence for inapplicability. |
| `unknown` | Missing evidence; blocks complete setup readiness. |

Output sheets add `inherited` and `override`. Inheritance preserves the input state,
value and baseline fingerprint in separate columns. A null inherited value may be
confirmed unset/unrestricted; it is not silently converted to zero or unknown.
Unknown is never promoted to unset, unrestricted or default.

Known input states require a reason, evidence reference, observation time and a basis:
`historical_observation`, `prospective_design`, `operational_constraint` or
`platform_requirement`. Observation time cannot follow recording time. The basis
explains why the value was chosen, not proof of its profitability. Unrestricted logged
history does not establish that an unrestricted scanner is profitable.

## Baseline and scanner translation

A complete sheet covers every known control, including unsupported/unknown ones.
A prospective parent baseline is explicitly evidenced and fingerprinted; current
screenshots and historical registry names do not establish past settings. A global
analytical slice cannot inherit an arbitrary saved-filter parent. Conflicting
applicable baselines block rather than selecting the newest convenient value.

Candidate terms remain analytical predicates. Translation into scanner controls
requires exact UI/version semantics and evidence known by the cutoff. A market
family is not automatically an exact scanner market set. A strict > endpoint must
not become >= or a rounded threshold. Overrides intersect supported parent
restrictions without silently widening them. Compound exclusion means NOT(the
whole conjunction), not removal of each term independently; unsupported compound
translations stay blocked. See the [configuration evidence schema and resolver](../DECISION_CONFIGURATION_RECORDS.md).

Ready means ready for operator configuration review, not an approval or external
application. A configuration-bound definition pins the exact publication, decision,
record hash and baseline. A changed setting requires a new publication/version;
legacy untyped definitions remain readable without gaining verified setup status.

Implementation: [continuous EV](../../src/oddsjam_ev/analysis/continuous_ev.py),
[control catalog](../../src/oddsjam_ev/analysis/decision_config.py),
[resolver](../../src/oddsjam_ev/analysis/decision_records.py).
Verification: [precision and endpoints](../../tests/test_continuous_ev.py),
[states, inheritance and unsupported mappings](../../tests/test_decision_records.py),
[bound definitions](../../tests/test_paper_definitions.py).

[Contract reference](README.md) · [Documentation home](../README.md)
