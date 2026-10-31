# Visual diagnostic publications

PR B provides reusable data for the requested notebook visuals. It does not render
new dashboard charts; that is PR D in the decision-workspace sequence. Publications
are descriptive evidence, not a new shortlist, corrected test, or strategy approval.
Existing statistical-prioritization outputs retain their own inference contract.

## Build from explicit verified inputs

```sh
poetry run python scripts/build_visual_diagnostics.py \
  --canonical data/processed/weekly_research/artifacts/canonical/<canonical-id> \
  --atlas data/processed/weekly_research/artifacts/atlas/<atlas-id> \
  --diagnostics data/processed/sizing_diagnostics/<diagnostics-id> \
  --config configs/visual_diagnostics.yaml \
  --output-root data/processed/visual_diagnostics
```

The three input publications must pin the same canonical manifest. No latest-file
search, acquisition, notebook execution, report replacement, or dashboard-default
change occurs. The CLI prints the resulting directory. Load an explicit publication
with `oddsjam_ev.io.visual_diagnostics.load_visual_diagnostics(Path(...))`.

The content ID includes input manifest hashes, visual code, continuous-EV parser,
configuration/evidence assertions and runtime versions. Tables and audit are hashed,
verified on read, and checked on identical replay. Inputs and older publications
are preserved. Membership, output-row and calendar-span limits abort the entire
build with a private failure audit; they never silently truncate chart data.

## Chart inventory and readiness

Each scope receives a readiness row for **every** requested chart, including blocked
ones. `supported` means the primary descriptive data exists; it is not statistical
validation. `partial` carries an explicit coverage denominator and remediation.
Read the separate closing-series readiness before drawing an EV/ROI composite.
Every cohort also carries its own minimum-performance-support flag; 100 eligible
wagers is a display policy, not a guarantee of power or independent observations.

| Requested notebook visual | Reusable dataset/view | Requirement or remediation |
| --- | --- | --- |
| Profit by sport/group | `visual_cohorts`: sport/platform/league/market | Eligible source-reported economics; counts and exclusions accompany profit |
| EV versus realized ROI bubbles | `visual_cohorts`: overall/platform | Compare EV and returns only on the EV-matched cohort; publish weighted and equal-stake measures |
| Filter-level EV/ROI bubbles, PROD/TEST comparison | Filter `overall` cohorts | Resolved identities; use pinned lifecycle records for PROD/TEST styling, never infer from names or funding |
| Advertised EV bucket summary | `ev_bucket` cohorts | Exact continuous source EV preserved; decimal left-closed/right-open display bins |
| Liquidity summary | `liquidity_bucket` cohorts | Verified decision-time USD quote semantics; missing/raw/proxy liquidity is unavailable |
| EV buckets within liquidity bands | `liquidity_ev` cohorts | Matched verified liquidity plus source EV; no missing-as-zero proxy |
| Time-to-event summary | `time_to_event_bucket` cohorts | Verified placement/event field meanings and timezone/offset evidence, DST-safe derivation |
| Cumulative profit by sportsbook | `visual_periods`: platform, D/W-SUN/M | Source-calendar attribution only unless stronger chronology evidence exists; never cash balance |
| Bet volume over time | `visual_periods`: overall/platform | Unique logged candidates, dated coverage; export gaps are unknown rather than zero |
| Average activity by weekday/hour | `visual_activity`: weekday/hour/weekday_hour | Mean per active source day **within the bucket**, not all covered days or available opportunities |

Closing-price EV is an optional series across these comparisons. The source field
`clv` is not assumed to be fair closing odds. Enabling its transform requires an
explicit source allowlist and evidence of fair American-price semantics and closing
timing. Offered decimal odds times fair closing probability minus one is a reference
EV estimate, **not realized EV or profit**. Without evidence, this series is blocked
and the remedy is recorded, even if the main ROI panel is supported.

## Data contract

Seven Parquet tables and `audit.json` are published under one immutable manifest:

- `visual_candidates`: one row per canonical wager candidate, economic eligibility
  and exclusion reasons, exact source EV text, derived features and quality states.
- `visual_memberships`: scope-to-candidate bridge. Population scopes partition unique
  candidates by recorded funding status and product; resolved filter scopes can overlap.
- `visual_cohorts`: overall and grouped descriptive economics, matched EV/closing/joint
  denominators, event-proxy breadth, concentration and minimum-support flags.
- `visual_periods`: daily, Monday–Sunday weekly and calendar-month summaries, including
  boundary flags, unknown gaps and cumulative **known dated** eligible profit.
- `visual_activity`: source weekday/hour counts and explicit active-day denominators.
- `visual_coverage`: per-scope feature quality states and population denominators,
  including source-date and source-clock missingness separately.
- `visual_readiness`: ten requested charts per scope, primary coverage, blockers,
  remediation, separate closing-series state and performance-display flags.

Money is source-reported USD; returns and derived EV are **fractions** (0.03 = 3%).
Original advertised EV is in **percentage points**, with exact decimal text retained.
A fixed-unit comparison is the sum of eligible per-wager profit/stake times the
configured notional unit (default $1); it is not a historical bankroll reconstruction.
Weighted ROI is total eligible profit/total eligible stake. Mean source EV and
stake-weighted source EV have corresponding matched return measures, so missing EV
cannot silently change one side of a comparison. Closing-only and joint EV/closing
cohorts are published separately. Missing denominators produce unavailable values.
No confidence intervals or hypothesis tests are invented for these visual tables.

All candidates remain in coverage/activity counts, while unsupported products,
nonpositive stakes and other sizing-diagnostic exclusions stay outside return metrics.
Population totals reconcile to unique candidates and eligible economics. Filter
memberships are deduplicated per candidate/filter but remain non-additive across
filters. Funding labels are provenance, not selection criteria or proof of real
execution; all recorded funding/product strata are published without blending paper
returns and real-money returns into one ROI.

## Calendar and optional evidence policy

Literal ISO source dates/hours are preserved without shifting to an assumed timezone.
These support retrospective calendar attribution only. Dates omitted from an export
are not proven inactive days. Source-date curves omit undated economics explicitly
reported in cohort/readiness coverage and cannot establish placement order, settlement
order, drawdowns or bankroll growth. Active-day averages cannot estimate market
opportunity arrival rates or expected future volume.

Default configuration leaves placement/event meanings, liquidity semantics and fair
closing-reference interpretation unverified. Optional derivations require an explicit
evidence reference in configuration, which is hashed into lineage. A nonempty reference
is an operator assertion, not independent verification by the program. Only enable
these assertions for a source population the evidence actually covers; publish
separate supported inputs for heterogeneous source semantics. Naive timestamps require
an explicit supported IANA timezone; ambiguous/nonexistent local DST times remain
unavailable. Negative time-to-event values are retained as `after_start`. Liquidity
zero is a valid quote only with verified units; it does not establish fill availability.

## Verified October snapshot smoke build

The local build against the pinned October publication family reconciled **70,635**
unique candidates and **53,853** eligible economic rows across **25** population/filter
scopes. It produced 1,303 cohort summaries, 52,723 period rows, 1,994 activity rows,
157 feature-coverage rows and 250 readiness rows. Of the eligible economic rows,
53,318 have valid source EV for matched comparisons. The source has 70,619 usable
literal calendar dates; undated rows remain in population coverage/economics.

All 25 scopes block the liquidity, liquidity/EV and time-to-event charts under the
default evidence policy. All closing-reference series are unavailable. Some scopes
also have no eligible economics; their performance charts remain blocked while
activity can still be available. This is explicit missing evidence, not zero returns.
These local output files are ignored and are not committed. The synthetic tests
exercise verified optional sources separately without inventing evidence for this
historical snapshot. Exact input IDs, code hashes and policy are in the private
publication manifest; the command above accepts those explicit paths for replay.
