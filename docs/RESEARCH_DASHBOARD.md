# Local research dashboard

The dashboard reads verified published research. It never rebuilds research on page
load, imports CSVs, executes notebooks, schedules work, acquires exports, changes
filters, registers tests, approves proposals or places wagers.

## Start or restart

From the repository root:

```bash
poetry install --with dashboard
poetry run streamlit run dashboard/app.py
```

Open the loopback URL printed by Streamlit, normally `http://127.0.0.1:8501`.
Stop an existing server with **Control+C**, restart and refresh after code changes:
file watching is disabled. The optional dashboard group supplies Streamlit and Altair.
Existing research commands do not require that group.

## Default experience

The newest recorded publication opens automatically, after full artifact verification.
Changing the report selection also loads automatically; no Open report button or
research-generation operation is needed. Failure clears cached results and displays
an error without falling back to another report.

Labels use the first successful completion timestamp recorded in local pipeline
attempt metadata, converted to the configured display timezone. Later replays do
not move publication time forward. These operational records are not authenticated
research evidence. Where completion time is missing, labels explicitly say generation
time is unavailable and show the analysis cutoff instead. Reports sort by completion
time (cutoff for missing completion); ties are deterministic and display version
ordinals, never user-facing hashes. Full run IDs remain in technical details.

Generation time is not source freshness. Analysis cutoff, data receipt and confirmed
export timing remain separate. A newly generated historical report can contain old
data. Missing source export timing stays unconfirmed, never fresh by implication.

All funding groups and products are visible initially. Optional Funding and Product
controls narrow the display. All-data browsing shows separate labeled cohorts, not a
blended financial return: paper outcomes are hypothetical and unclassified funding
is not silently reassigned. Source USD profit and fee-inclusion uncertainty remain
unchanged. Core Exchange history reflects selective trades, not all scanner quotes;
taker-only workflow is a user assumption, not independent fill evidence.

## Three destinations

| View | Purpose |
| --- | --- |
| Signals & leaks | Default landing view: positive historical signals, loss pockets and specific prospective TEST hypotheses; compare evidence interactively. |
| Filter health | Exact published saved-filter membership breakdowns by market or platform, keeping funding/product cohorts separate. |
| Evidence & methodology | Supporting coverage, exclusions, source provenance and methodological limits; detailed JSON is collapsed. |

Signals & leaks includes tabs for positive signals, potential leaks and an interactive
return-versus-volume comparison. Cards show up to five multidimensional candidates per
section first, with one-dimensional drivers in a separate expander. Specificity alone
does not imply stronger evidence. Cards show exact reported metrics, funding/product labels, published uncertainty and a
concrete TEST hypothesis. No positive signals or losses are invented for an empty
selection. Prospective test metrics remain available in a supporting expander.

Sidebar Signal filters can select Platform, Sport, League, Market, Odds range and
Advertised EV simultaneously, together with Funding and Product. Multiple values
within a subgroup field use OR; populated fields combine with AND. Empty fields mean
all. Active selections appear above the results; Reset signal filters clears the six
subgroup fields, retaining Funding/Product. Options reflect the funding/product
universe rather than disappearing when another subgroup field is selected.

A candidate must explicitly contain every selected dimension and value. Missing
predicate fields never match a constraint: separate Novig and Basketball summaries
cannot be substituted for Novig × Basketball. An unsupported combination produces a
clear no-published-evidence notice. Ranges are exact published bands, not arbitrary
numeric cutpoints. No intersections or unpublished bet types are calculated on load.
Signal filters apply to Signals & leaks; Filter health retains its separately labeled
published saved-filter breakdown controls.

## Reading order and interpretation

The queue orders existing candidates by event-proxy group count, then analyzed wager
count, then stable candidate identity. Positive historical signals and loss pockets
use a default minimum of 100 analyzed wagers for display, configurable in the sidebar
and `min_analyzed_wagers` configuration. Filter-health charts and tables use the same
setting. Hidden cohort counts are reported; lower the threshold explicitly to inspect
small samples. This is a reading convention,
not a significance, eligibility or promotion threshold. Sparse observations remain
in the comparison/full catalogue. Product-wide totals are context rather than new
TEST angles. Multiple proposal actions do not duplicate a candidate in the shortlist.
Related/overlapping segments remain separate and are not independent discoveries.

No new statistical edge score, p-value, confidence interval or approval decision is
computed. An interval including zero is identified on the card. A positive historical
return supports an exploratory TEST hypothesis, not PROD deployment. Negative results
support an exclusion comparison against an unchanged prospective baseline, not a
claim that excluding the subgroup will improve future profits. Widening settings
requires observations outside current settings and is not inferred here.

Filter health uses only the published filter-attribution diagnostics. It does not
substitute global sportsbook results for results within a saved filter. Current PROD
status is not established by historical names. CLV and advertised-edge-versus-realized
return diagnostics are unavailable at this published grain and are not fabricated.
More rigorous within-filter discovery and ranking require a separately tested research
publication; the dashboard must consume those artifacts rather than fit methods at
view time.

## Configuration and boundaries

`configs/dashboard.yaml` controls the trusted artifact root, display row cap and
`display_timezone` (default `America/Chicago`). Only aware timestamps are converted;
source timezone is not inferred. Alternative configuration:

```bash
ODDSJAM_DASHBOARD_CONFIG=/path/to/dashboard.yaml \
  poetry run streamlit run dashboard/app.py
```

There is no upload or arbitrary path picker. Artifact references and symlinks must
remain inside the configured root. Result/report hashes, checkpoint chains, stage
loaders, table bytes/counts, supported contracts and row-level lineage are verified.
Every rerun rehashes expected files before using session-cached data. Selection changes
clear cached results. No dashboard cache or research bundle is written to disk.

`.streamlit/config.toml` binds to loopback, disables static serving, usage telemetry
and file watching, and retains request protections. Public hosting/authentication is
out of scope. Existing notebooks, preprocessing defaults, raw data and research
snapshots remain untouched. Scheduling and programmatic acquisition remain disabled.

## Verification

```bash
poetry check --lock
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest -q -W error::FutureWarning
git diff --check
```

Synthetic tests cover automatic opening, missing/corrupt reports without fallback,
read-only navigation, population preservation, completion timestamps and replay,
source contracts, confinement, row lineage, metric equality, missing intervals,
empty data and deterministic reading order. Browser preview screenshots are private,
ignored local artifacts.

## Planned research extension

The [multidimensional discovery and saved-filter optimization plan](plans/MULTIDIMENSIONAL_SIGNAL_DISCOVERY.md)
defines the proposed follow-up after the dashboard is merged: bounded searches,
statistical families and gates, parent/complement comparisons, overlap handling,
versioned publication, logical commits, synthetic tests and acceptance criteria.
It is a design brief, not an implemented engine or approval for PROD changes.

Sizing interpretation: reported ROI is total source profit divided by total eligible
stake, so larger wagers carry more weight. No historical bankroll/Kelly normalization
is currently applied. The linked follow-up plan specifies parallel equal-stake and
sizing-regime diagnostics, with missing timing/history preserved. Source continuous
EV is retained; atlas bands are additional summaries, not replacements for raw EV.

## Saved-filter selection on Signals & leaks

The sidebar **Filter** selector defaults to All saved filters. Selecting a named
filter switches to exact persisted atlas filter-membership slices, retaining funding,
product, minimum-wager and subgroup constraints. Global discovery statistics are
never reused as within-filter evidence. Existing reports publish separate market and
platform breakdowns at this grain, not their intersections or sport/EV combinations.
Unsupported combinations return no-published-evidence notices. Membership slices have
no published subgroup confidence intervals; the UI labels these unavailable. They
are descriptive context, not statistically ranked filter-edit recommendations.
