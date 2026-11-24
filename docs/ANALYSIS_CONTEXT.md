# Shared analysis context (R07)

The version-1 optional contract connects exact evidence selection across charts,
frozen explanations and future risk views. It does not activate new pages, create
new cohort metrics, fit models or run simulations. See [R07 plan](roadmap/pr-plans/R07.md),
[operating regimes](OPERATING_REGIMES.md) and [venue capabilities](VENUE_CAPABILITIES.md).

## Scope and trust boundary

`analysis.analysis_context.AnalysisContext` is a frozen typed value. It carries:

- Source snapshot ID, configuration version and investigation identity.
- Named publication fingerprints. Pin operating-regime and venue-registry
  publications when their evidence supplies the population/configuration. Pin
  visual and explanation publications independently; navigation never replaces one.
- One funding group, product and selection regime. Unknown is a recorded state,
  not permission to reclassify funding. Use separate contexts for mixed populations.
- Cohort fingerprints, unique wager IDs and stable event IDs. IDs are OR within a
  field and AND across fields. `None` means unrestricted; `()` means explicitly empty.
- An optional half-open `[start,end)` aware timestamp window and explicit time column.
  Created time does not imply placement or settlement. The caller supplies supported
  chronology; missing or naive timestamps block window selection.
- Equal-stake or stake-weighted analysis, destination view and selection origin.
  The context transports weighting without converting denominators.

The context is a request, **not trusted evidence**. Load and verify immutable
publications with their existing readers first. Build `bindings` from those verified
manifests, not from incoming context fields. Source/configuration and every named
publication must match. Missing/stale references block selection; no latest-source
lookup, approximate ID match, source switch or broader fallback exists.

Canonical context JSON is suitable for a session-state value; URL-safe tokens use
base64 encoding and have a 16 KiB decoded JSON limit. Unknown/duplicate fields and
unsupported versions fail validation. Tokens contain potentially private wager,
source and investigation IDs: they are not encrypted, signed or public share links.
Do not place private tokens in public URLs, logs, analytics or screenshots. Large
selections need a future pinned server-side selection reference; do not truncate IDs.

## Example linked episode

```python
from oddsjam_ev.analysis.analysis_context import AnalysisContext
from oddsjam_ev.dashboard.context_queries import select_context_rows

context = AnalysisContext(
    source_id="synthetic-source", configuration_version="config-v1",
    investigation_id="investigation-1",
    publications=(("visuals", "verified-manifest-fingerprint"),),
    funding="paper", product="exchange", selection_regime="discretionary",
    wager_ids=("wager-a",), event_ids=("namespace:event-1",), focus="drawdown",
)
# verified_rows and trusted_bindings are independently loaded evidence, not token data.
result = select_context_rows(context, verified_rows, trusted_bindings)
next_context, changes = context.transition(view="risk")
session_json = next_context.to_json()
restored = AnalysisContext.from_json(session_json)
```

Render `result.reasons` and the exact context before showing metrics. `changes`
contains before/after values for a visible scope indicator. A view-only transition
retains all pins, IDs, weighting and window. Changing source/configuration explicitly
retains the previous IDs until validated (and may therefore block); it never silently
widens to all rows. `reset_selection()` explicitly clears IDs/window/focus while
preserving population, source/configuration pins and investigation. Show that reset
as a deliberate user action and display the returned field changes.

## Adapters and limitations

`select_context_rows` selects unique wager rows with explicit `funding`, `product`
and `selection_regime`. It validates selected IDs against that population before
intersecting. Event IDs must already be normalized, namespace-safe identifiers;
this PR does not derive clusters. Optional cohort membership maps must come from
the pinned publication. Cross-population memberships block rather than disappear.
A supported empty intersection returns zero rows with no error; blockers return
zero rows plus reason codes. Do not substitute another population for either case.

`context_explanations` calls the existing frozen `explanation_rows` query. It
requires an explanation pin and complete stored coverage of the selected IDs in
the requested published split; missing coverage is an explicit blocker. The existing explanation model uses equal-wager returns; stake-weighted
requests block explicitly rather than convert the estimand. Models,
backgrounds and split labels remain unchanged. Existing downstream explanation
summaries continue to separate model/reference populations.

`context_chart_rows` accepts exact published `cohort_fingerprint` values with
explicit population/regime/weighting columns. Wager/event/window restrictions cannot
filter an aggregate ROI: they return `unsupported_aggregate_intersection`. Legacy
visual tables lack regime/weighting scope assertions and remain explicitly
unsupported by this optional adapter until evidence-aware adoption. Existing
`decision_queries.chart_rows` and current pages remain unchanged and usable.
Do not manufacture scope columns merely to bypass that boundary.

Risk is a **handoff destination**, not a claim a risk publication/engine exists.
R23 derives exposure groups; R25–R29 implement scenario/risk calculations. R11/R12
own the visible workspace shell and R32 integrates the complete investigation/action
journey. Those consumers must verify their own publication pins and capabilities
and show unsupported states. No current chronology, profitability or production
approval can be inferred from a successfully serialized context.

## Validation and rollback

Synthetic fixtures exercise drawdown/event selection across all destination values,
round trips, half-open windows, explicit empties, stale references, unknown IDs,
cross-population membership, duplicates, unsupported aggregate intersections and
frozen explanation coverage. Existing dashboard tests protect the unchanged paths.
Adoption is opt-in: removing an adapter call rolls back its use without migrating
canonical tables, existing publication formats, notebooks or historical economics.
