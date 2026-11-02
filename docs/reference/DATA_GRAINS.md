# Data grains and identity

## Identity ladder

| Grain | Identity and safe use |
| --- | --- |
| Source observation | An archived CSV row/revision with import/row lineage. Repeated export receipts do not create new observations for identical accepted bytes. Observation count is not wager count. |
| Ingestion record / wager candidate | Reconciliation selects current state from observations. It is a candidate identity, not proof of an independent execution or economic eligibility. Ambiguities stay unresolved. |
| Canonical single | Current reconciled non-ticket candidate. Its canonical ID reuses the ingestion record identity; eligibility is separate. |
| Ticket candidate | Namespace plus explicit parlay ID; current and unresolved legs are audited. Ticket identity does not establish completeness or settlement. |
| Ticket leg / source feature | Source-grain context attached to a candidate. Never sum repeated ticket money or assign a leg's EV to a ticket. |
| Opportunity proxy | Selection/quote grouping of logged records. Not an inventory of scanner supply or independent tradable opportunities. |
| Filter membership | Unique resolved wager/filter relationship; multiple memberships can overlap. Missing/unresolved labels retain coverage without invented per-filter performance. |
| Event group | Namespace plus nonempty event/game identity used for clustered resampling/splits. An ID is a dependence proxy, not proof of event independence. |
| Signal candidate | Frozen analytical predicate plus parent/population, policy and source lineage. It is distinct from a wager candidate. |

An ingestion namespace separates declared sources/accounts. `game_id` alone is not
a bet ID; neither similar names nor adjacent timestamps identify a new execution.
The [ingestion identity algorithm](../INGESTION.md#identity-repeated-observations-and-updates)
retains its reconciliation-specific requirements. Legacy processed-row IDs in
`research_grains.py` are dataset-local and are not canonical record identities.

## Wagers, explicit tickets, and candidate legs

A nonempty `parlay_id` is explicit ticket evidence even if the source bet type is
inconsistent. Repeated ticket amounts are recorded once only when all current
legs agree numerically. Missing or conflicting ticket amounts remain null; no
first-row, maximum, or sum rule resolves them. Mixed venue/execution evidence is
not replaced with a guessed value.

Individual statuses/odds belong to legs. They are not used to determine ticket
status or combine odds. All ticket candidates have unverified completeness and
settlement eligibility; even one exported leg or all terminal leg statuses does
not establish a complete settled ticket. Current and unresolved leg counts are
reported separately. Historical superseded observations are source audit history,
not extra legs in the current ticket.

User clarification: Pick'em plays can be logged as individual bets before being
assembled into a real ticket. Therefore `bet_type=positive_ev` is not sufficient
to establish standalone-wager economics. Exact configured Pick'em products are
ineligible for standalone ROI regardless of their row label. Dated Pick'em filter
IDs/aliases provide additional negative eligibility evidence: they cannot make a
candidate leg eligible or establish payout mechanics. No tickets are reconstructed
from timestamps, shared events, nearby rows, or filter names.

`configs/canonical_facts.yaml` declares exact product mappings. These mappings
are research policy, not independently verified platform mechanics. Any unlisted
product remains `unknown`, including venues that may support multiple products.
The legacy exchange/sportsbook inference remains untouched; it is not used as a
fallback for canonical payout classification.

## Opportunities and filter attribution

Selection proxies use namespace, event identity, raw market label, and raw bet
selection. When game ID is missing, event name requires a raw event-start value;
otherwise an observation-specific fallback prevents unrelated rows collapsing.
Quote proxies add source venue, odds, created timestamp, and timestamp timezone.
Missing quote identity evidence also uses an observation-specific fallback.
These are grouping proxies, not confirmed scanner quotes or independent samples.

Different executions may link to the same proxy; their wagers are preserved.
Across prices/books, selection-group membership does not combine their economic
facts. Only invariant descriptive attributes are retained on shared proxies;
conflicting metadata stays in source features rather than choosing an execution.
The denominator is logged source records, not all available market opportunities.

Filter resolution gives precedence to exact names/aliases, preserving commas
inside names. Multiple memberships resolve only when the entire string has one
unambiguous segmentation using known names. Unknown/ambiguous labels are retained
as diagnostics. Repeated leg labels or aliases do not duplicate bet/filter links.

Registry validation rejects duplicate IDs, conflicting aliases, missing clone
references, and clone cycles. The Onyx clone reference now targets the already
committed `test_dk_pick6_player_props` ID; stable IDs were not renamed. Registry
snapshot date is provenance, not proof its settings applied to an earlier bet.
Canonical attribution never supplies historical UI settings or execution funding.

## Parent and overlap invariants

For a resolved saved-filter parent on one snapshot, funding/product and eligibility
scope, compute subgroup and disjoint complement from wager IDs once. Their counts,
stakes and profits reconcile to the parent; their ROIs are not additive. The parent
already contains both arms. A global platform association cannot substitute for
that platform within a particular filter. Equivalent-membership suppression changes
presentation only, never source money or correction-family membership.

Known membership can support descriptive analysis even when historical settings
are unknown. Identity reconciliation cannot infer funding, exact rules or PROD
status. Widening requires an observed outside-rule universe with comparable capture;
selected historical wagers alone cannot establish it.

Implementation: [bet facts](../../src/oddsjam_ev/analysis/bet_facts.py),
[opportunity proxies](../../src/oddsjam_ev/analysis/opportunity_facts.py),
[attribution](../../src/oddsjam_ev/analysis/filter_attribution.py),
[within-filter discovery](../../src/oddsjam_ev/analysis/multidimensional_discovery.py).
Verification: [bet facts](../../tests/test_bet_facts.py),
[canonical links](../../tests/test_canonical_links.py),
[attribution](../../tests/test_filter_attribution.py),
[within-filter partitioning](../../tests/test_within_filter_discovery.py).

[Contract reference](README.md) · [Documentation home](../README.md)
