# Operating regimes and configuration evidence

R05 adds an optional, version-1 evidence publication alongside canonical facts,
[sizing history](SIZING_HISTORY.md) and [prospective configuration records](DECISION_CONFIGURATION_RECORDS.md).
It records how opportunities were selected and tracked, when that practice reportedly
applied, and when evidence became known. Existing reports, readers, canonical schemas,
funding labels, accepted-fill rules and monetary totals retain their existing behavior.
The dashboard and weekly pipeline do not automatically consume this publication.

Read the [owner's dated regime narrative](operations/DECISION_REGIMES.md),
[source inventory](operations/SOURCE_FIELD_INVENTORY.md) and
[evidence gaps](operations/EVIDENCE_GAPS.md) before supplying private account evidence.
The implementation cannot recover opportunities, passes, settings or fills that were
never captured.

## Evidence model

A ledger has exactly `contract_version: 1`, an aware `knowledge_as_of`, nullable
`previous_history_id`, `records` and `wager_evidence` arrays. Each evidence ID must be
unique across both arrays. Every record has an aware `recorded_at` no later than the
knowledge cutoff and a nonempty `source_reference`. These identify evidence receipt,
not necessarily the time a setting applied or a wager was placed.

Each operating record has these fields:

| Fields | Meaning and requirements |
| --- | --- |
| `evidence_id`, `source_reference` | Stable assertion identity and source locator; preserve the original supporting material privately. |
| `namespace`, `filter_id`, `product` | Exact source/account namespace, nullable stable registry filter ID, nullable product scope. Null filter/product means all within the namespace, so use deliberately. Products are `exchange`, `fixed_odds`, `pickem` or `unknown`. A filter name is not a stable ID. |
| `effective_from`, `effective_to`, `precision`, `open_ended`, `effective_interval_verified` | Effective boundaries and their evidential precision; rules below. |
| `recorded_at`, `evidence_type`, `confidence` | Receipt time, assertion type and confidence. Types are `user_confirmed`, `contemporaneous_record`, `reconstructed_assumption`, `unknown`, `planned`; confidence is `confirmed`, `assumed` or `unknown`. |
| `selection_mode` | `broad_capture`, `discretionary` or `unknown`. This describes selection practice, not profitability or a complete opportunity universe. |
| `funding` | `paper`, `real_money` or `unknown` as independent regime evidence; never overwrites canonical funding. |
| `paper_sizing` | `kelly_recommendation` or `unknown`. The Kelly convention requires paper funding evidence and does not establish fillability or order acceptance. |
| `settings` | Existing typed scanner/account controls using the [setting-state contract](reference/EV_SETTINGS.md). Omitted controls remain unknown. Defaults, unset values and observations are distinct. |
| `venue_access` | Unique venue names supported by the record's source. Empty means no access assertion supplied; this is neither an API connector nor a verified product capability/permission model. |

`timestamp` precision requires aware ISO timestamps. Verified effective intervals
use `[effective_from, effective_to)`; the start is included and end excluded.
A verified interval with no end requires an explicit `open_ended: true`. An unknown
end is not automatically indefinite. `date` precision requires `YYYY-MM-DD`, may
retain a missing end, and cannot be marked verified for automatic wager linkage.
`unknown` precision requires null boundaries and `open_ended: false`.

Independent placement evidence has exactly `evidence_id`, `bet_id`, `placement_at`,
`placement_verified`, `recorded_at`, `source_reference`. A verified placement requires
an aware timestamp no later than recording. Repeated equivalent placements agree;
contradictory verified timestamps remain conflicting. Export creation, game start,
import time, a current filter label and the canonical `created_at_utc` are never
substitutes for placement evidence.

## Private starter and owner review

Run from the repository root with the normal Poetry environment:

```bash
poetry run python scripts/manage_operating_regimes.py template \
  --recorded-at 2026-11-24T00:00:00Z \
  --output data/processed/private/operating_history.json
```

The recording timestamp above is illustrative; supply the actual receipt time.
The generator writes a **starter requiring completion**, never an approved account
history. Replace its evidence ID, namespace and reference; choose the Core strategy's
stable filter ID and supported product scope. The starter's null filter scope is a
placeholder, not permission to apply this transition across an entire account.

It retains November 10, 2026 as a date-only owner transition: real-money execution
under the Core strategy stopped when PROD became TEST; subsequent paper execution
uses discretionary decisions and Kelly-recommended sizing. The exact transition
instant and end remain unverified. Do not manufacture midnight, a timezone or an
indefinite interval to obtain matches. Old broad TEST capture and other strategies
require separate scoped evidence. Proposed Passed tags remain TEST-only and are
not executions. They do not create historical passes or accepted amounts.

Add settings only with supporting provenance, using the existing typed states.
For example, confirmed Pinnacle/Worst Case configuration evidence should be scoped
to the observation and relevant effective period, not retroactively copied into old
bets. Support's pricing clarification does not establish every historical scanner
configuration. Record venue access separately from selected scanner sportsbooks.
No Kelly multiplier, bankroll amount, venue list or verified placement is guessed
by the starter. Review the private evidence before publishing or analytical use.

Generated template/migration destinations must be beneath this checkout's ignored
`data/processed` directory and must not already exist. No private ledger is committed.
Only template/migration writes are restricted by the CLI to that private subtree;
the publication API accepts an explicit separate root. Keep publication roots
private/ignored as well.

## Explicit legacy migration

Migration creates a new independent ledger; it does not alter old artifacts or
activate a consumer. The adapter accepts an **initial version-1 sizing JSON ledger**
(`previous_history_id: null`) and/or a version-1 prospective decision-evidence JSON
payload. Convert YAML deliberately to JSON if needed; this CLI does not parse YAML.
Retain originals, including their manifests and predecessor chains, privately.

```bash
poetry run python scripts/manage_operating_regimes.py migrate \
  --canonical data/processed/canonical/REPLACE_BUNDLE_ID \
  --sizing-history data/processed/private/sizing_history.json \
  --decision-evidence data/processed/private/decision_evidence.json \
  --knowledge-as-of 2026-11-24T00:00:00Z \
  --output data/processed/private/migrated_operating_history.json
```

Choose at least one legacy input; the two input flags are independently optional.
Sizing policies preserve interval, product, evidence/confidence and known bankroll/
Kelly settings, referencing the source payload fingerprint and original policy ID.
Selection, funding and paper-sizing convention remain unknown. A legacy funding
scope disables automatic linkage rather than broadening its meaning. Sizing
application, constraints and suggested/accepted amounts remain in the original
sizing contract; migration does not copy them into fill assertions. Settings use
the existing numeric configuration validator; original exact-decimal representations
remain available in the source sizing artifact.

Prospective baselines preserve their settings and effective dates, resolve an exact
unambiguous existing registry alias or stable ID, and remain `planned` with unknown
confidence and the placeholder namespace `unassigned_legacy_namespace`. Unknown or
ambiguous aliases fail instead of guessing. Baseline lifecycle/translation evidence
stays in the original decision contract. Migration does not convert proposed settings
into historical application. Detached sizing revisions are rejected because their
append history cannot be validated without their predecessor; retain/use their
existing reader, or prepare separately reviewed operating evidence.

## Immutable publication and unresolved linkage

```bash
poetry run python scripts/manage_operating_regimes.py publish \
  --canonical data/processed/canonical/REPLACE_BUNDLE_ID \
  --history data/processed/private/operating_history.json \
  --output-root data/processed/operating_regimes
```

The publisher verifies the canonical bundle and uses its pinned registry. It emits
`history.json`, `registry.json`, `linkage.parquet`, `audit.json` and `manifest.json`
under a content-addressed directory. Identity pins the logical history/registry,
exact input-history bytes, canonical manifest, predecessor manifest, implementation
hashes and Python/pandas runtime. Verified replay returns the existing directory;
corrupt artifacts fail, publication uses a writer lock and atomic staging, and output
roots must not overlap canonical publications or be inside a predecessor. A shared
parent directory may contain multiple immutable operating revisions.

`load_operating_regimes(path)` returns `(history, linkage, audit)`, verifying the
manifest, artifact hashes, candidate uniqueness and coverage. The reader does not
look up current configuration or require the old canonical input still to be present.
The publisher verifies the predecessor chain at revision creation; detached reads
verify stored payload integrity, not a recursively retrieved external chain.

The linkage table has one row per canonical candidate, with `bet_id`,
`regime_ids_json`, `linkage_status`, `selection_mode`, `funding_evidence`,
`paper_sizing`, `settings_json`, `venue_access_json`. It contains no money or verified
fill fields. Matching requires independent verified placement, a confirmed exact
interval from owner/contemporaneous evidence, matching namespace/product and, when
scoped to a filter, existing resolved stable membership. Labels are never membership.

| Status | Interpretation |
| --- | --- |
| `unknown_placement` | No independently verified placement; no date fallback. |
| `conflicting_placement` | Multiple incompatible verified placement timestamps. |
| `unknown_regime` | No trusted interval in scope contains placement. |
| `ambiguous_regime` | Multiple supported interval records match; keep all IDs, assign no regime facets. Even equivalent overlaps remain explicit for review. |
| `funding_conflict` | One interval matches but its funding claim contradicts known canonical funding; no facets assigned. |
| `matched` | One supported interval matches. Facets remain evidence, not canonical reassignment or verified application/fills. |

Canonical unknown funding remains unknown even if `funding_evidence` says paper.
Missing or conflicting facets remain unknown. The audit reports candidate/status
counts and `unlinked_record_ids` for records ineligible for automatic linkage
(date-only, planned or unconfirmed), not every trusted record without a matching bet.
It explicitly states that canonical classifications are unchanged and fills are not
inferred. Any later analytical consumer must disclose coverage/unresolved remainder
and preserve funding/product denominators; a sidecar match does not validate a
chronological holdout, corrected comparison or profitability claim.

## Append revisions, compatibility and rollback

Set `previous_history_id` to the prior ledger's logical fingerprint; retain its
`records` and `wager_evidence` arrays as exact prefixes, append new IDs and move the
knowledge cutoff only forward. Publish with `--previous` pointing to the prior
operating publication. The canonical input must remain identical. If source/canonical
scope changes, create a separately reviewed new lineage; never silently substitute a
new export. Additional conflicting evidence remains unresolved rather than taking
latest-wins precedence. Existing assertions cannot be edited or deleted in a revision.

Old sizing/configuration publications keep their versions/readers and accepted-fill
semantics. No default consumer is activated by R05. Roll back by omitting this
optional sidecar or selecting a previous publication, retaining all artifacts.
R06 retains venue capability/economics work, R07 shared analysis context, and later
capture work retains missing historical evidence requirements. See the
[R05 delivery plan](roadmap/pr-plans/R05.md) for validation and delivery records.

## Optional venue/product evidence

The [venue capability registry](VENUE_CAPABILITIES.md) preserves stable identities,
source/effective scope, units/fees/settlement gaps, access and operation permissions.
Historical labels and new brand requests never establish verified economics or
current account access. Existing canonical behavior remains unchanged.
