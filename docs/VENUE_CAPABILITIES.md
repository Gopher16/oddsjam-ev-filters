# Venue/product and source-capability evidence

R06 adds an optional version-1 registry for historical classification, product
contracts, account access and source-operation evidence. It does not connect accounts,
fetch prices, calculate payouts, place orders or change canonical facts. The existing
[canonical policy](CANONICAL_FACTS.md) and readers remain unchanged. A request for a
brand is neither a verified product nor permission to operate it.

This registry complements [operating-regime evidence](OPERATING_REGIMES.md): regime
venue names describe owner context; this contract provides stable product/source
identity and separately scoped capability evidence. Future R07 shared context and
later adapters can explicitly pin these publications. No dashboard/default runner
consumer is activated by this release.

## Identity, versions and time

Each JSON ledger has exactly `contract_version: 1`, `knowledge_as_of`,
`previous_registry_id`, `source_policy_id`, and these arrays:

| Array | Required record fields and meaning |
| --- | --- |
| `venues` | `venue_id`, `name`, `recorded_at`, `reference`. Stable identity; names are display text, not economics. |
| `venue_aliases` | `venue_id`, `alias`, `recorded_at`, `reference`. Exact explicit aliases; duplicate aliases and collisions with another venue's ID fail. Add a new assertion to extend aliases; no fuzzy/case/brand-stem matching. |
| `products` | `version_id`, `product_id`, `venue_id`, `source_id`, `product_type`, `aliases`, effective scope fields, `recorded_at`, `reference`, `contract`, `operations`. Stable product ownership/type cannot change; different products under one brand remain distinct. |
| `access` | `evidence_id`, `namespace`, `product_id`, `source_id`, `state`, effective scope fields, `recorded_at`, `reference`. States are `confirmed`, `denied`, `unknown`; product/source must exist. Never infer access from a historical wager. |
| `requests` | `request_id`, `name`, nullable `venue_id`, `product_type`, `recorded_at`, `reference`, `status: requested`, `notes`. A request can exist without a venue identity, product record, account or source. |

Product classes are `fixed_odds`, `exchange`, `pickem`, `dfs`, `unknown`. All IDs
are nonempty exact strings, unique within their record category. Version IDs differ
from stable product IDs. A source ID is an explicit identifier such as `oddsjam_csv`,
not an inferred vendor API capability. Product aliases may not belong to different
products in the same venue/source over overlapping or unknown effective periods.

Effective scope fields are `effective_from`, `effective_to`, `open_ended`,
`scope_verified`. Known timestamps must be aware ISO timestamps. Verified intervals
are half-open `[from, to)`; indefinite verified scope requires `open_ended: true`.
An unknown start requires null boundaries and both flags false. Never invent dates
or assume an unknown end is indefinitely valid. Unverified intervals cannot establish
readiness. All receipt times must be no later than the ledger's aware knowledge cutoff.
Scope/receipt time do not prove quote availability, decision chronology or fills.

Revisions must retain all old arrays as exact prefixes, keep the same source-policy
identity, reference the prior logical fingerprint and move the knowledge cutoff only
forward. Append versions/access/requests or alias assertions; never edit/delete old
claims. Unknown bounds conservatively overlap. Conflicting assertions remain explicit,
not latest-wins. A separately reviewed new lineage is required to revise earlier
assertions or seed from a different policy; old publications remain available.

## Contract units, fees and settlement

Every product version has a `contract` object with these exact fields:

- `evidence_state`: `unknown`, `confirmed`, `unsupported`; `reference` is nullable,
  but confirmation requires a nonempty source reference.
- `currency`: `unknown` or `USD` in this bounded version. This registry does not
  implement FX conversion or comparative non-USD economics.
- `price_unit`: `unknown`, `american_odds`, `decimal_odds`, `usd_per_contract`,
  `payout_multiplier`; `stake_unit`: `unknown`, `usd`, `contracts`, `ticket_usd`;
  `payout_unit`: `unknown`, `gross_usd`, `usd_per_contract`.
- Nullable `unit_payout`: finite nonnegative numeric value, strictly positive for
  exchange contract readiness. This is a declared amount per contract, not profit.
- `fee_model`: `unknown`, `none`, `rate`. Missing fees never mean zero. A rate model
  needs finite `fee_rate` from 0 through 1 and `fee_basis` of `stake`, `gross_profit`
  or `payout`. Other fee models require future adapter/contract work; leave readiness
  blocked rather than approximate them. Only rate models may supply a rate.
- `settlement_model`: `unknown`, `fixed_odds_win_loss`, `binary_contract`,
  `ticket_paytable`, `unsupported`; nullable `settlement_rules` retains a nonempty
  rule description when supplied. Capture refunds/pushes/voids, payout inclusions,
  settlement source/version and exceptional cases in supporting evidence.

The bounded evidence check recognizes coherent fixed-odds unit metadata
(odds price, USD stake, gross USD payout, fixed-odds settlement) and binary exchange
metadata (USD/contract price, contract quantity, USD/contract payout, positive unit
payout). Pick'em/ticket/DFS settlement is explicitly unsupported by this readiness
check even where historical classification exists. Brand identity cannot supply
missing units, fees, rules or payout schedules.

`economic_contract_ready` means only that these bounded metadata requirements and
verified effective scope are satisfied. **It is not an executable payout adapter,
ROI eligibility, validated economic calculation, contract legal review or deployment
approval.** No registry result enters canonical monetary calculations. Later adapters
must validate their actual settlement/fee rules independently before use.

## Account access and source operations

Each product's `operations` has all five keys: `manual_csv_import`, `quotes_read`,
`orders_submit`, `orders_cancel`, `automated_export`. Each has a `support` value
(`supported`, `unsupported`, `unknown`) and independent `permission` value
(`permitted`, `denied`, `unknown`). Product receipt/reference applies to these
assertions; retain the actual evidence privately. Metadata cannot grant permissions.

`evidence_ready` additionally requires a single confirmed access assertion for the
requested namespace/product/source/time, supported operation and permitted status.
An undated access assertion or conflicting active assertions blocks readiness.
No supplied namespace means access remains unverified. Technical support does not
imply permission; permission does not imply an implemented connector. Every lookup
always returns `execution_authorized: false`.

Historical seeds mark only offline `manual_csv_import` as supported/permitted in
repository research. This is a local import capability, not a vendor API entitlement
or account-access claim. All other operations remain unknown, including export
automation: the registry does not reverse support's manual-export limitation or
implement an automated OddsJam export route.

## Seed and inspect historical products

Run from the repository root with the locked Poetry environment:

```bash
poetry run python scripts/manage_venue_registry.py seed \
  --policy configs/canonical_facts.yaml \
  --recorded-at 2026-11-24T00:00:00Z \
  --output data/processed/private/venue_registry.json

poetry run python scripts/manage_venue_registry.py publish \
  --registry data/processed/private/venue_registry.json \
  --output-root data/processed/venue_registry
```

Supply the actual receipt time; the example timestamp is illustrative. Seed generation
copies every exact historical product/label from the explicit canonical policy,
pins its logical fingerprint and does not modify that policy. Stable IDs derive
from exact explicit identities/labels, not a guessed payout model. Only the two
DraftKings Pick 3/Pick 6 labels are explicitly grouped under a shared DraftKings
venue identity; other labels retain separate source identities rather than guessed
brand consolidations. Their products remain distinct.

All seeded effective scopes, fees, settlement units/rules and account access remain
unknown. Independent requested-evaluation entries for Robinhood, DraftKings and
FanDuel do not acquire support from same-brand historical products. The seed is a
private starter for supported source review, not completed venue due diligence.
Generated seeds/revisions must be beneath this checkout's ignored `data/processed`
and must not overwrite existing files. Do not commit account-specific evidence.

Use the exact publication directory printed by `publish`:

```bash
poetry run python scripts/manage_venue_registry.py lookup \
  --publication data/processed/venue_registry/REPLACE_PUBLICATION_ID \
  --venue DraftKings --source-id oddsjam_csv \
  --label 'DraftKings (Pick 6)'
```

With no label, DraftKings is ambiguous across its historical products. A selected
Pick 6 label resolves declared metadata but remains blocked for comparable economics.
Optional `--product-type`, `--at`, `--namespace`, `--operation` refine the query;
`--version-id` explicitly pins a supported version when unknown/overlapping history
would otherwise remain ambiguous. Pinning cannot bypass scope or evidence gates.
No timestamp means effective readiness remains unverified. Different sources require
separate assertions; none is inferred from another source's support.

## New requests and supported onboarding

```bash
poetry run python scripts/manage_venue_registry.py request \
  --previous data/processed/venue_registry/REPLACE_PUBLICATION_ID \
  --request-id evaluate-example-venue --name 'Example venue' \
  --product-type exchange --recorded-at 2026-11-24T00:00:00Z \
  --reference 'Owner research request' --notes 'Access and contract still unverified' \
  --output data/processed/private/venue_registry_next.json

poetry run python scripts/manage_venue_registry.py publish \
  --registry data/processed/private/venue_registry_next.json \
  --previous data/processed/venue_registry/REPLACE_PUBLICATION_ID \
  --output-root data/processed/venue_registry
```

`--venue-id` may attach an existing stable identity; absent identity is allowed.
Requests create no product mapping or access record. For actual onboarding, review
private product/source documentation, add distinct stable identities and exact aliases,
complete scoped contract/fee/rule/capability metadata and independent namespace access,
then publish an append-only revision. Use synthetic rule cases before a later economic
adapter adopts it. Preserve unsupported capabilities and missing evidence explicitly.

The read-only `resolve_product` API returns status (`unknown`, `ambiguous`, `resolved`),
candidate version IDs, declared product ID/type, contract/evidence readiness, blockers,
linked request IDs and an always-false execution flag. Blockers include unknown product,
ambiguous product, unverified effective scope, unknown currency/fees/contract, unsupported
units/settlement, missing rules/unit payout, unresolved access and operation support/
permission. Resolved identity alone is not readiness. No account P&L is produced.

## Publication integrity, compatibility and rollback

`publish_venue_registry` writes `registry.json` and `manifest.json` under a
content-addressed directory with writer lock and atomic staging. Identity pins logical
registry, exact source JSON bytes, predecessor manifest, implementation hashes and
Python runtime. Verified replay returns the same directory; corrupt registry,
manifest identity or summary fails. Roots cannot be inside a predecessor; multiple
revisions may share a private parent directory. `load_venue_registry` verifies stored
bytes/schema/summary; the publisher validates predecessor history. A detached read
checks stored integrity without recursively retrieving the external predecessor chain.

Old canonical, sizing and operating publications retain their readers and economics.
Rollback by omitting this optional registry or pinning a previous publication. Real
source/access evidence is still an owner task. R07 shared context, later venue adapters,
scenario engines and execution gates retain their roadmap destinations. See the
[R06 plan/results](roadmap/pr-plans/R06.md) and [delivery register](roadmap/DELIVERY_REGISTER.md).
