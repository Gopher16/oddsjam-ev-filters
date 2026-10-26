# Historical filter identity proposals

Prepare a proposal after publishing the [exact-label inventory](FILTER_LABEL_INVENTORY.md).
The command requires that inventory and its exact original canonical bundle:

```sh
poetry run python scripts/propose_filter_mappings.py \
  --inventory data/processed/filter_label_inventory/<inventory-id> \
  --canonical data/processed/canonical/<bundle-id>
```

The private, ignored destination is
`data/processed/filter_mapping_proposals/<proposal-id>/`. `proposal.json` carries
machine-readable proposals and pending questions; `review.md` provides a human
review view. The manifest pins inventory/canonical hashes, the registry identity,
proposal policy and implementation. Identical replay verifies existing outputs;
changes produce a new identity. Never edit a publication to record a decision.

## Proposed identity contract

Exact registered names and existing aliases take precedence. Existing unambiguous
registered memberships keep their IDs. Registered former names and explicit clone
relationships are reported as registry evidence, not inferred anew.

Every remaining nonblank exact string receives a deterministic provisional ID,
`historical_label_<full-label-fingerprint>`, and its original display string.
These IDs remain stable when other labels are added. They preserve separate
historical identities for review; they do not establish that a source string is
one filter. Null, empty and whitespace-only labels receive no invented membership.
New former-name lists remain empty until identity evidence supports an alias.

Name comparisons remove a leading PROD/TEST prefix, case-fold and collapse
whitespace **only for review leads**. Exact comparison values or SequenceMatcher
similarity of at least 0.82 create possible rename/separate-filter leads. Only
prefix/case/spacing equivalents enter the question queue on similarity grounds;
other name similarities remain supplemental context. This
heuristic is not a confidence probability. Similar devig or product names can be
separate filters; similarity never merges IDs or establishes a clone. Unmatched
short labels also enter the ambiguity queue because they may be manual/platform
tags. These heuristics do not exhaust all possible historical relationships.

For unresolved strings, the existing bounded resolver searches registered names
and other observed whole labels for a possible unambiguous segmentation. Such
memberships remain proposals. Commas inside names are never split blindly. A
whole exact registered name always wins over segmentation; ambiguous or incomplete
segmentation stays unresolved.

Each proposal links its exact inventory row, counts, import lineage and timestamp
bounds. Counts are non-additive, and lexical dates do not establish rename order.
Pinned current configurations are not evidence of historical parameter effectivity.
Funding is never inferred from names or from approval of an alias.

## Review and next publication

Review the ambiguity queue rather than reconstructing every old filter's settings.
For each decision, retain:

- Exact source label and proposed stable identity or membership segmentation.
- Relationship: same identity/rename, separate identity, explicit clone, non-filter
  tag, or unknown. Separate clones must retain distinct IDs.
- Evidence reference, decision, confirmed IDs and recorded-at timestamp. The pending
  JSON fields provide a review shape; decision ingestion/application is deferred.

Keep uncertain cases separate and unresolved. Current screenshots can support
current names/settings, but cannot alone prove a rename or historical settings.
Deleted filters may have known recorded membership with unknown configurations.
Missing labels cannot be assigned from economics or sportsbook similarity.

A later change will validate reviewed decisions, reject collisions and publish a
new registry/attribution version with before/after coverage audits. This command
neither updates the registry nor rebuilds canonical, atlas or dashboard reports.
Private labels and generated review artifacts must not be committed.
