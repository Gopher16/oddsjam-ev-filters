# OddsJam workflow image references

Use the [figure catalogue](CATALOGUE.md) and [documentation manifest](manifest.json)
to locate the source screenshots supporting the [operating workflow](../../operations/ODDSJAM_WORKFLOW.md)
and [case studies A–G](../../operations/case-studies/README.md).

## What is available now

The catalogue represents Figures 1–50 of the frozen owner R&D Word document.
Figure 11 explicitly contains two screenshots, so 51 stable slots are reserved.
**No account images are published in this directory.** Each slot is a readable text
placeholder, not a fabricated illustration, a broken image reference or a claim
that an image has been redacted/reviewed. Source figure numbers identify Word
captions, not independently verified image metadata. Existing
[synthetic dashboard images](../operator/README.md) illustrate a different workflow
and must not be substituted for these private account screenshots.

The owner source remains private and unchanged. Its SHA-256, caption and source
figure references provide traceability without publishing local account file paths,
raw exports, identifiers or the Word document itself. Case narratives intentionally
retain the selected quote/fill amounts needed to explain decisions; they do not
reproduce account records, balances, transaction IDs or complete wager histories.

## Manifest conventions

- `asset_id` is stable: `owf-001` through `owf-050`, with `owf-011-a` and
  `owf-011-b` for Figure 11. Slots are references, not a deduplicated media count.
- `source_caption` preserves the Word caption. `proposed_alt_text` describes the
  intended view for a future reviewed image; revise it against the actual approved
  image before publication. Figure 11's panel allocation is not invented.
- `original_capture_date` is null until established from trusted original evidence.
  `catalogued_on` is the inventory date. Neither this date, a filesystem modification
  time nor a case's opportunity date supplies the original capture date.
- `source_disposition` is private; `public_disposition` is currently
  `text_placeholder_only`. `public_asset_path` and `public_asset_sha256` are null.
  No file is promised by a null path.
- `publication_status` is `awaiting_image_review`; `visual_review.status` is
  `not_performed`. Text/caption checking is not visual privacy review.
- `case_id` and `context_path` connect examples to workflow decisions; null case
  means general controls or blended-quote context. The manifest is a documentation
  inventory, not a change to the application's canonical or publication schemas.

## Add or replace an image deliberately

1. Identify the stable figure/slot and privately compare the original with its source
   caption and narrative. Do not infer capture date, venue units or fill details.
2. Inspect the entire visible image for private account identity, balances, order IDs,
   history, browser tabs and unrelated material. Prepare a public-safe derivative
   only through a separately reviewed asset change; never commit a raw account image.
3. Ensure the approved derivative preserves the relevant price, depth, warning and
   execution distinctions. Do not edit a displayed value to reconcile a discrepancy.
   Document omissions/redactions and any limits on interpretation.
4. Before committing, visually review the actual derivative and obtain owner
   confirmation when release of sensitive evidence or its interpretation requires it.
   Merely attaching a document is not a recorded privacy review of every image.
5. Add a descriptive filename under this directory, set its relative manifest path
   and SHA-256, record the actual review date and disposition, and revise alt text
   to describe the visible approved view. Unknown original dates remain null.
6. Replace the slot's text placeholder with an actual image link and descriptive
   alt text in the catalogue. Preserve its source figure anchor and asset ID. State
   source, redaction/derivative status and capture-date uncertainty in its caption.
7. Run tracked Markdown checks and inspect rendered links, image readability and
   alt text. No private original path or raw bytes should enter Git history.

Keep source discrepancies explicit. In particular, Case G's approximate tagging
amount is not the sum of its two quoted sizes, and Case F's rounded display price
is not the execution price. Missing public images do not erase those distinctions.

[Catalogue](CATALOGUE.md) · [Cases](../../operations/case-studies/README.md) ·
[Documentation home](../../README.md)
