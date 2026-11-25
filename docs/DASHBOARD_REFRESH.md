# Local dashboard CSV refresh (R10)

R10 adds an opt-in upload/build workflow to the existing read-only research viewer.
It reuses [R09 staging](CSV_STAGING.md), immutable imports/snapshots and the pipeline.
It does not connect to OddsJam or place bets. The default dashboard stays read-only.

## Enable locally

In a private copy of `configs/dashboard.yaml`, set `refresh_pipeline_config` to the
trusted local pipeline YAML path (for example `configs/weekly_research.yaml`). Its
`output_root` must equal the dashboard `artifact_root`. Its source config identifies
the private import store. Existing stores must be owner-only (mode 0700); review
permissions deliberately rather than having the app change an existing store.
Do not enable this in a shared/hosted deployment: job ownership is scoped to a
local Streamlit session, not an authenticated multi-user authorization boundary.
Bind Streamlit to `127.0.0.1`. Paths/configuration are administrator-controlled;
no upload filename is used as a filesystem destination or command argument.

```sh
ODDSJAM_DASHBOARD_CONFIG=/absolute/private/dashboard.yaml PYTHONPATH=src poetry run streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8520 --server.headless true --browser.gatherUsageStats false
```

## Review, build and activate

1. Expand **Refresh Bet Tracker CSV**, drop a CSV or use the accessible file picker.
   R10 review increases the byte bound to 50,000,000 bytes; 200,000 rows and
   100 columns remain enforced. On first use, expand Source settings, confirm
   full/partial coverage and save your account alias and timezone once. Future
   uploads reuse those settings; overrides stay available in the expander.
   Export time is optional and never replaced with upload time. CSV timestamp
   offsets are preserved; naive timestamps retain the declared timezone or unknown. These are declarations,
   not verified completeness or freshness. No date is inferred from the filename.
2. Select **Preview CSV**. Inspect new/unchanged/revised/unresolved candidate rows,
   settlement revisions, source diagnostics and repeat status. Changes to controls
   require another preview. The displayed frozen preview is what confirmation binds.
   Unknown coverage cannot be confirmed. Formula-like cells remain literal text.
3. Check the confirmation box and select **Import and build report**. A fixed local
   subprocess performs the guarded import and builds an as-of snapshot report.
   Stale reviews fail closed; review again if another import changed the store.
   **Refresh build status** reads durable stage events. Progress is stage-based,
   not a fabricated percentage. The selected report remains usable while it runs.
4. **Cancel build** requests cooperative cancellation between stages and immediately
   before result publication. It cannot undo committed imports or a result already
   atomically published. **Retry frozen build** uses the same request, import cutoff
   and verified checkpoints; no source is silently substituted. Changed code/config
   requires a new reviewed request. An active job cannot be retried through the UI.
5. On success review receipt/export/analysis cutoff, build completion and analytical
   exclusion combinations, then choose **Activate verified report**. Integrity and
   display contracts must pass before switching. A newly populated empty catalogue
   also requires activation. Existing research/evidence attachments are not silently
   rebound to the new report. No optional model fit, scanner or study is launched.

Added/revised/unchanged/unresolved counts describe source reconciliation; analytical
exclusions apply to canonical wager candidates after building, and are displayed
separately. They have different grains and must not be summed. Full export remains
an owner declaration. This entry point uses historical snapshot reanalysis, even for
recent uploads; an unknown export time is retained as unknown. Build completion is
not source freshness and does not establish decision-time/settlement chronology.

## Recovery and rollback

Requests and attempt receipts are beneath `<artifact_root>/dashboard_refresh/jobs`;
staged files are separate under `dashboard_refresh/staging`. Directory mode 0700
protects uploaded bytes and session job IDs. Keep both trees out of Git/backups that
are not approved for private data. Session loss does not automatically cancel a
worker; inspect its PID and local records. A crash can leave a writer lock. Confirm
that its process is stopped before explicitly removing that job's stale lock; never
remove locks automatically or while a writer runs. Refresh reports a missing terminal
receipt distinctly. No automatic worker restart, import deletion or report fallback.

A failure after import can leave the accepted import and verified stage artifacts;
retry preserves them. An interrupted import-before-receipt window may require a
fresh preview/request; content identity still prevents duplicate imports. Do not edit
receipts to repair state. Stop launching new builds, let active work finish or cancel,
and remove `refresh_pipeline_config` to disable the UI. Old immutable reports and
existing CLI workflows remain available. Shared hosting, authentication, quotas,
retention/cleanup automation and distributed workers require later design work.

## Synthetic owner acceptance checkpoint

Run from the R10 worktree. The preparation directory must not already exist.
Only synthetic `Book`/`TEST - Demo` records are created; no account history is read.
**Use only the supplied synthetic CSVs with this configuration.** Changing the
source/account alias does not replace product mappings or research policies.
Before uploading real history, configure the repository research pipeline as
described above; the synthetic fixture is not a production-history configuration.

```sh
PYTHONPATH=src poetry run python scripts/acceptance/prepare_refresh.py /tmp/oddsjam-r10-owner-review --rows 1000
ODDSJAM_DASHBOARD_CONFIG=/tmp/oddsjam-r10-owner-review/dashboard.yaml PYTHONPATH=src poetry run streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8520 --server.headless true --browser.gatherUsageStats false
```

Open <http://127.0.0.1:8520>. If port 8520 is already running the prepared review
instance, use that instance with `/tmp/oddsjam-r10-review/history.csv` and
`revision.csv`; its catalogue may already contain the automated acceptance reports.
For a clean initial-empty check use a fresh directory and port 8521.

- Upload `history.csv`, declare `full`, preview, confirm and build. Refresh status
  until ready. Check readable counts/events and separate source/build dates. The
  report selector must stay unchanged until **Activate verified report**.
- Activate, then upload the same file with identical metadata. Preview must show
  repeated/unchanged, and a repeated build must preserve the selected report.
- Upload `revision.csv` and preview. Review the settlement revision; confirm/build,
  then activate only after inspecting the replacement. Verify report navigation.
- Upload `invalid.csv`: preview must fail visibly without changing the report.
  During another build, request cancellation; it may finish first on this small
  workload. In either case the selected report must remain usable. Retry a cancelled
  build, inspect terminal status and explicitly activate only if desired.
- Confirm whether the drop zone/picker, text, progress, error/retry controls and
  activation behavior are understandable and work as expected. Report any findings.

Owner acceptance is pending under the reusable delivery instructions' manual-review
checkpoint. Automated browser checks supplement, but do not substitute for it.

## Validation evidence

Local suite at implementation commit `3c8799c`: 652 passed in 137.73 seconds.
Seven affected cases passed after browser upload-size hardening.
Lock, Ruff check and format checks passed. Synthetic AppTest verifies initial empty
catalogue activation; backend tests verify stale reviews, cancellation, failure,
pinned retry, repeat/import idempotency, changed config and request integrity.
Isolated headless Chrome exercised the actual CSV file input, coverage selection,
preview/confirmation, subprocess build and explicit activation without JS exceptions. Repeated upload and an invalid optional-CLV build also
preserved the active report in Chrome.
Private browser screenshot evidence: `/tmp/oddsjam-r10-browser/activated.png`.

Declared workload: 1,000 synthetic independent event records, 110 KB CSV, 100
bootstrap resamples, one private empty import store, no optional job/model/study.
Measured staging+preview+import+core report: 2.86 s; Python process peak RSS
196,427,776 bytes (~187 MiB) on this macOS environment. This excludes browser/server
memory and is not a maximum-size or production-history capacity certification.

## Zero eligible wagers after a successful build

A completed build verifies processing, not analytical usefulness. Inspect exclusion
counts before accepting results. Extensive `unknown_product` exclusions can indicate
that a synthetic acceptance configuration was used with real platform history.
Use the vetted repository canonical mappings and research policies; do not remove
eligibility checks or guess classifications merely to increase coverage.

For an already accepted import, the local operator can materialize its snapshot
at the import receipt's original `imported_at` cutoff and run the pipeline with a
new, corrected configuration. This produces a distinct immutable run without
another upload. Preserve the old request, receipts and report; do not edit frozen
job inputs or resume the old run with changed settings. Verify eligible counts,
remaining exclusions and source limitations, then explicitly select the new report.

The core refresh does not create the optional review-queue publication. Its absence
is distinct from product classification failure. Generating a queue requires its
own configured research job and evidence; never attach another run's queue or
invent missing configuration/lifecycle evidence to populate the overview.
