# Local research pipeline

The pipeline connects manual ingestion, canonical facts, the descriptive atlas,
discovery, research proposals and optional experiment follow-up. It never acquires
exports, edits filters, approves proposals, enrolls observations, launches experiments,
or places wagers. **Scheduling is disabled and cannot be enabled by configuration.**
There is no scheduled workflow, network request, external alert or artifact upload.

## Inputs and commands

Run from the repository root in the Poetry environment. Existing notebooks,
`configs/preprocess.yaml`, and the historical notebook snapshot are independent of
this pipeline and remain unchanged. No notebook is executed by the runner.

Choose exactly one input mode:

```bash
# A manually downloaded export, with acquisition context declared in the source config.
poetry run python scripts/run_weekly.py \
  --config configs/weekly_research.yaml \
  --csv /path/to/manual-export.csv

# Reanalyze an explicit historical snapshot at a reproducible aware knowledge cutoff.
poetry run python scripts/run_weekly.py \
  --config configs/weekly_research.yaml \
  --snapshot data/raw/ingestion/snapshots/SNAPSHOT_ID \
  --as-of 2026-10-23T16:00:00Z

# Retry/replay the exact frozen plan identified in the previous attempt's output.
poetry run python scripts/run_weekly.py \
  --config configs/weekly_research.yaml \
  --resume RUN_ID
```

Replace the placeholder paths and IDs. No mode searches Downloads or selects a
"latest" file. `--snapshot` requires an explicit timezone-aware `--as-of`, no later
than the current system clock and no earlier than the selected snapshot's knowledge.
A new CSV run freezes its cutoff at that import's system receipt time, after the
source is accepted; the user cannot backdate it. Imports that arrive later are not
included in recovery of that run. Snapshot construction retains all accepted
imports visible at the cutoff, rather than discarding older sources automatically.

Source metadata comes from `configs/data_sources.yaml` (or an explicitly selected
source configuration in the pipeline policy). Confirm `exported_at` only from
actual evidence, including its UTC offset. Keep source `timestamp_timezone` and
`scope` unknown until established. Receipt times, filenames, the laptop timezone,
and the latest bet date cannot establish these fields. Importing identical bytes
with conflicting source metadata is rejected by the existing importer; never edit
an archived manifest to bypass this safeguard.

`--decisions /path/to/reviewed-resolutions.json` can supply existing reviewed
identity resolutions while constructing a CSV snapshot. It does not create or
backdate decisions. It cannot be applied to an existing snapshot. Optional
`--temporal-evidence /path/to/evidence.json` passes evidence to the existing discovery
gates without manufacturing it. Optional `--register /path/to/experiment-register`
produces experiment follow-up at the same cutoff. Omitting the register skips this
stage; an explicitly empty register produces empty experiment tables. No registration
or enrollment is inferred from historical source records.

Resume takes only `--config` and `--resume`. Frozen source/configuration/code/knowledge
references are reused, not whatever happens to be newest now. Changing code,
dependencies, configuration or the frozen register prefix requires a new run rather
than resuming an incompatible plan. Later valid register events remain outside the
original cutoff. Commit changes also affect recorded code provenance, so replay
validation across commits is a new run even if the analytical data is unchanged.

## Freshness, novelty and analytical readiness

The eight-day maximum confirmed export age is an operational convention, not proof
of complete coverage or statistical power. Each selected import is assessed at the
run's pinned cutoff. A recent export cannot conceal an older or undated selected
source: any stale source makes the aggregate stale; otherwise any undated source
makes it unverified. We do not presume that a newer export supersedes all older rows.
Confirmed scope is displayed separately and never upgraded into verified completeness.

| Situation | Behavior |
| --- | --- |
| No input or missing input path | Private `missing_input` notice; no analytical fallback |
| CSV already imported outside this unfinished run, or already frozen by this runner | `no_new_source_content` notice; no new observation-period claim |
| New CSV with unknown export timing | Preserve valid ingestion/snapshot; emit `freshness_unverified`; no current analytical report |
| New CSV with stale export timing | Preserve valid ingestion/snapshot; emit `stale_input`; no current analytical report |
| Explicit historical snapshot with stale/unknown freshness | Permit historical reanalysis with a prominent freshness assessment |
| Missing required headers, malformed rows, invalid numeric values or corrupted hashes | Fail; no successful final report |
| Added columns or missing optional fields | Preserve existing parser diagnostics and analytical missingness/exclusions; never invent fields |
| Unknown source chronology, identity ambiguity, unsupported products | Preserve existing reconciliation and canonical exclusions; show audits |
| Statistical/temporal prerequisites absent | Descriptive output only; existing gates remain blocked |

An unchanged source with changed policy/code is **reanalysis**, not new evidence.
Repeated imports still record the importer's existing attempt history. If a process
fails before freezing a plan, rerun the same CSV/configuration: the runner retains an
input-request receipt and can rebuild the interrupted snapshot at the original import
cutoff. After a plan is frozen, use the returned run ID with `--resume`. An old source
already imported before pipeline invocation is not treated as newly received; use an
explicit historical snapshot to analyze it. Undated October exports therefore belong
in historical mode, not a fabricated fresh weekly feed.

CLI exit codes:

- `0`: completed pipeline or historical replay; does not mean inference gates passed.
- `3`: missing, stale, unverified or repeated input notice; no new final analytical report.
- `2`: configuration, validation, lock, I/O or unexpected processing failure.

The terminal prints only attempt/run identifiers, status, exit code and stable error
codes. It does not print exception payloads, raw records, account information or money.
Inspect private attempt records for the failed stage and consult the stage's existing
documentation. Do not paste raw source files or credentials into debugging messages.

## Immutable results and execution attempts

All artifacts remain local under ignored `data/processed/weekly_research/` by default:

```text
input_requests/       Receipt/recovery records for explicit CSV requests
plans/RUN_ID.json     Frozen specification and private source/register locators
attempts/ATTEMPT_ID/  Start, stage events, outcome, and input notice when applicable
checkpoints/RUN_ID/   Verified stage-output references
artifacts/           Existing stage-specific immutable bundles
results/RUN_ID/      Final manifest.json and deterministic report.md
```

Run identity pins snapshot identity and manifest bytes; effective configurations and
policies; all package source hashes; CLI and dependency-lock hashes; Git revision;
library versions; source diagnostics; freshness assessment; aware cutoff; supplied
temporal evidence; and selected experiment events. Attempt timestamps and retry counts
are separate and do not change report content. Private local paths are locators, not
public artifacts. No report or run ledger should be committed or uploaded.

The final manifest records stage outputs, manifest hashes, row counts and aggregate
research evidence. Payload and report hashes detect accidental edits. A repeated run
with the same frozen specification verifies logical equality and returns the same
result. Results are published only after all required stages pass their verified
loaders. The report preserves separate economics cohorts, opportunity proxies,
reconciliation/exclusion counts, discovery gate status, proposal counts and optional
experiment follow-up. Overlapping views are not additive; no overall total is formed
by summing overlapping filter results.

## Failure and retry policy

Each attempt records stage start/completion/reuse/failure without arbitrary exception
text. A writer lock covers the pipeline run; stage publishers retain their own locks.
Concurrent writers fail rather than overwrite each other. Never remove a lock without
checking for a running writer and reviewing its ownership.

Completed stage bundles survive a later failure. Resume verifies their checkpoint,
manifest and output hashes before reuse. Corruption is a hard failure, not a signal to
silently rebuild or fall back. The final report is published through a staging directory
and atomic rename; failed publication cannot appear as a completed final result.
No mutable "latest successful" pointer is advanced.

Analytical-stage retries are bounded (one by default, maximum two) and restricted to
interrupted calls, temporary resource-unavailability, timeouts and stale file handles.
Schema/policy errors, lock conflicts, permission failures and full disks are not retried
automatically. Ingestion and snapshot failures are recovered by explicit rerun using
the receipt behavior above. A terminated process may leave a started attempt without
an outcome and an operator-review lock; neither is evidence of success. Preserve it
for investigation rather than treating it as an empty successful run.

Hashes and filesystem locks provide local integrity, not authenticated multi-user
storage or protection from a person rewriting the whole ledger. Backups, external
anchors, notifications and shared-service deployment are deferred.

## Research constraints

Currency is USD by user confirmation. Preserve reported profit without estimated fee
adjustments. Fee inclusion, source timezone/chronology/scope and true opportunity
identity remain unverified unless supported by evidence. Use the pinned ingestion
policy for historical data; new snapshots record the current versioned assumptions.
Only the exact historical Core Exchange Conservative attribution is assumed funded;
Core trades reflect selective liquidity and other judgments. Exchange workflow takes
existing orders only and does not prove fills or executable capacity. Unchecked paper
returns are hypothetical. Unsupported ticket economics, Pick'em candidates, refunds,
pending settlements and ambiguous products retain the canonical eligibility rules.
No automatic promotion or change to production follows from successful orchestration.

## Validation

Synthetic tests cover deterministic replay, optional empty-register reporting,
missing/repeated/stale/undated input notices, numeric/schema gates, frozen cutoffs,
later imports and events, configuration changes, per-stage failures, bounded transient
retries, permanent errors, corrupt checkpoints, tampered results, atomic publication,
locks and CLI status codes. Existing quality CI discovers these tests without secrets,
private data, network access or new workflow configuration.

Local historical integration on October 23, 2026 selected ingestion snapshot
`861924428766080b4468adad998805037901532c68f8bdd041e1aa722994c7d1`
at cutoff `2026-10-23T16:00:00Z`, with an explicitly empty experiment register.
Working-tree run
`714b3a5ad4f9fa3cc0959e6f2bb8f04cc6218637654ffcbdb91acfb2d77e7149`
completed and replayed with five verified reused stages and identical final report
and manifest bytes. It retained 70,635 wager candidates, 53,853 reported-ROI-eligible
candidates, 1,965 `COLLECT_MORE_DATA` proposals and 1,965 `REVIEW_MANUALLY` proposals.
All nine experiment tables were empty. Freshness remained **unverified** and scope
remained unknown. This is compatibility/replay evidence, not a prospective experiment
or fresh-feed validation. Artifacts stay ignored locally; a different Git revision
or source hash produces a different run identity by design.
