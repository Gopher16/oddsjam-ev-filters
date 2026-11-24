# Reproducible optional research jobs

The [core pipeline](WEEKLY_RESEARCH.md) can explicitly orchestrate existing optional
publishers. This is a local opt-in job, not a scheduler, upload UI, export service,
dashboard activation or trading action. See [R08 plan/results](roadmap/pr-plans/R08.md)
and [shared analysis context](ANALYSIS_CONTEXT.md).

## Select the work explicitly

Run from the repository using the intended locked environment:

```bash
poetry run python scripts/run_weekly.py --config configs/weekly_research.yaml \
  --snapshot /private/path/verified-snapshot --as-of 2026-11-24T12:00:00-06:00 \
  --research-job configs/research_job.yaml
```

Choose the actual snapshot/cutoff; the paths/time above are examples, not defaults
or new observations. CSV mode still requires the core runner's explicit fresh manual
export and does not redownload OddsJam data. To rebuild optional evidence for an
already imported CSV, choose its verified snapshot and explicit cutoff. Repeated
CSV imports remain notices and never manufacture fresh source data.

Copy the [example job](../configs/research_job.yaml) to a private location before
adding personal evidence. The version-1 YAML accepts only:

| Field | Meaning |
| --- | --- |
| `contract_version` | Exactly integer 1 |
| `stages` | Nonempty unique allowlisted stage names |
| `stage_configs` | Inline existing publisher policies, keyed only by selected dependency stages |
| `sizing_history_file` | Optional private sizing ledger; omission produces the explicit empty-history audit |
| `sizing_previous` | Verified immutable predecessor publication, only with an explicit history ledger |
| `decision_evidence_file` | Optional baseline/lifecycle/translation snapshot; omission produces unknown settings |
| `decision_effective_at` | Explicit aware effective time, no later than run cutoff; omission blocks sheets and queues |
| `model_policy`, `model_scope` | Existing bounded model policy and supported decision-time restrictions; only with models selected |

Relative evidence paths resolve against the job file's directory. Files are bounded
to 2 MB and their contents/hashes are frozen into the run specification. Policies
are validated and expanded to complete existing defaults. Review those defaults
before running; larger jobs can be expensive. This feature does not authorize new
statistical thresholds, truncate candidate families or bypass a builder's budgets.
Invalid settings/evidence or resource overflow fail the run instead of silently
reducing its work. No observed discovery partition is a validation failure, not an
invitation to widen scope or substitute another source.

## Dependency graph and receipt states

Core canonical/atlas/discovery/recommendation stages retain their original behavior.
The new **within-filter discovery** stage is distinct from core atlas discovery.
Requested optional stages expand to this fixed allowlisted closure:

| Stage | Prerequisites |
| --- | --- |
| `sizing` | canonical |
| `sizing_diagnostics` | canonical, sizing |
| `within_filter_discovery` | canonical, atlas, sizing diagnostics |
| `prioritization` | within-filter discovery, sizing diagnostics |
| `sheets` | within-filter discovery, prioritization; explicit effective time |
| `visuals` | canonical, atlas, sizing diagnostics |
| `queues` | within-filter discovery, prioritization, sheets, visuals |
| `models` | canonical, visuals; separate fitting opt-in |

No arbitrary executables, shell fragments, plugins or dynamic graph edges are
accepted. All preexisting explicit builder scripts remain available.

Each final `job.json` records the request, expanded graph/policies, source snapshot,
configuration/code fingerprints and all eight stage receipts. Completed receipts
contain immutable artifact/manifest/dependency hashes. Blocked receipts explain
missing effective time or model-build opt-in and propagate to dependent stages.
Unselected stages say `not_requested`. The Markdown report also lists these states.
A completed job may therefore have blocked capabilities: inspect receipts, not just
the process exit code. Completion does not establish profitability, chronology,
confirmed configuration settings or a usable fitted model.

Sizing without a ledger retains unknown history and replay blockers. Decision
sheets without baseline evidence retain unknown settings and setup blockers. These
are existing evidence audits, not substituted Kelly schedules or invented scanner
configurations. Invalid supplied evidence is a failure; it is not treated as absent.

## Model fitting is a separate choice

An ordinary refresh never trains a model. A request containing `models` without
`--fit-models` records `model_build_not_requested`. To explicitly build offline
models, include `models` in the job stages and add `--fit-models` to the first run:

```bash
poetry run python scripts/run_weekly.py --config configs/weekly_research.yaml \
  --snapshot /private/path/verified-snapshot --as-of 2026-11-24T12:00:00-06:00 \
  --research-job /private/path/model-job.yaml --fit-models
```

Model policy, scope and scikit-learn/SHAP runtime versions are frozen independently.
Existing support/resource gates may decline sparse populations. Models remain
exploratory equal-wager associations with their existing event-grouped split and
training background. No new forward validation or causal claim is added. Refresh
must not silently reuse a latest/older model: an earlier frozen publication stays
available through its explicit existing dashboard attachment. This PR does not
refresh that attachment or apply display-selected R07 cohorts as model-fit inputs.

## Resume, failure and rollback

Use the returned run ID, the same configuration/code/runtime and no replacement
inputs or model flag:

```bash
poetry run python scripts/run_weekly.py --config configs/weekly_research.yaml \
  --resume RUN_ID
```

Resume reads the saved plan, not a mutable job/evidence file. Original request-file
edits cannot silently change the resumed work. A changed policy/code revision or
model runtime rejects resume; start a new explicitly reviewed job instead. Sizing
predecessor publication hashes/bytes and source snapshot are reverified. Optional
checkpoints additionally pin their dependency manifests. Reused artifacts pass their
existing readers, and final-result loading verifies receipt inventory, dependency
lineage, job payload and linked optional artifact bytes.

Transient I/O retries use the existing bounded core policy. A failed builder writes
sanitized attempt-stage failure records; no new final result/report/job manifest is
published. Earlier completed artifacts remain private reusable checkpoints, not a
partially activated dashboard. Correct an interrupted transient failure and resume;
for corruption, restore independently verified bytes or create a new output root
and reviewed run. Never edit receipts/checksums to force a replay past verification.

The final `manifest.json`, `report.md` and `job.json` appear together through the
existing atomic directory rename. The last verified report remains usable. No
latest pointer is changed. Roll back by keeping/reselecting an earlier explicitly
verified report or omitting `--research-job` for the core-only workflow; retain the
immutable audit history. R09/R10 will provide import preview/upload and explicit
activation separately. No owner dashboard review is needed for this CLI/API release.
