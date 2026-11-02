# Troubleshooting and evidence remediation

Start with the exact selected run, publication IDs and configuration. Preserve
failing artifacts and receipts. A broader or newer-looking result is not a safe
fallback for a failed or unavailable attachment.

## Diagnose the symptom

| Symptom | Check first | Resolution |
| --- | --- | --- |
| No report catalogue / no published results | `artifact_root`, working directory, intended config/environment and completed core run | Publish or point to the exact intended local report. A clone contains no private data. |
| Optional evidence unavailable | Run-keyed attachment, `evidence_root`, matching source manifests and required upstream attachments | Build/attach that exact component. Missing optional data does not authorize guessing another report. |
| Integrity, corruption or source-lineage error | Byte hashes, confined paths, source manifests, selected run and same-source publication chain | Stop using the affected output. Preserve evidence; restore verified bytes or explicitly rebuild a new compatible publication from intact inputs. Never edit hashes to make it load. |
| Stale cache after file changes | Artifact verification error and cleared component state | Repair/republish the source chain, then explicitly reopen/restart. Cached metrics are not a substitute for verified bytes. |
| Requested intersection has no result | Declared views and exact published predicates/cohorts | Clear selectors or explicitly build the required supported intersection under a reviewed search contract. Separate one-dimensional summaries cannot be intersected to manufacture evidence. |
| Complete sheet shows unknown or setup blocked | Baseline ID/hash, each control's state and exact UI translation | Obtain source/configuration evidence. An omitted field is not unset/default; a current screenshot does not establish old settings. |
| Inherited setting is unexpected | Pinned baseline and effective/knowledge times | Review the actual baseline version. Create a new evidence/definition version when warranted; do not overwrite the earlier record. |
| A strict EV cutoff cannot be configured exactly | Predicate operator/endpoints/units versus verified scanner behavior | Preserve the exact hypothesis. Record a supported translation or keep it blocked; `>3%` and `>=3%` are not interchangeable. |
| Empty corrected/actionable queue | Both-arm support, event IDs, concentration, both return measures, correction family and setup/lifecycle | Empty is a valid result. Collect/inspect evidence; do not lower research gates after seeing outcomes. Display thresholds only hide/show existing cohorts. |
| Historical loss is only an investigation | Confirmed effective-dated PROD status and pinned unchanged baseline | Supply actual lifecycle/configuration evidence before treating it as a production-change proposal. A `PROD` name or funded trade is insufficient. |
| Models declined / explained rows missing | Population support, eligible target, event IDs, resource caps and publication audit | Keep declined populations visible. Add supported observations or an explicit new model publication; do not relax policy or train on a dashboard click. |
| Filtered explanations do not resemble a separate fit | Selected publication, model/background ID, fixed-model row scope | This is expected for fixed-model filtering. A subset refit must be explicitly built offline and selected as a separate publication. |
| Liquidity / timing / closing-price chart blocked | Readiness reason, units, observation timing, timezone/DST and source semantics | Obtain the specific evidence listed below. Never replace unavailable series with zero or infer capacity from a raw string. |
| Pipeline returns a stale/unknown/repeated-input notice | Import novelty, metadata, selected mode and cutoff | Use truthful metadata and explicit historical mode where appropriate; a changed policy or replay is not new source history. |
| Definition save/version/head rejected | Exact candidate source, configuration reference, expected version/head and material change | Inspect the current immutable register; reconcile concurrent edits before a new explicit action. Never invent a predecessor or erase the tail. |
| Enrollment rejected | Actual RUNNING state, event/start timestamps, decision-time feature availability and parent configuration hash | Confirm the real state/evidence. Historical exports cannot be enrolled retrospectively; never fake launch or timing to pass validation. |
| Review decision/approval becomes stale | New settlements, amendments, completed review and exact report/head | Publish current outcomes and record the required explicit review amendment/new decision. Do not reuse an obsolete approval. |
| Approval exists but applied status did not change | Separate CONFIRM_APPLICATION evidence and exact approved settings | Expected behavior. Confirm actual external application only after it happened, with its effective date. |
| Late settlement arrives after closure/change | Original observation and definition/application identity | Record the late fact/amendment under its original version. Reassess stale reviews; do not move it into a newer version. |
| Tutorial fails mid-step | Stderr, `commands.jsonl`, payloads and generated journal prefix | Preserve the failed workspace. Start a fresh `mktemp -d` directory; the example is not a resume tool. |
| Tutorial refuses a directory or acknowledgement | Empty temporary root, synthetic marker, state order, explicit flag | Use a new temporary directory and the documented order. Do not redirect the helper at real data or modify state to skip validation. |

## Missing evidence versus software defects

| Kind | Example | Appropriate next action |
| --- | --- | --- |
| **Missing source evidence** | Export timezone, closing-reference semantics or decision-time liquidity not established | Obtain provenance from the source/operator; keep unknowns explicit. Existing readiness gates may then support a new publication. |
| **Missing configuration/application evidence** | Unknown devig control, no pinned parent, no confirmed effective date | Record reviewed per-control/baseline/lifecycle evidence. Do not infer it from names or current settings. |
| **Insufficient prospective evidence** | Too few events, failed quote checks, pending results or protocol deviation | Continue only under an approved future protocol as appropriate, settle/review, or record inconclusive/stop. Do not alter past rules. |
| **Unsupported contract or feature** | A newly verified source requires an unimplemented field/translation | Scope an adapter/contract PR with units, provenance, compatibility and synthetic tests. |
| **Software defect** | Valid matching inputs fail a documented invariant, or synthetic reproduction demonstrates wrong reconciliation | Preserve the failing IDs/versions; make a minimal synthetic reproduction and a focused fix/test PR. No private raw records are needed in the issue. |
| **Operational configuration error** | Wrong artifact root or mismatched run attachment | Correct the local locator/configuration to verified matching publications. Avoid changing research contracts to mask a path error. |

A blocker is not automatically a bug. Conversely, a missing renderer, drill-through or
chronological evaluator is not “missing evidence” if the required implementation does
not exist. Use the [implementation ledger](plans/MULTIDIMENSIONAL_IMPLEMENTATION_STATUS.md)
to distinguish partial engineering from source-gated capabilities.

## Evidence checklist by capability

- **Scanner setup:** exact control value/state, baseline identity, observation/effective
  time, provenance and verified analytical-to-scanner translation. Do not treat every
  unspecified control as unrestricted.
- **Timing:** field meanings, placement/event/settlement distinction, timezone/offset,
  DST handling and information availability. Source-calendar attribution is weaker
  than verified placement or settlement order.
- **Liquidity:** unit/currency, measurement time, offer/market identity and meaning of
  available size for the evaluated stake. Historical display strings are not fills.
- **Closing-price diagnostics:** fair-reference odds, reference source, timestamp,
  odds format and supported transformation, with matched-cohort coverage.
- **Prospective review:** predeclared exact family/rules, approval, actual launch,
  parent/complement observations, available/unavailable quote checks, settlement and
  deviations, exact reviewed report and explicit scheduled review evidence.
- **Applied monitoring/rollback:** approved change, confirmed settings/date, previous
  and current application IDs, normalized configuration hash, version-linked outcomes
  and separate human follow-up. Retained rollback evidence is not rollback approval.

## Before reporting a problem

Record code revision, dependency lock, exact config/publication IDs, error category,
expected behavior and a synthetic reproduction if possible. Omit credentials, raw
exports, account screenshots, private bankroll values and register contents from
public reports. Review [release checks](RELEASE_ACCEPTANCE.md) and the relevant
[component map](CONFIGURATION_PUBLICATION_MAP.md) before rebuilding anything.

[Walkthrough home](walkthroughs/README.md) · [Documentation home](README.md)
