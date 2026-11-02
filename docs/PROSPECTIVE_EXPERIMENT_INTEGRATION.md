# Definition-bound prospective experiments

A frozen paper definition now connects to the existing human decision and experiment
registers. This is an explicit local workflow: publication does not approve a plan,
approval does not register or launch it, and elapsed dates do not create observations
or complete reviews. No live OddsJam settings, wagers, exports or jobs are changed.

## 1. Freeze, propose and review one version

Complete and publish a definition using [the definition workflow](PROSPECTIVE_PAPER_DEFINITIONS.md).
Set future start/end times, review dates, both return measures, fixed paper stake,
minimum support in each arm, overlap/availability policies and stopping criteria.
Preserve its exact version ID. The baseline configuration is taken from that version,
not from current screenshots or a historical label. Missing scanner provenance stays
unknown; an older untyped baseline remains explicitly operator-attested, not a verified
complete scanner setup.

Build a proposal for that exact version:

```bash
poetry run python scripts/build_definition_proposal.py \
  --definitions data/processed/paper_definition_publications/DEFINITION_PUBLICATION_ID \
  --version-id DEFINITION_VERSION_ID \
  --filter-id 'TEST - Your prospective scanner name' \
  --sampling-rule 'Capture every distinct unchanged-parent observation, including nonmatching candidate observations; retain failed and unknown quote checks'
```

Choose the sampling rule before capture. The tool does not ensure that a manual
operator observes every scanner opportunity. The resulting bundle's `proposals`
table contains the exact `proposal_id` and operational `prospective_protocol_json`.
The bundle embeds the entire definition, its publication ID and manifest hash, the
baseline, predicate, paper sizing and review rules. Changes need a new definition
version and a new approval. A superseded version in the selected snapshot cannot
be proposed. Supersession outside that snapshot is not inferred; explicitly publish
the updated snapshot and supersede the old approval when withdrawing a plan.

Review with the existing commands:

```bash
poetry run python scripts/record_research_decision.py \
  --register data/processed/decisions record \
  --bundle data/processed/definition_proposals/BUNDLE_ID \
  --proposal-id PROPOSAL_ID --to-status UNDER_REVIEW --expected-status PROPOSED \
  --reviewer 'Your name' --rationale 'Reviewed exact baseline, capture policy and paper protocol'

poetry run python scripts/record_research_decision.py \
  --register data/processed/decisions record \
  --bundle data/processed/definition_proposals/BUNDLE_ID \
  --proposal-id PROPOSAL_ID --to-status APPROVED --expected-status UNDER_REVIEW \
  --expected-event-id REVIEW_EVENT_ID \
  --reviewer 'Your name' --rationale 'Approve this exact version for paper tracking only'
```

Reviewer identities and supporting references are local attestations, not authenticated
permissions or tamper-proof evidence. Keep all private registers and payloads under
ignored `data/processed/` paths.

## 2. Register, then explicitly confirm actual launch

```bash
poetry run python scripts/manage_experiments.py \
  --register data/processed/experiments register \
  --bundle data/processed/definition_proposals/BUNDLE_ID \
  --decisions data/processed/decisions --proposal-id PROPOSAL_ID \
  --approval-event-id APPROVAL_EVENT_ID \
  --actor 'Your name' --rationale 'Register this approved future paper protocol'
```

For definition-bound bundles, omit `--protocol` to use the entire pinned protocol.
Legacy proposals still require that argument. Only the current exact APPROVED
paper proposal can register. One definition version can register once per register;
new experiments require new versions. Planned registration is not actual launch.

All subsequent commands use this pattern, replacing `KIND`, the payload file and
`LATEST_EXPERIMENT_EVENT_ID` each time:

```bash
poetry run python scripts/manage_experiments.py \
  --register data/processed/experiments event \
  --experiment-id EXPERIMENT_ID --kind KIND \
  --payload data/processed/payloads/event.json \
  --expected-event-id LATEST_EXPERIMENT_EVENT_ID \
  --actor 'Your name' --rationale 'Explain the observed action or evidence'
```

For `TRANSITION` to `RUNNING`, explicitly provide `launched_at_utc`, `filter_id`,
`filter_config` exactly equal to the pinned baseline, and `evidence_reference`.
Launch/resume confirmation must be recorded inside the planned enrollment window.
The local clock records knowledge time; it cannot be backdated through the CLI.

An enrollment's observation time must be **after the recorded launch/resume
confirmation**, and recording must precede event start and planned end. Backfilled
history and retrospective enrollment are rejected. `PAUSED` blocks captures;
`DEVIATION` records evidence and pauses a running experiment. A material rule or
baseline change requires a new approved definition, rather than editing the old plan.
Superseding approval blocks further captures and launches, while retaining outcomes
and audit evidence for observations already captured.

## 3. Capture parent observations, classify typed subgroup/complement

Use `ENROLL` with:

| Field | Meaning |
| --- | --- |
| `enrollment_id`, `observation_key` | Stable local enrollment and source observation identities; do not rename repeated exports/legs into new observations |
| `event_namespace`, `event_id` | Explicit source namespace and event identity; same ID in different namespaces is different |
| `observed_at_utc`, `event_start_utc` | Explicit aware decision and scheduled start timestamps |
| `feature_values` | Original decision-time scalar values for every candidate term; EV uses percentage points (`3` means 3%) |
| `feature_available_at_utc` | Per-feature aware knowledge time, no later than observation time |
| `scanner_quote` | `selection`, `line` (`not_applicable` where appropriate), and finite American `odds` |
| `baseline_config_sha256` | Repository `fingerprint(protocol['filter_config'])` of the exact approved baseline |
| `parent_membership_reference` | Contemporaneous evidence that this observation matched the unchanged parent |
| `evidence_reference` | Source capture reference, without credentials |

Optional `definition_version_id` must match; the writer attaches it when omitted.
AND predicates are evaluated from the frozen terms, with decimal numeric endpoints:
`>3` excludes exactly `3`, rather than rounding or using an EV band. Missing features
are rejected instead of being silently assigned to the complement. Unrestricted
parent settings are not reconstructed from a candidate's two-field predicate.

Every captured parent observation is assigned to subgroup or complement. Duplicate
observation keys and duplicate decision-time quote identities are rejected across
versions of the same test key **within the selected register**. Cross-register
identity checks are not possible: use one register for the research program.
Different tests may share an event; reports disclose that dependence. Event IDs and
parent membership are attestations and must be checked against their source.

## 4. Availability, source identity and settlement

`CHECK` retains `checked_available`, `checked_unavailable`, or `inconclusive`, with
`checked_at_utc`, reason and evidence reference. Available checks additionally need
selection, line, odds and positive `available_stake_usd`; execution-verification
checks also need `sample_selection_reference`. Changed quotes, insufficient stake
capacity, unchecked observations and late-recorded checks never qualify as original
quote availability. All remain in the hypothetical cohort. An availability check
is not a fill, acceptance or realized funded return.

`LINK` associates an enrolled observation with `source_record_id` and evidence
reference. Duplicate source identities within an experiment are rejected. For these
new protocols, this link is an operator attestation, **not independent canonical
identity verification** and not automatic outcome import. Historical exports cannot
create enrollments or inject their stakes/profits into the new paper metrics.

Use `SETTLE` to record an explicit operator-attested paper result:

```json
{
  "enrollment_id": "your-stable-enrollment-id",
  "result": "won",
  "outcome_available_at_utc": "2030-01-03T23:00:00Z",
  "evidence_reference": "Reference to the published selection-specific outcome"
}
```

The date above is illustrative; provide actual aware evidence times. Results are
`won`, `lost`, `push`, `void`, or `pending`. Outcome availability must follow recorded
event start and not exceed recording time. Win return uses the original scanner
odds, loss is −1 unit, push is 0, void is excluded, and pending remains missing.
Paper profit uses only the definition's predeclared fixed stake. Fees are not
estimated; these are hypothetical quote-price returns, not verified net economics.

Later results/corrections are additional `SETTLE` events with `supersedes_event_id`
and `correction_reason`. Earlier events and reports remain intact. No result is
inferred from the game having ended, a link existing or a follow-up period elapsing.

## 5. End capture, publish an as-of report and record scheduled review

Explicitly transition to `AWAITING_OUTCOMES` with `ended_at_utc` and evidence reference.
The end cannot precede a captured observation. Early ending remains a blocker; the
planned end alone never changes lifecycle state.

```bash
poetry run python scripts/build_experiment_report.py \
  --register data/processed/experiments \
  --canonical data/processed/canonical/EXPLICIT_CANONICAL_BUNDLE_ID \
  --as-of 'ACTUAL_AWARE_KNOWLEDGE_CUTOFF' \
  --output-root data/processed/experiment_reports
```

The canonical snapshot is verified and pinned independently. Its import knowledge
must precede the cutoff; it does not retroactively enroll observations or settle
new protocols. Old experiment reports and legacy linked-outcome behavior remain
supported. Reports with bound definitions add:

- `prospective_observations`: exact experiment/version, subgroup/complement,
  availability, settlement event, source-link status and shared-event exposure.
- `prospective_metrics`: separate parent/subgroup/complement rows in hypothetical
  and original-quote-checked cohorts, support, pending/void counts, fixed paper
  profit, weighted ROI and equal-stake mean return. Returns are fractions, not
  percentage points. With identical fixed stakes, the two measures coincide.
- `prospective_reviews`: each predeclared review date, lifecycle, due/recorded flags,
  support/availability/deviation/pending blockers and follow-up status.

Parent overlaps subgroup/complement and must not be added to them. No results are
pooled across versions. Settled support counts exclude voids and pending outcomes;
missing quote checks remain visible. Human-review readiness is not a significance
test, proof of edge, causal comparison or promotion recommendation. Exploratory
historical corrected results are not changed or repurposed as prospective proof.

Record `REVIEW` only when its exact `scheduled_review_utc` is due. Include
`review_assessment_reference` pointing to a current immutable report, plus notes and
evidence reference. The report must end at the current experiment event head and
contain the same definition version and scheduled date. A review can explicitly
record inconclusive evidence; it does not claim support or profitability. Subsequent
reports show the review as performed. A stale report cannot support another event.

`READY_FOR_REVIEW` similarly requires current report evidence and either complete
readiness or elapsed follow-up (allowing an inconclusive human assessment). Human
continue/modify/stop decisions, production changes, promotion and monitoring remain
in the next workstream; this bridge does not close or promote definition-bound tests.
Scheduled dates are report fields, not an installed scheduler or notification service.

## Dashboard attachment

Set explicit immutable paths relative to `evidence_root` in the trusted dashboard
configuration:

```yaml
paper_definitions_publication: paper_definition_publications/DEFINITION_PUBLICATION_ID
prospective_report_publication: experiment_reports/REPORT_ID
```

The Paper-test definitions page displays lifecycle, per-version arm/cohort economics,
returns chart, availability, pending outcomes and scheduled reviews. The decision
workspace shows follow-up status for the exact candidate. Unattached operational
evidence is unknown, not proof that an experiment never launched. Tables are
filtered to definitions verified against the selected historical report. A newer
prospective canonical snapshot is permitted; similar filter names are never used to
join versions. Files are confined to the trusted root and reverified on each load;
missing/corrupt attachments are cleared without consulting a live register.

## Validation and boundaries

Synthetic tests exercise the complete cycle, decimal thresholds, unsupported
features, changed baselines, stale approvals/reports, duplicate/cross-version
captures, pauses, late checks, pending/corrected outcomes, source links, overlapping
events and immutable cutoff behavior. Existing historical and legacy experiment
tests remain applicable. No actual operator approval, launch, enrollment or settlement
was created to implement this feature. For a live study, freeze the plan, review its
full setup and record launch before capturing the first observation.
