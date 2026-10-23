# Prospective experiments and outcome tracking

The experiment registry records paper research, not external trading operations.
It cannot create or activate OddsJam filters, fund an experiment, place wagers,
promote a strategy, or schedule exports. Existing notebooks and defaults remain
unchanged. Actual protocols, evidence, registers and reports belong in ignored
`data/processed/` paths; the checked-in example is synthetic only.

## Signal potential and execution potential

Two experiment types are supported:

- `signal_screening`: broad scanner observations may remain `not_checked`.
  Hypothetical paper outcomes can identify hypotheses about books, markets, prices
  or advertised EV, without claiming the quote was obtainable.
- `execution_verification`: a declared sampling rule and selection references are
  required. Unavailable and inconclusive checks are retained alongside successes.

Funding (`paper`, assumed `real_money`, or `unknown`) is separate from availability.
A sportsbook check is not an accepted wager. A changed price is not confirmation
of the original scanner quote. No blanket availability haircut is applied to ROI;
which observations are unavailable can matter systematically.

## Historical funding policy

The user clarified that through October 22, 2026 inclusive, only the exact label
`PROD - Exchanges: Core Exchange Conservative (Low Risk, Slow Build)` identifies
assumed real-money trades. Other **named** historical filters are paper, regardless
of their PROD/TEST spelling. Empty attribution, mixed Core/other attribution, and
unparseable dates remain unknown. Current names do not establish past funding.
Core Exchange results describe selectively chosen trades influenced by liquidity
and other judgment; they do not represent all opportunities surfaced by that filter.

Future designations remain user-assumed metadata, not independently verified fills.
The updated assumption is versioned in ingestion policy. New ingestion snapshots
carry it; existing snapshots, archives and historical notebook datasets are never
rewritten. Experiment reports preserve the pinned source mode separately from the
current assumption, including the policy/version in report provenance. Regenerating
older canonical snapshots is an explicit separate operation, not automatic here.

The user also confirmed on October 22, 2026 that all exchange filters use the
positive-EV tool only to take existing orders, never to post resting limit orders.
This taker-only workflow is recorded as an assumption in versioned policy; it is
not evidence that a quote was available or an order filled. Verification should
record the offered price and available quantity at the time of the check. No maker
fills, queue priority, or estimated taker fees are inferred; reported profit is
preserved and fee inclusion remains unverified.

## Registration is not launch

Registration requires a verified `CREATE_TEST` proposal and its latest exact
`APPROVED` decision, with `draft_paper_protocol_only` scope. Research-review approvals
cannot register experiments. Superseded approvals cannot launch/resume an experiment
or enroll new observations. Settlement checks and audit history remain recordable.
No approvals or real registrations are fabricated from existing October history.

The operational protocol preserves every approved draft field and adds:

- Experiment name and type, exact predicate, filter identity/configuration.
- Timezone-aware planned start/end matching the approved duration.
- Explicit comparison and overlap plans, including an explicit no-control plan.
- Expected liquidity assessment; unknown is permitted for paper screening.
- Stop/pause rules, review criteria and settlement follow-up duration.
- A sampling rule for execution-verification experiments.

Only paper funding with zero funded risk is accepted. Protocol predicate and
reviewed UI control values must match the approval; an explicitly selected registry
target must also match. Additional settings are pinned, not inferred. Material
changes require a newly approved proposal and new registration. Protocols are
immutable; configuration deviations are separate events.

Copy `configs/experiments/example.yaml` to a private path and replace all synthetic
values with the complete approved protocol. Its numbers are illustrations, not
research defaults. Registration must occur before planned enrollment. No start time
or group target is inferred from historical outcomes.

## CLI workflow

Global `--register` selects the private experiment register and precedes a subcommand.

```bash
poetry run python scripts/manage_experiments.py \
  --register data/processed/experiments register \
  --bundle data/processed/recommendations/BUNDLE_ID \
  --decisions data/processed/decisions \
  --proposal-id PROPOSAL_ID \
  --approval-event-id APPROVAL_EVENT_ID \
  --protocol data/processed/private_protocol.yaml \
  --actor "YOUR_NAME" --rationale "Register the approved paper protocol"
```

The experiment ID and event digest are returned. Subsequent commands require the
latest **per-experiment** digest, preventing stale writes:

```bash
poetry run python scripts/manage_experiments.py \
  --register data/processed/experiments event \
  --experiment-id EXPERIMENT_ID --kind TRANSITION \
  --payload data/processed/launch.json \
  --expected-event-id PREVIOUS_EVENT_ID \
  --actor "YOUR_NAME" --rationale "Confirm the externally launched paper filter"
```

Launch payload fields are `to_status: RUNNING`, `launched_at_utc`, `filter_id`,
`filter_config`, and `evidence_reference`. Configuration must match registration.
Launch time must follow registration and be within the declared window; it cannot
be in the future relative to recording time. Resume also requires configuration
evidence. The command records a human attestation; it does not activate anything.

Inspect with `poetry run python scripts/manage_experiments.py inspect`.
No actor is inferred from OS identity. Recording times come from the system clock,
not a user-supplied backdating option. Evidence references are not independently
authenticated by this offline program. Never include credentials or raw requests.

## Lifecycle

- `REGISTERED` → `RUNNING` or `CANCELLED`.
- `RUNNING` → `PAUSED` or `AWAITING_OUTCOMES`.
- `PAUSED` → `RUNNING` or `AWAITING_OUTCOMES`.
- `AWAITING_OUTCOMES` → `READY_FOR_REVIEW`.
- `READY_FOR_REVIEW` → `CLOSED`.

A `DEVIATION` records details and evidence and pauses a running experiment. Enrollment
is blocked while paused or after planned end. A fixed sample target does not stop
enrollment automatically. Early enrollment closure is audited and prevents a
normal readiness result. Existing observations continue through outcome tracking.

To enter `READY_FOR_REVIEW`, provide `review_assessment_reference` pointing to a
verified pinned experiment report including the latest event for this experiment.
Complete observations, duration and sample requirements permit human readiness
review; otherwise the declared follow-up must elapse before inconclusive review.
Future/stale reports are rejected. This is not a statistical validation certificate.
Closure requires one of `continue_new_protocol`, `modify_new_protocol`, `stop`, or
`inconclusive`. No promotion or live filter change is performed.

## Explicit enrollment and checks

Use `--kind ENROLL` with payload:

- Unique `enrollment_id` and experiment-local `observation_key`.
- `observed_at_utc` and `event_start_utc`, each explicitly aware or null if unknown.
- `feature_values` matching the exact approved predicate.
- `scanner_quote` containing `selection`, `line` (or an explicit not-applicable
  label), and valid American `odds` captured at decision time.
- `evidence_reference` for the observation.

Enrollment recorded after launch and before event start, with observation before
recording and after launch, is labelled prospective under those attestations.
After-event or pre-launch observations remain retrospective. Missing timing remains
`timing_unverified`. Neither retrospective nor timing-unverified observations enter
prospective paper ROI. A future observation timestamp is rejected.

Checks use `--kind CHECK`, linking an enrollment to `status` (`checked_available`,
`checked_unavailable`, `inconclusive`), `checked_at_utc`, `reason`, and
`evidence_reference`. Verification experiments also require
`sample_selection_reference` for **failed checks too**. Availability success requires
observed selection, line, finite American odds and positive available stake limit.
Delays are recorded where scanner observation time is known. Every attempt remains
in history; summaries use the latest recorded check, not an unbiased first-attempt
availability estimate. Checks after event start or with unknown timing cannot be
represented as timely quote confirmation.

The original selection/line/price is compared with the checked quote. A difference
produces `checked_available_changed_quote`, not original quote confirmation. No
settlement economics are recomputed at the different checked price.

## Source linkage and snapshot outcomes

`--kind LINK` requires `enrollment_id`, exact `source_record_id`, and
`evidence_reference`. Identity links are explicit human assertions; they are not
inferred from filter names, nearby times, similar selections or outcomes. Missing
or unresolved current source identities remain in the audit. Later link corrections
are appended and do not alter previously pinned reports. A source identity cannot
be assigned to two enrollments in one experiment. Multiple leg records mapping to
the same canonical ticket cannot multiply eligible economics.

Build an explicit as-of report:

```bash
poetry run python scripts/build_experiment_report.py \
  --register data/processed/experiments \
  --canonical data/processed/canonical/CANONICAL_ID \
  --as-of EXPLICIT_ISO_TIMESTAMP_WITH_OFFSET
```

Source imports after the recording cutoff are rejected. Register events are selected
by when they were recorded, not by a retroactively supplied launch date. Reports
pin the entire selected event history, source manifest hash, canonical ID, execution
assumption policy, transformation hashes and library versions. Published reports
are immutable and hash-verified; replay checks logical table equality. New settlement
observations or link events produce new reports without changing prior ones.

Tables: protocols, events, enrollments, links, checks, availability counts, outcomes,
metrics and readiness audits. Funding cohorts and availability statuses remain
separate. Counts include pending, excluded, retrospective and unresolved observations;
ROI uses only the identical eligible prospective-paper stake/profit population.
No observations produce empty tables, not fabricated experiments or zero-return
performance. Empty eligible cohorts have null economics.

Source selection and odds conflicting with enrollment, canonical economic exclusions,
mode conflicts and duplicate canonical wagers cannot enter prospective paper ROI.
Core/other historical funding follows the explicit user assumption, not current UI
names. Funded and unknown source conflicts remain visible with zero authorized
funded execution; they are never relabelled paper.

Checked availability is not verified funding, fillability or net profitability.
Unchecked paper returns remain hypothetical. Metrics grouped by availability may be
selection-biased; no representativeness claim or availability haircut is made.
Liquidity, fees, actual acceptance and cross-event dependence remain unverified.
Unsupported parlay/Pick'em products, pending outcomes and refunds/cancellations retain
canonical exclusions. Event-group counts are dependence proxies, not independent
sample-size proof. Reports never label a strategy promoted or statistically validated.

## Validation and limits

The register uses hash-chained events, explicit actors/rationales, exclusive atomic
publication and single-writer locks. This is local auditability, not authentication
or tamper-proof storage. Tail deletion needs an external anchor to detect. A lock
must be investigated before manual recovery.

Synthetic tests cover approvals, protocol changes, launch/timing evidence, unchecked
signals, changed quotes, failed checks, sampling references, overlap/duplicates,
mode conflicts, pending outcomes, cutoff isolation, settlement views, corruption,
locks, atomic failures and report replay. No network or private credentials are
required. Real experiments require genuine approvals and evidence; existing October
results do not supply them. No weekly scheduling or notebook changes are included.

Local validation on October 22, 2026 used canonical bundle
`df99ad86af5f1c7b09b7fb93fdab8c039306217239fe528242cf39b30f886dc7`
and an empty experiment register. At knowledge cutoff
`2026-10-22T23:22:51.541141+00:00`, report
`e120237fe84b5b8e6a8975405c5e40027e72a2e5800d4557801e5f5f56e8c30f`
replayed identically with all nine tables empty. This verifies compatibility with
the pinned October source without fabricating registrations, enrollments or
prospective outcomes. The report remains in ignored local data storage.
