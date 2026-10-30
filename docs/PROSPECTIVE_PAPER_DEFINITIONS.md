# Prospective paper-test definitions

Freeze a reviewable plan before collecting new paper observations. Definitions pin
exact source rules and preserve revisions; they do not approve or launch experiments,
place bets, infer fills, or evaluate future outcomes. A candidate can be explicitly
chosen for exploratory paper study even when the historical corrected shortlist is
empty. That decision never changes its historical category or p-values.

## Review and draft

1. Open **Within-filter evidence**, choose a candidate, and inspect both return
   measures, parent/subgroup/complement, uncertainty, concentration and overlap.
2. Select **Download paper-test definition draft**. This downloads JSON without
   writing to the register. JSON is accepted by the YAML-reading CLI.
3. Complete the `protocol` fields. Leave `candidate_id` and `exact_predicate_json`
   unchanged. Exact source rules and parent identity come from verified publications;
   source rules cannot be edited through a prospective definition.

The equivalent command prints an incomplete YAML draft:

```bash
poetry run python scripts/manage_paper_definitions.py draft \
  --discovery data/processed/within_filter_discovery/<publication-id> \
  --candidate-id '<candidate-id>' > data/processed/my-paper-definition.yaml
```

## Required choices

| Field | Meaning |
| --- | --- |
| `name`, `hypothesis` | The question being studied, without a deployment claim. |
| `design` | `subgroup_tracking` or `exclusion_comparison`. |
| `baseline` | `unchanged_parent`, explicit prospective `filter_config`, and contemporaneous `evidence_reference`. Historical settings are not reconstructed. |
| `planned_start_utc`, `planned_end_utc` | Aware timestamps defining the prospective enrollment window. Save before the start; end must follow start. |
| `review_dates_utc` | Ordered unique aware review times. No review predates start; final review includes the declared settlement follow-up. |
| `sizing` | `fixed_paper_stake`, positive finite `stake_usd`, `applies_to: both_arms`. This initial workflow keeps paper exposure comparable; fractional Kelly replay is unsupported. |
| `min_wagers_per_arm`, `min_event_groups_per_arm` | Explicit minimum support for the planned review. Draft values of 100 are editable conventions, not power or promotion guarantees. |
| `primary_metric`, `secondary_metric` | Predeclare both `equal_stake_mean_return` and `stake_weighted_roi`, choosing which is primary. |
| `review_criteria` | Explicit `continue`, `stop`, and `inconclusive` criteria. Describe decisions before seeing future outcomes. |
| `stop_rules` | Predeclared conditions for stopping or pausing capture, such as rule/configuration mismatch. |
| `overlap_policy` | How shared opportunities/events and overlap with other tests will be recorded and reviewed. |
| `quote_availability_policy` | Capture decision-time UTC, event/selection identity, platform, line, odds, failed/unknown availability and supporting evidence. Never turn an unchecked quote into a fill. |
| `settlement_followup_days` | Nonnegative time reserved after enrollment ends for outcome follow-up. |
| `funding`, `max_funded_stake_usd` | Must be `paper_only` and integer `0`. |

`subgroup_tracking` means follow the exact candidate inside the unchanged parent.
`exclusion_comparison` means compare parent-minus-subgroup with the unchanged parent.
These arms overlap and are not independent samples. The baseline configuration is
new prospective evidence, not proof that the same configuration existed historically.
No current scanner/UI translation is automatically verified or applied.

## Save an immutable version

```bash
poetry run python scripts/manage_paper_definitions.py \
  --register data/processed/paper_definition_register save \
  --discovery data/processed/within_filter_discovery/<publication-id> \
  --prioritization data/processed/signal_prioritization/<publication-id> \
  --definition data/processed/my-paper-definition.yaml \
  --test-key platform-market-paper-study \
  --actor '<your name>' \
  --rationale '<why this exploratory test is worth prospective study>'
```

The CLI prints the stable test key, version, version ID and explicit non-launch
status. It uses the system recording clock; there is no backdating option. The
source publications must have exact matching discovery and diagnostic lineage.
Support-blocked or unconfirmed historical candidates may be saved intentionally;
registration does not manufacture statistical approval.

To revise, edit the human-intent fields and repeat `save` with
`--expected-version-id '<current-version-id>'`. A changed candidate requires a new
candidate-bound draft. Every material revision creates a new version with a future
start and explicit rationale; stale writers, duplicate/no-op definitions and in-place
rewrites are rejected. Prior versions remain recorded, including their original
start/sizing/review choices. Do not pool observations across versions. Revising the
plan does not silently stop an external test or change an existing experiment.

## Publish a frozen dashboard copy

```bash
poetry run python scripts/manage_paper_definitions.py \
  --register data/processed/paper_definition_register publish \
  --as-of '<explicit-aware-knowledge-cutoff>'
```

Choose a cutoff no later than the current system time. The command publishes the
known event prefix to `data/processed/paper_definitions/<publication-id>/`, preserving
all previous publications. Empty registers publish an explicit empty state. Replay
verifies content; the live log is checked before applying the knowledge cutoff.
Hashes cover events and implementation. As with the existing experiment log, an
external head anchor is needed to detect tail deletion.

Set this optional field in your trusted dashboard YAML (relative to `evidence_root`):

```yaml
paper_definitions_publication: paper_definitions/<publication-id>
```

Open **Paper-test definitions**. It shows matching source versions, prospective rules,
baseline, sizing, review dates, criteria and status. It preserves superseded versions,
including when the replacement references another report. The dashboard only reads
the published copy, rechecks integrity on every rerun and never reads/writes the live
register. Missing snapshots and report mismatches have explicit unavailable states.

## What happens next

Choose a small number of distinct hypotheses and complete their definitions. No
real tests are preselected or started by this change. Downloading, saving, or the
passage of planned dates does not establish launch, enrollment, availability, outcomes
or completed review. The status remains `definition_only_not_launched` and
`no_prospective_observations_linked` until a separate operational workflow supplies
evidence; it is not inferred from historical paper exports.

The existing `manage_experiments.py` approval/launch/enrollment/link/outcome workflow
is unchanged. Definition IDs do not bypass its proposal approval requirements, and
this change does not automatically translate multidimensional definitions into that
legacy proposal contract. Operational linkage and evaluation of these new versions
must explicitly preserve version identity, decision-time capture, shared events and
review rules. No prospective performance, holdout pass, significance or production
promotion is claimed here.

Registers, completed definitions, baseline settings and generated publications belong
under ignored `data/processed`, not in Git. No acquisition, scheduling, notebook
execution or live scanner changes are introduced.
