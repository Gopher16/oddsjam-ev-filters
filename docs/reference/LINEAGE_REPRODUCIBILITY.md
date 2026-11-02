# Lineage, replay and verified loading

## Trace a displayed value

Follow dashboard attachment → exact publication manifest/table → canonical wager
IDs or membership digest → canonical bundle → ingestion snapshot → source observation
and archived import bytes. Also retain policy/configuration, code/runtime versions,
knowledge/effective times, eligibility, units, weighting and cohort fingerprint.
A similar filter name or a recent file modification time is not a lineage join.
The [configuration/publication map](../CONFIGURATION_PUBLICATION_MAP.md) lists the
actual CLI inputs/outputs and dashboard attachment keys.

| Layer | Identity/evidence retained |
| --- | --- |
| Ingestion | Original byte hash, declared namespace/metadata, accepted imports, reconciliation choices, snapshot as-of and source rows. Repeated receipts are not new source data. |
| Canonical and atlas | Source manifests, policy/registry, code/runtime versions, candidate and membership IDs, output hashes/counts. Registry date does not backdate settings. |
| Optional research | Exact parent publication hashes, search/policy/scope and applicable evidence/history inputs, seeds, code/runtime identity and output audits. Source references must agree across attached stages. |
| Definitions and registers | Exact definition/configuration references and append-only event hashes, predecessors, recorded-at times and explicit expected heads. Decision/approval/application identities remain distinct. |
| Outcome/lifecycle reports | Cutoff snapshot of exact source heads, policies, definitions/applications, settlement/review events, output hashes and statistical provenance where applicable. |
| Dashboard | Trusted root, explicit compatible attachments and verified table identities. No implicit newest substitute, live journal read or background model fit. |

Hashes establish byte integrity and internal linkage, not the truth of operator
attestations, actual fills, source completeness or authentic reviewer identity.
Journal hash chains detect inconsistent predecessors; detecting deleted tails needs
an external head anchor. Cross-journal duplicate/budget checks cannot recover omitted
records. Keep private backups and an appropriate external anchor separately.

## Replay, immutability and failure

Publication builders generally stage under a single-writer lock and atomically
publish a complete directory. Existing matching specifications are verified/reused;
changes in declared source, policy, scope, evidence, implementation or applicable
runtime versions create new identities. The exact manifest specification is the
stage-specific authority: not every builder hashes every repository file or uses
the same schema. Do not assume output bytes are portable across library versions.
Ingestion/canonical replay can check logical table equality; other stages additionally
or instead verify stored output bytes. Retain code and locked runtime for replay.

Do not overwrite historical sources/publications or hand-edit manifests to repair
integrity failures. Build a new explicit publication for new evidence; preserve old
paths and links. Interrupted/failed builds must not advertise completeness. Budgets
fail rather than publish an unreported subset family. Core pipeline attempts/resume
apply to supported pinned stages; newer optional builders still require their
explicit CLI steps, not an assumed full-pipeline orchestrator.

Loaders verify expected schemas, hashes/counts and lineage as required by their
contract. Dashboard adapters additionally confine paths/symlinks to the trusted root,
check report compatibility and clear invalid attachments instead of retaining stale
cached evidence. An absent optional attachment means unavailable evidence for that
view, not zero economics, no launch or a known false status. Old compatible reports
remain usable with explicit missing-stage notices. See [remediation](../TROUBLESHOOTING.md)
for corruption versus missing source evidence; never loosen gates to make a view load.

Implementation: [shared manifests](../../src/oddsjam_ev/io/manifests.py),
[canonical writer/reader](../../src/oddsjam_ev/io/canonical.py),
[pipeline manifests](../../src/oddsjam_ev/pipeline/manifests.py),
[dashboard evidence](../../src/oddsjam_ev/dashboard/evidence.py).
Verification: [canonical replay/corruption](../../tests/test_canonical_bundle.py),
[optional publication integrity](../../tests/test_prioritization_publication.py),
[dashboard confinement](../../tests/test_dashboard_artifacts.py),
[pipeline failures/resume](../../tests/test_weekly_pipeline.py),
[immutable tutorial refresh](../../tests/test_operator_walkthrough.py).

[Contract reference](README.md) · [Documentation home](../README.md)
