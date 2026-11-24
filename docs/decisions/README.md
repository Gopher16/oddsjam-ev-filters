# Architecture decision records

Use an ADR for a significant technical choice with durable alternatives/tradeoffs,
such as a framework, storage/hosting boundary or shared model/service contract.
Routine edits belong in the PR plan; scope/schedule changes belong in the
[programme decision register](../roadmap/DECISION_REGISTER.md) and change log.
These documents are separate from runtime research/experiment decision journals.

No new architecture choice is accepted by creating this directory. The existing
[architecture guide](../ARCHITECTURE.md) describes current behavior; the R11 framework
choice and R34 independent-platform assessment remain pending gates.

Start from [TEMPLATE.md](TEMPLATE.md), assign the next stable ADR ID and link affected
R/REQ IDs and the actual decision authority. Keep proposed/accepted/deferred/rejected/
superseded states explicit; superseding decisions link both records. Include evidence,
alternatives (including retaining the current design), consequences, compatibility,
operating cost, validation and rollback. Do not populate an acceptance date or claim
owner agreement until the decision occurs.
