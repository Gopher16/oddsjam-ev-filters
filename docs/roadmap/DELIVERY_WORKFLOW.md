# Delivering one roadmap PR

Apply this workflow when the owner selects a PR or requests the next eligible one
under the [versioned delivery agreement and adopted amendments](SOURCE_BASELINES.md). A plan is saved and
presented before implementation; routine approval pauses are not required. A request
to continue the next unfinished commit authorizes finishing the current PR, not
skipping a pending owner checkpoint or automatically starting another PR.

## 1. Establish the current state

1. Read both baseline roadmaps and applied delivery instructions; verify their
   fingerprints and inspect the selected card's detailed acceptance requirements.
2. Read applicable repository instructions and current PR template. Inspect branch,
   remote main/PR state, working-tree changes, relevant code/contracts/tests and
   delivery/decision records. Preserve private and unrelated work; isolate the branch
   or worktree when needed.
3. Verify dependencies are actually merged into main, including required post-merge
   checks. Record actual PR numbers/merge SHAs. Evaluate the register's conditional
   clauses explicitly; do not convert feasibility into activation permission.
4. Identify current capability to reuse and unresolved evidence. Apply accepted scheduling amendments first: N001 is the checkpoint before R10.
   Otherwise pick the lowest eligible original R ID if none was specified. Later
   N items need explicit selection or recorded owner-authorized scheduling; document
   why the selected item is eligible. Missing
   evidence or a material baseline conflict is not permission to invent a workaround.

## 2. Save and present the detailed plan

Use [pr-plans/TEMPLATE.md](pr-plans/TEMPLATE.md) as `pr-plans/Rxx.md`
or `pr-plans/Nxxx.md` for an adopted additive PR. Include the
problem/before-and-after behavior; source/REQ/legacy IDs; dependency evidence;
exact files, contracts and adapters; ordered Conventional Commits; acceptance
examples; tests and owner checkpoints; failure/empty states; migration, rollout,
rollback and operational implications. Distinguish private evidence dependencies
from implementation still missing. Present the main decisions/commit sequence to
the owner, then proceed within clear selected scope.

A plan does not authorize paid feeds, account connections, deployment or wagering.
For material scope/architecture/dependency/acceptance changes, first record the
reason, alternatives (including retaining the baseline), downstream impact and
original work's destination in the change log/decision register. Ask for owner
agreement before affected work. Continue independent authorized work while awaiting
an answer; silence and elapsed time are not approval. Routine implementation details
can be resolved and recorded without stopping the work.

## 3. Implement and validate commit by commit

Preserve the approved scope and unrelated work. Review each diff for private data,
credentials, unintended generated artifacts and contract drift. Run increment checks
and the repository's required checks before commits; record actual outcomes and
failures without claiming unavailable validation. Keep meaningful commits, not one
commit per arbitrary counter or bookkeeping updates that chase their own SHA.

For the currently documented environment, required checks are:

```bash
poetry check --lock
poetry run ruff check . --fix
poetry run ruff format .
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest -q
git diff --check
python3 scripts/check_documentation.py
```

Re-read current repo requirements rather than treating this list as permanent.
Use the documented locked environment; the [maintenance guide](../DOCUMENTATION_MAINTENANCE.md)
explains optional CLI help checks. Stage new Markdown before tracked-file checks.
Do not execute shell snippets scraped from documentation. For research changes,
verify economics/grains/denominators, chronology/leakage, dependence, missingness,
policy/family accounting, replay and independent reference cases. Run affected
notebooks against the intended snapshot when necessary; preserve useful outputs
and do not run private-data work just to fill a checkbox.

## 4. Establish necessary owner acceptance

Ask only when owner validation is required to establish acceptance or a material
choice blocks progress. For a dashboard/checkpoint provide launch command or URL,
exact publication/snapshot, selections, steps, expected behavior, limitations and
whether merge is blocked. For data/evidence changes specify the record/period and
interpretation needing confirmation. Record the owner's actual answer and findings;
fix issues, rerun affected checks and request revalidation where needed.

Never mark a required manual review passed on the basis of silence, screenshots
from a different version or automated checks alone. Conversely, documentation-only
link/content validation can establish acceptance without a manufactured UI walkthrough.
Explain why manual acceptance is inapplicable for that selected PR.

## 5. Open, review and merge

Use the current [.github/pull_request_template.md](../../.github/pull_request_template.md).
Keep its sections, with a concrete objective, roadmap/plan links, impact classification,
research applicability, exact checks, unresolved follow-ups and merge checklist.
Check boxes only for completed successful checks; mark inapplicable work honestly.
A draft can be opened early; final title/body describe the delivered scope.

Inspect the full diff and relevant feedback. Address CI/review findings, with larger
ideas preserved in the backlog rather than silently broadening the PR. Verify the
target is main, tested head is current, required reviews/checks passed, required
owner acceptance is recorded and no conflict remains. If main moved, integrate it
through normal repository workflow and rerun affected checks. Do not bypass branch
protection, force-push over unrelated work or merge failing checks. The current
single-maintainer instructions do not require an approving reviewer, but future
repo requirements may change.

Merge using a supported method, then verify remote merged state and that the actual
merge/squash commit is on main. Verify applicable post-merge CI. If blocked, state
the exact permission/review/check blocker and keep status accurate; do not report
completion or start another roadmap PR while required merge verification remains open.

## 6. Close the delivery trail

The plan records implemented behavior, validations, owner checkpoints, schema/config/
model versions, deviations and remaining work. The register links the actual PR and
merge evidence. Before merge use an accurate pre-merge status. After merge report
the actual SHA and CI results in the completion response/PR; reconcile the repo ledger
in the next authorized PR or a transparent separately reviewed documentation PR.
Never alter a merged commit just to insert its own or its future merge SHA.

Report the selected ID, result, PR/merge SHA, checks, limitations, deviations, owner
acceptance status and next eligible ID concisely. Keep both Word baselines unchanged.
A new owner instruction is required to begin the next roadmap PR.

## Change control and coverage reconciliation

Use [programme decisions](DECISION_REGISTER.md) for disposition and links,
[change log](CHANGE_LOG.md) for chronology and [ADRs](../decisions/README.md) for
significant technical choices. Preserve the parent's R ID for splits (R26a/R26b),
link children and dependencies, and retain moved work's new destination. N001-style
ideas stay in the [backlog](OPPORTUNITY_BACKLOG.md) until explicitly prioritized.
Do not confuse these records with runtime research decision journals or experiment
approval/application events.

At each phase boundary reconcile all original requirements and detailed cards:
show delivered evidence, planned work, moved destination, reasoned deferral or
owner-approved retirement. Check cross-cutting evidence/compatibility constraints
and all partial items in the legacy crosswalk. New requirements get appended IDs;
never renumber history. Optional dated Word delivery-status reports can supplement
Git records if requested, without editing either frozen baseline.

[Programme index](README.md) · [Delivery register](DELIVERY_REGISTER.md)

## Adopted source amendments

Preserve archived originals and applied-source fingerprints. An explicit owner
request may produce a dated additive version; applying it authorizes only the
selected PR under the delivery agreement. Read the full original sections and
adopted addenda, compare changed content and record authority/disposition. A changed
hash is neither automatic rejection nor automatic approval of all new work. PD005
adopts tracker scope and N001-before-R10 sequencing; unchanged R01–R60 remain intact.
Use [source versions](SOURCE_BASELINES.md) and [tracker cards](TRACKER_ADDITION.md).
Do not edit supplied Word files during implementation or publish private captures.
