# Dashboard release acceptance — November 2, 2026

Scope: the A–I decision-workspace sequence, including the reviewed-filter lifecycle
merged into `main`. This record contains aggregate acceptance evidence only. Private
publications, raw data, screenshots and browser profiles remain outside Git history.

## Automated and private-snapshot evidence

- Locked dependencies, Ruff lint/format and `git diff --check`: passed.
- Complete synthetic suite: **522 passed** in **95.85 seconds** locally. The release
  tests add full-family positive, negative, unresolved and no-winner scenarios;
  cross-publication reconciliation/replay/corruption checks; and a paper definition
  → review → modification → confirmed TEST application → monitoring cycle. Existing
  prospective/promotion tests also exercise the verified configuration-to-pilot path,
  explicit caps, stale evidence, closure, late settlement, rollback and version isolation.
- Explicit private report:
  `a5d1070494285b648f452222f004b4f6f5ed96217112860fb1d83c96519aeffd`,
  using the exact links in `configs/dashboard.yaml`. Historical sources were neither
  refreshed nor relabeled; acceptance time is not the data cutoff or a holdout period.
- **70,635** unique canonical and visual wager candidates reconcile. **53,853** are
  diagnostic eligible. Source stake/profit and population counts, weighted ROI and
  equal-stake means reconcile across **11** funding/product populations. Saved-filter
  memberships and overlapping scopes are not added to produce a wager total.
- Within-filter evidence, configuration sheets, queues, visuals and models: verified.
  Definitions, prospective outcomes and lifecycle reports: **not attached** in the
  default private config. Their workflow is tested synthetically; no real studies,
  approvals, confirmations or monitoring observations were created for acceptance.
- Frozen global model: **29,975** explained wagers; eight declined populations remain
  explicit. Separately fitted offline subset model: **2,948** explained wagers.
  Loaders check exact lineage, unique identities, target agreement, event-disjoint
  splits, training-only background and additive contributions. Existing model tests
  additionally cover bounded fitting, baseline utility, feature leakage controls,
  unsupported scopes, corruption and no training during ordinary interactions.

These counts describe source coverage, not future opportunity supply or statistical
power. No corrected historical comparisons or confirmed PROD proposals are fabricated
to fill an empty headline queue. A qualifying synthetic fixture is not private-data
validation and does not authorize a real pilot.

## Requested visual inventory

Each chart has a published readiness record for every scope. The counts below are
**supported scopes / 25 published scopes**, not independent samples. Partial coverage
is disclosed within supported scopes. All ten implementations are exercised with
synthetic data, including evidence-enabled liquidity, timing and closing-price cases.

| Requested notebook visual | Private snapshot support | Remaining source limitation / remediation |
| --- | --- | --- |
| Profit by sport/platform/league/market | 14 / 25 | Economic exclusions and unsupported products remain visible |
| Platform advertised EV versus return | 14 / 25 | Numeric EV must match eligible economics |
| Saved-filter advertised EV versus return | 8 / 25 | Resolved attribution and matched EV required; no status inference from name |
| EV bands with return and volume | 14 / 25 | Exact source EV endpoints retained; missing EV is not zero |
| Liquidity bands | 0 / 25 | Verify decision-time USD liquidity units and capacity semantics |
| Liquidity × EV | 0 / 25 | Same liquidity evidence plus matched EV coverage |
| Time-to-event bands | 0 / 25 | Verify placement/event timestamp meanings, timezone and DST |
| Cumulative reported / fixed-unit profit by platform | 14 / 25 | Source-calendar attribution only; no settlement-order bankroll claim |
| Bet volume over time | 25 / 25 | Undated rows and partial boundaries disclosed |
| Average weekday/hour activity | 25 / 25 | Active source days only; not verified placement-time activity |

The optional closing-reference EV series remains unavailable on this private snapshot:
verify fair closing-price meaning, source and observation timing before enabling it.
Raw CLV labels are not substituted for that evidence. These are specific data tasks,
not unimplemented chart UI placeholders.

## Browser and visual acceptance

Real local Chrome, isolated profile, read-only dashboard on loopback. Harness:
`scripts/acceptance/browser.mjs`; private evidence saved under
`/private/tmp/oddsjam-release-browser` during this acceptance run.

| Check | Result |
| --- | --- |
| Overview, opportunities and production review | Rendered with explicit evidence/setup limitations |
| 1440×900 and 1366×768 layouts | No page horizontal overflow or uncaught browser/application exception |
| Card → complete setup → exact candidate deep link → fresh navigation | Exact idea retained; setup states and blockers visible |
| Reset signal filters | Clears the candidate deep link |
| Actual chart hover | Tooltip includes wager/economic denominators |
| Bucket top-panel selection | Linked marks dim; double-click clears selection |
| All ten history diagnostics | Rendered chart or explicit source blocker/remediation |
| Explain returns with sidebar platform selection | Fixed-model view; existing unit/background contract retained |
| Explicit offline subset publication selection | Separately fitted model remains explicitly labeled |
| Screenshots inspected | Overview, historical EV panel, setup and explanation layouts reviewed locally |

AppTest separately covers draft contents, read-only review displays, exact selectors,
empty scopes and corrupt-cache clearing. The browser harness does not click an approval,
launch or application control. Wide parameter/data tables remain scrollable with data
and downloads available; they are not compressed into unreadable text. Dense all-data
chart legends are exploratory views; narrow the population for inspection.

## Performance measurements and limits

Measured locally on macOS with Python **3.11.7**, against the explicit private workload
above. The publication checker does not fit or evaluate a new model. Its cold timer
starts **after imports**. Browser results use an already-running local Streamlit process;
this is not a clean-machine cold-start or network/hosted-service benchmark.

| Measure | Observed | Explicit desktop release budget |
| --- | --- | --- |
| Initial publication load | 1.85 s | ≤10 s |
| Byte-verified warm rechecks | 0.42 / 0.30 / 0.49 s | Each ≤5 s |
| Process peak RSS including imports | 1,968 MiB | ≤3,072 MiB |
| Browser initial navigation to settled overview | About 3–4 s | Measured, no cross-machine SLA |

The explicit loader budgets passed. They are engineering smoke budgets, unrelated to
statistical screening or promotion rules. Peak RSS is a process high-water mark, not
incremental request memory. This workload needs desktop-scale memory; small hosted
instances require a separate deployment benchmark and potentially memory optimization.
No claim of multi-user load capacity, mobile layout acceptance or production uptime
is made.

## Release interpretation and next work

The planned implementation sequence is complete for supported review and prospective
paper-study preparation. You may review ideas now; unknown scanner settings, baseline
provenance, chronology or lifecycle evidence can still block a faithful launch or a
production-change proposal. Record and publish real prospective evidence through the
existing tools; do not replace it with synthetic acceptance fixtures.

Next optional project: [documentation consolidation](../plans/DOCUMENTATION_OVERHAUL.md).
Next operational work: confirm missing settings/lifecycle/source evidence for selected
studies, register future protocols and attach explicit outcome publications. No generic
extra feature PR is required before using the review workflow. Defects, unsupported
source contracts or a new deployment environment merit focused follow-up changes.
