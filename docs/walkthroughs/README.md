# Reproducible operator walkthroughs

These walkthroughs use **synthetic data only**, in an isolated temporary directory.
They exercise current CLI parsers, writers, loaders and time/state validators. They
are not instructions to approve or launch any real candidate automatically.

| Walkthrough | Result |
| --- | --- |
| [Review a candidate](REVIEW_CANDIDATE.md) | Verified historical report, evidence, setup and blockers |
| [Define and launch a paper study](PAPER_STUDY.md) | Separate definition, paper approval/registration, launch, observations and review |
| [Review and modify a filter](MODIFY_FILTER.md) | Future definition, exact before/after request, separate approval/application and monitoring |
| [Refresh from a new export](REFRESH_RESEARCH.md) | New snapshot/publications with old sources/reports intact |

Supporting references: [configuration/publication map](../CONFIGURATION_PUBLICATION_MAP.md),
[troubleshooting and remediation](../TROUBLESHOOTING.md),
[current architecture](../ARCHITECTURE.md), [glossary](../GLOSSARY.md).

## Create a fresh workspace

Run from the repository root with Python 3.11 and the locked Poetry environment:

```bash
poetry install --with dashboard
export WALKTHROUGH_ROOT="$(mktemp -d)"
poetry run python docs/examples/operator_demo.py --workspace "$WALKTHROUGH_ROOT" init
poetry run python docs/examples/operator_demo.py --workspace "$WALKTHROUGH_ROOT" build
```

Keep `WALKTHROUGH_ROOT` for subsequent steps. `init` refuses a nonempty directory;
it does not clear existing files. The tutorial only accepts a directory under the
system temporary directory (or `/private/tmp`), verifies its synthetic marker and
rejects external paths in its state. It has no account, CSV, register or policy-path
option for operating on your real history. Do not edit tutorial state to redirect it.

`build` creates 240 fictional $1 wagers split between **Demo A** and **Demo B** under
**TEST - Synthetic parent**, then runs real CLIs for ingestion, historical core
research, sizing, discovery, prioritization, setup sheets, visuals, queues, declined
model publication and reconciliation. The intentionally patterned outcomes illustrate
UI/contract behavior; they are not evidence about any sportsbook.

## Simulated time and explicit decisions

The teaching calendar is January–March **2030**. Within this example process only,
the adapter temporarily substitutes the relevant clock functions and review timestamp
provider while executing the actual scripts with `runpy`. Production scripts and time
gates are unchanged. Import receipts still describe the actual isolated import; the
historical knowledge cutoff must include them. This fixed calendar is intended for
the current pre-2030 environment, not for claims about real acquisition/placement time.

There is **no time-travel option in the production CLI**. Never replay these future
payloads against real registers, backdate a real launch, or reduce support gates to
complete a study early. Ordinary operations use actual system time and supported
observed/effective times. All tutorial decisions require an explicit
`--acknowledge-synthetic-decision` flag and remain local fictional records.

Steps run in order once. After a failure, inspect stderr and the generated files;
start with a **new** temporary directory if a mutating step partially wrote state.
The helper is not a resume engine. Do not delete locks or edit immutable publications
to force progress.

## Inspect what actually ran

The adapter writes:

| File | Purpose |
| --- | --- |
| `state.json` | Completed steps, exact publications and current version/event IDs |
| `commands.jsonl` | Each successful actual script, argv, simulated UTC and captured stdout |
| `payloads/*.json` | Exact synthetic enrollment, check, review and change payloads |
| `definition*.json` | Frozen intent and future follow-up inputs |
| `dashboard*.yaml` | Explicit view configurations for each publication/lineage stage |
| `rollback-evidence.json` | Previous application/settings retained as evidence, not an approved rollback |

For a readable command transcript:

```bash
poetry run python - <<'PY'
import json, os, shlex
from pathlib import Path
root = Path(os.environ['WALKTHROUGH_ROOT'])
for line in (root / 'commands.jsonl').read_text().splitlines():
    row = json.loads(line)
    print(row['simulated_utc'], shlex.join(['python', 'scripts/' + row['script'], *row['argv']]))
PY
```

Those are the real CLI interfaces, but copying the transcript alone does not supply
the synthetic clock. The staged tutorial commands are the reproducible route. For a
real study, use the [production command reference](../PROSPECTIVE_EXPERIMENT_INTEGRATION.md)
with separately reviewed real inputs and system time.

## Reproduce the screenshots

Complete the walkthrough through `monitor` first. The capture helper requires Node
22 or newer and Chrome; it uses built-in CDP support without npm dependencies.
In one terminal, from the repository root with `WALKTHROUGH_ROOT` still exported:

```bash
ODDSJAM_DASHBOARD_CONFIG="$WALKTHROUGH_ROOT/dashboard-monitored.yaml" \
  poetry run streamlit run dashboard/app.py --server.address 127.0.0.1 \
  --server.port 8541 --server.headless true --browser.gatherUsageStats false
```

In another terminal, start a separate Chrome profile (macOS example; on Linux use
your installed Chrome executable), leaving it running during capture:

```bash
export TUTORIAL_CHROME_PROFILE="$(mktemp -d)"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9235 --user-data-dir="$TUTORIAL_CHROME_PROFILE" \
  --no-first-run --no-default-browser-check about:blank
```

In a third terminal, export the same synthetic workspace path and run:

```bash
export TUTORIAL_SCREENSHOTS="$(mktemp -d)"
node docs/examples/capture_synthetic.mjs
```

The helper verifies the synthetic workspace marker, fictional platform and completed
configuration, then captures four 1440 × 1000 views. It adds a fictional-data label
to the local browser DOM only. Stop the two servers with Ctrl-C when finished.
The [committed image provenance](../images/operator/README.md) identifies the stage
shown; these screenshots illustrate navigation, not real account performance.

## Verification

```bash
poetry run pytest tests/test_operator_walkthrough.py -q
```

The tests execute every step through the same adapter, check approval/launch and
approval/application separation, preserve the 100-wager policy, verify version-linked
reports and prove the refresh leaves first-generation publications unchanged.
The synthetic dashboard is also exercised with AppTest. Committed screenshots were
captured from an isolated Chrome profile against synthetic configurations only.
No raw/private records or account screenshots are included.

[Documentation home](../README.md) · [Operator guide](../guides/OPERATOR.md)
