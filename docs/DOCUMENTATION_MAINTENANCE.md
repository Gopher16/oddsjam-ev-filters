# Documentation maintenance

Current instructions start at the [documentation home](README.md). Shared semantics
belong in the [contract reference](reference/README.md); component guides own their
CLI, schema and artifact details. [Historical plans](plans/README.md) preserve prior
intent and link to the current replacement. Do not update old evidence to look like
current defaults.

## Checks and their scope

Run from the repository root:

```bash
python scripts/check_documentation.py
poetry run python scripts/check_documentation.py --check-cli
poetry run pytest tests/test_documentation.py tests/test_operator_walkthrough.py -q
```

The standard-library link check scans **Git-tracked Markdown**, including root,
component, reference and historical pages. New documentation must be staged before
it is included. It checks inline/image and reference-definition local destinations,
repository confinement, ATX heading fragments (including duplicate slugs), explicit
HTML anchors, closed code fences and prose immediately following Mermaid blocks.
External URLs are skipped: the check makes no network requests. Directory links
are valid. Private/untracked roadmap notes and datasets are not scanned.

This is a focused checker for this repository's Markdown conventions, not a complete
CommonMark renderer. Use ATX headings and simple inline/reference links; keep URL
parentheses encoded, avoid nested Markdown link labels and use explicit HTML IDs for
legacy anchors that need unusual syntax. It does not validate external URLs, render
Mermaid, measure screenshot quality or prove that prose matches source semantics.
Review those explicitly. No generic Markdown formatting dependency is required.

The separate `--check-cli` mode uses the locked application environment. Its explicit
[interface allowlist](examples/cli_interfaces.json) runs only the named script and
optional subcommand followed by `--help`, then checks selected documented options.
No source files, publications, registers, credentials or network services are used.
The focused tests also pass documented dashboard keys through the real parser using
a temporary configuration with no attachments. Existing synthetic operator tests
exercise actual builders and versioned lifecycle behavior in temporary directories.
CI runs link/presentation checks as a lightweight `documentation` job; CLI checks
and synthetic tests run in the application test job.

## Example safety classes

| Example | How it is verified | What a reader must do |
| --- | --- | --- |
| Read-only CLI interface | Allowlisted `--help`; no payload is supplied | Install the locked environment; provide explicit inputs for actual use |
| Synthetic end-to-end workflow | [Operator tests](../tests/test_operator_walkthrough.py), fresh temporary root, fictional data and explicit synthetic acknowledgment | Follow the staged tutorial; fictional acknowledgments are not real approvals |
| Report build / attachment | Synthetic builders, immutable replay and configuration tests | Select trusted compatible publications and reconcile source coverage |
| Human decision, launch or application | Safe synthetic fixtures and negative-state tests; help only for selected interfaces | Review exact real evidence and explicitly record each authorized action separately |
| Historical/private-source example | Preserved as context; not executed by documentation CI | Supply local evidence if deliberately reproducing it; do not commit it |
| Browser capture | Explicit isolated local setup, fictional-data guard and watermark | Start local synthetic services intentionally; not an automatic CI requirement |

A code fence is illustrative text, **not an execution directive**. Do not add a tool
that discovers and runs all shell blocks. Approval, registration, actual launch,
enrollment and external application examples require human context. The repository
cannot apply settings inside OddsJam. Synthetic clocks/acknowledgments belong only
to the tutorial fixtures and must never bypass real decision gates.

## Keep contracts and documentation synchronized

When changing a contract:

1. Identify its authoritative reference page, versioned configuration, implementation
   and synthetic tests. Change methods only through the reviewed policy/schema process.
2. Update the owning explanation once, linking component guides instead of copying
   it. Preserve units, eligibility, missingness, chronology and exact version identity.
3. Update affected CLI help and the selected interface allowlist together. Add a safe
   synthetic behavior test when semantics change; help checks alone do not prove them.
4. Update the [configuration/publication map](CONFIGURATION_PUBLICATION_MAP.md),
   [glossary](GLOSSARY.md), navigation and implementation ledger when affected.
5. Keep one H1, sentence-case ATX section headings, descriptive relative links and
   simple tables with a header/separator. Label code fences (`bash`, `python`, `yaml`,
   `json`, `text`, `mermaid`); language labels do not grant execution permission.
6. Give each diagram an immediately following prose explanation of the same path,
   including human gates. Use meaningful image alt text and synthetic provenance.
7. Preserve historical files/URLs. Add a historical-status banner and replacement
   links when guidance is superseded; update inbound links or preserve heading anchors.
8. Run these checks and the repository's required quality checks, inspect the diff,
   and record actual validation/limitations in the PR. Never stage private notes,
   historical datasets, credentials or account screenshots.

The checks detect stale interfaces and broken navigation; they do not establish new
research evidence or authorize methodological changes.

[Contributor guide](guides/CONTRIBUTOR.md) · [Documentation home](README.md)
