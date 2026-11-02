"""Isolated synthetic operator tutorial using the real CLI parsers and validators.

All generated inputs, registers and publications stay in a fresh temporary workspace.
Only this tutorial process substitutes a simulated clock. No production time override,
real-account input, automatic scheduler or scanner integration is introduced.
"""

import argparse
import copy
import csv
import io
import json
import runpy
import sys
import tempfile
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.decision_config import CONTROLS
from oddsjam_ev.dashboard.artifacts import DashboardConfig
from oddsjam_ev.io.decision_records import load_decisions
from oddsjam_ev.io.paper_definitions import read_definitions
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery

REPO = Path(__file__).resolve().parents[2]
STEPS = (
    "init",
    "build",
    "define",
    "approve-study",
    "launch",
    "capture-review",
    "propose-change",
    "approve-change",
    "confirm-application",
    "monitor",
    "refresh",
)
CLOCKS = (
    "oddsjam_ev.pipeline.runner.utc_now",
    "oddsjam_ev.io.paper_definitions.utc_now",
    "oddsjam_ev.io.experiment_register.utc_now",
    "oddsjam_ev.io.filter_lifecycle.utc_now",
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


class Demo:
    """Confine all tutorial state to one explicitly initialized temporary directory."""

    def __init__(self, root, initialize=False):
        supplied = Path(root).absolute()
        self.root = supplied.resolve()
        allowed = (Path(tempfile.gettempdir()).resolve(), Path("/private/tmp").resolve())
        if supplied.is_symlink() or not any(self.root.is_relative_to(p) for p in allowed):
            raise ValueError("Use a real directory beneath the system temporary directory")
        if initialize:
            if not self.root.is_dir() or any(self.root.iterdir()):
                raise ValueError("init requires a fresh empty temporary directory")
            write(self.root / "SYNTHETIC_ONLY.json", {"kind": "operator-demo", "version": 1})
            self.state = {"completed": [], "paths": {}, "generation": "first"}
        else:
            marker = json.loads((self.root / "SYNTHETIC_ONLY.json").read_text())
            if marker != {"kind": "operator-demo", "version": 1}:
                raise ValueError("Not a synthetic tutorial workspace")
            self.state = json.loads((self.root / "state.json").read_text())
            self.validate_paths(self.state)

    def validate_paths(self, value):
        if isinstance(value, dict):
            for v in value.values():
                self.validate_paths(v)
        elif isinstance(value, list):
            for v in value:
                self.validate_paths(v)
        elif isinstance(value, str) and value.startswith("/"):
            if not Path(value).resolve().is_relative_to(self.root):
                raise ValueError("Tutorial state contains an external path")

    def file(self, relative):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Tutorial path escape")
        return path

    def cli(self, script, arguments, now="2030-01-01T00:00:00Z"):
        """Execute the actual CLI with explicit paths, retaining argv/stdout/time."""
        args = [str(a) for a in arguments]
        stream = io.StringIO()

        class SimulatedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.fromisoformat(now).astimezone(tz)

        with ExitStack() as stack:
            stack.enter_context(
                patch("oddsjam_ev.io.decision_register.datetime", SimulatedDateTime)
            )
            for clock in CLOCKS:
                stack.enter_context(patch(clock, return_value=now))
            stack.enter_context(patch.object(sys, "argv", [script, *args]))
            stack.enter_context(redirect_stdout(stream))
            try:
                runpy.run_path(str(REPO / "scripts" / script), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError(
                        f"Tutorial CLI {script} failed with exit {exc.code}"
                    ) from exc
        record = {
            "script": script,
            "argv": args,
            "simulated_utc": now,
            "stdout": stream.getvalue(),
            "synthetic_only": True,
        }
        with self.file("commands.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        return stream.getvalue().strip()

    def one(self, relative):
        paths = list(self.file(relative).glob("*/manifest.json"))
        if len(paths) != 1:
            raise ValueError("Expected one explicit tutorial publication, not latest lookup")
        return str(paths[0].parent)

    def payload(self, label, value):
        return write(self.file("payloads/" + label + ".json"), value)

    def experiment(self, kind, payload, now):
        result = json.loads(
            self.cli(
                "manage_experiments.py",
                [
                    "--register",
                    self.file("experiments"),
                    "event",
                    "--experiment-id",
                    self.state["experiment_id"],
                    "--kind",
                    kind,
                    "--payload",
                    self.payload(kind + "-" + str(len(self.state.get("events", []))), payload),
                    "--expected-event-id",
                    self.state["experiment_head"],
                    "--actor",
                    "SYNTHETIC operator",
                    "--rationale",
                    "Tutorial simulation only",
                ],
                now,
            )
        )
        self.state["experiment_head"] = result["event_id"]
        self.state.setdefault("events", []).append(result["event_id"])
        return result

    def lifecycle(self, kind, payload, now):
        args = [
            "--register",
            self.file("lifecycle"),
            "record",
            "--kind",
            kind,
            "--payload",
            self.payload("lifecycle-" + kind, payload),
            "--actor",
            "SYNTHETIC operator",
            "--rationale",
            "Tutorial simulation only",
        ]
        if self.state.get("lifecycle_head"):
            args += ["--expected-event-id", self.state["lifecycle_head"]]
        result = json.loads(self.cli("manage_filter_lifecycle.py", args, now))
        self.state["lifecycle_head"] = result["event_id"]
        return result

    def publish_definitions(self, now, output):
        return self.cli(
            "manage_paper_definitions.py",
            [
                "--register",
                self.file("definitions-register"),
                "publish",
                "--as-of",
                now,
                "--output-root",
                self.file(output),
            ],
            now,
        )

    def dashboard(self, name, attachments=None):
        p = self.state["paths"]
        cfg = asdict(
            DashboardConfig(
                artifact_root=p["pipeline_root"],
                evidence_root=str(self.root),
                min_analyzed_wagers=100,
                evidence_publications={
                    p["run_id"]: {"discovery": p["discovery"], "prioritization": p["priority"]}
                },
                decision_records_publication=p["sheets"],
                workspace_publications={
                    p["run_id"]: {"queues": p["queues"], "visuals": p["visuals"]}
                },
                explanation_publications={p["run_id"]: {"global": p["models"], "subsets": []}},
            )
        )
        cfg.update(attachments or {})
        self.file(name).write_text(yaml.safe_dump(cfg))

    def init(self):
        row = {
            "created_at": "2025-01-01T12:00:00Z",
            "sportsbook": "Demo A",
            "game_id": "demo-0",
            "market_name": "Moneyline",
            "bet_name": "Away",
            "odds": "100",
            "stake": "1",
            "bet_profit": "1",
            "potential_payout": "2",
            "status": "won",
            "bet_type": "positive_ev",
            "saved_filter_names": "TEST - Synthetic parent",
            "is_live_bet": "False",
            "is_free_bet": "False",
            "is_odds_boost": "False",
            "percentage": "3.25",
            "sport": "basketball",
            "league": "SYNTHETIC",
            "event_start_date": "2025-01-02T18:00:00Z",
        }
        rows = []
        for i in range(240):
            won = i % 10 < (8 if i < 120 else 2)
            rows.append(
                {
                    **row,
                    "game_id": f"synthetic-{i}",
                    "sportsbook": "Demo A" if i < 120 else "Demo B",
                    "bet_profit": "1" if won else "-1",
                    "status": "won" if won else "lost",
                }
            )
        self.csv("source.csv", rows)
        write(
            self.file("source-config.json"),
            {
                "store": str(self.file("store")),
                "source": {"namespace": "synthetic-operator-tutorial"},
            },
        )
        write(
            self.file("canonical-config.json"),
            asdict(
                CanonicalConfig({"fixed_odds": ["Demo A", "Demo B"], "exchange": [], "pickem": []})
            ),
        )
        write(
            self.file("registry.json"),
            {
                "schema_version": 1,
                "snapshot_date": "2026-10-22",
                "filters": [{"id": "synthetic-parent", "name": "TEST - Synthetic parent"}],
            },
        )
        write(self.file("search.json"), {"contract_version": 1, "views": [["sportsbook"]]})

    def csv(self, name, rows):
        with self.file(name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def build(self):
        generation = self.state["generation"]
        out = generation + "/"
        self.cli(
            "ingest_bet_tracker.py",
            [
                "--config",
                self.file("source-config.json"),
                "import",
                self.file("source.csv" if generation == "first" else "source-refresh.csv"),
            ],
        )
        self.cli(
            "ingest_bet_tracker.py",
            [
                "--config",
                self.file("source-config.json"),
                "snapshot",
                "--as-of",
                "2030-01-01T00:00:00Z",
            ],
        )
        # The snapshot is an explicit CLI result, not a latest-directory search.
        transcript = json.loads(self.file("commands.jsonl").read_text().splitlines()[-1])["stdout"]
        # The importer prints the exact locator on its first output line.
        snapshot = transcript.splitlines()[0]
        config = {
            "contract_version": 1,
            "output_root": str(self.file(out + "pipeline")),
            "source_config": str(self.file("source-config.json")),
            "canonical_config": str(self.file("canonical-config.json")),
            "registry": str(self.file("registry.json")),
        }
        write(self.file(out + "pipeline.json"), config)
        result = json.loads(
            self.cli(
                "run_weekly.py",
                [
                    "--config",
                    self.file(out + "pipeline.json"),
                    "--snapshot",
                    snapshot,
                    "--as-of",
                    "2030-01-01T00:00:00Z",
                ],
            )
        )
        run_id = result["run_id"]
        manifest = json.loads(
            self.file(out + "pipeline/results/" + run_id + "/manifest.json").read_text()
        )
        p = {
            "run_id": run_id,
            "pipeline_root": str(self.file(out + "pipeline")),
            "canonical": manifest["outputs"]["canonical"]["path"],
            "atlas": manifest["outputs"]["atlas"]["path"],
        }
        self.state["paths"] = p
        self.cli(
            "build_sizing_history.py",
            [
                "--canonical",
                p["canonical"],
                "--knowledge-as-of",
                "2030-01-01T00:00:00Z",
                "--output-root",
                self.file(out + "sizing"),
            ],
        )
        self.cli(
            "build_sizing_diagnostics.py",
            [
                "--canonical",
                p["canonical"],
                "--sizing",
                self.one(out + "sizing"),
                "--output-root",
                self.file(out + "diagnostics"),
            ],
        )
        self.cli(
            "build_within_filter_discovery.py",
            [
                "--canonical",
                p["canonical"],
                "--atlas",
                p["atlas"],
                "--diagnostics",
                self.one(out + "diagnostics"),
                "--config",
                self.file("search.json"),
                "--output-root",
                self.file(out + "discovery"),
            ],
        )
        p["discovery"] = self.one(out + "discovery")
        self.cli(
            "build_signal_prioritization.py",
            [
                "--discovery",
                p["discovery"],
                "--diagnostics",
                self.one(out + "diagnostics"),
                "--output-root",
                self.file(out + "priority"),
            ],
        )
        p["priority"] = self.one(out + "priority")
        self.state["candidate"] = next(
            r
            for r in load_within_filter_discovery(Path(p["discovery"]))[
                "signal_candidates"
            ].to_dict("records")
            if json.loads(r["predicate_json"])[0]["value"] == "Demo A"
        )
        evidence = self.configuration()
        write(self.file(out + "evidence.json"), evidence)
        self.cli(
            "build_decision_records.py",
            [
                "--discovery",
                p["discovery"],
                "--prioritization",
                p["priority"],
                "--evidence",
                self.file(out + "evidence.json"),
                "--as-of",
                "2030-01-01T00:00:00Z",
                "--effective-at",
                "2030-01-01T00:00:00Z",
                "--output-root",
                self.file(out + "sheets"),
            ],
        )
        p["sheets"] = self.one(out + "sheets")
        self.cli(
            "build_visual_diagnostics.py",
            [
                "--canonical",
                p["canonical"],
                "--atlas",
                p["atlas"],
                "--diagnostics",
                self.one(out + "diagnostics"),
                "--output-root",
                self.file(out + "visuals"),
            ],
        )
        p["visuals"] = self.one(out + "visuals")
        self.cli(
            "build_review_queues.py",
            [
                "--discovery",
                p["discovery"],
                "--prioritization",
                p["priority"],
                "--decisions",
                p["sheets"],
                "--visuals",
                p["visuals"],
                "--output-root",
                self.file(out + "queues"),
            ],
        )
        p["queues"] = self.one(out + "queues")
        self.cli(
            "build_explanatory_models.py",
            [
                "--canonical",
                p["canonical"],
                "--visuals",
                p["visuals"],
                "--output-root",
                self.file(out + "models"),
            ],
        )
        p["models"] = self.one(out + "models")
        self.dashboard("dashboard.yaml" if generation == "first" else "dashboard-refresh.yaml")
        self.cli(
            "verify_release.py",
            [
                "--config",
                self.file("dashboard.yaml" if generation == "first" else "dashboard-refresh.yaml"),
                "--run-id",
                run_id,
                "--repeats",
                "1",
            ],
        )

    def configuration(self):
        def setting(value, state="observed"):
            return {
                "state": state,
                "value": value,
                "basis": "prospective_design",
                "reason": "Explicit synthetic scanner setting; not OddsJam evidence",
            }

        settings = {key: setting(None, "unset") for key in CONTROLS}
        settings["sportsbooks"] = setting(["Demo A", "Demo B"])
        settings["paper_sizing"] = setting(
            {"method": "fixed_paper_stake", "stake_usd": 1, "applies_to": "both_arms"}
        )
        return {
            "contract_version": 1,
            "baselines": [
                {
                    "baseline_id": "synthetic-parent-v1",
                    "filter_id": "synthetic-parent",
                    "kind": "prospective_parent",
                    "recorded_at": "2030-01-01T00:00:00Z",
                    "effective_from": "2030-01-01T00:00:00Z",
                    "effective_until": None,
                    "evidence_reference": "Synthetic complete baseline, no real scanner settings",
                    "settings": settings,
                }
            ],
            "lifecycle": [],
            "translations": [
                {
                    "term": json.loads(self.state["candidate"]["predicate_json"])[0],
                    "control": "sportsbooks",
                    "value": ["Demo A"],
                    "recorded_at": "2030-01-01T00:00:00Z",
                    "evidence_reference": "Synthetic UI mapping only",
                    "platform_version": "demo-v1",
                    "semantics_verified": True,
                }
            ],
        }

    def define(self):
        p = self.state["paths"]
        candidate = self.state["candidate"]
        draft = yaml.safe_load(
            self.cli(
                "manage_paper_definitions.py",
                [
                    "draft",
                    "--discovery",
                    p["discovery"],
                    "--candidate-id",
                    candidate["candidate_id"],
                ],
            )
        )
        sheet_pub = load_decisions(Path(p["sheets"]))
        sheet = next(
            r
            for r in sheet_pub["records"]
            if r["candidate_id"] == candidate["candidate_id"] and r["design"] == "subgroup_tracking"
        )
        protocol = draft["protocol"]
        protocol.update(
            name="SYNTHETIC Demo A paper study",
            hypothesis="Synthetic subgroup/complement demonstration",
            planned_start_utc="2030-01-02T00:00:00Z",
            planned_end_utc="2030-02-02T00:00:00Z",
            review_dates_utc=["2030-02-04T00:00:00Z"],
            settlement_followup_days=2,
            review_criteria={
                "continue": "Review evidence; new future version only",
                "stop": "Protocol mismatch",
                "inconclusive": "Insufficient support or unknown quotes",
            },
            stop_rules=["Pause on baseline mismatch"],
            overlap_policy="Record shared event IDs, never pool versions",
            quote_availability_policy="Record original selection line odds timestamp capacity and failed checks; do not assume fills",
        )
        protocol["baseline"] = {
            "kind": "unchanged_parent",
            "filter_config": {r["control"]: r["baseline_value"] for r in sheet["settings"]},
            "evidence_reference": sheet["baseline"]["evidence_reference"],
        }
        protocol["sizing"]["stake_usd"] = 1
        protocol["configuration_reference"] = {
            "contract_version": 1,
            "publication_id": sheet_pub["manifest"]["publication_id"],
            "decision_id": sheet["decision_id"],
            "record_sha256": sheet["record_sha256"],
        }
        write(self.file("definition.json"), draft)
        self.save_definition(draft)
        self.state["definitions"] = self.publish_definitions("2030-01-01T00:00:00Z", "definitions")
        proposal = self.cli(
            "build_definition_proposal.py",
            [
                "--definitions",
                self.state["definitions"],
                "--version-id",
                self.state["version_id"],
                "--filter-id",
                "synthetic-scanner",
                "--sampling-rule",
                "Capture both parent books including complement and failed quotes",
                "--output-root",
                self.file("proposals"),
            ],
        )
        self.state["proposal"] = proposal
        row = load_recommendations(Path(proposal))["proposals"].iloc[0]
        self.state["proposal_id"] = row.proposal_id
        self.state["protocol"] = json.loads(row.prospective_protocol_json)
        self.dashboard(
            "dashboard-defined.yaml", {"paper_definitions_publication": self.state["definitions"]}
        )

    def save_definition(self, draft, now="2030-01-01T00:00:00Z", previous=None):
        p = self.state["paths"]
        args = [
            "--register",
            self.file("definitions-register"),
            "save",
            "--discovery",
            p["discovery"],
            "--prioritization",
            p["priority"],
            "--definition",
            write(self.file("definition-next.json"), draft),
            "--test-key",
            "synthetic-study",
            "--actor",
            "SYNTHETIC operator",
            "--rationale",
            "Tutorial paper intent only",
            "--decision-publication",
            p["sheets"],
        ]
        if previous:
            args += ["--expected-version-id", previous]
        result = json.loads(self.cli("manage_paper_definitions.py", args, now))
        self.state["version_id"] = result["version_id"]

    def approve_study(self):
        args = [
            "--register",
            self.file("approvals"),
            "record",
            "--bundle",
            self.state["proposal"],
            "--proposal-id",
            self.state["proposal_id"],
            "--reviewer",
            "SYNTHETIC reviewer",
            "--rationale",
            "Explicit synthetic paper decision",
        ]
        review = json.loads(
            self.cli(
                "record_research_decision.py",
                args + ["--to-status", "UNDER_REVIEW", "--expected-status", "PROPOSED"],
            )
        )
        approval = json.loads(
            self.cli(
                "record_research_decision.py",
                args
                + [
                    "--to-status",
                    "APPROVED",
                    "--expected-status",
                    "UNDER_REVIEW",
                    "--expected-event-id",
                    review["event_id"],
                ],
            )
        )
        registration = json.loads(
            self.cli(
                "manage_experiments.py",
                [
                    "--register",
                    self.file("experiments"),
                    "register",
                    "--bundle",
                    self.state["proposal"],
                    "--decisions",
                    self.file("approvals"),
                    "--proposal-id",
                    self.state["proposal_id"],
                    "--approval-event-id",
                    approval["event_id"],
                    "--actor",
                    "SYNTHETIC operator",
                    "--rationale",
                    "Explicit synthetic registration",
                ],
                "2030-01-01T12:00:00Z",
            )
        )
        self.state.update(
            experiment_id=registration["experiment_id"], experiment_head=registration["event_id"]
        )
        policy = json.loads((REPO / "configs/filter_review_policy.example.json").read_text())
        policy["evidence_reference"] = (
            "Explicit synthetic descriptive review; no production approval"
        )
        self.state["policy_id"] = self.lifecycle("POLICY", policy, "2030-01-01T13:00:00Z")[
            "event_id"
        ]

    def launch(self):
        self.experiment(
            "TRANSITION",
            {
                "to_status": "RUNNING",
                "launched_at_utc": "2030-01-02T00:00:00Z",
                "filter_id": "synthetic-scanner",
                "filter_config": self.state["protocol"]["filter_config"],
                "evidence_reference": "SYNTHETIC actual launch confirmation; no real platform action",
            },
            "2030-01-02T00:01:00Z",
        )

    def report(self, now, output):
        self.cli(
            "build_experiment_report.py",
            [
                "--register",
                self.file("experiments"),
                "--canonical",
                self.state["paths"]["canonical"],
                "--as-of",
                now,
                "--output-root",
                self.file(output),
            ],
            now,
        )
        return self.one(output)

    def capture_review(self):
        for key, book, status in (
            ("trial", "Demo A", "checked_available"),
            ("other", "Demo B", "checked_unavailable"),
        ):
            self.experiment(
                "ENROLL",
                {
                    "enrollment_id": key,
                    "observation_key": "synthetic-" + key,
                    "observed_at_utc": "2030-01-02T00:02:00Z",
                    "event_start_utc": "2030-01-02T18:00:00Z",
                    "event_namespace": "synthetic",
                    "event_id": "event-" + key,
                    "feature_values": {"sportsbook": book},
                    "feature_available_at_utc": {"sportsbook": "2030-01-02T00:02:00Z"},
                    "scanner_quote": {"selection": "Away", "line": "not_applicable", "odds": 100},
                    "baseline_config_sha256": fingerprint(self.state["protocol"]["filter_config"]),
                    "parent_membership_reference": "Synthetic parent includes both books",
                    "evidence_reference": "Synthetic decision-time quote",
                },
                "2030-01-02T00:03:00Z",
            )
            self.experiment(
                "CHECK",
                {
                    "enrollment_id": key,
                    "status": status,
                    "checked_at_utc": "2030-01-02T00:03:00Z",
                    "selection": "Away",
                    "line": "not_applicable",
                    "odds": 100,
                    "available_stake_usd": 1 if key == "trial" else 0,
                    "reason": "Synthetic original offer available"
                    if key == "trial"
                    else "Synthetic offer unavailable",
                    "evidence_reference": "Synthetic check, not accepted fill",
                },
                "2030-01-02T00:03:00Z",
            )
        for key, result in (("trial", "lost"), ("other", "won")):
            self.experiment(
                "SETTLE",
                {
                    "enrollment_id": key,
                    "result": result,
                    "outcome_available_at_utc": "2030-01-02T20:00:00Z",
                    "evidence_reference": "Synthetic outcome",
                },
                "2030-01-02T20:01:00Z",
            )
        self.experiment(
            "TRANSITION",
            {
                "to_status": "AWAITING_OUTCOMES",
                "ended_at_utc": "2030-02-02T00:00:00Z",
                "evidence_reference": "Synthetic actual end",
            },
            "2030-02-02T00:01:00Z",
        )
        report = self.report("2030-02-04T00:00:00Z", "outcomes-before-review")
        review = self.experiment(
            "REVIEW",
            {
                "scheduled_review_utc": "2030-02-04T00:00:00Z",
                "review_assessment_reference": report,
                "evidence_reference": "Explicit synthetic review",
                "notes": "Two observations, missing capacity in complement; inconclusive, no promotion",
            },
            "2030-02-04T00:02:00Z",
        )
        self.state["review_id"] = review["event_id"]
        report = self.report("2030-02-04T00:02:00Z", "outcomes-reviewed")
        self.experiment(
            "TRANSITION",
            {
                "to_status": "READY_FOR_REVIEW",
                "review_assessment_reference": report,
                "evidence_reference": "Explicit review readiness, not a statistical pass",
            },
            "2030-02-04T00:03:00Z",
        )
        self.state["outcomes"] = self.report("2030-02-04T00:03:00Z", "outcomes")
        self.dashboard(
            "dashboard-reviewed.yaml",
            {
                "paper_definitions_publication": self.state["definitions"],
                "prospective_report_publication": self.state["outcomes"],
            },
        )

    def propose_change(self):
        original = read_definitions(self.file("definitions-register"))[0]
        draft = json.loads(self.file("definition.json").read_text())
        draft["protocol"].update(
            planned_start_utc="2030-02-05T00:00:00Z",
            planned_end_utc="2030-03-05T00:00:00Z",
            review_dates_utc=["2030-03-07T00:00:00Z"],
        )
        self.save_definition(draft, "2030-02-04T00:10:00Z", original["version_id"])
        future = self.publish_definitions("2030-02-04T00:10:00Z", "definitions-followup")
        decision = self.lifecycle(
            "DECISION",
            {
                "policy_event_id": self.state["policy_id"],
                "experiment_register": str(self.file("experiments")),
                "experiment_id": self.state["experiment_id"],
                "review_report": self.state["outcomes"],
                "review_event_id": self.state["review_id"],
                "action": "modify",
                "followup_publication": future,
                "followup_version_id": self.state["version_id"],
                "evidence_reference": "Synthetic future TEST proposal despite inconclusive support; never PROD",
            },
            "2030-02-04T01:00:00Z",
        )
        settings = copy.deepcopy(self.configuration()["baselines"][0]["settings"])
        before = copy.deepcopy(settings)
        before["market_width"].update(state="observed", value=25)
        baseline = self.lifecycle(
            "BASELINE",
            {
                "filter_id": "synthetic-test-v2",
                "policy_event_id": self.state["policy_id"],
                "status": "TEST",
                "effective_at_utc": "2030-01-01T00:00:00Z",
                "settings": before,
                "legacy_context": "Synthetic preexisting TEST with market width 25",
                "evidence_reference": "Synthetic operator confirmation, not actual account evidence",
            },
            "2030-02-04T01:00:30Z",
        )
        change = self.lifecycle(
            "CHANGE",
            {
                "filter_id": "synthetic-test-v2",
                "decision_event_id": decision["event_id"],
                "change_kind": "modify",
                "target_status": "TEST",
                "previous_application_id": baseline["event_id"],
                "planned_effective_at_utc": "2030-02-04T02:00:00Z",
                "after_settings": settings,
                "ui_evidence": [
                    {
                        "control": k,
                        "state": v["state"],
                        "value": v["value"],
                        "semantics_verified": True,
                        "platform_version": "SYNTHETIC-v1",
                        "evidence_reference": "Synthetic UI evidence",
                        "observed_at_utc": "2030-02-04T01:00:00Z",
                    }
                    for k, v in settings.items()
                ],
                "evidence_reference": "Synthetic before/after request; no external application",
            },
            "2030-02-04T01:01:00Z",
        )
        write(
            self.file("rollback-evidence.json"),
            {
                "synthetic_only": True,
                "baseline_application_id": baseline["event_id"],
                "change_event_id": change["event_id"],
                "before_settings": baseline["payload"]["settings"],
                "interpretation": "Retained evidence only; not a rollback proposal or approval",
            },
        )
        self.state["change_id"] = change["event_id"]
        self.state["after_settings"] = settings

    def approve_change(self):
        self.state["change_approval"] = self.lifecycle(
            "APPROVE_CHANGE",
            {
                "change_event_id": self.state["change_id"],
                "evidence_reference": "Explicit synthetic change approval",
            },
            "2030-02-04T01:02:00Z",
        )["event_id"]

    def confirm_application(self):
        applied = self.lifecycle(
            "CONFIRM_APPLICATION",
            {
                "change_event_id": self.state["change_id"],
                "approval_event_id": self.state["change_approval"],
                "filter_id": "synthetic-test-v2",
                "status": "TEST",
                "settings": self.state["after_settings"],
                "effective_at_utc": "2030-02-04T02:00:00Z",
                "evidence_reference": "SYNTHETIC external application confirmation; no real scanner modified",
            },
            "2030-02-04T02:01:00Z",
        )
        self.state["application_id"] = applied["event_id"]
        self.state["after_settings"] = applied["payload"]["settings"]

    def monitor(self):
        observed = self.lifecycle(
            "OBSERVE",
            {
                "application_id": self.state["application_id"],
                "configuration_sha256": fingerprint(self.state["after_settings"]),
                "observation_key": "synthetic-applied-observation",
                "observed_at_utc": "2030-02-04T02:02:00Z",
                "event_start_utc": "2030-02-04T18:00:00Z",
                "event_namespace": "synthetic",
                "event_id": "monitor-event",
                "execution_mode": "paper",
                "stake_usd": 1,
                "american_odds": 100,
                "selection": "Away",
                "line": "not_applicable",
                "availability_status": "not_checked",
                "evidence_reference": "Synthetic monitoring only",
            },
            "2030-02-04T02:03:00Z",
        )
        self.lifecycle(
            "SETTLE",
            {
                "observation_event_id": observed["event_id"],
                "result": "won",
                "outcome_available_at_utc": "2030-02-04T20:00:00Z",
                "evidence_reference": "Synthetic monitoring outcome",
            },
            "2030-02-04T20:01:00Z",
        )
        self.cli(
            "build_filter_lifecycle_report.py",
            [
                "--register",
                self.file("lifecycle"),
                "--as-of",
                "2030-02-04T20:01:00Z",
                "--output-root",
                self.file("lifecycle-report"),
            ],
            "2030-02-04T20:01:00Z",
        )
        self.state["lifecycle_report"] = self.one("lifecycle-report")
        self.lifecycle(
            "MONITOR_REVIEW",
            {
                "application_id": self.state["application_id"],
                "action": "investigate",
                "monitoring_report": self.state["lifecycle_report"],
                "evidence_reference": "Synthetic human review: one wager is insufficient; no automatic rollback",
            },
            "2030-02-04T20:02:00Z",
        )
        self.cli(
            "build_filter_lifecycle_report.py",
            [
                "--register",
                self.file("lifecycle"),
                "--as-of",
                "2030-02-04T20:02:00Z",
                "--output-root",
                self.file("lifecycle-reviewed"),
            ],
            "2030-02-04T20:02:00Z",
        )
        self.state["lifecycle_report"] = self.one("lifecycle-reviewed")
        self.dashboard(
            "dashboard-monitored.yaml",
            {
                "paper_definitions_publication": self.state["definitions"],
                "prospective_report_publication": self.state["outcomes"],
                "lifecycle_report_publication": self.state["lifecycle_report"],
            },
        )

    def refresh(self):
        self.state["previous_paths"] = copy.deepcopy(self.state["paths"])
        with self.file("source.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        rows.append({**rows[0], "game_id": "synthetic-new-export-only"})
        self.csv("source-refresh.csv", rows)
        self.state["generation"] = "refresh"
        self.build()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--acknowledge-synthetic-decision", action="store_true")
    args = parser.parse_args()
    index = STEPS.index(args.step)
    demo = Demo(args.workspace, initialize=index == 0)
    if demo.state["completed"] != list(STEPS[:index]):
        parser.error("Run steps in order exactly once; use a new workspace after a failed mutation")
    if (
        args.step
        in {
            "approve-study",
            "launch",
            "capture-review",
            "propose-change",
            "approve-change",
            "confirm-application",
            "monitor",
        }
        and not args.acknowledge_synthetic_decision
    ):
        parser.error("This synthetic decision step requires --acknowledge-synthetic-decision")
    getattr(demo, args.step.replace("-", "_"))()
    demo.state["completed"].append(args.step)
    demo.validate_paths(demo.state)
    write(demo.file("state.json"), demo.state)
    print(json.dumps({"synthetic_only": True, "completed": args.step, "workspace": str(demo.root)}))


if __name__ == "__main__":
    main()
