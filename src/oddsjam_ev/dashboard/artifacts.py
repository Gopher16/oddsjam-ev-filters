"""Verify a whole persisted run and its lineage before presenting any research."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.io.atlas import load_atlas
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.discovery import load_discovery
from oddsjam_ev.io.experiment_reports import load_experiment_report
from oddsjam_ev.io.manifests import verify_bytes
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.pipeline.manifests import load_result

LOADERS = {
    "canonical": load_canonical_bundle,
    "atlas": load_atlas,
    "discovery": load_discovery,
    "recommendations": load_recommendations,
    "experiments": load_experiment_report,
}
ID = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class DashboardConfig:
    """Trusted local configuration; the UI does not accept arbitrary filesystem paths."""

    artifact_root: str = "data/processed/weekly_research"
    max_display_rows: int = 200
    min_analyzed_wagers: int = 100
    contract_version: int = 1
    display_timezone: str = "America/Chicago"
    evidence_root: str = "data/processed"
    paper_definitions_publication: str | None = None
    evidence_publications: dict = field(default_factory=dict)

    def __post_init__(self):
        from zoneinfo import ZoneInfo

        ZoneInfo(self.display_timezone)
        if self.paper_definitions_publication is not None and (
            not isinstance(self.paper_definitions_publication, str)
            or not self.paper_definitions_publication
        ):
            raise ValueError("Explicit paper definition publication required")
        if not isinstance(self.evidence_root, str) or not self.evidence_root:
            raise ValueError("Explicit evidence root required")
        if ".local" in Path(self.evidence_root).resolve().parts:
            raise ValueError("Credential directories are not evidence roots")
        if not isinstance(self.evidence_publications, dict):
            raise ValueError("Evidence publications must be a run-ID mapping")
        for run_id, links in self.evidence_publications.items():
            if not ID.fullmatch(str(run_id)) or not isinstance(links, dict):
                raise ValueError("Invalid evidence report mapping")
            if set(links) != {"discovery", "prioritization"} or not all(
                isinstance(v, str) and v for v in links.values()
            ):
                raise ValueError("Explicit discovery and prioritization paths required")
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported dashboard contract")
        if not isinstance(self.artifact_root, str) or not self.artifact_root:
            raise ValueError("Explicit artifact root required")
        if ".local" in Path(self.artifact_root).resolve().parts:
            raise ValueError("Credential directories are not artifact roots")
        if (
            type(self.min_analyzed_wagers) is not int
            or not 1 <= self.min_analyzed_wagers <= 1000000
        ):
            raise ValueError("Minimum analyzed wagers must be between 1 and 1000000")
        if type(self.max_display_rows) is not int or not 1 <= self.max_display_rows <= 1000:
            raise ValueError("Display row limit must be between 1 and 1000")


def read_dashboard_config(path: Path) -> DashboardConfig:
    """Load known local settings only; credentials are never consulted."""
    return DashboardConfig(**yaml.safe_load(path.read_text()))


def within(root: Path, path: Path) -> Path:
    """Reject path and symlink escapes from the explicitly configured artifact root."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Artifact outside configured root")
    return resolved


def discover_runs(root: Path) -> list[str]:
    """List candidate result IDs only; no selection, metrics, or 'latest' inference."""
    results = root / "results"
    if not results.exists():
        return []
    within(root, results)
    return sorted(
        p.name
        for p in results.iterdir()
        if ID.fullmatch(p.name) and within(root, p).is_dir() and (p / "manifest.json").is_file()
    )


@dataclass
class PublishedRun:
    """Session-local verified tables and the exact bytes required for cache reuse."""

    result: dict
    manifests: dict
    tables: dict[str, dict[str, pd.DataFrame]]
    expected_files: dict[Path, str]
    root: Path

    def verify(self) -> None:
        """Rehash files before each cached display; mtimes are not integrity evidence."""
        for path, digest in self.expected_files.items():
            verify_bytes(within(self.root, path), digest)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition):
    if not condition:
        raise ValueError("Unsupported or inconsistent published research lineage")


def load_published_run(root: Path, run_id: str) -> PublishedRun:
    """Load verified bundles only. Never invoke builders or read current configs.

    Old code versions are valid provenance, not a reason to rebuild a historical run.
    Raw source CSVs, credentials and live decision/experiment registers are not read.
    """
    if not ID.fullmatch(run_id):
        raise ValueError("Explicit published run ID required")
    directory = within(root, root / "results" / run_id)
    for file in ("manifest.json", "report.md"):
        within(root, directory / file)
    result = load_result(directory)
    spec = result["specification"]
    _require(result["run_id"] == run_id and spec["contract_version"] == 1)
    _require(spec["mode"] in {"manual_import", "historical_reanalysis"})
    aware(spec["as_of"])
    _require(spec["freshness"]["export_status"] in {"recent", "stale", "unverified", "missing"})
    _require(isinstance(spec["freshness"]["sources"], list))
    _require(isinstance(spec["source_diagnostics"], list) and isinstance(spec["code"], dict))
    outputs = result["outputs"]
    _require(
        set(outputs)
        in (
            {"canonical", "atlas", "discovery", "recommendations"},
            {"canonical", "atlas", "discovery", "recommendations", "experiments"},
        )
    )
    files = {directory / name: _digest(directory / name) for name in ("manifest.json", "report.md")}
    manifests, tables = {}, {}
    hashes = {}
    for stage, record in outputs.items():
        _require(record["stage"] == stage and record["run_id"] == run_id)
        _require(
            fingerprint({k: v for k, v in record.items() if k != "checkpoint_sha256"})
            == record["checkpoint_sha256"]
        )
        path = within(root, Path(record["path"]))
        manifest_path = within(root, path / "manifest.json")
        verify_bytes(manifest_path, record["manifest_sha256"])
        manifest = json.loads(manifest_path.read_text())
        files[manifest_path] = record["manifest_sha256"]
        hashes[stage] = record["manifest_sha256"]
        for name, digest in manifest["output_sha256"].items():
            if Path(name).name != name or not name.endswith(".parquet"):
                raise ValueError("Unexpected artifact filename")
            file = within(root, path / name)
            verify_bytes(file, digest)
            files[file] = digest
        tables[stage] = LOADERS[stage](path)
        manifests[stage] = manifest
    c, a, d, r = (manifests[n] for n in ("canonical", "atlas", "discovery", "recommendations"))
    ingestion = c["specification"]["ingestion_manifest"]
    _require(c["specification"]["canonical_contract_version"] == 1)
    _require(
        ingestion["snapshot_id"] == spec["snapshot_id"] == fingerprint(ingestion["specification"])
    )
    serialized = json.dumps(ingestion, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _require(hashlib.sha256(serialized.encode()).hexdigest() == spec["snapshot_manifest_sha256"])
    _require(
        all(
            aware(i["imported_at"]) <= aware(spec["as_of"])
            for i in ingestion["specification"]["imports"]
        )
    )
    if ingestion["specification"]["as_of"]:
        _require(aware(ingestion["specification"]["as_of"]) <= aware(spec["as_of"]))
    for manifest in (a, d, r):
        _require(manifest["specification"]["policy"]["contract_version"] == 1)
    _require(a["specification"]["canonical_bundle_id"] == c["bundle_id"])
    _require(a["specification"]["canonical_manifest_sha256"] == hashes["canonical"])
    _require(a["specification"]["ingestion_snapshot_id"] == spec["snapshot_id"])
    _require(d["specification"]["atlas_id"] == a["atlas_id"])
    _require(d["specification"]["atlas_manifest_sha256"] == hashes["atlas"])
    _require(d["specification"]["canonical_bundle_id"] == c["bundle_id"])
    _require(r["specification"]["source_report_id"] == d["report_id"])
    _require(r["specification"]["atlas_id"] == a["atlas_id"])
    _require(r["specification"]["canonical_bundle_id"] == c["bundle_id"])
    _require(
        r["specification"]["source_manifest_sha256"]
        == {
            "canonical": hashes["canonical"],
            "atlas": hashes["atlas"],
            "report": hashes["discovery"],
        }
    )
    for stage, stage_tables in tables.items():
        version_column = {
            "canonical": "canonical_contract_version",
            "atlas": "atlas_contract_version",
            "discovery": "discovery_contract_version",
            "recommendations": "recommendation_contract_version",
            "experiments": "experiment_contract_version",
        }[stage]
        parent_columns = {
            "canonical": {"ingestion_snapshot_id": spec["snapshot_id"]},
            "atlas": {
                "canonical_bundle_id": c["bundle_id"],
                "ingestion_snapshot_id": spec["snapshot_id"],
            },
            "discovery": {"atlas_id": a["atlas_id"]},
            "recommendations": {"recommendation_bundle_id": r["bundle_id"]},
            "experiments": {"canonical_bundle_id": c["bundle_id"]},
        }[stage]
        for frame in stage_tables.values():
            _require(version_column in frame and frame[version_column].eq(1).all())
            for column, identity in parent_columns.items():
                _require(column in frame and frame[column].eq(identity).all())
    if "experiments" in manifests:
        e = manifests["experiments"]["specification"]
        _require(
            e["canonical_bundle_id"] == c["bundle_id"]
            and e["canonical_manifest_sha256"] == hashes["canonical"]
        )
        _require(aware(e["as_of"]) == aware(spec["as_of"]))
        _require(e["events"] == spec["experiment_events"])
    else:
        _require(spec["experiment_events"] is None)
    snapshot = PublishedRun(result, manifests, tables, files, root.resolve())
    snapshot.verify()
    return snapshot


def operational_attempts(root: Path) -> list[dict]:
    """Return allowlisted operational metadata, never treat attempts as research results."""
    attempts = root / "attempts"
    if not attempts.exists():
        return []
    within(root, attempts)
    allowed = {
        "completed",
        "completed_historical",
        "missing_input",
        "no_new_source_content",
        "stale_input",
        "freshness_unverified",
        "failed",
    }
    rows = []
    for path in sorted(attempts.iterdir()):
        if not re.fullmatch(r"[0-9a-f]{32}", path.name):
            continue
        outcome = within(root, path / "outcome.json")
        try:
            record = json.loads(outcome.read_text())
            status = record.get("status")
            rows.append(
                {
                    "attempt_id": path.name,
                    "status": status if status in allowed else "unrecognized",
                    "run_id": record.get("run_id")
                    if ID.fullmatch(str(record.get("run_id")))
                    else None,
                }
            )
        except (OSError, ValueError, TypeError):
            rows.append(
                {"attempt_id": path.name, "status": "incomplete_or_unreadable", "run_id": None}
            )
    return rows


def catalogue_labels(root: Path, zone="America/Chicago") -> dict[str, str]:
    """Newest recorded publication first; operational times never imply source freshness."""
    import pandas as pd

    from oddsjam_ev.dashboard.labels import local_time

    completed = {}
    attempts = root / "attempts"
    if attempts.exists():
        within(root, attempts)
        for directory in attempts.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                record = json.loads(within(root, directory / "outcome.json").read_text())
                time = pd.Timestamp(record["completed_at"])
                if (
                    record["status"] not in {"completed", "completed_historical"}
                    or time.tzinfo is None
                ):
                    continue
                identity = record["run_id"]
                # First successful completion identifies publication, not later replay.
                completed[identity] = min(time, completed.get(identity, time))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    rows = []
    for identity in discover_runs(root):
        try:
            directory = within(root, root / "results" / identity)
            within(root, directory / "manifest.json")
            within(root, directory / "report.md")
            record = load_result(directory)
            if record["run_id"] != identity:
                raise ValueError("Catalogue identity mismatch")
            cutoff = pd.Timestamp(record["specification"]["as_of"])
            if cutoff.tzinfo is None:
                raise ValueError("Unconfirmed cutoff")
            stamp = completed.get(identity)
            label = (
                "Generated " + local_time(stamp, zone)
                if stamp is not None
                else "Generation time unavailable · Cutoff " + local_time(cutoff, zone)
            )
            rows.append((stamp if stamp is not None else cutoff, identity, label))
        except (OSError, ValueError, KeyError, TypeError):
            rows.append(
                (pd.Timestamp.min.tz_localize("UTC"), identity, "Report metadata unavailable")
            )
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    labels = {}
    seen = {}
    for _, identity, label in rows:
        seen[label] = seen.get(label, 0) + 1
        labels[identity] = label + (f" · version {seen[label]}" if seen[label] > 1 else "")
    return labels
