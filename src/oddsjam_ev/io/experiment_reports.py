"""
experiment_reports.py
===============================================================================
Purpose
-------
Publish versioned experiment outcomes without rewriting earlier knowledge.

What this module provides
-------------------------
Pinned register/canonical reports, integrity verification and atomic replay.

Design Notes
------------
Import time is knowledge time, not settlement or placement time.
===============================================================================
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.experiment_outcomes import build_experiment_outcomes
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.experiment_register import read_experiment_events
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.source_csv import fingerprint

TABLES = (
    "experiment_availability",
    "experiment_protocols",
    "experiment_events",
    "experiment_enrollments",
    "experiment_links",
    "experiment_checks",
    "experiment_outcomes",
    "experiment_metrics",
    "experiment_audit",
)


def load_experiment_report(path: Path) -> dict[str, pd.DataFrame]:
    """Verify all output tables in an explicitly selected report.

    Parameters
    ----------
    path : pathlib.Path
        Published experiment report directory.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Verified tables.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["report_id"]:
        raise ValueError("Experiment report integrity failure")
    result = {}
    names = (
        (*TABLES, "prospective_observations", "prospective_metrics", "prospective_reviews")
        if manifest["specification"].get("prospective_contract_version") == 2
        else TABLES
    )
    for name in names:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = read_parquet_file(path / file)
        if len(result[name]) != manifest["table_rows"][name]:
            raise ValueError("Experiment report count mismatch")
    return result


def publish_experiment_report(
    register: Path, canonical: Path, output_root: Path, as_of: str
) -> Path:
    """Pin system-known experiment events and source observations at a cutoff.

    Parameters
    ----------
    register, canonical, output_root : pathlib.Path
        Explicit register, canonical snapshot and private output destination.
    as_of : str
        Aware knowledge cutoff. Later-imported snapshots are rejected.

    Returns
    -------
    pathlib.Path
        Immutable report with source/evidence hashes and execution policy.
    """
    events = read_experiment_events(register, as_of)
    tables = load_canonical_bundle(canonical)
    manifest = json.loads((canonical / "manifest.json").read_text())
    imports = manifest["specification"]["ingestion_manifest"]["specification"]["imports"]
    if any(aware(i["imported_at"]) > aware(as_of) for i in imports):
        raise ValueError("Canonical snapshot contains knowledge after report cutoff")
    analysis = Path(__file__).resolve().parents[1] / "analysis"
    files = [
        analysis / (n + ".py") for n in ["experiment_config", "experiments", "experiment_outcomes"]
    ] + [
        Path(__file__),
        Path(__file__).with_name("parquet.py"),
        Path(__file__).with_name("experiment_register.py"),
        Path(__file__).with_name("reconcile.py"),
    ]
    from oddsjam_ev.io.reconcile import POLICY

    spec = {
        "as_of": aware(as_of).isoformat(),
        "events": events,
        "canonical_bundle_id": manifest["bundle_id"],
        "canonical_manifest_sha256": hashlib.sha256(
            (canonical / "manifest.json").read_bytes()
        ).hexdigest(),
        "execution_policy": POLICY,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
    }
    bound_ids = {
        e["experiment_id"]
        for e in events
        if e["kind"] == "REGISTER" and e["payload"]["protocol"].get("contract_version") == 2
    }
    if bound_ids:
        from oddsjam_ev.analysis import prospective
        from oddsjam_ev.io import prospective as prospective_io
        from oddsjam_ev.io.experiment_register import utc_now

        if aware(as_of) > aware(utc_now()):
            raise ValueError("Prospective report cutoff cannot be in the future")
        spec["prospective_contract_version"] = 2
        for file in (Path(prospective.__file__), Path(prospective_io.__file__)):
            spec["code_sha256"][str(file.relative_to(analysis.parent))] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
    ident = fingerprint(spec)
    built = build_experiment_outcomes(
        [e for e in events if e["experiment_id"] not in bound_ids], tables, as_of
    )
    if bound_ids:
        built.update(prospective.build_prospective_tables(events, as_of))
        # Preserve a discoverable protocol/event trail alongside separate v2 economics.
        protocols = []
        for event in events:
            if event["experiment_id"] not in bound_ids:
                continue
            if event["kind"] == "REGISTER":
                p = event["payload"]["protocol"]
                protocols.append(
                    {
                        "experiment_id": event["experiment_id"],
                        "status": "see_prospective_reviews",
                        "protocol_sha256": fingerprint(p),
                        "protocol_json": json.dumps(p, sort_keys=True),
                        "experiment_type": p["experiment_type"],
                        "registered_at": event["recorded_at"],
                    }
                )
        built["experiment_protocols"] = pd.concat(
            [built["experiment_protocols"], pd.DataFrame(protocols)], ignore_index=True
        )
        event_rows = [
            {
                "experiment_id": e["experiment_id"],
                "event_id": e["event_id"],
                "kind": e["kind"],
                "recorded_at": e["recorded_at"],
                "payload_json": json.dumps(e["payload"], sort_keys=True),
            }
            for e in events
            if e["experiment_id"] in bound_ids
        ]
        built["experiment_events"] = pd.concat(
            [built["experiment_events"], pd.DataFrame(event_rows)], ignore_index=True
        )
    for frame in built.values():
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].where(frame[column].notna(), None)
        frame["canonical_bundle_id"] = manifest["bundle_id"]
        frame["experiment_contract_version"] = 2 if bound_ids else 1
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_experiment_report(destination)
            for name in built:
                assert_frame_equal(built[name], old[name])
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            hashes = {}
            for name, frame in built.items():
                file = name + ".parquet"
                frame.to_parquet(stage / file, index=False)
                hashes[file] = hashlib.sha256((stage / file).read_bytes()).hexdigest()
            write_json(
                stage / "manifest.json",
                {
                    "report_id": ident,
                    "specification": spec,
                    "output_sha256": hashes,
                    "table_rows": {k: len(v) for k, v in built.items()},
                },
            )
            stage.rename(destination)
    return destination
