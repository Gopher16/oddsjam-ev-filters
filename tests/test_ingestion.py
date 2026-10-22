"""Synthetic contracts for immutable imports, conservative reconciliation, and replay."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.io.ingestion import SourceMetadata, import_csv, load_observations
from oddsjam_ev.io.manifests import writer_lock
from oddsjam_ev.io.reconcile import REAL_FILTER, classify_execution, reconcile
from oddsjam_ev.io.snapshots import load_snapshot, materialize_snapshot
from oddsjam_ev.io.source_csv import read_source


@pytest.fixture
def store(tmp_path):
    return tmp_path / "private"


def row(**updates):
    return {
        "created_at": "2026-01-01 12:00:00",
        "sportsbook": "Book",
        "game_id": "game-1",
        "event_name": "Event",
        "market_name": "Moneyline",
        "bet_name": "Away",
        "odds": "120",
        "stake": "10.0",
        "bet_type": "positive_ev",
        "parlay_id": "",
        "status": "pending",
        "bet_profit": "-10.0",
        "saved_filter_names": REAL_FILTER,
        "notes": "NA",
        **updates,
    }


def write_csv(tmp_path, rows, name="source.csv"):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = tmp_path / name
    path.write_bytes(stream.getvalue().encode())
    return path


def ingest(
    tmp_path,
    store,
    rows,
    *,
    name="source.csv",
    exported="2026-02-01T12:00:00Z",
    zone="unknown",
    namespace="test",
):
    return import_csv(
        write_csv(tmp_path, rows, name), store, SourceMetadata(namespace, exported, zone)
    )


def test_lossless_and_idempotent(tmp_path, store):
    original = row(notes='NA, "quoted"\nsecond line', extra="NULL")
    path = write_csv(tmp_path, [original])
    result = import_csv(path, store, SourceMetadata("test"))
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(path.read_bytes())
    repeated = import_csv(renamed, store, SourceMetadata("test"))
    assert repeated["repeated_import"]
    assert result["import_id"] == repeated["import_id"]
    frame, manifests = load_observations(store)
    assert len(frame) == len(manifests) == 1
    for key, value in original.items():
        assert frame.loc[0, key] == value
    assert result["diagnostics"]["added_columns"] == ["extra"]
    assert (store / "objects" / f"{result['source_sha256']}.csv").read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match="different metadata"):
        import_csv(path, store, SourceMetadata("test", timestamp_timezone="UTC"))


@pytest.mark.parametrize(
    "data, match",
    [
        (b"", "header"),
        (b"a,a\n1,2\n", "duplicate"),
        (b"a,b\n1,2\n", "required"),
        (b"\xff", "UTF-8"),
        (b'created_at,sportsbook,market_name,bet_name,odds,stake,status\n"bad', "Invalid"),
        (b"created_at,sportsbook,market_name,bet_name,odds,stake,status\n1,2\n", "row 1"),
    ],
)
def test_invalid_sources(data, match):
    with pytest.raises(ValueError, match=match):
        read_source(data)


def test_bad_import_never_publishes(tmp_path, store):
    path = tmp_path / "bad.csv"
    path.write_text("a\n1\n")
    with pytest.raises(ValueError):
        import_csv(path, store, SourceMetadata("test"))
    assert not (store / "imports").exists()


def test_multiplicity_and_distinct_executions(tmp_path, store):
    ingest(tmp_path, store, [row(), row(), row(created_at="2026-01-01 12:01:00")])
    observations, _ = load_observations(store)
    current, audit, summary = reconcile(observations)
    assert len(audit) == 3
    assert len(current) == 1
    assert summary["unresolved_observations"] == 2
    assert set(audit._ingest_reason) == {"", "within_source_identity_collision"}


def test_revision_and_rebuild(tmp_path, store):
    ingest(tmp_path, store, [row()], zone="UTC")
    ingest(
        tmp_path,
        store,
        [row(status="won", bet_profit="12.0", notes="updated")],
        name="later.csv",
        exported="2026-03-01T12:00:00Z",
        zone="UTC",
    )
    observations, _ = load_observations(store)
    current, audit, summary = reconcile(observations)
    assert current.iloc[0].status == "won"
    assert sorted(audit._ingest_revision) == [1, 2]
    assert summary["dispositions"] == {"new": 1, "revised": 1}
    path = materialize_snapshot(store)
    assert materialize_snapshot(store) == path
    assert_frame_equal(current, load_snapshot(path))
    assert summary == json.loads((path / "manifest.json").read_text())["summary"]


def test_out_of_order_export_uses_source_chronology(tmp_path, store):
    ingest(tmp_path, store, [row(status="won")], exported="2026-03-01T00:00:00Z")
    ingest(tmp_path, store, [row()], name="older.csv", exported="2026-02-01T00:00:00Z")
    current, _, _ = reconcile(load_observations(store)[0])
    assert current.iloc[0].status == "won"


@pytest.mark.parametrize("exported", [None, "2026-02-01T12:00:00Z"])
def test_conflicting_payload_requires_chronology(tmp_path, store, exported):
    ingest(tmp_path, store, [row()], exported=exported)
    ingest(tmp_path, store, [row(status="won")], exported=exported, name="later.csv")
    current, audit, summary = reconcile(load_observations(store)[0])
    assert current.empty
    assert len(audit) == summary["unresolved_observations"] == 2


def test_unknown_timezone_shift_stays_unresolved(tmp_path, store):
    ingest(tmp_path, store, [row()])
    ingest(
        tmp_path,
        store,
        [row(created_at="2026-01-01 07:00:00")],
        name="later.csv",
        exported="2026-03-01T12:00:00Z",
    )
    current, audit, _ = reconcile(load_observations(store)[0])
    assert current.empty
    assert set(audit._ingest_reason) == {"uncertain_timestamp_overlap"}


def test_confirmed_timezones_can_match(tmp_path, store):
    ingest(tmp_path, store, [row()], zone="UTC")
    ingest(
        tmp_path,
        store,
        [row(created_at="2026-01-01 07:00:00", status="won")],
        name="later.csv",
        zone="America/New_York",
        exported="2026-03-01T12:00:00Z",
    )
    current, _, _ = reconcile(load_observations(store)[0])
    assert len(current) == 1
    assert current.iloc[0].status == "won"


def test_namespaces_never_merge(tmp_path, store):
    ingest(tmp_path, store, [row()], namespace="one")
    ingest(tmp_path, store, [row()], namespace="two")
    current, _, _ = reconcile(load_observations(store)[0])
    assert len(current) == 2


def test_as_of_blocks_later_imports(tmp_path, store):
    first = ingest(tmp_path, store, [row()])
    ingest(tmp_path, store, [row(status="won")], name="later.csv", exported="2026-03-01T12:00:00Z")
    observations, imports = load_observations(store, as_of=first["imported_at"])
    assert len(imports) == len(observations) == 1
    assert (
        load_snapshot(materialize_snapshot(store, as_of=first["imported_at"])).iloc[0].status
        == "pending"
    )
    assert load_observations(store, as_of="2000-01-01T00:00:00Z")[0].empty


def test_reviewed_resolution_is_persisted(tmp_path, store):
    result = ingest(tmp_path, store, [row(), row()])
    observations, _ = load_observations(store)
    decisions = [
        {
            "observation_id": ident,
            "record_id": str(i),
            "reason": "confirmed separate bets",
            "decided_at": result["imported_at"],
        }
        for i, ident in enumerate(observations._ingest_observation_id)
    ]
    path = materialize_snapshot(store, decisions=decisions)
    assert len(load_snapshot(path)) == 2
    assert json.loads((path / "manifest.json").read_text())["specification"]["decisions"] == sorted(
        decisions, key=lambda d: d["observation_id"]
    )


def test_lock_and_interrupted_staging(tmp_path, store):
    with writer_lock(store), pytest.raises(ValueError, match="locked"):
        ingest(tmp_path, store, [row()])
    abandoned = store / ".stage-interrupted"
    abandoned.mkdir()
    (abandoned / "manifest.json").write_text("broken")
    ingest(tmp_path, store, [row()])
    assert len(load_observations(store)[1]) == 1


def test_corrupt_source_and_snapshot_fail(tmp_path, store):
    result = ingest(tmp_path, store, [row()])
    path = materialize_snapshot(store)
    (path / "current.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        load_snapshot(path)
    (store / "objects" / f"{result['source_sha256']}.csv").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        load_observations(store)


@pytest.mark.parametrize(
    "label,created,expected",
    [
        (REAL_FILTER, "2026-01-01", "real_money"),
        ("PROD - Other", "2026-01-01", "paper"),
        ("TEST - Other", "2026-01-01", "paper"),
        (REAL_FILTER + ", TEST - Other", "2026-01-01", "unknown"),
        ("", "2026-01-01", "unknown"),
        ("PROD - Other", "2026-10-23", "real_money"),
        ("TEST - Other", "2026-10-23", "paper"),
        ("PROD - Other, TEST - Other", "2026-10-23", "unknown"),
        ("Unrecognized", "2026-10-23", "unknown"),
        ("PROD - Other", "bad", "unknown"),
        ("PROD - Other", "2026-10-22", "unknown"),
    ],
)
def test_execution_policy(label, created, expected):
    assert classify_execution(label, created) == expected


def test_missing_identity_and_optional_schema_drift(tmp_path, store):
    original = row(game_id="", event_name="")
    del original["notes"]
    result = ingest(tmp_path, store, [original])
    current, audit, _ = reconcile(load_observations(store)[0])
    assert current.empty
    assert audit.iloc[0]._ingest_reason == "missing_identity_or_invalid_timestamp"
    assert "notes" in result["diagnostics"]["missing_optional_columns"]


def test_cli_import_and_error(tmp_path):
    config = tmp_path / "source.yaml"
    config.write_text(f"store: {tmp_path / 'store'}\nsource:\n  namespace: test\n")
    path = write_csv(tmp_path, [row()])
    script = Path(__file__).resolve().parents[1] / "scripts" / "ingest_bet_tracker.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config), "import", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["diagnostics"]["row_count"] == 1
    bad = subprocess.run(
        [sys.executable, str(script), "--config", str(config), "import", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert "Ingestion failed" in bad.stderr


def test_future_resolution_not_visible_as_of(tmp_path, store):
    result = ingest(tmp_path, store, [row(), row()])
    observations, _ = load_observations(store)
    decisions = [
        {
            "observation_id": observations.iloc[0]._ingest_observation_id,
            "record_id": "reviewed",
            "reason": "reviewed later",
            "decided_at": "2099-01-01T00:00:00Z",
        }
    ]
    path = materialize_snapshot(store, as_of=result["imported_at"], decisions=decisions)
    assert load_snapshot(path).empty
    assert json.loads((path / "manifest.json").read_text())["specification"]["decisions"] == []


def test_pending_not_converted_to_settled(tmp_path, store):
    ingest(tmp_path, store, [row()])
    current, _, summary = reconcile(load_observations(store)[0])
    assert current.iloc[0].status == "pending"
    assert current.iloc[0].bet_profit == "-10.0"
    assert summary["settlement_status"] == {"pending": 1}


def test_revisions_report_changed_fields(tmp_path, store):
    ingest(tmp_path, store, [row()])
    ingest(
        tmp_path,
        store,
        [row(status="won", saved_filter_names="TEST - renamed")],
        name="later.csv",
        exported="2026-03-01T00:00:00Z",
    )
    current, _, _ = reconcile(load_observations(store)[0])
    assert set(json.loads(current.iloc[0]._ingest_changed_fields)) == {
        "status",
        "saved_filter_names",
    }


def test_no_absence_deletion(tmp_path, store):
    ingest(tmp_path, store, [row(), row(game_id="game-2")])
    ingest(tmp_path, store, [row(status="won")], name="later.csv", exported="2026-03-01T00:00:00Z")
    current, _, _ = reconcile(load_observations(store)[0])
    assert len(current) == 2


def test_atomic_preprocessing_bridge_and_snapshot_selection(tmp_path, store, monkeypatch):
    from oddsjam_ev.analysis.snapshot_config import resolve_snapshot_config
    from oddsjam_ev.io.preprocess import run_preprocessing

    ingest(tmp_path, store, [row(status="won", bet_profit="12.0")], zone="UTC")
    snapshot = materialize_snapshot(store)
    output = tmp_path / "data" / "processed"
    config = {
        "raw_csv_path": str(tmp_path / "source.csv"),
        "output_dir": str(output),
        "as_of_date": "03-21-2026",
        "timestamps": {"assume_tz_for_naive": "UTC"},
        "opportunity": {"enabled": True},
        "liquidity": {},
        "ev_features": {"enabled": False},
        "status_settlement": {"enabled": True},
    }
    # Compare the extracted legacy path against the ingestion bridge.
    legacy_config = {**config, "output_dir": str(tmp_path / "legacy")}
    run_preprocessing(legacy_config)
    run_preprocessing(config, ingestion_snapshot=snapshot)
    manifests = list(output.glob("ingestion-*/research-manifest.json"))
    assert len(manifests) == 1
    monkeypatch.setenv("ODDSJAM_RESEARCH_MANIFEST", str(manifests[0]))
    selected = resolve_snapshot_config(project_root=tmp_path)
    assert selected.provenance_path == manifests[0]
    legacy = pd.read_parquet(tmp_path / "legacy" / selected.processed_path.name)
    derived = pd.read_parquet(selected.processed_path)
    assert_frame_equal(legacy, derived[list(legacy.columns)])
    with pytest.raises(ValueError, match="already exists"):
        run_preprocessing(config, ingestion_snapshot=snapshot)
    assert not list(output.glob(".stage-*"))


def test_preprocessing_refuses_unknown_timezone(tmp_path, store):
    from oddsjam_ev.io.preprocess import run_preprocessing

    ingest(tmp_path, store, [row()])
    snapshot = materialize_snapshot(store)
    with pytest.raises(ValueError, match="confirmed source timezone"):
        run_preprocessing({"output_dir": str(tmp_path / "processed")}, ingestion_snapshot=snapshot)
    assert not (tmp_path / "processed").exists()


def test_possible_stake_correction_not_silently_new(tmp_path, store):
    ingest(tmp_path, store, [row()])
    ingest(tmp_path, store, [row(stake="20.0")], name="later.csv", exported="2026-03-01T00:00:00Z")
    current, audit, _ = reconcile(load_observations(store)[0])
    assert current.empty
    assert set(audit._ingest_reason) == {"possible_identity_field_correction"}


def test_resolutions_cannot_predate_related_import(tmp_path, store):
    ingest(tmp_path, store, [row()])
    observations, _ = load_observations(store)
    with pytest.raises(ValueError, match="predate"):
        materialize_snapshot(
            store,
            decisions=[
                {
                    "observation_id": observations.iloc[0]._ingest_observation_id,
                    "record_id": "review",
                    "reason": "review",
                    "decided_at": "2000-01-01T00:00:00Z",
                }
            ],
        )


@pytest.mark.parametrize("created", ["2026-03-08 02:30:00", "2026-11-01 01:30:00"])
def test_dst_ambiguities_are_unresolved(tmp_path, store, created):
    ingest(tmp_path, store, [row(created_at=created)], zone="America/New_York")
    current, audit, _ = reconcile(load_observations(store)[0])
    assert current.empty
    assert audit.iloc[0]._ingest_reason == "missing_identity_or_invalid_timestamp"


def test_interrupted_import_does_not_publish(tmp_path, store, monkeypatch):
    import oddsjam_ev.io.ingestion as ingestion

    original_writer = ingestion.write_json

    def fail(*args, **kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(ingestion, "write_json", fail)
    with pytest.raises(OSError, match="simulated"):
        ingest(tmp_path, store, [row()])
    assert not list((store / "imports").glob("*/manifest.json"))
    assert not list(store.glob(".stage-*"))
    assert not (store / ".writer.lock").exists()
    monkeypatch.setattr(ingestion, "write_json", original_writer)
    assert not ingest(tmp_path, store, [row()])["repeated_import"]


def test_interrupted_snapshot_does_not_publish(tmp_path, store, monkeypatch):
    import oddsjam_ev.io.snapshots as snapshots

    ingest(tmp_path, store, [row()])
    original_writer = snapshots.write_json

    def fail(*args, **kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(snapshots, "write_json", fail)
    with pytest.raises(OSError, match="simulated"):
        materialize_snapshot(store)
    assert not list((store / "snapshots").iterdir())
    assert not list(store.glob(".stage-*"))
    monkeypatch.setattr(snapshots, "write_json", original_writer)
    assert len(load_snapshot(materialize_snapshot(store))) == 1


def test_future_resolution_rejected_for_latest(tmp_path, store):
    ingest(tmp_path, store, [row()])
    observations, _ = load_observations(store)
    with pytest.raises(ValueError, match="future-dated"):
        materialize_snapshot(
            store,
            decisions=[
                {
                    "observation_id": observations.iloc[0]._ingest_observation_id,
                    "record_id": "review",
                    "reason": "review",
                    "decided_at": "2099-01-01T00:00:00Z",
                }
            ],
        )


def test_numeric_drift_and_reserved_headers(tmp_path):
    frame, report = read_source(write_csv(tmp_path, [row(stake="bad", odds="inf")]).read_bytes())
    assert frame.iloc[0].stake == "bad"
    assert report["invalid_numeric_values"]["stake"] == 1
    assert report["invalid_numeric_values"]["odds"] == 1
    with pytest.raises(ValueError, match="reserved"):
        read_source(write_csv(tmp_path, [row(_ingest_fake="x")]).read_bytes())


@pytest.mark.parametrize(
    "updates",
    [
        {"namespace": "private/path"},
        {"namespace": None},
        {"scope": []},
        {"timestamp_timezone": "not-a-zone"},
        {"exported_at": "2026-01-01"},
        {"exported_at": 1},
    ],
)
def test_source_metadata_validation(updates):
    with pytest.raises(ValueError):
        SourceMetadata(**{"namespace": "test", **updates})
