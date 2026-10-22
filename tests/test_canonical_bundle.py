"""End-to-end pinned-source publication, replay, and temporal boundary contracts."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.io.canonical import build_canonical_bundle, load_canonical_bundle
from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.snapshots import materialize_snapshot


def registry():
    return {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [{"id": "a", "name": "TEST - A"}],
    }


def config():
    return CanonicalConfig({"fixed_odds": ["Book"], "exchange": [], "pickem": []})


def source(tmp_path, profit="12", status="won", name="source.csv", duplicate=False):
    row = {
        "created_at": "2026-01-01 12:00:00",
        "sportsbook": "Book",
        "game_id": "g",
        "market_name": "Moneyline",
        "bet_name": "Away",
        "odds": "120",
        "stake": "10",
        "bet_profit": profit,
        "potential_payout": "22",
        "status": status,
        "bet_type": "positive_ev",
        "saved_filter_names": "TEST - A",
        "is_live_bet": "False",
        "is_free_bet": "False",
        "is_odds_boost": "False",
    }
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    if duplicate:
        writer.writerow(row)
    path = tmp_path / name
    path.write_text(stream.getvalue())
    return path


@pytest.mark.filterwarnings("error::FutureWarning")
def test_pinned_bundle_rebuild_and_hashes(tmp_path):
    store = tmp_path / "store"
    import_csv(source(tmp_path), store, SourceMetadata("test"))
    snapshot = materialize_snapshot(store)
    output = build_canonical_bundle(snapshot, tmp_path / "out", config(), registry())
    tables = load_canonical_bundle(output)
    assert len(tables["bet_facts"]) == 1
    assert tables["bet_facts"].iloc[0].reported_roi_eligible
    assert tables["source_audit"].canonical_disposition.tolist() == ["current_single"]
    assert build_canonical_bundle(snapshot, tmp_path / "out", config(), registry()) == output
    assert_frame_equal(tables["bet_facts"], load_canonical_bundle(output)["bet_facts"])
    (output / "bet_facts.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_canonical_bundle(output)


def test_earlier_snapshot_cannot_use_later_settlement(tmp_path):
    store = tmp_path / "store"
    first = import_csv(
        source(tmp_path, profit="0", status="pending"),
        store,
        SourceMetadata("test", "2026-02-01T00:00:00Z"),
    )
    early = materialize_snapshot(store, as_of=first["imported_at"])
    import_csv(
        source(tmp_path, name="later.csv"), store, SourceMetadata("test", "2026-03-01T00:00:00Z")
    )
    late = materialize_snapshot(store)
    earlier = load_canonical_bundle(
        build_canonical_bundle(early, tmp_path / "out", config(), registry())
    )
    later = load_canonical_bundle(
        build_canonical_bundle(late, tmp_path / "out", config(), registry())
    )
    assert earlier["bet_facts"].iloc[0].settlement_status == "pending"
    assert not earlier["bet_facts"].iloc[0].reported_roi_eligible
    assert later["bet_facts"].iloc[0].reported_roi_eligible
    assert len(later["source_audit"]) == 2


def test_all_unresolved_input_publishes_empty_facts(tmp_path):
    store = tmp_path / "store"
    import_csv(source(tmp_path, duplicate=True), store, SourceMetadata("test"))
    snapshot = materialize_snapshot(store)
    path = build_canonical_bundle(snapshot, tmp_path / "out", config(), registry())
    tables = load_canonical_bundle(path)
    assert tables["bet_facts"].empty
    assert len(tables["source_audit"]) == 2
    assert set(tables["source_audit"].canonical_disposition) == {"unresolved_source"}


def test_publication_failure_leaves_no_bundle(tmp_path, monkeypatch):
    import oddsjam_ev.io.canonical as canonical

    store = tmp_path / "store"
    import_csv(source(tmp_path), store, SourceMetadata("test"))
    snapshot = materialize_snapshot(store)

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(canonical, "write_json", fail)
    with pytest.raises(OSError):
        build_canonical_bundle(snapshot, tmp_path / "out", config(), registry())
    assert not list((tmp_path / "out").iterdir())


def test_cli_and_checked_in_registry(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    store = tmp_path / "store"
    import_csv(source(tmp_path), store, SourceMetadata("test"))
    snapshot = materialize_snapshot(store)
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/build_canonical_facts.py"),
            "--snapshot",
            str(snapshot),
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    published = next((tmp_path / "out").glob("*/manifest.json"))
    manifest = json.loads(published.read_text())
    assert manifest["summary"]["table_rows"]["source_audit"] == 1
    assert manifest["specification"]["economic_policy"]["currency"] == "USD"
    from oddsjam_ev.analysis.filter_attribution import validate_registry

    validate_registry(
        yaml.safe_load(
            (
                repo / "configs/filter_history/oddsjam_filter_configurations_2026-10-22.yaml"
            ).read_text()
        )
    )


def test_feature_catalog_does_not_fabricate_optional_source_columns(tmp_path):
    store = tmp_path / "store"
    import_csv(source(tmp_path), store, SourceMetadata("test"))
    snapshot = materialize_snapshot(store)
    tables = load_canonical_bundle(
        build_canonical_bundle(snapshot, tmp_path / "out", config(), registry())
    )
    catalog = tables["feature_catalog"].set_index("source_column")
    assert not catalog.loc["clv", "column_present"]
    assert "clv" not in tables["source_features"]


def test_configuration_change_creates_new_immutable_bundle(tmp_path):
    store = tmp_path / "store"
    import_csv(source(tmp_path), store, SourceMetadata("test"))
    snapshot = materialize_snapshot(store)
    first = build_canonical_bundle(snapshot, tmp_path / "out", config(), registry())
    updated = CanonicalConfig(config().products, profit_tolerance_usd=0.02)
    second = build_canonical_bundle(snapshot, tmp_path / "out", updated, registry())
    assert first != second
    assert len(list((tmp_path / "out").glob("*/manifest.json"))) == 2
