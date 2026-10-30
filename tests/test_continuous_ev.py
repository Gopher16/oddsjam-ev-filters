"""Continuous EV survives parsing/publication and exact endpoints remain distinct."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical

from oddsjam_ev.analysis.continuous_ev import build_ev_tables, ev_threshold, parse_ev
from oddsjam_ev.analysis.opportunity_atlas import numeric_band
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.continuous_ev import load_continuous_ev, publish_continuous_ev
from oddsjam_ev.io.load import load_bet_tracker_csv, normalize_bet_tracker_frame


def test_exact_thresholds_units_and_float_collision():
    tiny = "3.000000000000000000000000000001"
    values = pd.Series(["2.999", "3", "3.001", tiny, "3.125", " 3.0000 "])
    parsed = parse_ev(values)
    assert ev_threshold(values, ">", "3").tolist() == [False, False, True, True, True, False]
    assert ev_threshold(values, ">=", "3").tolist() == [False, True, True, True, True, True]
    assert ev_threshold(values, "==", "3").tolist() == [False, True, False, False, False, True]
    assert ev_threshold(values, "<", "3").tolist() == [True, False, False, False, False, False]
    assert ev_threshold(values, "<=", "3").tolist() == [True, True, False, False, False, True]
    assert parsed.loc[3, "advertised_ev_percent"] == 3.0
    assert parsed.loc[3, "advertised_ev_percent_exact"] == tiny
    assert parsed.loc[3, "advertised_ev_fraction_exact"] == "0.03000000000000000000000000000001"
    assert Decimal(parsed.loc[4, "advertised_ev_fraction_exact"]) == Decimal("0.03125")
    assert parsed.loc[5, "source_ev_raw"] == " 3.0000 "
    assert numeric_band(values, (0, 2.5, 5)).iloc[:4].nunique() == 1
    assert parsed.advertised_ev_percent.dtype == "Float64"


def test_quality_states_and_boundaries_do_not_match():
    values = pd.Series(
        [
            None,
            "",
            " ",
            "<conflict>",
            "NaN",
            "inf",
            "-Infinity",
            "3%",
            "1_0",
            "oops",
            True,
            "1e9999",
            "1e-9999",
        ]
    )
    parsed = parse_ev(values)
    assert parsed.ev_status.tolist() == ["missing"] * 3 + ["conflict"] + ["invalid"] * 9
    assert not ev_threshold(values, ">=", "-100").any()
    assert parsed.advertised_ev_percent.isna().all()
    assert ev_threshold(pd.Series(["-1", "0", "+3.0", "3e0"]), ">=", "0").tolist() == [
        False,
        True,
        True,
        True,
    ]
    assert (
        parse_ev(pd.Series([3.1])).source_representation.iloc[0]
        == "typed_input_original_lexeme_unavailable"
    )
    assert parse_ev(pd.Series([], dtype=object)).empty
    assert ev_threshold(pd.Series([], dtype=object), ">", "3").empty


@pytest.mark.parametrize(
    "op,endpoint", [("!=", "3"), (">", "inf"), (">", "3%"), (">", 3), (">", ""), (">", "1_0")]
)
def test_invalid_thresholds(op, endpoint):
    with pytest.raises(ValueError):
        ev_threshold(pd.Series(["3"]), op, endpoint)


def test_legacy_reader_preserves_lexeme_and_numeric_preview(tmp_path):
    text = "3.000000000000000000000000000001"
    path = tmp_path / "source.csv"
    path.write_text("percentage,stake\n" + text + ",100\n,100\ninvalid,100\n")
    frame = load_bet_tracker_csv(path)
    assert frame.percentage_source_raw.tolist() == [text, "", "invalid"]
    assert frame.percentage.iloc[0] == 3
    assert frame.percentage.iloc[1:].isna().all()
    assert ev_threshold(frame.percentage_source_raw, ">", "3").tolist() == [True, False, False]
    original = frame.copy(deep=True)
    assert_frame_equal(normalize_bet_tracker_frame(frame), frame)
    assert_frame_equal(frame, original)


def test_verified_canonical_publication_replay_cli_and_source_preservation(tmp_path):
    bundle = canonical(tmp_path)
    before = {p: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    data = load_canonical_bundle(bundle)
    output = publish_continuous_ev(bundle, tmp_path / "ev")
    assert publish_continuous_ev(bundle, tmp_path / "ev") == output
    views = load_continuous_ev(output)
    assert (
        views["source_ev"].source_ev_raw.tolist()
        == data["source_features"]
        .reindex(columns=["percentage"])
        .percentage.astype(object)
        .where(lambda x: x.notna(), None)
        .tolist()
    )
    assert (
        views["wager_ev"].reported_roi_eligible.tolist()
        == data["bet_facts"].reported_roi_eligible.tolist()
    )
    assert all(p.read_bytes() == v for p, v in before.items())
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/build_continuous_ev.py"),
            "--canonical",
            str(bundle),
            "--output-root",
            str(tmp_path / "ev"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="source bundle"):
        publish_continuous_ev(bundle, bundle / "ev")
    (output / "source_ev.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_continuous_ev(output)


def test_ticket_ev_never_inherits_leg_ev_and_invalid_provenance(tmp_path):
    data = load_canonical_bundle(canonical(tmp_path))
    bets = data["bet_facts"]
    bets.loc[bets.index[0], "fact_grain"] = "ticket_candidate"
    bets.loc[bets.index[0], "percentage"] = "3.125"
    source_before = data["source_features"].copy(deep=True)
    result = build_ev_tables(data)
    assert result["wager_ev"].ev_status.iloc[0] == "unavailable_ticket_model"
    assert pd.isna(result["wager_ev"].advertised_ev_percent.iloc[0])
    assert result["wager_ev"].advertised_ev_percent_exact.iloc[0] is None
    assert_frame_equal(data["source_features"], source_before)
    data["source_features"] = pd.concat([data["source_features"], data["source_features"]])
    with pytest.raises(ValueError, match="provenance"):
        build_ev_tables(data)


def test_manifest_tampering(tmp_path):
    output = publish_continuous_ev(canonical(tmp_path), tmp_path / "ev")
    path = output / "manifest.json"
    data = json.loads(path.read_text())
    data["specification"]["ev_contract"]["units"] = "fraction"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        load_continuous_ev(output)


def test_source_text_to_canonical_to_typed_parquet_exact_roundtrip(tmp_path):
    import csv

    from test_canonical_bundle import config, registry, source

    from oddsjam_ev.io.canonical import build_canonical_bundle
    from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
    from oddsjam_ev.io.snapshots import materialize_snapshot

    csv_path = source(tmp_path)
    with csv_path.open() as stream:
        template = next(csv.DictReader(stream))
    values = ["2.999", "3", "3.001", "3.000000000000000000000000000001"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*template, "percentage"])
        writer.writeheader()
        for i, value in enumerate(values):
            writer.writerow({**template, "bet_name": f"Selection {i}", "percentage": value})
    store = tmp_path / "store"
    import_csv(csv_path, store, SourceMetadata("synthetic_ev"))
    snapshot = materialize_snapshot(store)
    bundle = build_canonical_bundle(snapshot, tmp_path / "canonical", config(), registry())
    tables = load_canonical_bundle(bundle)
    assert set(tables["source_features"].percentage) == set(values)
    assert set(tables["bet_facts"].percentage) == set(values)
    views = load_continuous_ev(publish_continuous_ev(bundle, tmp_path / "ev"))
    assert set(views["source_ev"].source_ev_raw) == set(values)
    assert set(views["wager_ev"].advertised_ev_percent_exact) == set(values)
    assert ev_threshold(views["wager_ev"].advertised_ev_percent_exact, ">", "3").sum() == 2
    assert ev_threshold(views["wager_ev"].advertised_ev_percent_exact, ">=", "3").sum() == 3


def test_both_preprocessing_routes_preserve_exact_ev_text(tmp_path):
    from test_ingestion import ingest, row

    from oddsjam_ev.io.parquet import read_parquet_file
    from oddsjam_ev.io.preprocess import run_preprocessing
    from oddsjam_ev.io.snapshots import materialize_snapshot

    exact = "3.000000000000000000000000000001"
    store = tmp_path / "store"
    ingest(tmp_path, store, [row(status="won", bet_profit="12.0", percentage=exact)], zone="UTC")
    snapshot = materialize_snapshot(store)
    cfg = {
        "raw_csv_path": str(tmp_path / "source.csv"),
        "output_dir": str(tmp_path / "legacy"),
        "as_of_date": "03-21-2026",
        "timestamps": {"assume_tz_for_naive": "UTC"},
        "opportunity": {"enabled": True},
        "liquidity": {},
        "ev_features": {"enabled": False},
        "status_settlement": {"enabled": True},
    }
    run_preprocessing(cfg)
    run_preprocessing({**cfg, "output_dir": str(tmp_path / "derived")}, ingestion_snapshot=snapshot)
    paths = list((tmp_path / "legacy").rglob("*.parquet")) + list(
        (tmp_path / "derived").rglob("*.parquet")
    )
    assert paths
    for path in paths:
        frame = read_parquet_file(path)
        if "percentage" in frame:
            assert frame.percentage_source_raw.tolist() == [exact]
            assert ev_threshold(frame.percentage_source_raw, ">", "3").tolist() == [True]
