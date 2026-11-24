"""Publication must preserve old readers, economics, history and private destinations."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical
from test_operating_regimes import history

from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.operating_regimes import load_operating_regimes, publish_operating_regimes
from oddsjam_ev.io.sizing_history import load_sizing_history, publish_sizing_history
from oddsjam_ev.io.source_csv import fingerprint


def ledger(tmp_path, bundle):
    h = history()
    h["records"][0].update(filter_id="a", product="fixed_odds")
    ident = load_canonical_bundle(bundle)["bet_facts"].bet_id.iloc[0]
    h["wager_evidence"] = [
        dict(
            evidence_id="receipt",
            bet_id=ident,
            placement_at="2026-11-11T12:00:00Z",
            placement_verified=True,
            recorded_at=h["knowledge_as_of"],
            source_reference="synthetic receipt",
        )
    ]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(h))
    return path, h


def test_replay_revision_and_legacy_reader_economics_unchanged(tmp_path):
    bundle = canonical(tmp_path)
    original = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    tables = load_canonical_bundle(bundle)
    old_sizing = publish_sizing_history(
        bundle, tmp_path / "sizing", knowledge_as_of="2026-11-24T00:00:00Z"
    )
    old_history, old_frame, old_audit = load_sizing_history(old_sizing)
    path, h = ledger(tmp_path, bundle)
    output = publish_operating_regimes(bundle, tmp_path / "operating", path)
    assert publish_operating_regimes(bundle, tmp_path / "operating", path) == output
    stored, frame, audit = load_operating_regimes(output)
    assert stored == h and audit["candidate_count"] == 1
    assert frame.linkage_status.tolist() == ["matched"]
    assert frame.paper_sizing.tolist() == ["kelly_recommendation"]
    for name, before in tables.items():
        assert_frame_equal(before, load_canonical_bundle(bundle)[name])
    assert original == {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    after_history, after_frame, after_audit = load_sizing_history(old_sizing)
    assert old_history == after_history and old_audit == after_audit
    assert_frame_equal(old_frame, after_frame)
    updated = copy.deepcopy(h)
    updated["previous_history_id"] = fingerprint(h)
    updated["records"].append({**h["records"][0], "evidence_id": "conflicting-record"})
    path.write_text(json.dumps(updated))
    with pytest.raises(ValueError, match="Predecessor"):
        publish_operating_regimes(bundle, tmp_path / "operating", path)
    newer = publish_operating_regimes(bundle, tmp_path / "operating", path, previous=output)
    assert newer != output
    assert load_operating_regimes(newer)[1].linkage_status.tolist() == ["ambiguous_regime"]
    assert_frame_equal(frame, load_operating_regimes(output)[1])
    updated["records"][0]["selection_mode"] = "broad_capture"
    path.write_text(json.dumps(updated))
    with pytest.raises(ValueError, match="append-only"):
        publish_operating_regimes(bundle, tmp_path / "operating", path, previous=output)


def test_corruption_overlap_and_atomic_failure(tmp_path, monkeypatch):
    import oddsjam_ev.io.operating_regimes as module

    bundle = canonical(tmp_path)
    path, _ = ledger(tmp_path, bundle)
    with pytest.raises(ValueError, match="overlap"):
        publish_operating_regimes(bundle, bundle / "nested", path)
    root = tmp_path / "operating"
    original_write = module.write_json

    def fail(*args, **kwargs):
        raise OSError("synthetic interrupted publication")

    monkeypatch.setattr(module, "write_json", fail)
    with pytest.raises(OSError, match="interrupted"):
        publish_operating_regimes(bundle, root, path)
    assert not list(root.iterdir())
    monkeypatch.setattr(module, "write_json", original_write)
    output = publish_operating_regimes(bundle, root, path)
    (output / "history.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_operating_regimes(output)
    with pytest.raises(ValueError, match="integrity"):
        publish_operating_regimes(bundle, root, path)


def test_cli_template_private_only_exclusive_and_help(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/manage_operating_regimes.py"
    spec = importlib.util.spec_from_file_location("operating_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/manage_operating_regimes.py"))
    target = tmp_path / "data/processed/private/history.json"
    args = [
        str(script),
        "template",
        "--recorded-at",
        "2026-11-24T00:00:00Z",
        "--output",
        str(target),
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    h = json.loads(target.read_text())
    assert h["records"][0]["effective_from"] == "2026-11-10"
    assert h["records"][0]["effective_interval_verified"] is False
    assert h["wager_evidence"] == []
    with pytest.raises(SystemExit) as err:
        module.main()
    assert err.value.code == 2
    monkeypatch.setattr(sys, "argv", args[:-1] + [str(tmp_path / "configs/public.json")])
    with pytest.raises(SystemExit) as err:
        module.main()
    assert err.value.code == 2 and not (tmp_path / "configs").exists()
    for command in ([], ["template"], ["migrate"], ["publish"]):
        result = subprocess.run(
            [sys.executable, str(script), *command, "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_cli_migration_keeps_original_settings_and_source_untouched(tmp_path, monkeypatch):
    from test_sizing_config import history as sizing_history

    script = Path(__file__).resolve().parents[1] / "scripts/manage_operating_regimes.py"
    spec = importlib.util.spec_from_file_location("operating_migration_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/manage_operating_regimes.py"))
    bundle = canonical(tmp_path)
    old = sizing_history()
    original = tmp_path / "old.json"
    original.write_text(json.dumps(old))
    target = tmp_path / "data/processed/migrated.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "migrate",
            "--canonical",
            str(bundle),
            "--sizing-history",
            str(original),
            "--knowledge-as-of",
            "2026-11-24T00:00:00Z",
            "--output",
            str(target),
        ],
    )
    module.main()
    new = json.loads(target.read_text())
    assert json.loads(original.read_text()) == old
    assert new["records"][0]["settings"]["kelly_multiplier"]["value"] == 0.5
    assert new["records"][0]["funding"] == "unknown"
    assert new["records"][0]["selection_mode"] == "unknown"
    output = publish_operating_regimes(bundle, tmp_path / "operating", target)
    assert load_operating_regimes(output)[1].linkage_status.tolist() == ["unknown_placement"]
