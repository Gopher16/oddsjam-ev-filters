"""Explicit capability publications preserve old economics and reject corrupt history."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical
from test_venue_registry import STAMP, policy, ready_registry

from oddsjam_ev.analysis.venue_registry import resolve_product, seed_from_policy
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.venue_registry import load_venue_registry, publish_venue_registry


def test_replay_revisions_and_canonical_economics_unchanged(tmp_path):
    bundle = canonical(tmp_path)
    old_bytes = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    old_tables = load_canonical_bundle(bundle)
    r, _ = ready_registry()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(r))
    root = tmp_path / "registry"
    output = publish_venue_registry(path, root)
    assert publish_venue_registry(path, root) == output
    assert load_venue_registry(output) == r
    changed = copy.deepcopy(r)
    changed["previous_registry_id"] = fingerprint(r)
    changed["requests"].append(
        {**r["requests"][0], "request_id": "new-request", "name": "New Brand"}
    )
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="predecessor"):
        publish_venue_registry(path, root)
    newer = publish_venue_registry(path, root, previous=output)
    assert newer != output and load_venue_registry(output) == r
    assert load_venue_registry(newer) == changed
    assert publish_venue_registry(path, root, previous=output) == newer
    changed["products"][0]["product_type"] = "unknown"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="append-only"):
        publish_venue_registry(path, root, previous=output)
    assert old_bytes == {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    for name, table in old_tables.items():
        assert_frame_equal(table, load_canonical_bundle(bundle)[name])


def test_corruption_and_failed_staging_do_not_publish(tmp_path, monkeypatch):
    import oddsjam_ev.io.venue_registry as module

    path = tmp_path / "input.json"
    path.write_text(json.dumps(seed_from_policy(policy(), STAMP)))
    root = tmp_path / "out"
    original = module.write_json

    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(module, "write_json", fail)
    with pytest.raises(OSError, match="synthetic"):
        publish_venue_registry(path, root)
    assert not list(root.iterdir())
    monkeypatch.setattr(module, "write_json", original)
    output = publish_venue_registry(path, root)
    with pytest.raises(ValueError, match="inside predecessor"):
        publish_venue_registry(path, output / "nested", previous=output)
    (output / "registry.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_venue_registry(output)
    with pytest.raises(ValueError, match="integrity"):
        publish_venue_registry(path, root)


def test_manifest_specification_and_summary_integrity(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(seed_from_policy(policy(), STAMP)))
    output = publish_venue_registry(path, tmp_path / "out")
    manifest = output / "manifest.json"
    m = json.loads(manifest.read_text())
    m["summary"]["request_count"] = 999
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="coverage"):
        load_venue_registry(output)
    m["specification"]["contract_version"] = True
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="specification"):
        load_venue_registry(output)


def test_cli_private_seed_publish_request_and_lookup(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/manage_venue_registry.py"
    spec = importlib.util.spec_from_file_location("venue_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/manage_venue_registry.py"))
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy()))
    target = tmp_path / "data/processed/private/seed.json"
    args = [
        str(script),
        "seed",
        "--policy",
        str(policy_path),
        "--recorded-at",
        STAMP,
        "--output",
        str(target),
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    capsys.readouterr()
    with pytest.raises(SystemExit) as err:
        module.main()
    assert err.value.code == 2
    monkeypatch.setattr(sys, "argv", args[:-1] + [str(tmp_path / "configs/private.json")])
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path / "configs").exists()
    root = tmp_path / "publications"
    monkeypatch.setattr(
        sys, "argv", [str(script), "publish", "--registry", str(target), "--output-root", str(root)]
    )
    module.main()
    output = Path(capsys.readouterr().out.strip())
    next_input = tmp_path / "data/processed/private/request.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "request",
            "--previous",
            str(output),
            "--request-id",
            "evaluate-new",
            "--name",
            "New Brand",
            "--product-type",
            "exchange",
            "--recorded-at",
            STAMP,
            "--reference",
            "synthetic owner request",
            "--notes",
            "No access or API verified",
            "--output",
            str(next_input),
        ],
    )
    module.main()
    capsys.readouterr()
    next_output = publish_venue_registry(next_input, root, previous=output)
    r = load_venue_registry(next_output)
    got = resolve_product(r, "New Brand", source_id="unknown")
    assert got["status"] == "unknown" and got["request_ids"] == ["evaluate-new"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "lookup",
            "--publication",
            str(output),
            "--venue",
            "DraftKings",
            "--source-id",
            "oddsjam_csv",
        ],
    )
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ambiguous"
    for cmd in ([], ["seed"], ["publish"], ["request"], ["lookup"]):
        result = subprocess.run(
            [sys.executable, str(script), *cmd, "--help"], text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr


def test_repository_policy_seed_preserves_all_exact_mappings():
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs/canonical_facts.yaml").read_text())
    r = seed_from_policy(data, STAMP)
    actual = {(p["product_type"], label) for p in r["products"] for label in p["aliases"]}
    expected = {(kind, label) for kind, labels in data["products"].items() for label in labels}
    assert actual == expected
    assert all(p["contract"]["fee_model"] == "unknown" for p in r["products"])
    assert not r["access"]
