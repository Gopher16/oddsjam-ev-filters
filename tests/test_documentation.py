"""Offline documentation contracts; examples are never treated as shell programs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_documentation.py"
spec = importlib.util.spec_from_file_location("documentation_check", SCRIPT)
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def test_local_links_anchors_and_fences(tmp_path):
    """Files, images, directories, references and duplicate Unicode anchors resolve."""
    (tmp_path / "target file.md").write_text("# Café & returns\n\n## Same\n\n## Same\n")
    (tmp_path / "images").mkdir()
    (tmp_path / "images/demo.png").write_bytes(b"synthetic fixture")
    page = tmp_path / "README.md"
    page.write_text(
        "# Home\n[unicode](target%20file.md#café--returns)\n"
        "[duplicate](<target file.md#same-1>)\n![image](images/demo.png)\n"
        "[directory](images/)\n[reference][r]\n[r]: <target file.md#same>\n"
        "[external](https://example.invalid/no-network)\n"
        '```bash\n[not a link](missing.md)\nprintf "not executed"\n```\n'
    )
    assert docs.check_links(tmp_path, [Path("README.md")]) == []
    page.write_text(page.read_text() + "[bad](target%20file.md#absent)\n[missing](no.md)\n")
    errors = docs.check_links(tmp_path, [Path("README.md")])
    assert len(errors) == 2
    assert any("missing heading" in e for e in errors)
    assert any("missing local target" in e for e in errors)


def test_repository_escape_and_explicit_anchors(tmp_path):
    """Confinement applies to symlinks; HTML anchor destinations are accepted."""
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "outside.md").write_text("# Private\n")
    (root / "escape.md").symlink_to(tmp_path / "outside.md")
    (root / "README.md").write_text('<a id="legacy"></a>\n[ok](#legacy)\n[bad](escape.md)\n')
    errors = docs.check_links(root, [Path("README.md")])
    assert len(errors) == 1
    assert "escapes repository" in errors[0]


def test_cli_allowlist_runs_only_help(tmp_path, monkeypatch):
    """Even decision-writing commands receive only --help, with no input payloads."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/manage.py").write_text("raise AssertionError('not executed')")
    (tmp_path / "docs/examples").mkdir(parents=True)
    (tmp_path / "docs/guide.md").write_text("--definition")
    case = {
        "script": "scripts/manage.py",
        "command": ["save"],
        "options": ["--definition"],
        "docs": ["docs/guide.md"],
    }
    manifest = tmp_path / "docs/examples/cli_interfaces.json"
    manifest.write_text(json.dumps([case]))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv[1:] == [str(tmp_path / "scripts/manage.py"), "save", "--help"]
        assert kwargs["cwd"] == tmp_path
        return SimpleNamespace(returncode=0, stdout="--definition", stderr="")

    monkeypatch.setattr(docs.subprocess, "run", fake_run)
    assert docs.check_cli(tmp_path) == []
    case["command"] = ["--definition", "private.json"]
    manifest.write_text(json.dumps([case]))
    assert "Unsafe help subcommand" in docs.check_cli(tmp_path)[0]
    assert len(calls) == 1
    case["script"] = "../outside.py"
    manifest.write_text(json.dumps([case]))
    assert "Unsafe script locator" in docs.check_cli(tmp_path)[0]


def test_missing_cli_options_and_documentation_fail(tmp_path, monkeypatch):
    """A changed argparse interface or stale selected guide fails explicitly."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/example.py").touch()
    (tmp_path / "docs/examples").mkdir(parents=True)
    (tmp_path / "docs/guide.md").write_text("No old option here")
    (tmp_path / "docs/examples/cli_interfaces.json").write_text(
        json.dumps([dict(script="scripts/example.py", options=["--old"], docs=["docs/guide.md"])])
    )
    monkeypatch.setattr(
        docs.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="--new", stderr=""),
    )
    errors = docs.check_cli(tmp_path)
    assert len(errors) == 2
    assert "missing documented option" in errors[0]
    assert "absent from selected guide" in errors[1]


def test_mermaid_needs_prose_and_fences_close():
    assert (
        docs.check_presentation("```mermaid\nflowchart TD\n A --> B\n```\n\nA moves to B.\n") == []
    )
    assert "prose explanation" in docs.check_presentation("```mermaid\nA --> B\n```\n## Next")[0]
    assert docs.check_presentation("~~~python\nprint('fixture')") == ["Unclosed code fence"]


def test_documented_dashboard_fields_use_actual_parser(tmp_path):
    """Selected public attachment keys are checked without loading any publications."""
    import yaml

    from oddsjam_ev.dashboard.artifacts import read_dashboard_config

    fields = {
        "artifact_root": str(tmp_path / "reports"),
        "evidence_root": str(tmp_path / "evidence"),
        "decision_records_publication": None,
        "paper_definitions_publication": None,
        "prospective_report_publication": None,
        "lifecycle_report_publication": None,
        "evidence_publications": {},
        "workspace_publications": {},
        "explanation_publications": {},
        "min_analyzed_wagers": 100,
        "max_display_rows": 200,
    }
    guide = (SCRIPT.parents[1] / "docs/CONFIGURATION_PUBLICATION_MAP.md").read_text()
    assert all(key in guide for key in fields)
    path = tmp_path / "dashboard.yaml"
    path.write_text(yaml.safe_dump(fields))
    config = read_dashboard_config(path)
    assert all(getattr(config, key) == value for key, value in fields.items())
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "evidence").exists()


def test_main_ignores_untracked_private_notes(tmp_path, monkeypatch):
    """The scanner uses Git's explicit file list rather than walking local notes."""
    (tmp_path / "README.md").write_text("# Fixture\n")
    (tmp_path / "private-roadmap.md").write_bytes(b"\xff private fixture: do not read")
    monkeypatch.setattr(docs, "ROOT", tmp_path)
    monkeypatch.setattr(docs.sys, "argv", ["check_documentation.py"])
    monkeypatch.setattr(docs.subprocess, "check_output", lambda *a, **kw: b"README.md\0")
    assert docs.main() == 0
