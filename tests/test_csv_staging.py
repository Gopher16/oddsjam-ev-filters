import io
from concurrent.futures import ThreadPoolExecutor

import pytest

from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.staging import load_stage, preview_csv, stage_csv

HEADER = "created_at,sportsbook,market_name,bet_name,odds,stake,status,game_id,bet_type,notes\n"
ROW = "2026-01-01T12:00:00Z,Book,total,over,100,10,pending,g1,single,=SUM(A1)\n"


def stage(tmp_path, data=None, **kwargs):
    return stage_csv(
        io.BytesIO((HEADER + ROW).encode() if data is None else data),
        tmp_path / "stages",
        filename=kwargs.pop("filename", "bets.csv"),
        metadata=kwargs.pop("metadata", SourceMetadata("owner", scope="full")),
        **kwargs,
    )


def test_stage_only_private_repeat(tmp_path):
    receipt = stage(tmp_path)
    assert stage(tmp_path) == receipt
    assert not (tmp_path / "imports").exists()
    assert (tmp_path / "stages").stat().st_mode & 0o077 == 0
    data, _ = load_stage(tmp_path / "stages", receipt["stage_id"])
    assert b"=SUM(A1)" in data
    preview = preview_csv(tmp_path / "stages", receipt["stage_id"], tmp_path / "imports")
    assert preview["candidate_rows"] == dict(new=1, unchanged=0, revised=0, unresolved=0)
    assert preview["formula_like_cells"] == 1


@pytest.mark.parametrize("filename", ["../bets.csv", "x/b.csv", "x\\b.csv", "bets.exe", ".csv"])
def test_bad_names(tmp_path, filename):
    with pytest.raises(ValueError):
        stage(tmp_path, filename=filename)


@pytest.mark.parametrize(
    "data",
    [
        b"bad",
        b"\xff",
        b"a,a\n1,2",
        b"\x00",
        (HEADER + '"bad').encode(),
        (HEADER + "short\n").encode(),
    ],
)
def test_invalid_source(tmp_path, data):
    with pytest.raises(ValueError):
        stage(tmp_path, data)


def test_limits(tmp_path, monkeypatch):
    monkeypatch.setattr("oddsjam_ev.io.staging.MAX_BYTES", 5)
    with pytest.raises(ValueError):
        stage(tmp_path)


def test_shape_limits(tmp_path, monkeypatch):
    monkeypatch.setattr("oddsjam_ev.io.staging.MAX_ROWS", 1)
    with pytest.raises(ValueError):
        stage(tmp_path, (HEADER + ROW + ROW).encode())
    monkeypatch.setattr("oddsjam_ev.io.staging.MAX_COLUMNS", 2)
    with pytest.raises(ValueError):
        stage(tmp_path)


def test_unknown_and_revision(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text(HEADER + ROW)
    import_csv(
        path,
        tmp_path / "imports",
        SourceMetadata("owner", exported_at="2026-01-02T00:00:00Z", scope="full"),
    )
    r = stage(
        tmp_path,
        (HEADER + ROW.replace("pending", "won")).encode(),
        metadata=SourceMetadata("owner", exported_at="2026-01-03T00:00:00Z"),
    )
    p = preview_csv(tmp_path / "stages", r["stage_id"], tmp_path / "imports")
    assert p["coverage_state"] == "unknown_requires_declaration"
    assert p["candidate_rows"]["revised"] == 1
    assert p["settlement_revision_rows"] == 1


def test_corruption_and_path(tmp_path):
    r = stage(tmp_path)
    with pytest.raises(ValueError):
        load_stage(tmp_path / "stages", "../x")
    (tmp_path / "stages" / r["stage_id"] / "source.csv").write_text(
        HEADER + ROW.replace("10,", "20,")
    )
    with pytest.raises(ValueError):
        load_stage(tmp_path / "stages", r["stage_id"])


def test_concurrent_stage(tmp_path):
    def attempt(_):
        try:
            return stage(tmp_path)["stage_id"]
        except ValueError as e:
            assert "locked" in str(e)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert any(results)
    assert len(list((tmp_path / "stages").glob("*/receipt.json"))) == 1


def test_confirm_explicit_idempotent_and_stale(tmp_path):
    from oddsjam_ev.io.staging import confirm_import

    r = stage(tmp_path)
    root, imports = tmp_path / "stages", tmp_path / "imports"
    p = preview_csv(root, r["stage_id"], imports)
    with pytest.raises(ValueError, match="confirmation"):
        confirm_import(root, r["stage_id"], imports, preview_sha256=p["preview_sha256"])
    result = confirm_import(
        root, r["stage_id"], imports, preview_sha256=p["preview_sha256"], confirmed=True
    )
    assert not result["repeated_import"]
    with pytest.raises(ValueError, match="Preview changed"):
        confirm_import(
            root, r["stage_id"], imports, preview_sha256=p["preview_sha256"], confirmed=True
        )
    p = preview_csv(root, r["stage_id"], imports)
    assert p["candidate_rows"]["unchanged"] == 1
    assert confirm_import(
        root, r["stage_id"], imports, preview_sha256=p["preview_sha256"], confirmed=True
    )["repeated_import"]
    assert len(list((imports / "imports").iterdir())) == 1


def test_unknown_confirmation_blocked(tmp_path):
    from oddsjam_ev.io.staging import confirm_import

    r = stage(tmp_path, metadata=SourceMetadata("owner"))
    p = preview_csv(tmp_path / "stages", r["stage_id"], tmp_path / "imports")
    with pytest.raises(ValueError, match="Restage"):
        confirm_import(
            tmp_path / "stages",
            r["stage_id"],
            tmp_path / "imports",
            preview_sha256=p["preview_sha256"],
            confirmed=True,
        )
    assert not (tmp_path / "imports").exists()


def test_import_store_lock_rechecks_receipts(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text(HEADER + ROW)
    import_csv(path, tmp_path / "imports", SourceMetadata("owner", scope="full"))
    with pytest.raises(ValueError, match="store changed"):
        import_csv(
            path, tmp_path / "imports", SourceMetadata("owner", scope="full"), expected_imports=[]
        )


def test_ambiguous_revision_and_namespace(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text(HEADER + ROW)
    import_csv(path, tmp_path / "imports", SourceMetadata("owner", scope="full"))
    r = stage(tmp_path, (HEADER + ROW.replace("pending", "won")).encode())
    assert (
        preview_csv(tmp_path / "stages", r["stage_id"], tmp_path / "imports")["candidate_rows"][
            "unresolved"
        ]
        == 1
    )
    r = stage(tmp_path, metadata=SourceMetadata("other", scope="partial"))
    assert (
        preview_csv(tmp_path / "stages", r["stage_id"], tmp_path / "imports")["candidate_rows"][
            "new"
        ]
        == 1
    )


def test_concurrent_confirm_preserves_single_import(tmp_path):
    from oddsjam_ev.io.staging import confirm_import

    r = stage(tmp_path)
    root, imports = tmp_path / "stages", tmp_path / "imports"
    p = preview_csv(root, r["stage_id"], imports)

    def attempt(_):
        try:
            return confirm_import(
                root, r["stage_id"], imports, preview_sha256=p["preview_sha256"], confirmed=True
            )
        except ValueError as exc:
            assert "locked" in str(exc) or "Preview changed" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(x is not None for x in results) == 1
    assert len(list((imports / "imports").iterdir())) == 1


def test_cli_stage_preview_confirm(tmp_path):
    import json
    import os
    import subprocess
    import sys

    path = tmp_path / "bets.csv"
    path.write_text(HEADER + ROW)
    base = [
        sys.executable,
        "scripts/stage_csv.py",
        "--staging-root",
        str(tmp_path / "stages"),
        "--ingestion-root",
        str(tmp_path / "imports"),
    ]

    def run(args):
        return json.loads(
            subprocess.run(
                base + args,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": "src"},
            ).stdout
        )

    r = run(["stage", str(path), "--namespace", "owner", "--scope", "partial"])
    p = run(["preview", r["stage_id"]])
    assert not run(
        ["confirm", r["stage_id"], "--preview-sha256", p["preview_sha256"], "--confirm-import"]
    )["repeated_import"]


def test_symlink_stage_and_public_directory_rejected(tmp_path):
    r = stage(tmp_path)
    source = tmp_path / "stages" / r["stage_id"] / "source.csv"
    source.unlink()
    target = tmp_path / "target.csv"
    target.write_text(HEADER + ROW)
    source.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_stage(tmp_path / "stages", r["stage_id"])
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private"):
        stage_csv(
            io.BytesIO((HEADER + ROW).encode()),
            public,
            filename="bets.csv",
            metadata=SourceMetadata("owner"),
        )


def test_export_above_previous_byte_limit(monkeypatch):
    import pandas as pd

    from oddsjam_ev.io import staging

    monkeypatch.setattr(staging, "read_source", lambda data: (pd.DataFrame({"id": [1]}), {}))
    frame, _ = staging._parse(b"x" * 28_000_000)
    assert len(frame) == 1
    with pytest.raises(ValueError, match="byte limit"):
        staging._parse(b"x" * (staging.MAX_BYTES + 1))


def test_row_limit_reports_counts(monkeypatch):
    import pandas as pd

    from oddsjam_ev.io import staging

    monkeypatch.setattr(
        staging, "read_source", lambda data: (pd.DataFrame({"id": range(106_974)}), {})
    )
    assert len(staging._parse(b"csv")[0]) == 106_974
    monkeypatch.setattr(staging, "MAX_ROWS", 100_000)
    with pytest.raises(staging.UploadLimitError, match="106,974 rows and 1 columns"):
        staging._parse(b"csv")
