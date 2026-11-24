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
