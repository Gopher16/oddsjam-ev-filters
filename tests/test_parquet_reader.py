"""Native single-file reads preserve data and allow normal subprocess teardown."""

import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.io.parquet import read_parquet_file


@pytest.mark.parametrize("empty", [False, True])
def test_native_reader_preserves_pandas_metadata_and_values(tmp_path, empty):
    frame = pd.DataFrame(
        {
            "category": pd.Categorical(["a", "b"]),
            "nullable": pd.array([1, None], dtype="Int64"),
            "time": pd.to_datetime(["2030-01-01", None], utc=True),
            "text": ["value", None],
        },
        index=pd.Index([4, 7], name="row"),
    )
    if empty:
        frame = frame.iloc[:0]
    path = tmp_path / "table.parquet"
    frame.to_parquet(path)
    # Empty Parquet dictionaries may not retain unused category values. Match
    # the existing pandas read semantics, including this storage limitation.
    assert_frame_equal(pd.read_parquet(path), read_parquet_file(path))
    if not empty:
        assert_frame_equal(frame, read_parquet_file(path))


def test_native_reader_repeated_fresh_process_exit(tmp_path):
    path = tmp_path / "table.parquet"
    pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]}).to_parquet(path, index=False)
    code = "from pathlib import Path; from oddsjam_ev.io.parquet import read_parquet_file; import sys; f=read_parquet_file(Path(sys.argv[1])); assert f.a.sum()==3; print('complete')"
    for _ in range(12):
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "complete"


def test_native_reader_rejects_corrupt_file(tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(pa.ArrowInvalid, match="Parquet|parquet|size"):
        read_parquet_file(path)
