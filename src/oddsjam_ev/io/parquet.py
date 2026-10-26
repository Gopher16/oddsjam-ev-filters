"""
parquet.py
===============================================================================
Purpose
-------
Read local immutable Parquet files without Python-backed dataset worker lifetimes.

What this module provides
-------------------------
A synchronous native-file reader with explicit resource closure.

Design Notes
------------
Avoid pandas.read_parquet's Python file-handle path at interpreter shutdown.
Artifact integrity checks remain the responsibility of the calling bundle loader.
===============================================================================
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def read_parquet_file(path: Path) -> pd.DataFrame:
    """Read one local file, preserving pandas metadata and index semantics.

    Parameters
    ----------
    path : pathlib.Path
        Local Parquet file, not a partitioned dataset or remote URL.

    Returns
    -------
    pandas.DataFrame
        Fully materialized data with no asynchronous read or conversion workers.
    """
    with pq.ParquetFile(str(path), pre_buffer=False, memory_map=False) as source:
        return source.read(use_threads=False).to_pandas(use_threads=False)
