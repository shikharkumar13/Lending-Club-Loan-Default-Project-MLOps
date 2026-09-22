"""Stage 1: raw CSV -> parquet, keeping only the columns we are allowed to use.

Why a separate stage: the raw file is a 1.6 GB CSV with 151 columns. Reading it
takes minutes and a lot of memory. We read it once, keep the allowlisted
columns plus a few bookkeeping ones, and write parquet. Everything downstream
reads that parquet in seconds.

Why the column selection happens HERE: the forbidden post-issuance columns
never even enter the pipeline, so they cannot leak by accident later (D-006).
"""

from __future__ import annotations

import polars as pl

from lending_club.config import load_params, path_of


def ingest() -> pl.DataFrame:
    params = load_params()
    features = params["features"]
    columns = features["allowlist"] + features["bookkeeping"]

    src, dst = path_of("raw_csv"), path_of("raw_parquet")
    dst.parent.mkdir(parents=True, exist_ok=True)

    frame = (
        # scan_csv is lazy: with .select() below, polars reads only these columns
        pl.scan_csv(src, infer_schema_length=10_000, ignore_errors=True)
        .select(columns)
        # The file ends with a few summary rows that have no loan_status.
        .filter(pl.col("loan_status").is_not_null())
        .collect()
    )
    frame.write_parquet(dst)
    print(f"ingest: {frame.height:,} rows x {frame.width} cols -> {dst}")
    return frame


if __name__ == "__main__":
    ingest()
