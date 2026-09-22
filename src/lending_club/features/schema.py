"""Data contract for the processed splits.

A schema turns assumptions into checks that fail loudly. Without it, a broken
upstream file shows up much later as a strangely bad model score, and it can
take hours to trace. Here it fails immediately, naming the column.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from lending_club.config import load_params


def processed_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "target": pa.Column(pl.Int8, pa.Check.isin([0, 1])),
            "issue_date": pa.Column(pl.Date, nullable=False),
            "loan_amnt": pa.Column(float, pa.Check.in_range(500, 40_000)),
            "int_rate": pa.Column(float, pa.Check.in_range(0, 40)),
            "installment": pa.Column(float, pa.Check.gt(0)),
            "grade": pa.Column(str, pa.Check.isin(list("ABCDEFG"))),
            # 4 loans from 2007 have no reported income; the pipeline imputes them.
            "annual_inc": pa.Column(float, pa.Check.ge(0), nullable=True),
            # 999 was a "not available" sentinel and is converted to null upstream.
            "dti": pa.Column(float, pa.Check.lt(999), nullable=True),
            "fico_mid": pa.Column(float, pa.Check.in_range(600, 900)),
            "term_months": pa.Column(pl.Int16, pa.Check.eq(36)),
            "realized_profit": pa.Column(float, nullable=False),
        },
        strict=False,  # extra columns are fine; the forbidden ones are checked below
        coerce=False,
    )


def assert_no_forbidden_columns(frame: pl.DataFrame, params: dict | None = None) -> None:
    """The leakage contract (D-008).

    Post-issuance columns may exist for computing realized profit, but nothing
    else from that family is allowed anywhere near the processed data.
    """
    params = params or load_params()
    forbidden = tuple(params["features"]["forbidden_prefixes"])
    allowed = {"total_pymnt", "realized_profit", "funded_amnt"}
    leaked = [c for c in frame.columns if c.startswith(forbidden) and c not in allowed]
    if leaked:
        raise ValueError(f"post-issuance columns found in processed data: {leaked}")


def validate(frame: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    """Raise if the frame breaks the contract; otherwise return it unchanged."""
    assert_no_forbidden_columns(frame, params)
    try:
        processed_schema().validate(frame, lazy=True)
    except (SchemaError, SchemaErrors) as exc:  # pragma: no cover
        raise ValueError(f"schema validation failed: {exc}") from exc
    return frame
