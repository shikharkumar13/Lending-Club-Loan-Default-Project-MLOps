"""Tests for the data contract (D-037) and the missingness guard."""

import copy
from datetime import date

import polars as pl
import pytest

from lending_club.config import load_params
from lending_club.features.schema import assert_no_forbidden_columns, check_missingness, validate

PARAMS = load_params()


def valid_frame(n: int = 10, **overrides) -> pl.DataFrame:
    data = {
        "target": [0] * n,
        "issue_date": [date(2013, 6, 1)] * n,
        "loan_amnt": [10_000.0] * n,
        "int_rate": [12.0] * n,
        "installment": [330.0] * n,
        "grade": ["B"] * n,
        "annual_inc": [60_000.0] * n,
        "dti": [15.0] * n,
        "fico_mid": [692.0] * n,
        "term_months": [36] * n,
        "realized_profit": [1_500.0] * n,
    }
    data.update({k: ([v] * n if not isinstance(v, list) else v) for k, v in overrides.items()})
    return pl.DataFrame(data).with_columns(
        pl.col("target").cast(pl.Int8), pl.col("term_months").cast(pl.Int16)
    )


def test_valid_frame_passes():
    assert validate(valid_frame()).height == 10


def test_impossible_interest_rate_is_rejected():
    with pytest.raises(ValueError, match="schema validation failed"):
        validate(valid_frame(int_rate=99.0))


def test_sixty_month_loan_is_rejected():
    """v1 is 36-month loans only; a 60-month loan means an upstream bug (D-004)."""
    with pytest.raises(ValueError, match="schema validation failed"):
        validate(valid_frame(term_months=60))


def test_forbidden_column_is_rejected():
    frame = valid_frame().with_columns(pl.lit(1.0).alias("last_pymnt_amnt"))
    with pytest.raises(ValueError, match="post-issuance columns"):
        assert_no_forbidden_columns(frame, PARAMS)


def test_missingness_guard_flags_a_mostly_empty_column():
    params = copy.deepcopy(PARAMS)
    params["features"]["groups"] = {
        "skewed_amount": [],
        "numeric": ["dti"],
        "numeric_with_indicator": [],
        "nominal": [],
        "structural_missing": [],
    }
    mostly_empty = valid_frame(n=10, dti=[15.0] + [None] * 9)
    with pytest.raises(ValueError, match="max_missing_fraction"):
        check_missingness(mostly_empty, params)


def test_structural_columns_are_exempt_from_the_guard():
    """mths_since_last_record is 90% missing by design and must not trip the guard."""
    params = copy.deepcopy(PARAMS)
    params["features"]["groups"] = {
        "skewed_amount": [],
        "numeric": ["mths_since_last_record"],
        "numeric_with_indicator": [],
        "nominal": [],
        "structural_missing": ["mths_since_last_record"],
    }
    frame = valid_frame(n=10, mths_since_last_record=[12.0] + [None] * 9)
    check_missingness(frame, params)  # must not raise
