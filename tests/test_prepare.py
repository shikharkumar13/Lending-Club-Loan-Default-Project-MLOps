"""Tests for the cleaning/labeling stage.

These run on a few handmade rows, so they are instant and need no data file.
Each test pins down one rule from DECISIONS.md.
"""

import copy

import polars as pl
import pytest

from lending_club.config import load_params
from lending_club.data.prepare import transform


def _row(**overrides):
    """A minimal valid raw row; tests override only the field under test."""
    base = {
        "loan_amnt": 10000.0,
        "term": " 36 months",
        "int_rate": 12.0,
        "installment": 330.0,
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "3 years",
        "home_ownership": "RENT",
        "annual_inc": 60000.0,
        "verification_status": "Verified",
        "purpose": "credit_card",
        "addr_state": "CA",
        "dti": 15.0,
        "delinq_2yrs": 0.0,
        "earliest_cr_line": "Aug-2003",
        "fico_range_low": 690.0,
        "fico_range_high": 694.0,
        "inq_last_6mths": 1.0,
        "mths_since_last_delinq": None,
        "mths_since_last_record": None,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_bal": 5000.0,
        "revol_util": 50.0,
        "total_acc": 20.0,
        "initial_list_status": "f",
        "pub_rec_bankruptcies": 0.0,
        "acc_now_delinq": 0.0,
        "collections_12_mths_ex_med": 0.0,
        "issue_d": "Jun-2013",
        "loan_status": "Fully Paid",
        "total_pymnt": 11500.0,
        "recoveries": 0.0,
        "funded_amnt": 10000.0,
    }
    base.update(overrides)
    return base


def run(*rows) -> pl.DataFrame:
    return transform(pl.DataFrame([_row(**r) for r in rows]), load_params())


def test_labels_map_correctly():
    out = run(
        {"loan_status": "Fully Paid"}, {"loan_status": "Charged Off"}, {"loan_status": "Default"}
    )
    assert out["target"].to_list() == [0, 1, 1]


def test_unfinished_loans_are_dropped():
    """Current/late loans have no known outcome, so they must not be labeled."""
    out = run(
        {"loan_status": "Current"},
        {"loan_status": "Late (31-120 days)"},
        {"loan_status": "In Grace Period"},
    )
    assert out.height == 0


def test_credit_policy_loans_kept_and_flagged():
    out = run({"loan_status": "Does not meet the credit policy. Status:Charged Off"})
    assert out.height == 1
    assert out["target"][0] == 1
    assert out["not_credit_policy"][0] is True


def test_sixty_month_loans_excluded():
    out = run({"term": " 60 months"})
    assert out.height == 0


def test_loans_issued_after_maturity_cutoff_excluded():
    """A 2017 loan cannot have finished by the 2018Q4 snapshot (D-004)."""
    assert run({"issue_d": "Jun-2017"}).height == 0
    assert run({"issue_d": "Jun-2015"}).height == 1


@pytest.mark.parametrize(
    "raw,expected", [("10+ years", 10), ("< 1 year", 0), ("3 years", 3), (None, None)]
)
def test_emp_length_parsing(raw, expected):
    # A second row keeps the column typed as string; a lone None row would make
    # polars infer a Null column, which never happens with the real data.
    out = run({"emp_length": raw}, {"emp_length": "5 years"})
    assert out["emp_length_years"][0] == expected


def test_derived_features():
    out = run(
        {
            "fico_range_low": 690.0,
            "fico_range_high": 694.0,
            "annual_inc": 60000.0,
            "loan_amnt": 12000.0,
        }
    )
    assert out["fico_mid"][0] == 692.0
    assert out["loan_to_income"][0] == pytest.approx(0.2)
    # Aug-2003 -> Jun-2013 is about 118 months
    assert out["credit_history_months"][0] == pytest.approx(118, abs=1)


def test_zero_income_does_not_divide_by_zero():
    out = run({"annual_inc": 0.0})
    assert out["loan_to_income"][0] is None


def test_dti_sentinel_becomes_missing():
    """dti = 999 means 'not available', not a 999% debt ratio."""
    assert run({"dti": 999.0})["dti"][0] is None


def test_no_forbidden_column_survives():
    """The leakage guard: post-issuance columns must not be feature candidates."""
    out = run({})
    forbidden = tuple(load_params()["features"]["forbidden_prefixes"])
    bookkeeping = {"total_pymnt", "realized_profit", "funded_amnt"}  # evaluation only
    leaked = [c for c in out.columns if c.startswith(forbidden) and c not in bookkeeping]
    assert not leaked, f"leaked columns: {leaked}"


def test_unknown_status_raises():
    """If a refreshed dataset adds a status we have never seen, fail loudly
    rather than silently dropping those loans."""
    with pytest.raises(ValueError, match="unclassified loan_status"):
        run({"loan_status": "Settled In Full"})


def test_credit_policy_loans_can_be_excluded():
    """The keep_credit_policy_loans switch actually does something (D-026)."""
    params = copy.deepcopy(load_params())
    params["data"]["keep_credit_policy_loans"] = False
    rows = pl.DataFrame(
        [
            _row(loan_status="Does not meet the credit policy. Status:Charged Off"),
            _row(loan_status="Fully Paid"),
        ]
    )
    assert transform(rows, params).height == 1
