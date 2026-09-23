"""The request contract: exactly what a loan listing shows an investor.

The API asks for **raw listing fields**, never for engineered ones. A caller
cannot be expected to know what `fico_mid` or `credit_history_months` mean, and
if they computed them slightly differently the model would silently score
garbage. The service derives them with the same function training used (D-056).
"""

from __future__ import annotations

from typing import Annotated, Literal

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from lending_club.data.prepare import derive_features

Money = Annotated[float, Field(ge=0)]
Count = Annotated[float, Field(ge=0)]


class LoanApplication(BaseModel):
    """One Lending Club listing, as an investor sees it before funding."""

    # --- the loan ---
    loan_amnt: Annotated[float, Field(gt=0, le=40_000)]
    term_months: Literal[36]  # v1 covers 36-month loans only (D-004)
    int_rate: Annotated[float, Field(gt=0, le=40)]
    installment: Annotated[float, Field(gt=0)]
    grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    sub_grade: Annotated[str, Field(pattern=r"^[A-G][1-5]$")]
    purpose: str
    initial_list_status: Literal["f", "w"]
    application_date: Annotated[str, Field(pattern=r"^[A-Z][a-z]{2}-\d{4}$")]  # e.g. "Jun-2015"

    # --- the borrower ---
    annual_inc: Money
    emp_length: str | None = None  # e.g. "10+ years", "< 1 year", null = unknown
    home_ownership: str
    verification_status: str
    addr_state: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    dti: float | None = None

    # --- credit bureau, at application time ---
    earliest_cr_line: Annotated[str, Field(pattern=r"^[A-Z][a-z]{2}-\d{4}$")]
    fico_range_low: Annotated[float, Field(ge=300, le=900)]
    fico_range_high: Annotated[float, Field(ge=300, le=900)]
    delinq_2yrs: Count = 0
    inq_last_6mths: Count = 0
    open_acc: Count = 0
    pub_rec: Count = 0
    revol_bal: Money = 0
    revol_util: float | None = None
    total_acc: Count = 0
    acc_now_delinq: Count = 0
    pub_rec_bankruptcies: float | None = 0
    collections_12_mths_ex_med: float | None = 0
    # Missing means "never happened", which the pipeline encodes explicitly (D-031).
    mths_since_last_delinq: float | None = None
    mths_since_last_record: float | None = None

    # Lending Club's own flag; modern listings meet its credit policy (D-026).
    not_credit_policy: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_amnt": 12_000,
                "term_months": 36,
                "int_rate": 11.99,
                "installment": 398.52,
                "grade": "B",
                "sub_grade": "B3",
                "purpose": "debt_consolidation",
                "initial_list_status": "w",
                "application_date": "Jun-2015",
                "annual_inc": 72_000,
                "emp_length": "10+ years",
                "home_ownership": "MORTGAGE",
                "verification_status": "Source Verified",
                "addr_state": "CA",
                "dti": 16.4,
                "earliest_cr_line": "Aug-2003",
                "fico_range_low": 690,
                "fico_range_high": 694,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 9,
                "pub_rec": 0,
                "revol_bal": 8_400,
                "revol_util": 42.1,
                "total_acc": 21,
                "acc_now_delinq": 0,
                "pub_rec_bankruptcies": 0,
                "collections_12_mths_ex_med": 0,
            }
        }
    }


class LoanDecision(BaseModel):
    decision: Literal["fund", "decline"]
    probability_of_default: float
    expected_return_usd: float
    risk_score: float
    threshold: float
    model_version: str


def to_model_frame(applications: list[LoanApplication]) -> pd.DataFrame:
    """API payload -> the exact frame the bundle expects.

    The adapter renames two fields to the names the raw dataset used, so the
    shared derivation code runs unchanged.
    """
    rows = []
    for application in applications:
        row = application.model_dump()
        row["issue_d"] = row.pop("application_date")
        row["term"] = f" {row.pop('term_months')} months"
        rows.append(row)

    frame = derive_features(pl.DataFrame(rows))
    return frame.to_pandas()
