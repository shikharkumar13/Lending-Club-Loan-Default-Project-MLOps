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
from pydantic import BaseModel, Field, model_validator

from lending_club.data.prepare import derive_features

Money = Annotated[float, Field(ge=0)]
Count = Annotated[float, Field(ge=0)]

# Closed sets, taken from the training vocabulary (D-072). A value outside
# them is a bug in the caller, not a new kind of borrower, so it is rejected
# rather than quietly mapped to "infrequent" and scored with confidence.
# `app.py` asserts at startup that these still match the loaded model.
HomeOwnership = Literal["MORTGAGE", "RENT", "OWN", "OTHER", "NONE"]
VerificationStatus = Literal["Verified", "Source Verified", "Not Verified"]
Purpose = Literal[
    "car",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "house",
    "major_purchase",
    "medical",
    "moving",
    "other",
    "renewable_energy",
    "small_business",
    "vacation",
    "wedding",
]
EmpLength = Literal[
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]

# `addr_state` is validated against the 51 real USPS codes, NOT against the
# training vocabulary, which happens to contain only 50 (North Dakota never
# appears before 2016). A valid state the model has not seen is a genuine
# business event that the drift monitor should report — not a client error to
# reject with a 422 (D-072).
US_STATES = frozenset(
    "AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS "
    "MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY".split()
)

# How far the stated installment may sit from the amortization formula before
# the listing is treated as corrupt (D-073).
INSTALLMENT_TOLERANCE = 0.05


def expected_installment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    """The scheduled monthly payment implied by amount, rate and term."""
    monthly = annual_rate_pct / 1200.0
    if monthly == 0:
        return principal / term_months
    return principal * monthly / (1 - (1 + monthly) ** -term_months)


class LoanApplication(BaseModel):
    """One Lending Club listing, as an investor sees it before funding."""

    # --- the loan ---
    loan_amnt: Annotated[float, Field(gt=0, le=40_000)]
    term_months: Literal[36]  # v1 covers 36-month loans only (D-004)
    int_rate: Annotated[float, Field(gt=0, le=40)]
    installment: Annotated[float, Field(gt=0)]
    grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    sub_grade: Annotated[str, Field(pattern=r"^[A-G][1-5]$")]
    purpose: Purpose
    initial_list_status: Literal["f", "w"]
    application_date: Annotated[str, Field(pattern=r"^[A-Z][a-z]{2}-\d{4}$")]  # e.g. "Jun-2015"

    # --- the borrower ---
    annual_inc: Money
    emp_length: EmpLength | None = None  # null = not supplied by the borrower
    home_ownership: HomeOwnership
    verification_status: VerificationStatus
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

    @model_validator(mode="after")
    def _consistent(self):
        """Reject listings whose fields contradict each other (D-073).

        Every field above is individually plausible; these checks look at
        combinations, which is where real corruption shows up. The source data
        contains 1,015 loans (0.16%) whose stated installment disagrees with
        the amortization formula by more than 1% — one lists $14.77 a month on
        $6,000 at 6.89%, which implies a payment of $185. Scored as-is, that
        loan produced a "fund" decision alongside an expected return of
        -$13,066.
        """
        if self.fico_range_high < self.fico_range_low:
            raise ValueError(
                f"fico_range_high ({self.fico_range_high}) is below "
                f"fico_range_low ({self.fico_range_low})"
            )
        if self.addr_state not in US_STATES:
            raise ValueError(f"addr_state {self.addr_state!r} is not a US state code")

        scheduled = expected_installment(self.loan_amnt, self.int_rate, self.term_months)
        if abs(self.installment / scheduled - 1) > INSTALLMENT_TOLERANCE:
            raise ValueError(
                f"installment {self.installment:.2f} is inconsistent with "
                f"{self.loan_amnt:.0f} at {self.int_rate}% over {self.term_months} "
                f"months, which amortizes to {scheduled:.2f}"
            )
        return self

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
    """The funding decision, and every number behind it.

    `decision` is **not** a simple function of `expected_return_usd`, and the
    `decision_basis` field says which rule applied (D-074). Funding requires
    both a risk score below the tuned threshold and a positive expected
    return; the threshold is what binds in practice, because at Lending Club's
    interest rates almost every loan has positive expected value.
    """

    decision: Literal["fund", "decline"]
    decision_basis: Literal[
        "score_below_threshold",
        "score_above_threshold",
        "negative_expected_return",
    ]
    probability_of_default: float = Field(description="Calibrated, for a human to read")
    expected_return_usd: float = Field(description="At the listed amount, over the full term")
    risk_score: float = Field(description="Uncalibrated model output; this is what the rule uses")
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
