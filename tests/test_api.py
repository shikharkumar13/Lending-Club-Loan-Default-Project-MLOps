"""Tests for the serving layer.

The decisive one is `test_api_matches_offline_scoring`: a served prediction must
equal what the offline pipeline produces for the same loan. Training/serving
skew is the classic production failure, and it is silent — the API keeps
returning plausible numbers that are simply wrong.
"""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from lending_club.config import load_params, path_of
from lending_club.features.pipeline import feature_columns
from lending_club.serving.app import app
from lending_club.serving.bundle import LoanDecisionModel, default_bundle_path
from lending_club.serving.schema import LoanApplication, to_model_frame

pytestmark = pytest.mark.skipif(
    not default_bundle_path().exists(), reason="bundle not built (run: dvc repro package)"
)

APPLICATION = {
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


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_family"] == "lightgbm"
    assert body["model_version"]  # a deploy check needs to see WHICH model is live


def test_predict_returns_a_complete_decision(client):
    body = client.post("/predict", json=APPLICATION).json()
    assert body["decision"] in {"fund", "decline"}
    assert 0 <= body["probability_of_default"] <= 1
    assert isinstance(body["expected_return_usd"], float)
    assert body["threshold"] == pytest.approx(LoanDecisionModel.load().metadata.threshold)


def test_decision_follows_the_threshold(client):
    body = client.post("/predict", json=APPLICATION).json()
    expected = "fund" if body["risk_score"] < body["threshold"] else "decline"
    assert body["decision"] == expected


def test_a_riskier_borrower_is_scored_higher(client):
    """Sanity: a G-grade, high-rate, low-income application must not look safer."""
    risky = APPLICATION | {
        "grade": "G",
        "sub_grade": "G5",
        "int_rate": 28.0,
        "annual_inc": 20_000,
        "dti": 35.0,
        "fico_range_low": 660,
        "fico_range_high": 664,
        "delinq_2yrs": 3,
        "inq_last_6mths": 5,
    }
    safe = client.post("/predict", json=APPLICATION).json()
    risky_body = client.post("/predict", json=risky).json()
    assert risky_body["probability_of_default"] > safe["probability_of_default"]


def test_batch_matches_single_predictions(client):
    single = client.post("/predict", json=APPLICATION).json()
    batch = client.post("/predict/batch", json=[APPLICATION, APPLICATION]).json()
    assert len(batch) == 2
    assert batch[0]["probability_of_default"] == single["probability_of_default"]


def test_invalid_input_is_rejected_with_a_useful_error(client):
    bad = APPLICATION | {"grade": "Z"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
    assert "grade" in response.text


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in APPLICATION.items() if k != "annual_inc"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_sixty_month_loan_is_rejected(client):
    """v1 was trained on 36-month loans only; scoring a 60-month loan would be
    out of distribution, so the contract refuses it (D-004)."""
    assert client.post("/predict", json=APPLICATION | {"term_months": 60}).status_code == 422


def test_request_schema_matches_the_model_inputs():
    """Guard against drift: every engineered feature must be derivable from the
    fields the API actually asks for."""
    derived = to_model_frame([LoanApplication(**APPLICATION)])
    missing = set(feature_columns(load_params())) - set(derived.columns)
    assert not missing, f"API cannot produce these model inputs: {missing}"


@pytest.mark.skipif(
    not (path_of("processed_dir") / "test.parquet").exists(), reason="processed data missing"
)
def test_api_matches_offline_scoring(client):
    """The served prediction must equal the offline one for the same loans."""
    bundle = LoanDecisionModel.load()
    raw = (
        pl.read_parquet(path_of("raw_parquet"))
        .filter(pl.col("issue_d").str.contains("2015"), pl.col("term").str.contains("36"))
        .head(25)
    )

    applications, offline_rows = [], []
    for row in raw.iter_rows(named=True):
        payload = {
            "loan_amnt": row["loan_amnt"],
            "term_months": 36,
            "int_rate": row["int_rate"],
            "installment": row["installment"],
            "grade": row["grade"],
            "sub_grade": row["sub_grade"],
            "purpose": row["purpose"],
            "initial_list_status": row["initial_list_status"],
            "application_date": row["issue_d"],
            "annual_inc": row["annual_inc"],
            "emp_length": row["emp_length"],
            "home_ownership": row["home_ownership"],
            "verification_status": row["verification_status"],
            "addr_state": row["addr_state"],
            "dti": row["dti"],
            "earliest_cr_line": row["earliest_cr_line"],
            "fico_range_low": row["fico_range_low"],
            "fico_range_high": row["fico_range_high"],
            "delinq_2yrs": row["delinq_2yrs"],
            "inq_last_6mths": row["inq_last_6mths"],
            "open_acc": row["open_acc"],
            "pub_rec": row["pub_rec"],
            "revol_bal": row["revol_bal"],
            "revol_util": row["revol_util"],
            "total_acc": row["total_acc"],
            "acc_now_delinq": row["acc_now_delinq"],
            "pub_rec_bankruptcies": row["pub_rec_bankruptcies"],
            "collections_12_mths_ex_med": row["collections_12_mths_ex_med"],
            "mths_since_last_delinq": row["mths_since_last_delinq"],
            "mths_since_last_record": row["mths_since_last_record"],
        }
        applications.append(payload)
        offline_rows.append(LoanApplication(**payload))

    served = client.post("/predict/batch", json=applications).json()
    offline = bundle.decide(to_model_frame(offline_rows))

    np.testing.assert_allclose(
        [row["probability_of_default"] for row in served],
        offline["probability_of_default"].to_numpy(),
        rtol=0,
        atol=1e-6,
    )
    assert [row["decision"] for row in served] == list(offline["decision"])
