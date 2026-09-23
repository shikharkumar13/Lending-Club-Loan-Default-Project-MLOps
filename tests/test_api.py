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
from lending_club.serving.schema import LoanApplication, expected_installment, to_model_frame

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
        # The installment has to move with the rate, or the request is
        # internally inconsistent and is rejected before it reaches the model
        # (D-073) — which is exactly what this fixture used to be.
        "installment": round(
            expected_installment(APPLICATION["loan_amnt"], 28.0, APPLICATION["term_months"]), 2
        ),
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


# --- input validation: the audit's C2 and M2 -------------------------------


def test_an_unknown_category_is_rejected_not_silently_scored(client):
    """The classic silent production failure: an upstream rename shifts every
    prediction while the service keeps returning 200s (D-072)."""
    response = client.post("/predict", json=APPLICATION | {"home_ownership": "MORGAGE"})
    assert response.status_code == 422


def test_an_unknown_purpose_is_rejected(client):
    assert (
        client.post("/predict", json=APPLICATION | {"purpose": "wedding_cake"}).status_code == 422
    )


def test_a_valid_but_unseen_state_is_accepted(client):
    """North Dakota never appears in training, but it is a real state (D-072).

    Rejecting it would turn a genuine business event into a client error. The
    drift monitor reports it instead.
    """
    response = client.post("/predict", json=APPLICATION | {"addr_state": "ND"})
    assert response.status_code == 200
    assert response.json()["decision"] in {"fund", "decline"}


def test_a_state_that_does_not_exist_is_rejected(client):
    assert client.post("/predict", json=APPLICATION | {"addr_state": "ZZ"}).status_code == 422


def test_an_inverted_fico_band_is_rejected(client):
    """Each field is individually valid; the combination is impossible (D-073)."""
    bad = APPLICATION | {"fico_range_low": 800, "fico_range_high": 400}
    assert client.post("/predict", json=bad).status_code == 422


def test_an_installment_that_contradicts_the_loan_is_rejected(client):
    """The source data really contains these, and one produced a 'fund'
    decision next to an expected return of -$13,066 (D-073)."""
    response = client.post("/predict", json=APPLICATION | {"installment": 14.77})
    assert response.status_code == 422
    assert "amortizes" in response.json()["detail"][0]["msg"]


def test_a_consistent_installment_is_accepted(client):
    """The tolerance must not reject legitimate rounding."""
    exact = expected_installment(APPLICATION["loan_amnt"], APPLICATION["int_rate"], 36)
    response = client.post("/predict", json=APPLICATION | {"installment": round(exact, 2)})
    assert response.status_code == 200


# --- the decision contract: the audit's C1 ---------------------------------


def test_the_response_says_which_rule_decided(client):
    body = client.post("/predict", json=APPLICATION).json()
    assert body["decision"] == "fund"
    assert body["decision_basis"] == "score_below_threshold"


def test_a_funded_loan_never_carries_a_negative_expected_return(client):
    """The two numbers in the response cannot contradict each other (D-074)."""
    params = load_params()
    test = pl.read_parquet(path_of("processed_dir") / "test.parquet").head(2_000)
    bundle = LoanDecisionModel.load()
    from lending_club.serving.bundle import prepare_for_decision

    decisions = bundle.decide(prepare_for_decision(test, params))
    funded = decisions["decision"] == "fund"
    assert (decisions.loc[funded, "expected_return_usd"] > 0).all()


def test_decision_basis_explains_every_decline(client):
    params = load_params()
    test = pl.read_parquet(path_of("processed_dir") / "test.parquet").head(5_000)
    bundle = LoanDecisionModel.load()
    from lending_club.serving.bundle import prepare_for_decision

    decisions = bundle.decide(prepare_for_decision(test, params))
    declined = decisions[decisions["decision"] == "decline"]
    assert set(declined["decision_basis"]) <= {
        "score_above_threshold",
        "negative_expected_return",
    }
    funded = decisions[decisions["decision"] == "fund"]
    assert set(funded["decision_basis"]) == {"score_below_threshold"}


# --- readiness: the audit's M3 ---------------------------------------------


def test_requests_before_the_model_loads_return_503_not_500():
    """503 means 'retry'; 500 means 'page someone'. A startup race is the
    former, and a KeyError would have reported the latter (D-075)."""
    from lending_club.serving import app as app_module

    saved = dict(app_module.state)
    app_module.state.clear()
    try:
        with TestClient(app_module.app, raise_server_exceptions=False) as _:
            pass
        # Exercise the handlers directly: the lifespan would repopulate state.
        app_module.state.clear()
        for call in (app_module.health, app_module.model_metadata):
            with pytest.raises(Exception) as excinfo:
                call()
            assert getattr(excinfo.value, "status_code", None) == 503
    finally:
        app_module.state.update(saved)


def test_the_bundle_records_what_categories_it_was_trained_on():
    """The schema's accepted values are checked against this at startup (D-072)."""
    vocabulary = LoanDecisionModel.load().metadata.categorical_vocabulary
    assert "MORTGAGE" in vocabulary["home_ownership"]
    assert "debt_consolidation" in vocabulary["purpose"]
    assert "MORGAGE" not in vocabulary["home_ownership"]
