"""The whole pipeline on synthetic data, in a couple of seconds.

CI has no access to the 1.6 GB dataset, but "every unit passes" is not the same
as "the pipeline works". This test generates loans with a known risk structure
and runs the real code end to end: clean -> label -> split -> preprocess ->
train -> calibrate -> choose a policy -> bundle -> serve over HTTP.

It is the check that catches wiring mistakes — a renamed column, a stage that
no longer matches the one after it — which unit tests cannot see.
"""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from lending_club.config import load_params
from lending_club.data.prepare import transform
from lending_club.data.split import assign_split, cv_folds
from lending_club.features.pipeline import build_preprocessor, feature_columns, to_model_frame
from lending_club.features.schema import validate
from lending_club.models.promote import should_promote
from lending_club.policy.profit import choose_threshold, fit_profit_model
from lending_club.serving.bundle import (
    BundleMetadata,
    LoanDecisionModel,
    categorical_vocabulary,
    prepare_for_decision,
)
from lending_club.testing.synthetic import synthetic_raw


@pytest.fixture(scope="module")
def pipeline_outputs():
    params = load_params()
    prepared = transform(synthetic_raw(), params)
    assert prepared.height > 0, "every synthetic loan was filtered out"
    validate(prepared, params)

    frame = assign_split(prepared, params)
    train = frame.filter(pl.col("split") == "train")
    validation = frame.filter(pl.col("split") == "validation")
    return params, train, validation


def test_pipeline_produces_usable_splits(pipeline_outputs):
    _, train, validation = pipeline_outputs
    assert train.height > 100 and validation.height > 50
    assert 0.02 < train["target"].mean() < 0.6
    assert train["issue_date"].max() < validation["issue_date"].min()


def test_cv_folds_are_valid_on_synthetic_data(pipeline_outputs):
    params, train, _ = pipeline_outputs
    folds = cv_folds(train, params)
    assert len(folds) == len(params["split"]["cv_folds"])


def test_model_learns_the_planted_signal(pipeline_outputs):
    """If the wiring is right, the model must beat random on data where risk
    truly depends on the features."""
    from sklearn.metrics import roc_auc_score

    params, train, validation = pipeline_outputs
    X_train, X_val = to_model_frame(train, params), to_model_frame(validation, params)

    model = Pipeline(
        [
            ("preprocess", build_preprocessor(params, "tree")),
            ("model", LGBMClassifier(n_estimators=60, verbose=-1, random_state=1)),
        ]
    ).fit(X_train, train["target"].to_numpy())

    auc = roc_auc_score(validation["target"].to_numpy(), model.predict_proba(X_val)[:, 1])
    assert auc > 0.60, f"pipeline is wired wrong or signal is lost: AUC={auc:.3f}"


def test_full_decision_path_including_the_api(pipeline_outputs):
    """Train -> calibrate -> pick a policy -> bundle -> serve, then check the
    API returns the same answer as the bundle."""
    params, train, validation = pipeline_outputs
    X_train, X_val = to_model_frame(train, params), to_model_frame(validation, params)

    model = Pipeline(
        [
            ("preprocess", build_preprocessor(params, "tree")),
            ("model", LGBMClassifier(n_estimators=60, verbose=-1, random_state=1)),
        ]
    ).fit(X_train, train["target"].to_numpy())
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(
        X_val, validation["target"].to_numpy()
    )

    scores = model.predict_proba(X_val)[:, 1]
    threshold, _ = choose_threshold(
        validation,
        scores,
        np.arange(**params["profit"]["threshold_grid"]),
        min_share_funded=params["profit"]["min_share_funded"],
    )
    profit = fit_profit_model(train)

    bundle = LoanDecisionModel(
        pipeline=model,
        calibrator=calibrator,
        profit=profit,
        metadata=BundleMetadata(
            model_family="lightgbm",
            threshold=threshold,
            calibrated_risk_at_threshold=0.15,
            lgd=profit.lgd,
            prepay_factor=profit.prepay_factor,
            feature_columns=feature_columns(params),
            categorical_vocabulary=categorical_vocabulary(model, params),
            train_window=params["split"]["train"],
            cv_log_loss=0.0,
            test_return_per_dollar=0.0,
        ),
    )

    decisions = bundle.decide(prepare_for_decision(validation, params))
    assert set(decisions["decision"]) <= {"fund", "decline"}
    assert decisions["probability_of_default"].between(0, 1).all()
    # A funded loan must have positive expected value under the fitted economics.
    assert decisions.loc[decisions["decision"] == "fund", "expected_return_usd"].mean() > 0

    # The gate must accept this model when there is no champion yet.
    assert should_promote(bundle, None, validation, params).promote


def test_served_api_agrees_with_the_bundle(tmp_path, pipeline_outputs, monkeypatch):
    """Serve the synthetic bundle through the real app and compare answers."""
    params, train, validation = pipeline_outputs
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(params, "tree")),
            ("model", LGBMClassifier(n_estimators=40, verbose=-1, random_state=1)),
        ]
    ).fit(to_model_frame(train, params), train["target"].to_numpy())
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(
        to_model_frame(validation, params), validation["target"].to_numpy()
    )
    bundle = LoanDecisionModel(
        pipeline=model,
        calibrator=calibrator,
        profit=fit_profit_model(train),
        metadata=BundleMetadata(
            model_family="lightgbm",
            threshold=0.2,
            calibrated_risk_at_threshold=0.2,
            lgd=0.37,
            prepay_factor=0.83,
            feature_columns=feature_columns(params),
            categorical_vocabulary=categorical_vocabulary(model, params),
            train_window=params["split"]["train"],
            cv_log_loss=0.0,
            test_return_per_dollar=0.0,
        ),
    )
    path = tmp_path / "loan_decision_model.joblib"
    bundle.save(path)

    import lending_club.serving.app as app_module

    monkeypatch.setattr(app_module.LoanDecisionModel, "load", staticmethod(lambda p=None: bundle))
    with TestClient(app_module.app) as client:
        payload = {
            "loan_amnt": 12_000,
            "term_months": 36,
            "int_rate": 11.99,
            "installment": 398.52,
            "grade": "B",
            "sub_grade": "B3",
            "purpose": "debt_consolidation",
            "initial_list_status": "w",
            "application_date": "Jun-2012",
            "annual_inc": 72_000,
            "emp_length": "10+ years",
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "addr_state": "CA",
            "dti": 16.4,
            "earliest_cr_line": "Aug-1999",
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
        served = client.post("/predict", json=payload).json()

    from lending_club.serving.schema import LoanApplication
    from lending_club.serving.schema import to_model_frame as api_frame

    offline = bundle.decide(api_frame([LoanApplication(**payload)]))
    assert served["probability_of_default"] == pytest.approx(
        offline["probability_of_default"][0], abs=1e-6
    )
    assert served["decision"] == offline["decision"][0]
