"""Synthetic loans with a known risk structure.

Used by the end-to-end test and by CI, which has no access to the real 1.6 GB
dataset but still needs to prove that the pipeline, the bundle and the
container all work together.
"""

from __future__ import annotations

import numpy as np
import polars as pl

STATES = ["CA", "NY", "TX", "FL"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement"]
GRADES = list("ABCDEFG")


def synthetic_raw(n: int = 4_000, seed: int = 7) -> pl.DataFrame:
    """Loans whose default risk genuinely depends on grade, income and FICO,
    so a model has something real to learn."""
    rng = np.random.default_rng(seed)
    grade_idx = rng.integers(0, len(GRADES), n)
    fico = rng.normal(700, 30, n).clip(620, 840).round()
    income = rng.lognormal(11, 0.5, n).clip(15_000, 400_000)
    amount = rng.integers(1_000, 35_000, n).astype(float)
    rate = 6 + grade_idx * 3 + rng.normal(0, 1, n)

    # True risk: worse grade, lower FICO and a bigger loan relative to income.
    logit = -3.2 + 0.32 * grade_idx - 0.012 * (fico - 700) + 2.5 * (amount / income)
    defaulted = rng.random(n) < 1 / (1 + np.exp(-logit))

    # Spread issue dates across 2008-2015 so the train/validation/test windows
    # in params.yaml all receive loans.
    months = rng.integers(0, 96, n)
    issue = [f"{['Jan', 'Mar', 'Jun', 'Sep', 'Dec'][m % 5]}-{2008 + m // 12}" for m in months]
    installment = amount * (1 + rate / 100 * 3) / 36

    return pl.DataFrame(
        {
            "loan_amnt": amount,
            "funded_amnt": amount,
            "term": [" 36 months"] * n,
            "int_rate": rate.round(2),
            "installment": installment.round(2),
            "grade": [GRADES[i] for i in grade_idx],
            "sub_grade": [f"{GRADES[i]}{rng.integers(1, 6)}" for i in grade_idx],
            "emp_length": rng.choice(["< 1 year", "3 years", "10+ years", None], n).tolist(),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n).tolist(),
            "annual_inc": income.round(2),
            "verification_status": rng.choice(["Verified", "Not Verified"], n).tolist(),
            "issue_d": issue,
            "loan_status": np.where(defaulted, "Charged Off", "Fully Paid").tolist(),
            "purpose": rng.choice(PURPOSES, n).tolist(),
            "addr_state": rng.choice(STATES, n).tolist(),
            "dti": rng.uniform(1, 35, n).round(2),
            "delinq_2yrs": rng.integers(0, 3, n).astype(float),
            "earliest_cr_line": ["Aug-1999"] * n,
            "fico_range_low": fico,
            "fico_range_high": fico + 4,
            "inq_last_6mths": rng.integers(0, 4, n).astype(float),
            "mths_since_last_delinq": rng.choice([None, 12.0, 40.0], n).tolist(),
            "mths_since_last_record": rng.choice([None, 60.0], n).tolist(),
            "open_acc": rng.integers(2, 20, n).astype(float),
            "pub_rec": rng.integers(0, 2, n).astype(float),
            "revol_bal": rng.uniform(0, 40_000, n).round(2),
            "revol_util": rng.uniform(0, 100, n).round(1),
            "total_acc": rng.integers(5, 40, n).astype(float),
            "initial_list_status": rng.choice(["f", "w"], n).tolist(),
            "pub_rec_bankruptcies": rng.integers(0, 2, n).astype(float),
            "acc_now_delinq": np.zeros(n),
            "collections_12_mths_ex_med": np.zeros(n),
            # Repaid loans return the contracted interest; defaults lose ~40%.
            "total_pymnt": np.where(defaulted, amount * 0.6, amount * (1 + rate / 100 * 2.4)).round(
                2
            ),
            "recoveries": np.zeros(n),
        }
    )


def build_demo_bundle(path=None, force: bool = False):
    """Train a small model on synthetic loans and write a deployable bundle.

    CI uses this to build and smoke-test the Docker image without the real
    dataset. It is a *demo* artifact: it proves the plumbing works, and its
    numbers mean nothing.
    """
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.pipeline import Pipeline

    from lending_club.config import load_params
    from lending_club.data.prepare import transform
    from lending_club.data.split import assign_split
    from lending_club.features.pipeline import build_preprocessor, feature_columns, to_model_frame
    from lending_club.policy.profit import fit_profit_model
    from lending_club.serving.bundle import (
        BundleMetadata,
        LoanDecisionModel,
        categorical_vocabulary,
        default_bundle_path,
    )

    params = load_params()
    frame = assign_split(transform(synthetic_raw(), params), params)
    train = frame.filter(pl.col("split") == "train")
    validation = frame.filter(pl.col("split") == "validation")

    model = Pipeline(
        [
            ("preprocess", build_preprocessor(params, "tree")),
            ("model", LGBMClassifier(n_estimators=60, verbose=-1, random_state=1)),
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
            model_family="lightgbm-demo",
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
    destination = path or default_bundle_path()
    if destination.exists() and not force:
        # Guard rail: this demo artifact must never silently replace a real
        # trained bundle. CI starts from a clean checkout, so nothing exists
        # there and the guard never fires.
        raise FileExistsError(
            f"{destination} already exists - refusing to overwrite a real bundle. "
            "Pass force=True (or `python -m lending_club.testing --force`) if that is intended."
        )
    bundle.save(destination)
    print(f"demo bundle written to {destination}")
    return destination
