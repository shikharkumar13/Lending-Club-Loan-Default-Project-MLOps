"""Phase 4: calibrate, choose a funding policy, and score the test year once.

Order matters here, and it is the whole point of the phase:

1. **Fit the profit model on training loans** (loss given default, prepayment).
2. **Calibrate on the first half of 2014.** The models are trained for ranking;
   calibration makes "0.2" actually mean "20% of these default" (D-024).
3. **Choose the threshold on the second half of 2014.** Choosing it on the same
   rows used to fit the calibrator would flatter the result.
4. **Score 2015 once.** Nothing before this point has seen it.

Every policy is compared on realized dollars, against the two strategies an
investor could follow with no model at all (D-010).
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lending_club.config import load_params, path_of
from lending_club.features.pipeline import to_model_frame
from lending_club.policy.profit import (
    bootstrap_difference,
    choose_threshold,
    evaluate_policy,
    fit_profit_model,
    threshold_for_share,
)

FAMILIES = ("grade_prior", "logistic_regression", "lightgbm")


def load_splits(params: dict) -> dict[str, pl.DataFrame]:
    directory = path_of("processed_dir")
    return {
        name: pl.read_parquet(directory / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }


def quality(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def evaluate() -> dict:
    params = load_params()
    splits = load_splits(params)
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    # --- 1. economics, measured on training loans only -----------------------
    profit_model = fit_profit_model(train)
    print(
        f"profit model: LGD={profit_model.lgd:.4f}  prepay_factor={profit_model.prepay_factor:.4f}"
    )

    # --- 2. split the validation year: calibrate on H1, tune the policy on H2 -
    mid = pl.date(2014, 7, 1)
    calib = validation.filter(pl.col("issue_date") < mid)
    tune = validation.filter(pl.col("issue_date") >= mid)
    print(f"calibration rows: {calib.height:,}   policy-tuning rows: {tune.height:,}")

    X = {
        name: to_model_frame(frame, params)
        for name, frame in (("calib", calib), ("tune", tune), ("test", test))
    }
    y = {
        name: frame["target"].to_numpy()
        for name, frame in (("calib", calib), ("tune", tune), ("test", test))
    }

    grid = np.arange(**params["profit"]["threshold_grid"])
    results: dict[str, dict] = {}
    curves: dict[str, list] = {}
    policy_masks: dict[str, dict[str, np.ndarray]] = {}

    for family in FAMILIES:
        model = joblib.load(path_of("models_dir") / f"{family}.joblib")

        raw_test = model.predict_proba(X["test"])[:, 1]
        # Isotonic regression: a flexible, monotone map from predicted score to
        # observed frequency. Fitted on held-out 2014 loans, never on test.
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(
            X["calib"], y["calib"]
        )

        # Calibrators belong to the evaluate stage, not the train stage: two
        # DVC stages must never write into the same output directory.
        calibrators = path_of("calibrators_dir")
        calibrators.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrator, calibrators / f"{family}.joblib")

        p_tune = calibrator.predict_proba(X["tune"])[:, 1]
        p_test = calibrator.predict_proba(X["test"])[:, 1]
        raw_tune = model.predict_proba(X["tune"])[:, 1]

        # Ranking policies use the RAW score, not the calibrated probability.
        # Isotonic regression is a step function: it maps 283k test loans onto
        # only ~255 distinct values, one of which covers 39,792 loans. A cut-off
        # landing inside such a block moves 14% of the portfolio at once. The raw
        # score is continuous, so the policy degrades smoothly (D-053).
        threshold, curve = choose_threshold(
            tune,
            raw_tune,
            grid,
            objective="return_per_dollar",
            min_share_funded=params["profit"]["min_share_funded"],
        )
        curves[family] = curve
        # The raw score has no natural meaning, so report the calibrated risk
        # level it corresponds to: "this cut-off funds loans up to about X%
        # predicted default risk".
        funded_on_tune = raw_tune < threshold
        risk_at_threshold = float(p_tune[funded_on_tune].max()) if funded_on_tune.any() else 0.0

        # Same selectivity as the grade A-B rule, so the comparison is not
        # "more selective wins". The share to match is measured on the tuning
        # rows, never on test.
        grade_ab_share = float(tune["grade"].is_in(["A", "B"]).mean())
        matched_threshold = threshold_for_share(raw_tune, grade_ab_share)

        policies = {
            # Fund when expected value is positive: no tuning at all.
            "expected_value": profit_model.expected_return(test, p_test) > 0,
            # Fund when predicted risk is below the threshold tuned on 2014 H2.
            "threshold": raw_test < threshold,
            # Fund the same share of loans the grade A-B rule would.
            "matched_to_grade_AB": raw_test < matched_threshold,
        }

        policy_masks[family] = policies
        results[family] = {
            "quality_test_uncalibrated": quality(y["test"], raw_test),
            "quality_test_calibrated": quality(y["test"], p_test),
            "threshold": float(threshold),
            "matched_threshold": matched_threshold,
            "calibrated_risk_at_threshold": risk_at_threshold,
            "policies": {name: evaluate_policy(test, mask) for name, mask in policies.items()},
        }
        print(
            f"\n{family}: threshold={threshold:.3f} (~{risk_at_threshold:.1%} risk)  "
            f"test log_loss {results[family]['quality_test_uncalibrated']['log_loss']:.5f} "
            f"-> {results[family]['quality_test_calibrated']['log_loss']:.5f} after calibration"
        )
        for name, metrics in results[family]["policies"].items():
            print(
                f"   {name:15s} funded {metrics['share_funded']:6.1%}  "
                f"return/dollar {metrics['return_per_dollar']:+.4f}  "
                f"profit ${metrics['total_profit']:,.0f}"
            )

    # --- 3. model-free baselines (D-010) -------------------------------------
    baselines = {
        "fund_all": np.ones(test.height, dtype=bool),
        "grade_AB": test["grade"].is_in(["A", "B"]).to_numpy(),
    }
    results["baselines"] = {name: evaluate_policy(test, mask) for name, mask in baselines.items()}

    # --- 4. is the edge real, or noise? (D-054) ------------------------------
    draws = params["profit"]["bootstrap_draws"]
    comparisons = {
        "lightgbm_threshold_vs_fund_all": (
            policy_masks["lightgbm"]["threshold"],
            baselines["fund_all"],
        ),
        "lightgbm_threshold_vs_grade_AB": (
            policy_masks["lightgbm"]["threshold"],
            baselines["grade_AB"],
        ),
        "lightgbm_matched_vs_grade_AB": (
            policy_masks["lightgbm"]["matched_to_grade_AB"],
            baselines["grade_AB"],
        ),
        "logreg_matched_vs_grade_AB": (
            policy_masks["logistic_regression"]["matched_to_grade_AB"],
            baselines["grade_AB"],
        ),
        "lightgbm_vs_logreg_matched": (
            policy_masks["lightgbm"]["matched_to_grade_AB"],
            policy_masks["logistic_regression"]["matched_to_grade_AB"],
        ),
    }
    results["significance"] = {
        name: bootstrap_difference(test, a, b, draws=draws, seed=params["seed"])
        for name, (a, b) in comparisons.items()
    }
    print("\nbootstrap (return per dollar, 95% CI):")
    for name, stats in results["significance"].items():
        print(
            f"   {name:34s} {stats['mean_difference']:+.4f} "
            f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
            f"P(better)={stats['prob_better']:.3f}"
        )
    for name, metrics in results["baselines"].items():
        print(
            f"{name:20s} funded {metrics['share_funded']:6.1%}  "
            f"return/dollar {metrics['return_per_dollar']:+.4f}  "
            f"profit ${metrics['total_profit']:,.0f}"
        )

    _figures(params, test, y["test"], curves, threshold_family="lightgbm")
    top_features = _explain(params, test, family="lightgbm")

    payload = {
        "profit_model": {"lgd": profit_model.lgd, "prepay_factor": profit_model.prepay_factor},
        "test_loans": test.height,
        "results": results,
        "threshold_curve_lightgbm": curves["lightgbm"],
        "top_features_lightgbm": top_features,
    }
    reports = path_of("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _figures(params, test, y_test, curves, threshold_family: str) -> None:
    """Calibration curve and profit curve: the two plots that explain the policy."""
    figures = path_of("figures_dir")
    figures.mkdir(parents=True, exist_ok=True)

    model = joblib.load(path_of("models_dir") / f"{threshold_family}.joblib")
    X_test = to_model_frame(test, params)
    raw = model.predict_proba(X_test)[:, 1]

    calibrator = joblib.load(path_of("calibrators_dir") / f"{threshold_family}.joblib")
    calibrated = calibrator.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, probabilities in [("uncalibrated", raw), ("calibrated", calibrated)]:
        true, predicted = calibration_curve(y_test, probabilities, n_bins=15, strategy="quantile")
        axes[0].plot(predicted, true, "o-", label=label)
    axes[0].plot([0, 0.6], [0, 0.6], "--", color="grey", label="perfect")
    axes[0].set_xlabel("predicted default probability")
    axes[0].set_ylabel("observed default rate")
    axes[0].set_title("Calibration on the 2015 test year")
    axes[0].legend()

    curve = curves[threshold_family]
    thresholds = [row["threshold"] for row in curve]
    axes[1].plot(thresholds, [row["return_per_dollar"] for row in curve], color="#2a9d8f")
    axes[1].set_xlabel("fund when predicted probability < t")
    axes[1].set_ylabel("return per dollar", color="#2a9d8f")
    twin = axes[1].twinx()
    twin.plot(thresholds, [row["share_funded"] for row in curve], color="#e76f51")
    twin.set_ylabel("share of loans funded", color="#e76f51")
    axes[1].set_title("Profit curve on 2014 H2 (policy tuning)")
    plt.tight_layout()
    plt.savefig(figures / "policy.png", dpi=120)


def _explain(params, test, family: str, sample: int = 3000) -> list[dict]:
    """SHAP: which columns drive the predictions, and in which direction.

    Credit decisions have to be explainable — "the model said no" is not an
    acceptable answer to a rejected borrower, or to a regulator.
    """
    import shap

    model = joblib.load(path_of("models_dir") / f"{family}.joblib")
    rows = to_model_frame(test.sample(sample, seed=params["seed"]), params)
    matrix = model.named_steps["preprocess"].transform(rows)
    names = list(model.named_steps["preprocess"].get_feature_names_out())

    values = shap.TreeExplainer(model.named_steps["model"]).shap_values(matrix)
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(-importance)[:15]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(values, matrix, feature_names=names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(path_of("figures_dir") / "shap_summary.png", dpi=120)
    plt.close()

    return [{"feature": names[i], "mean_abs_shap": float(importance[i])} for i in order]


if __name__ == "__main__":
    evaluate()
