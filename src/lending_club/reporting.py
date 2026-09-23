"""Figures and explanations for the evaluation report (m5).

Split out of `evaluate.py`, which had grown to do calibration, policy choice,
baselines, bootstrapping, plotting and SHAP in one 330-line function. Plotting
is presentation, not evaluation logic, and mixing them made the part that
actually decides the policy harder to read.
"""

from __future__ import annotations

import joblib
import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # no display in CI or in a container
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from lending_club.config import path_of  # noqa: E402
from lending_club.features.pipeline import to_model_frame  # noqa: E402


def figures(
    params: dict,
    test: pl.DataFrame,
    y_test: np.ndarray,
    curves: dict[str, list],
    threshold_family: str,
) -> None:
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


def explain(params: dict, test: pl.DataFrame, family: str, sample: int = 3_000) -> list[dict]:
    """SHAP: which columns drive the predictions, and in which direction.

    Credit decisions have to be explainable — "the model said no" is not an
    acceptable answer to a rejected borrower, or to a regulator.

    The sample comes from the TEST year. Nothing is fitted on it, so this is
    not leakage: it explains the predictions that were actually reported. It is
    noted because a reader is entitled to know which rows a published
    explanation describes (m1).
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
