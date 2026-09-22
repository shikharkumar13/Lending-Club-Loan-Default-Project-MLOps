"""Tests for the baseline, the metric set and the cross-validation loop."""

import numpy as np
import pandas as pd
import pytest

from lending_club.config import load_params
from lending_club.models.baselines import GradePriorClassifier
from lending_club.models.train import run_cv, sample_configs, score

PARAMS = load_params()


# --------------------------------------------------------------------------
# The grade baseline
# --------------------------------------------------------------------------
def test_grade_prior_predicts_the_training_default_rate_per_grade():
    X = pd.DataFrame({"grade": ["A"] * 10 + ["G"] * 10})
    y = np.array([0] * 10 + [1] * 5 + [0] * 5)  # A: 0% default, G: 50%

    model = GradePriorClassifier().fit(X, y)
    proba = model.predict_proba(pd.DataFrame({"grade": ["A", "G"]}))[:, 1]
    assert proba == pytest.approx([0.0, 0.5])


def test_grade_prior_falls_back_for_an_unseen_grade():
    """A grade absent from training must not crash the baseline at predict time."""
    X = pd.DataFrame({"grade": ["A"] * 8 + ["B"] * 2})
    y = np.array([0] * 9 + [1])
    model = GradePriorClassifier().fit(X, y)
    assert model.predict_proba(pd.DataFrame({"grade": ["G"]}))[0, 1] == pytest.approx(0.1)


def test_probabilities_sum_to_one():
    X = pd.DataFrame({"grade": ["A", "B", "C"] * 5})
    model = GradePriorClassifier().fit(X, np.array([0, 1] * 7 + [0]))
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_score_rewards_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    perfect = score(y, np.array([0.01, 0.01, 0.99, 0.99]))
    assert perfect["roc_auc"] == 1.0
    assert perfect["log_loss"] < 0.02
    assert perfect["brier"] < 0.01


def test_score_punishes_confident_mistakes_more_than_ranking_does():
    """Why we tune on log loss, not ROC-AUC (D-025): both of these rank the
    loans identically, but one is badly calibrated."""
    y = np.array([0, 0, 1, 1])
    calibrated = score(y, np.array([0.2, 0.3, 0.7, 0.8]))
    miscalibrated = score(y, np.array([0.5, 0.6, 0.9, 0.95]))
    assert calibrated["roc_auc"] == miscalibrated["roc_auc"]
    assert calibrated["log_loss"] < miscalibrated["log_loss"]


# --------------------------------------------------------------------------
# Randomized search
# --------------------------------------------------------------------------
def test_sample_configs_is_deterministic_and_unique():
    search = {"a": [1, 2, 3], "b": [0.1, 0.2]}
    first = sample_configs(search, 4, seed=42)
    assert first == sample_configs(search, 4, seed=42)  # same seed -> same search
    assert len({tuple(sorted(c.items())) for c in first}) == 4  # no repeats
    assert all(c["a"] in search["a"] and c["b"] in search["b"] for c in first)


# --------------------------------------------------------------------------
# The cross-validation loop
# --------------------------------------------------------------------------
class SpyEstimator:
    """Records what it was fitted on, so a test can prove it never saw the
    validation rows of its fold."""

    instances: list["SpyEstimator"] = []

    def __init__(self):
        SpyEstimator.instances.append(self)
        self.fitted_rows = None

    def fit(self, X, y):
        self.fitted_rows = set(np.asarray(X["row_id"]))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), 0.14)
        return np.column_stack([1 - p, p])


def test_cv_never_fits_on_its_own_validation_rows():
    SpyEstimator.instances.clear()
    n = 60
    X = pd.DataFrame({"row_id": np.arange(n), "grade": ["B"] * n})
    y = np.array([0, 1] * (n // 2))
    folds = [(np.arange(0, 20), np.arange(20, 40)), (np.arange(0, 40), np.arange(40, 60))]

    result = run_cv(SpyEstimator, "tree", X, y, folds, PARAMS, preprocess=False)

    assert len(result["per_fold"]) == 2
    for spy, (train_idx, val_idx) in zip(SpyEstimator.instances, folds, strict=True):
        assert spy.fitted_rows == set(train_idx)
        assert spy.fitted_rows.isdisjoint(val_idx)


def test_cv_returns_the_expected_metrics():
    X = pd.DataFrame({"row_id": np.arange(40), "grade": ["B"] * 40})
    y = np.array([0, 1] * 20)
    folds = [(np.arange(0, 20), np.arange(20, 40))]
    result = run_cv(SpyEstimator, "tree", X, y, folds, PARAMS, preprocess=False)
    assert set(result["mean"]) == {"log_loss", "roc_auc", "pr_auc", "brier"}
