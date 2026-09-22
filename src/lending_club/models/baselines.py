"""Baselines the model has to beat (D-010).

A model is only worth its complexity if it beats what an investor could do
without any machine learning. The most honest baseline here is Lending Club's
own risk grade: it is assigned before the listing appears, so it costs nothing
to use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class GradePriorClassifier(ClassifierMixin, BaseEstimator):
    """Predicts the historical default rate of the loan's Lending Club grade.

    No features, no fitting beyond a group average: "grade C loans defaulted
    14% of the time in the training years, so this grade C loan defaults with
    probability 0.14". Any real model must beat this to justify itself.

    The rates come from the training rows it is fitted on, so inside
    cross-validation they are recomputed per fold, exactly like every other
    learned statistic (D-016).
    """

    def __init__(self, grade_column: str = "grade"):
        self.grade_column = grade_column

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        frame = pd.DataFrame({"grade": np.asarray(X[self.grade_column]), "y": y})
        self.classes_ = np.array([0, 1])
        self.rates_ = frame.groupby("grade")["y"].mean().to_dict()
        self.default_rate_ = float(y.mean())  # fallback for an unseen grade
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "rates_")
        grades = pd.Series(np.asarray(X[self.grade_column]))
        p = grades.map(self.rates_).fillna(self.default_rate_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
