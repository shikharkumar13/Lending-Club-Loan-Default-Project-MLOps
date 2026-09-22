"""Layer B preprocessing: the transforms that LEARN from data (D-016).

Medians, percentile caps, category lists and scaling factors are all statistics
estimated from the data. If they are computed on the full dataset, information
from the validation and test years silently reaches the training set — the
classic, invisible form of leakage.

Putting them in an sklearn Pipeline solves this structurally rather than by
discipline: cross-validation refits the whole pipeline on each fold, so a
statistic can only ever come from that fold's training rows.

Two variants are built from the same configuration:
  * "linear"  -> scaled features for logistic regression
  * "tree"    -> unscaled features for LightGBM (trees are scale-invariant)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

GRADES = list("ABCDEFG")
SUB_GRADES = [f"{g}{i}" for g in GRADES for i in range(1, 6)]


class CapAndLog(BaseEstimator, TransformerMixin):
    """Clip at a percentile learned from training data, then apply log1p.

    Income and revolving balance are extremely skewed (incomes up to $9M). A
    linear model would be dominated by a handful of rows. Capping uses the
    TRAINING percentile only — using the full data's percentile would leak.
    """

    def __init__(self, percentile: float = 0.995):
        self.percentile = percentile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.caps_ = np.nanquantile(values, self.percentile, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.log1p(np.clip(values, a_min=0, a_max=self.caps_))

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"log_{name}" for name in input_features], dtype=object)


def build_preprocessor(params: dict, kind: str = "tree") -> Pipeline:
    """Assemble the ColumnTransformer described in params.yaml -> features.groups."""
    groups = params["features"]["groups"]
    cfg = params["features"]

    skewed = Pipeline(
        [
            ("cap_log", CapAndLog(cfg["cap_percentile"])),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    numeric = Pipeline(
        # add_indicator keeps a 0/1 column saying "this value was missing",
        # because missingness itself can predict default (D-017).
        # keep_empty_features: if a column happens to be entirely missing in one
        # CV fold, sklearn would otherwise DROP it, changing the number of
        # features between folds and breaking a served model. Keeping it means
        # a stable output shape, always.
        [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    )
    structural = Pipeline(
        # Missing here means "never happened" -> a sentinel far outside the real
        # range, plus the indicator (D-031).
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=cfg["never_happened_sentinel"],
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        ]
    )
    ordinal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    categories=[GRADES, SUB_GRADES],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encode",
                OneHotEncoder(
                    # Levels below 1% of training rows collapse into one
                    # "infrequent" column, and a level never seen in training
                    # joins them instead of raising an error (D-018).
                    min_frequency=cfg["rare_category_min_frequency"],
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformer = ColumnTransformer(
        [
            ("skewed", skewed, groups["skewed_amount"]),
            ("numeric", numeric, groups["numeric"]),
            ("structural", structural, groups["structural_missing"]),
            ("ordinal", ordinal, ["grade", "sub_grade"]),
            ("nominal", nominal, groups["nominal"]),
            ("boolean", "passthrough", groups["boolean"]),
        ],
        remainder="drop",  # anything not listed above can never reach the model
        verbose_feature_names_out=False,
    )

    steps = [("columns", transformer)]
    if kind == "linear":
        # Logistic regression with L2 needs comparable scales, otherwise the
        # penalty falls unevenly across features. Trees do not care.
        steps.append(("scale", StandardScaler()))
    elif kind != "tree":
        raise ValueError(f"unknown kind: {kind!r}")
    return Pipeline(steps)


def feature_columns(params: dict) -> list[str]:
    """Every column the model is allowed to see, in a stable order."""
    groups = params["features"]["groups"]
    return (
        list(groups["skewed_amount"])
        + list(groups["numeric"])
        + list(groups["structural_missing"])
        + ["grade", "sub_grade"]
        + list(groups["nominal"])
        + list(groups["boolean"])
    )


def to_model_frame(frame, params: dict) -> pd.DataFrame:
    """polars -> pandas, keeping only model-input columns (the D-015 boundary)."""
    columns = feature_columns(params)
    pandas_frame = frame.select(columns).to_pandas()
    # not_credit_policy is boolean; sklearn wants a number.
    for column in params["features"]["groups"]["boolean"]:
        pandas_frame[column] = pandas_frame[column].astype("float64")
    return pandas_frame
