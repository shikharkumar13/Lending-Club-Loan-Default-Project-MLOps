"""Tests for Layer B preprocessing.

These are the leakage tests that matter most: every statistic the pipeline uses
must come from the rows it was fitted on, never from the data it later scores.
"""

import numpy as np
import pandas as pd
import pytest

from lending_club.config import load_params
from lending_club.features.pipeline import build_preprocessor, feature_columns

PARAMS = load_params()
COLUMNS = feature_columns(PARAMS)

DEFAULTS = {
    "annual_inc": 60_000.0,
    "revol_bal": 5_000.0,
    "loan_amnt": 10_000.0,
    "int_rate": 12.0,
    "installment": 330.0,
    "dti": 15.0,
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 1.0,
    "open_acc": 8.0,
    "pub_rec": 0.0,
    "revol_util": 50.0,
    "total_acc": 20.0,
    "pub_rec_bankruptcies": 0.0,
    "acc_now_delinq": 0.0,
    "collections_12_mths_ex_med": 0.0,
    "emp_length_years": 3.0,
    "fico_mid": 692.0,
    "credit_history_months": 120.0,
    "loan_to_income": 0.17,
    "installment_to_income": 0.07,
    "mths_since_last_delinq": None,
    "mths_since_last_record": None,
    "grade": "B",
    "sub_grade": "B3",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "credit_card",
    "addr_state": "CA",
    "initial_list_status": "f",
    "not_credit_policy": 0.0,
}


STRING_COLUMNS = {
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
}


def frame(n: int = 200, **overrides) -> pd.DataFrame:
    data = {c: [DEFAULTS[c]] * n for c in COLUMNS}
    for column, values in overrides.items():
        data[column] = list(values) if isinstance(values, list) else [values] * n
    out = pd.DataFrame(data)[COLUMNS]
    # An all-None column would otherwise be inferred as object dtype, which the
    # real parquet data never produces.
    numeric = [c for c in COLUMNS if c not in STRING_COLUMNS]
    return out.astype({c: "float64" for c in numeric})


def index_of(pipeline, name: str) -> int:
    return list(pipeline.get_feature_names_out()).index(name)


def test_percentile_cap_comes_from_training_data():
    """An income far above anything seen in training must be clipped to the
    training cap, not re-scaled by the new data (D-019)."""
    train = frame(n=200, annual_inc=[50_000.0] * 199 + [100_000.0])
    pipeline = build_preprocessor(PARAMS, "tree").fit(train)

    out = pipeline.transform(frame(n=1, annual_inc=9_000_000.0))
    cap = pipeline.named_steps["columns"].named_transformers_["skewed"][0].caps_[0]
    assert out[0, index_of(pipeline, "log_annual_inc")] == pytest.approx(np.log1p(cap))
    assert cap <= 100_000.0


def test_median_imputation_comes_from_training_data():
    train = frame(n=100, dti=[10.0] * 100)
    pipeline = build_preprocessor(PARAMS, "tree").fit(train)

    # A later batch where dti is missing must be filled with the TRAIN median.
    out = pipeline.transform(frame(n=1, dti=np.nan))
    assert out[0, index_of(pipeline, "dti")] == pytest.approx(10.0)


def test_scaling_statistics_come_from_training_data():
    train = frame(n=100, loan_amnt=[10_000.0] * 50 + [20_000.0] * 50)
    pipeline = build_preprocessor(PARAMS, "linear").fit(train)
    scaler = pipeline.named_steps["scale"]
    position = list(pipeline.named_steps["columns"].get_feature_names_out()).index("loan_amnt")
    assert scaler.mean_[position] == pytest.approx(15_000.0)


def test_unseen_category_does_not_crash():
    """A state that never appeared in training must not raise at predict time."""
    train = frame(n=100, addr_state=["CA"] * 60 + ["NY"] * 40)
    pipeline = build_preprocessor(PARAMS, "tree").fit(train)
    out = pipeline.transform(frame(n=1, addr_state="WY"))
    assert out.shape[0] == 1 and not np.isnan(out).any()


def test_rare_categories_are_grouped():
    """Levels under the 1% threshold collapse into one 'infrequent' column (D-018)."""
    train = frame(n=1000, purpose=["credit_card"] * 995 + ["wedding"] * 5)
    pipeline = build_preprocessor(PARAMS, "tree").fit(train)
    names = list(pipeline.get_feature_names_out())
    assert "purpose_infrequent_sklearn" in names
    assert "purpose_wedding" not in names


def test_structural_missing_becomes_sentinel_plus_flag():
    """Missing 'months since last delinquency' means 'never happened' (D-031)."""
    pipeline = build_preprocessor(PARAMS, "tree").fit(frame(n=50))
    out = pipeline.transform(frame(n=1, mths_since_last_delinq=None))
    names = list(pipeline.get_feature_names_out())
    sentinel = PARAMS["features"]["never_happened_sentinel"]
    assert out[0, names.index("mths_since_last_delinq")] == sentinel
    assert any("missingindicator_mths_since_last_delinq" in n for n in names)


def test_output_has_no_missing_values():
    pipeline = build_preprocessor(PARAMS, "tree").fit(frame(n=100))
    assert not np.isnan(pipeline.transform(frame(n=50))).any()


def test_only_allowed_columns_reach_the_model():
    """Bookkeeping columns must never become features (D-006, D-008)."""
    forbidden = tuple(PARAMS["features"]["forbidden_prefixes"])
    banned = {"issue_date", "target", "realized_profit", "funded_amnt", "total_pymnt"}
    assert not [c for c in COLUMNS if c.startswith(forbidden) or c in banned]


def test_train_and_serve_produce_the_same_column_order():
    """Column order must be stable, or a served model silently reads the wrong values."""
    pipeline = build_preprocessor(PARAMS, "tree").fit(frame(n=100))
    shuffled = frame(n=1)[list(reversed(COLUMNS))]
    assert np.allclose(pipeline.transform(frame(n=1)), pipeline.transform(shuffled))
