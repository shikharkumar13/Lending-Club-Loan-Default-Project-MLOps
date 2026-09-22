"""Tests for the time-based split and the cross-validation folds (D-005, D-022)."""

from datetime import date

import polars as pl
import pytest

from lending_club.config import load_params
from lending_club.data.split import assign_split, cv_folds

PARAMS = load_params()


def frame_with_dates(*dates: str) -> pl.DataFrame:
    return pl.DataFrame({"issue_date": [date.fromisoformat(d) for d in dates]})


def test_split_boundaries():
    out = assign_split(
        frame_with_dates(
            "2013-12-01",  # last training month
            "2014-01-01",  # first validation month
            "2014-12-01",  # last validation month
            "2015-01-01",  # first test month
            "2016-06-01",  # after the maturity cutoff
        ),
        PARAMS,
    )
    assert out["split"].to_list() == [
        "train",
        "validation",
        "validation",
        "test",
        "unused",
    ]


def test_each_row_gets_exactly_one_split():
    """A row can only belong to one split; overlapping windows would leak."""
    out = assign_split(frame_with_dates("2010-05-01", "2014-06-01", "2015-06-01"), PARAMS)
    assert out["split"].to_list() == ["train", "validation", "test"]


def test_cv_folds_never_train_on_the_future():
    train = frame_with_dates("2009-06-01", "2010-06-01", "2011-06-01", "2012-06-01", "2013-06-01")
    folds = cv_folds(train, PARAMS)
    assert len(folds) == 3

    dates = train["issue_date"].to_list()
    for (train_idx, val_idx), cfg in zip(folds, PARAMS["split"]["cv_folds"], strict=True):
        latest_train = max(dates[i] for i in train_idx)
        earliest_val = min(dates[i] for i in val_idx)
        assert latest_train < earliest_val, "fold trains on data after its validation year"
        assert set(train_idx).isdisjoint(val_idx), "a loan appears in both sides of a fold"
        assert all(dates[i].year == cfg["validate_year"] for i in val_idx)


def test_cv_fold_with_no_rows_raises():
    """Silently empty folds would make a CV score meaningless."""
    with pytest.raises(ValueError, match="empty fold"):
        cv_folds(frame_with_dates("2013-06-01"), PARAMS)
