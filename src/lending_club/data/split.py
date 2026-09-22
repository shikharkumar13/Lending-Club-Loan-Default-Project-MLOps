"""Stage 3: split by issue date, and define the cross-validation folds.

Everything here is about TIME. A random split would let the model learn from
2013 loans and be scored on 2010 loans, which is impossible in production and
makes results look better than they are (D-005).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lending_club.config import load_params, path_of
from lending_club.features.schema import check_missingness, validate

SPLIT_NAMES = ("train", "validation", "test")


def assign_split(frame: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Add a `split` column: train / validation / test / unused."""
    cfg = params["split"]
    expr = pl.lit("unused")
    for name in reversed(SPLIT_NAMES):  # reversed so `train` is checked first
        bounds = cfg[name]
        expr = (
            pl.when(
                pl.col("issue_date").is_between(
                    pl.lit(bounds["start"]).str.to_date(),
                    pl.lit(bounds["end"]).str.to_date(),
                )
            )
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return frame.with_columns(expr.alias("split"))


def cv_folds(train: pl.DataFrame, params: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: train on all earlier years, validate on the next (D-022).

    Returned as plain index arrays, which scikit-learn accepts directly as `cv=`.
    sklearn's own TimeSeriesSplit is not usable here because it splits by row
    count, ignoring the calendar.
    """
    years = train["issue_date"].dt.year().to_numpy()
    dates = train["issue_date"].to_numpy()

    folds = []
    for fold in params["split"]["cv_folds"]:
        train_end = np.datetime64(fold["train_end"])
        train_idx = np.flatnonzero(dates <= train_end)
        val_idx = np.flatnonzero(years == fold["validate_year"])
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"empty fold: {fold}")
        folds.append((train_idx, val_idx))
    return folds


def split() -> dict[str, pl.DataFrame]:
    params = load_params()
    frame = assign_split(pl.read_parquet(path_of("prepared")), params)

    out_dir = path_of("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = {}
    for name in SPLIT_NAMES:
        part = frame.filter(pl.col("split") == name).drop("split")
        # Fail here, loudly, rather than during training three stages later.
        validate(part, params)
        if name == "train":
            # Whether a column is usable is decided on training data only.
            check_missingness(part, params)
        part.write_parquet(out_dir / f"{name}.parquet")
        parts[name] = part
        print(
            f"{name:11s} {part.height:>7,} loans  "
            f"default rate {part['target'].mean():.4f}  "
            f"{part['issue_date'].min()} -> {part['issue_date'].max()}"
        )

    for i, (tr, va) in enumerate(cv_folds(parts["train"], params), start=1):
        print(f"  cv fold {i}: train {len(tr):>6,} rows | validate {len(va):>6,} rows")
    return parts


if __name__ == "__main__":
    split()
