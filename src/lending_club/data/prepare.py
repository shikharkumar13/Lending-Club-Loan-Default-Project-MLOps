"""Stage 2: clean, label and filter. This is "Layer A" preprocessing (D-016).

Every transform here uses ONE row's own values, so running it before the
train/test split cannot leak information. Anything that learns a statistic from
the data (medians, category lists, caps) belongs in the sklearn pipeline
instead, where it is refitted on each training fold.

The stage also applies the two filters that make the labels honest:
  * only loans with a known final outcome (D-003)
  * only 36-month loans issued early enough to have finished (D-004)
"""

from __future__ import annotations

import polars as pl

from lending_club.config import load_params, path_of

CREDIT_POLICY_PREFIX = "Does not meet the credit policy. Status:"


def _emp_length_to_years(col: str = "emp_length") -> pl.Expr:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing stays missing.

    Kept as a number because employment length is ordered; one-hot encoding
    would throw that ordering away.
    """
    return (
        pl.when(pl.col(col).str.contains("10\\+"))
        .then(pl.lit(10))
        .when(pl.col(col).str.contains("< 1"))
        .then(pl.lit(0))
        .otherwise(pl.col(col).str.extract(r"(\d+)").cast(pl.Int8))
        .cast(pl.Int8)
        .alias("emp_length_years")
    )


def transform(frame: pl.DataFrame, params: dict) -> pl.DataFrame:
    """All the cleaning logic, as a pure function.

    Kept separate from file reading/writing so tests can feed it a handful of
    handmade rows instead of the real 1.6 GB dataset. Fast tests get run; slow
    ones get skipped.
    """
    data_cfg, feature_cfg = params["data"], params["features"]

    frame = frame.with_columns(
        # Dates arrive as 'Dec-2015'.
        pl.col("issue_d").str.to_date("%b-%Y").alias("issue_date"),
        pl.col("earliest_cr_line").str.to_date("%b-%Y").alias("earliest_cr_date"),
        # ' 36 months' -> 36
        pl.col("term").str.strip_chars().str.head(2).cast(pl.Int16).alias("term_months"),
        # Loans Lending Club flags as outside its own credit policy: keep the
        # outcome, keep the flag as a feature (D-026).
        pl.col("loan_status").str.starts_with("Does not meet").alias("not_credit_policy"),
        pl.col("loan_status")
        .str.replace(CREDIT_POLICY_PREFIX, "", literal=True)
        .alias("status_clean"),
        _emp_length_to_years(),
    )

    # --- label (D-003) -------------------------------------------------------
    positive = data_cfg["positive_statuses"]
    negative = data_cfg["negative_statuses"]
    frame = frame.filter(pl.col("status_clean").is_in(positive + negative)).with_columns(
        pl.col("status_clean").is_in(positive).cast(pl.Int8).alias("target")
    )

    # --- maturity filter (D-004) --------------------------------------------
    frame = frame.filter(
        pl.col("term_months").is_in(data_cfg["terms_months"]),
        pl.col("issue_date") >= pl.lit(data_cfg["min_issue_date"]).str.to_date(),
        pl.col("issue_date") <= pl.lit(data_cfg["max_issue_date"]).str.to_date(),
    )

    # --- derived features (still row-by-row) --------------------------------
    frame = frame.with_columns(
        # FICO is reported as a 4-5 point band; the midpoint carries the same
        # information in one column instead of two nearly identical ones.
        ((pl.col("fico_range_low") + pl.col("fico_range_high")) / 2).alias("fico_mid"),
        # How long the borrower has had credit, at the time of THIS loan.
        # Using issue_date (not today) keeps the feature point-in-time correct.
        ((pl.col("issue_date") - pl.col("earliest_cr_date")).dt.total_days() / 30.44).alias(
            "credit_history_months"
        ),
        # Debt burden relative to income. annual_inc == 0 would divide by zero,
        # so those (2 loans) become missing and get imputed downstream.
        pl.when(pl.col("annual_inc") > 0)
        .then(pl.col("loan_amnt") / pl.col("annual_inc"))
        .alias("loan_to_income"),
        pl.when(pl.col("annual_inc") > 0)
        .then(pl.col("installment") * 12 / pl.col("annual_inc"))
        .alias("installment_to_income"),
        # dti uses 999 as a "not available" sentinel in a handful of rows.
        pl.when(pl.col("dti") >= 999).then(None).otherwise(pl.col("dti")).alias("dti"),
        # Realized profit: EVALUATION ONLY, never a feature (D-008).
        (pl.col("total_pymnt") - pl.col("funded_amnt")).alias("realized_profit"),
    )

    keep = (
        [c for c in feature_cfg["allowlist"] if c in frame.columns]
        + [
            "emp_length_years",
            "fico_mid",
            "credit_history_months",
            "loan_to_income",
            "installment_to_income",
            "not_credit_policy",
            "term_months",
        ]
        + ["issue_date", "target", "realized_profit", "funded_amnt", "total_pymnt"]
    )
    return frame.select(sorted(set(keep), key=keep.index))


def prepare() -> pl.DataFrame:
    """Read the ingested parquet, transform it, write the prepared parquet."""
    frame = transform(pl.read_parquet(path_of("raw_parquet")), load_params())

    dst = path_of("prepared")
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(dst)
    rate = frame["target"].mean()
    print(f"prepare: {frame.height:,} rows, default rate {rate:.4f} -> {dst}")
    return frame


if __name__ == "__main__":
    prepare()
