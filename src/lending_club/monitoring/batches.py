"""Replay 2016-2018 loans as a stream of monthly production batches (D-064).

The model was trained on 2007-2013 and reported on 2015. Everything after that
was excluded on purpose: a 36-month loan issued in 2016 has not finished by the
2018Q4 snapshot, so most of these loans have no label.

That is not a gap in the exercise — it is the whole reason monitoring exists.
In production the outcome of today's decision arrives in three years. Waiting
for it is not a monitoring strategy. So we watch the two things that ARE
visible immediately:

    * the inputs   - do incoming applications still look like training data?
    * the outputs  - is the model's own score distribution moving?

Each month is scored with the deployed bundle exactly as the API would score
it, so what we monitor is what production would actually have produced.
"""

from __future__ import annotations

import polars as pl

from lending_club.config import load_params, path_of
from lending_club.data.prepare import CREDIT_POLICY_PREFIX, derive_features
from lending_club.serving.bundle import LoanDecisionModel, prepare_for_decision

RESOLVED_STATUSES = ["Fully Paid", "Charged Off", "Default"]


def production_frame(frame: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Row-level cleaning for loans whose outcome is NOT known yet.

    Deliberately not `prepare.transform()`: that function drops every loan
    without a final status, which would throw away the entire population we
    want to monitor. It shares `derive_features`, so the features themselves
    are computed by the same code that trained the model (D-056).
    """
    cfg = params["data"]
    monitoring = params["monitoring"]

    return (
        derive_features(frame)
        .with_columns(
            pl.col("loan_status").str.starts_with("Does not meet").alias("not_credit_policy"),
            pl.col("loan_status")
            .str.replace(CREDIT_POLICY_PREFIX, "", literal=True)
            .alias("status_clean"),
        )
        .filter(
            pl.col("term_months").is_in(cfg["terms_months"]),
            pl.col("issue_date") >= pl.lit(monitoring["start"]).str.to_date(),
            pl.col("issue_date") <= pl.lit(monitoring["end"]).str.to_date(),
        )
        .with_columns(
            pl.col("issue_date").dt.strftime("%Y-%m").alias("batch"),
            # Known outcomes exist for some of these loans, but only because
            # they ended EARLY - paid off ahead of schedule, or charged off in
            # the first year. Scoring a model on them would flatter or punish
            # it for reasons that have nothing to do with its quality, so this
            # column is reported as coverage and never used as a metric (D-065).
            pl.col("status_clean").is_in(RESOLVED_STATUSES).alias("outcome_known"),
        )
    )


def score_batches(frame: pl.DataFrame, bundle: LoanDecisionModel, params: dict) -> pl.DataFrame:
    """Attach the deployed model's score, probability and funding decision."""
    decisions = bundle.decide(prepare_for_decision(frame, params))
    return frame.with_columns(
        pl.Series("risk_score", decisions["risk_score"].to_numpy()),
        pl.Series("probability_of_default", decisions["probability_of_default"].to_numpy()),
        pl.Series("decision", decisions["decision"].to_numpy()),
    )


def reference_frame(bundle: LoanDecisionModel, params: dict) -> pl.DataFrame:
    """The "normal" the monthly batches are compared against.

    The training split, scored by the same bundle. Comparing against the data
    the model learned from is what makes a drift alarm actionable: its imputation
    medians, percentile caps and category lists all come from exactly these rows.
    Using 2015 instead would answer a different question ("has anything changed
    since the last report?") and would not tell you the model's fitted
    statistics had gone stale.
    """
    split = params["monitoring"]["reference_split"]
    frame = pl.read_parquet(path_of("processed_dir") / f"{split}.parquet")
    return score_batches(frame, bundle, params)


def build() -> pl.DataFrame:
    """Score every 2016-2018 month and write one parquet per month."""
    params = load_params()
    bundle = LoanDecisionModel.load()

    raw = pl.read_parquet(path_of("raw_parquet"))
    frame = score_batches(production_frame(raw, params), bundle, params)

    out_dir = path_of("monitoring_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    for month, part in frame.partition_by("batch", as_dict=True).items():
        part.write_parquet(out_dir / f"{month[0]}.parquet")

    summary = (
        frame.group_by("batch")
        .agg(
            pl.len().alias("loans"),
            pl.col("outcome_known").mean().alias("outcome_known"),
            (pl.col("decision") == "fund").mean().alias("fund_rate"),
            pl.col("probability_of_default").mean().alias("mean_probability"),
        )
        .sort("batch")
    )
    print(f"monitoring: {frame.height:,} loans across {summary.height} months -> {out_dir}")
    print(summary.head(3))
    return summary


if __name__ == "__main__":
    build()
