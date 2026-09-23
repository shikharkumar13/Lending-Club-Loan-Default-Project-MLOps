"""Who gets funded, and where is the model wrong? (D-071)

The dataset contains no protected attributes, so disparate treatment cannot be
measured directly. What can be measured is whether outcomes differ across
`addr_state` — a well-documented proxy for race and income in the US.

Two questions, and the second is the one people forget:

1. **Does the funding rate differ by group?** Some difference is legitimate:
   groups really do carry different risk.
2. **Is the model equally RIGHT about each group?** If loans funded in one state
   default at 5.8% and in another at 14.3%, the stated threshold means something
   different depending on where the borrower lives. A single headline accuracy
   number hides this completely.

The audit is deliberately cheap to run and writes a JSON report, so the claims
in MODEL_CARD.md can be regenerated rather than trusted.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from lending_club.config import load_params, path_of
from lending_club.serving.bundle import LoanDecisionModel, prepare_for_decision

MIN_GROUP_SIZE = 1_000


def by_group(frame: pl.DataFrame, column: str, min_count: int = MIN_GROUP_SIZE) -> pl.DataFrame:
    """Per-group funding rate, predicted risk and realized outcome.

    `frame` must already carry `funded` and `risk_score`. Groups smaller than
    `min_count` are dropped: a rate measured on 40 loans is noise, and reporting
    it as a disparity would be worse than saying nothing.
    """
    return (
        frame.group_by(column)
        .agg(
            pl.len().alias("n"),
            pl.col("funded").mean().alias("fund_rate"),
            pl.col("risk_score").mean().alias("mean_score"),
            pl.col("target").mean().alias("default_rate"),
            # The key column: among loans we DID fund, how often were we wrong?
            pl.col("target").filter(pl.col("funded")).mean().alias("default_if_funded"),
            pl.col("funded").sum().alias("n_funded"),
        )
        .filter(pl.col("n") >= min_count)
        .sort("fund_rate")
    )


def disparity(groups: pl.DataFrame, column: str) -> dict:
    """Reduce a group table to the few numbers a model card should state."""
    fund = groups["fund_rate"]
    funded_default = groups.filter(pl.col("n_funded") >= 800)["default_if_funded"]
    # How much of the between-group difference in score is justified by a real
    # between-group difference in default rate? 1.0 would mean "entirely".
    correlation = float(np.corrcoef(groups["mean_score"], groups["default_rate"])[0, 1])

    return {
        "column": column,
        "groups": int(groups.height),
        "fund_rate_min": {
            "group": groups[column][int(fund.arg_min())],
            "value": round(float(fund.min()), 4),
        },
        "fund_rate_max": {
            "group": groups[column][int(fund.arg_max())],
            "value": round(float(fund.max()), 4),
        },
        "fund_rate_spread": round(float(fund.max() - fund.min()), 4),
        "default_if_funded_min": round(float(funded_default.min()), 4),
        "default_if_funded_max": round(float(funded_default.max()), 4),
        "score_vs_outcome_correlation": round(correlation, 3),
    }


def scored_test_frame(params: dict | None = None) -> pl.DataFrame:
    """The test year with the deployed bundle's decisions attached."""
    params = params or load_params()
    bundle = LoanDecisionModel.load()
    frame = pl.read_parquet(path_of("processed_dir") / "test.parquet")
    decisions = bundle.decide(prepare_for_decision(frame, params))
    return frame.with_columns(
        pl.Series("funded", (decisions["decision"] == "fund").to_numpy()),
        pl.Series("risk_score", decisions["risk_score"].to_numpy()),
    )


def audit() -> dict:
    params = load_params()
    frame = scored_test_frame(params)

    payload = {
        "test_loans": frame.height,
        "overall_fund_rate": round(float(frame["funded"].mean()), 4),
        "overall_default_if_funded": round(
            float(frame.filter(pl.col("funded"))["target"].mean()), 4
        ),
        "columns": {},
    }
    for column in ("addr_state", "home_ownership", "purpose"):
        groups = by_group(frame, column)
        payload["columns"][column] = {
            "summary": disparity(groups, column),
            "groups": groups.to_dicts(),
        }
        summary = payload["columns"][column]["summary"]
        print(
            f"{column:16s} {summary['groups']:>3} groups  "
            f"fund rate {summary['fund_rate_min']['value']:.3f} "
            f"({summary['fund_rate_min']['group']}) -> "
            f"{summary['fund_rate_max']['value']:.3f} "
            f"({summary['fund_rate_max']['group']})  "
            f"score/outcome corr {summary['score_vs_outcome_correlation']:+.2f}"
        )

    reports = path_of("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "fairness.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    audit()
