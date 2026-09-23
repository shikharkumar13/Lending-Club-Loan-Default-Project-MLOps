"""Prefect flow: the retraining run, end to end, with a promotion gate.

Run it locally with:

    PREFECT_API_URL= PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=true python flows/retrain.py

(The empty PREFECT_API_URL ignores any Prefect server configured for another
project on this machine and runs the flow in-process instead.)


    ingest -> prepare -> split -> train -> evaluate -> candidate -> gate -> promote

Why an orchestrator rather than a shell script: each step becomes a unit with
its own logs, retries and timing, a failure names the step that failed, and the
same flow can be run on a schedule or triggered by drift (Phase 7) without
rewriting anything.

Why the gate is the point: a scheduled retrain without one is a way to ship a
worse model automatically. Every task can succeed while the result is bad.
"""

from __future__ import annotations

import polars as pl
from prefect import flow, get_run_logger, task

from lending_club.config import load_params, path_of
from lending_club.data.ingest import ingest
from lending_club.data.prepare import prepare
from lending_club.data.split import split
from lending_club.evaluate import evaluate
from lending_club.models.promote import PromotionDecision, should_promote
from lending_club.models.train import train
from lending_club.serving.bundle import (
    LoanDecisionModel,
    build_bundle,
    default_bundle_path,
)


# Reading a 1.6 GB CSV can fail on a transient I/O error; retry once.
@task(name="ingest", retries=1, retry_delay_seconds=10)
def ingest_task() -> int:
    return ingest().height


@task(name="prepare")
def prepare_task() -> int:
    return prepare().height


@task(name="split")
def split_task() -> dict[str, int]:
    return {name: frame.height for name, frame in split().items()}


@task(name="train", timeout_seconds=3600)
def train_task() -> dict:
    return train()


@task(name="evaluate", timeout_seconds=1800)
def evaluate_task() -> dict:
    """Calibrates, tunes the funding policy, and writes the report.

    The test-year numbers it produces are for the report only. Nothing in this
    flow makes a decision from them (D-061).
    """
    return evaluate()


@task(name="promotion_gate")
def gate_task() -> PromotionDecision:
    params = load_params()
    candidate = build_bundle()

    # The gate judges on the policy-tuning half of 2014 - validation data,
    # never the test year.
    validation = pl.read_parquet(path_of("processed_dir") / "validation.parquet")
    tuning_rows = validation.filter(pl.col("issue_date") >= pl.date(2014, 7, 1))

    champion_path = default_bundle_path()
    champion = LoanDecisionModel.load(champion_path) if champion_path.exists() else None

    decision = should_promote(candidate, champion, tuning_rows, params)
    if decision.promote:
        candidate.save(champion_path)
    return decision


@flow(name="lending-club-retrain", log_prints=True)
def retrain(skip_ingest: bool = False) -> PromotionDecision:
    logger = get_run_logger()

    if not skip_ingest:
        rows = ingest_task()
        logger.info("ingested %s raw loans", f"{rows:,}")
    logger.info("prepared %s labeled loans", f"{prepare_task():,}")
    logger.info("splits: %s", split_task())

    selected = train_task()["selected"]
    for model in selected:
        logger.info("cv %s: log_loss=%.5f", model["family"], model["cv"]["log_loss"])

    report = evaluate_task()
    logger.info(
        "test (report only): lightgbm return/dollar %.4f",
        report["results"]["lightgbm"]["policies"]["threshold"]["return_per_dollar"],
    )

    decision = gate_task()
    outcome = "PROMOTED" if decision.promote else "REJECTED"
    logger.info("promotion: %s - %s", outcome, decision.reason)
    return decision


if __name__ == "__main__":
    retrain()
