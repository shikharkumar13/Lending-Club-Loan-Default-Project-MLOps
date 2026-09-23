"""Phase 7: drift monitoring and the retraining trigger.

The trigger tests use handmade months rather than real reports: the decision
logic is what has to be right, and it must be testable without a two-hour
pipeline run.
"""

from __future__ import annotations

import polars as pl
import pytest

from lending_club.config import load_params
from lending_club.features.pipeline import feature_columns
from lending_club.monitoring.batches import production_frame
from lending_club.monitoring.drift import PREDICTION_COLUMN, monitored_columns, summarize_batch
from lending_club.monitoring.trigger import Alert, BatchSummary, data_quality_alerts, evaluate
from lending_club.testing.synthetic import synthetic_raw

TRIGGER = {
    "drift_share": 0.30,
    "prediction_drift": 0.15,
    "consecutive_months": 2,
    "max_missing_increase": 0.20,
}


def month(batch: str, drift_share: float = 0.0, prediction_drift: float = 0.0, **kwargs):
    return BatchSummary(
        batch=batch,
        loans=1000,
        drift_share=drift_share,
        prediction_drift=prediction_drift,
        **kwargs,
    )


def kinds(alerts: list[Alert], kind: str) -> list[str]:
    return [a.batch for a in alerts if a.kind == kind]


# --- the retraining trigger ------------------------------------------------


def test_one_bad_month_does_not_trigger_a_retrain():
    """A single spike is usually an artifact, not a changed world."""
    alerts = evaluate([month("2016-01", drift_share=0.9), month("2016-02")], TRIGGER)
    assert kinds(alerts, "retrain") == []


def test_two_consecutive_bad_months_trigger_a_retrain():
    alerts = evaluate(
        [month("2016-01", drift_share=0.9), month("2016-02", drift_share=0.8)], TRIGGER
    )
    assert kinds(alerts, "retrain") == ["2016-02"]


def test_prediction_drift_alone_is_enough():
    """The inputs can look fine while the model's own output moves."""
    alerts = evaluate(
        [month("2016-01", prediction_drift=0.4), month("2016-02", prediction_drift=0.4)], TRIGGER
    )
    assert kinds(alerts, "retrain") == ["2016-02"]
    assert "prediction drift" in alerts[0].reason
    # Only the threshold that actually broke is named.
    assert "drift share" not in alerts[0].reason


def test_a_continuing_episode_alerts_once():
    """Six bad months in a row is one problem, not six pages."""
    months = [month(f"2016-{i:02d}", drift_share=0.9) for i in range(1, 7)]
    assert kinds(evaluate(months, TRIGGER), "retrain") == ["2016-02"]


def test_a_clean_month_resets_the_run_and_a_later_episode_alerts_again():
    months = [
        month("2016-01", drift_share=0.9),
        month("2016-02", drift_share=0.9),
        month("2016-03"),
        month("2016-04", drift_share=0.9),
        month("2016-05", drift_share=0.9),
    ]
    assert kinds(evaluate(months, TRIGGER), "retrain") == ["2016-02", "2016-05"]


# --- data quality: a different alert, with a different owner ---------------


def test_a_missing_value_jump_is_investigate_not_retrain():
    """Retraining on a broken feed just trains the next model on broken data."""
    alerts = evaluate([month("2016-01", missing_increase={"revol_util": 0.8})], TRIGGER)
    assert kinds(alerts, "investigate") == ["2016-01"]
    assert kinds(alerts, "retrain") == []


def test_a_small_missing_increase_is_ignored():
    alerts = evaluate([month("2016-01", missing_increase={"revol_util": 0.05})], TRIGGER)
    assert alerts == []


def test_a_known_new_category_is_reported_once():
    """The first run paged every month about the same state. Never again."""
    months = [month(f"2016-{i:02d}", new_categories={"addr_state": ["ND"]}) for i in range(1, 7)]
    alerts = evaluate(months, TRIGGER)
    assert kinds(alerts, "investigate") == ["2016-01"]


def test_a_genuinely_new_category_alerts_again():
    months = [
        month("2016-01", new_categories={"addr_state": ["ND"]}),
        month("2016-02", new_categories={"addr_state": ["ND"]}),
        month("2016-03", new_categories={"addr_state": ["ND"], "home_ownership": ["ANY"]}),
    ]
    alerts = evaluate(months, TRIGGER)
    assert kinds(alerts, "investigate") == ["2016-01", "2016-03"]
    assert "home_ownership=ANY" in alerts[-1].reason
    assert "addr_state" not in alerts[-1].reason


def test_a_missing_jump_alerts_again_after_the_column_recovers():
    months = [
        month("2016-01", missing_increase={"dti": 0.9}),
        month("2016-02", missing_increase={"dti": 0.9}),
        month("2016-03"),
        month("2016-04", missing_increase={"dti": 0.9}),
    ]
    assert kinds(evaluate(months, TRIGGER), "investigate") == ["2016-01", "2016-04"]


def test_data_quality_alerts_can_be_called_on_a_single_month():
    alerts = data_quality_alerts(month("2016-01", new_categories={"purpose": ["wedding"]}), TRIGGER)
    assert len(alerts) == 1 and alerts[0].kind == "investigate"


# --- what gets monitored ----------------------------------------------------


def test_every_model_input_is_monitored():
    """A feature the model uses but nobody watches is the gap drift hides in."""
    params = load_params()
    numerical, categorical = monitored_columns(params)
    assert set(numerical + categorical) == set(feature_columns(params))


# --- building the production batches ----------------------------------------


@pytest.fixture
def unlabeled_2016() -> pl.DataFrame:
    """Loans issued in 2016 that have not finished: no label, like production."""
    return synthetic_raw(1_500, seed=3).with_columns(
        pl.lit("Mar-2016").alias("issue_d"), pl.lit("Current").alias("loan_status")
    )


def test_production_frame_keeps_loans_with_no_outcome(unlabeled_2016):
    """`prepare.transform` would drop all of these; monitoring must not."""
    params = load_params()
    frame = production_frame(unlabeled_2016, params)
    assert frame.height == unlabeled_2016.height
    assert not frame["outcome_known"].any()
    assert frame["batch"].unique().to_list() == ["2016-03"]


def test_production_frame_excludes_loans_outside_the_window(unlabeled_2016):
    params = load_params()
    older = unlabeled_2016.with_columns(pl.lit("Mar-2015").alias("issue_d"))
    assert production_frame(older, params).height == 0


def test_production_frame_flags_resolved_outcomes(unlabeled_2016):
    params = load_params()
    mixed = unlabeled_2016.with_columns(
        pl.Series("loan_status", ["Fully Paid"] * 500 + ["Current"] * 1_000)
    )
    frame = production_frame(mixed, params)
    assert frame["outcome_known"].sum() == 500


# --- the drift report itself ------------------------------------------------


def scored(frame: pl.DataFrame, params: dict, score: float) -> pl.DataFrame:
    """A production_frame with the prediction columns a summary needs."""
    return production_frame(frame, params).with_columns(
        pl.lit(score).alias(PREDICTION_COLUMN),
        pl.lit(score).alias("probability_of_default"),
        pl.lit("fund").alias("decision"),
    )


def test_identical_months_show_no_drift(unlabeled_2016):
    params = load_params()
    frame = scored(unlabeled_2016, params, 0.1)
    summary = summarize_batch(frame, frame, params)
    assert summary.drift_share == 0.0
    assert summary.prediction_drift == 0.0
    assert summary.new_categories == {}


def test_a_shifted_month_is_detected(unlabeled_2016):
    """Triple every income and the drift report has to notice."""
    params = load_params()
    reference = scored(unlabeled_2016, params, 0.1)
    current = scored(unlabeled_2016.with_columns(pl.col("annual_inc") * 3), params, 0.1)
    summary = summarize_batch(current, reference, params)
    assert "annual_inc" in summary.drifted_columns
    assert summary.drift_share > 0


def test_an_unseen_category_is_reported(unlabeled_2016):
    params = load_params()
    reference = scored(unlabeled_2016, params, 0.1)
    current = scored(unlabeled_2016.with_columns(pl.lit("ND").alias("addr_state")), params, 0.1)
    summary = summarize_batch(current, reference, params)
    assert summary.new_categories == {"addr_state": ["ND"]}
