"""When should the model be retrained? (D-067)

A drift number on a dashboard is not a decision. This module turns the monthly
reports into one of three outcomes: do nothing, retrain, or call an engineer.

Two rules shape it:

1. **Persistence, not spikes.** A single month is often an artifact - a
   marketing push, a holiday, a product change that reverts. Requiring the same
   breach in consecutive months trades a month of delay for far fewer false
   alarms, and retraining is not free: it consumes the promotion gate, and every
   model swap carries risk.

2. **Drift and breakage are different alerts with different owners.** Retraining
   fixes a world that has moved. It does not fix a column that suddenly arrives
   empty, or a new category nobody implemented - those need a human to look at
   the pipeline (D-068). Firing "retrain" at a broken feed would train the next
   model on the same broken data.

Kept as pure functions over plain summaries so the logic can be tested against
handmade months instead of a two-hour pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BatchSummary:
    """One month of production, reduced to the numbers a decision needs."""

    batch: str  # "2016-03"
    loans: int
    drift_share: float  # share of monitored features flagged as drifted
    prediction_drift: float  # movement of the model's own score distribution
    drifted_columns: list[str] = field(default_factory=list)
    fund_rate: float = 0.0
    mean_probability: float = 0.0
    outcome_known: float = 0.0  # share with a final status; NOT a quality metric
    missing_increase: dict[str, float] = field(default_factory=dict)
    new_categories: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    kind: str  # "retrain" or "investigate"
    batch: str
    reason: str


def breaches(summary: BatchSummary, cfg: dict) -> list[str]:
    """Which drift thresholds this month crosses, named for the alert text."""
    crossed = []
    if summary.drift_share > cfg["drift_share"]:
        crossed.append(f"drift share {summary.drift_share:.2f} > {cfg['drift_share']}")
    if summary.prediction_drift > cfg["prediction_drift"]:
        crossed.append(
            f"prediction drift {summary.prediction_drift:.3f} > {cfg['prediction_drift']}"
        )
    return crossed


def missing_breaches(summary: BatchSummary, cfg: dict) -> dict[str, float]:
    limit = cfg["max_missing_increase"]
    return {c: v for c, v in summary.missing_increase.items() if v > limit}


def unseen_categories(summary: BatchSummary) -> set[tuple[str, str]]:
    return {(c, v) for c, values in summary.new_categories.items() for v in values}


def data_quality_alerts(
    summary: BatchSummary,
    cfg: dict,
    reported_categories: frozenset[tuple[str, str]] = frozenset(),
    reported_missing: frozenset[str] = frozenset(),
) -> list[Alert]:
    """Breakage, not drift: fires immediately, without waiting for persistence.

    `reported_*` carry what has already been raised, so a condition that is
    understood and accepted stops paging. Without this the first run of these
    reports produced the same "addr_state has a new value: ND" alert for
    thirty-six consecutive months, which is not an alert, it is a log line —
    and a page people learn to ignore is worse than no page at all.
    """
    alerts = []
    jumped = {c: v for c, v in missing_breaches(summary, cfg).items() if c not in reported_missing}
    if jumped:
        worst = ", ".join(f"{c} +{v:.0%}" for c, v in sorted(jumped.items(), key=lambda kv: -kv[1]))
        alerts.append(
            Alert("investigate", summary.batch, f"missing values jumped vs reference: {worst}")
        )

    new_pairs = unseen_categories(summary) - reported_categories
    if new_pairs:
        named = ", ".join(f"{c}={v}" for c, v in sorted(new_pairs))
        alerts.append(
            Alert("investigate", summary.batch, f"categories never seen in training: {named}")
        )
    return alerts


def evaluate(summaries: list[BatchSummary], cfg: dict) -> list[Alert]:
    """Walk the months in order and return the alerts that would have fired.

    A retrain alert fires on the month that COMPLETES a run of breaching months
    and does not repeat while the run continues: one alert per episode, so an
    on-call engineer is not paged every month by the same known problem. The run
    resets on a clean month, so a later episode does page again.
    """
    needed = cfg["consecutive_months"]
    alerts: list[Alert] = []
    run, fired = 0, False
    reported_categories: set[tuple[str, str]] = set()
    reported_missing: set[str] = set()

    for summary in summaries:
        alerts.extend(
            data_quality_alerts(
                summary, cfg, frozenset(reported_categories), frozenset(reported_missing)
            )
        )
        # A new category is permanent knowledge. A missing-value jump is not:
        # once the column recovers, a fresh jump deserves a fresh alert.
        reported_categories |= unseen_categories(summary)
        reported_missing = set(missing_breaches(summary, cfg))

        crossed = breaches(summary, cfg)
        if not crossed:
            run, fired = 0, False
            continue

        run += 1
        if run >= needed and not fired:
            fired = True
            alerts.append(
                Alert(
                    "retrain",
                    summary.batch,
                    f"{run} consecutive months above threshold ({'; '.join(crossed)})",
                )
            )
    return alerts
