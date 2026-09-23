"""Monthly drift and data-quality reports (D-066).

Evidently does the statistically awkward part - comparing two distributions per
column and giving a human-readable HTML report. Two choices make its output
usable as a *signal* rather than a dashboard:

* **The test is pinned.** Left alone, Evidently picks a statistical test based
  on sample size, so a 40,000-loan month and a 900-loan month would be measured
  with different tests and their numbers would not be comparable. We fix
  Wasserstein for numeric and Jensen-Shannon for categorical columns: both are
  distances, both bounded below by 0, both "higher = more drift".

* **Drift is measured on raw inputs, not the 73 transformed features.** A
  transformed column is partly an artifact of the fitted preprocessor, and
  "feature 47 drifted" is not something anyone can act on. "Income verification
  mix changed" is.

Missing-value shares and unseen categories are computed directly in Polars
rather than read back out of the report: they are exact counts, and keeping
them in our own code means the retraining trigger does not depend on the shape
of a third-party library's output.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import polars as pl
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from lending_club.config import load_params, path_of
from lending_club.monitoring.batches import reference_frame
from lending_club.monitoring.trigger import BatchSummary, evaluate
from lending_club.serving.bundle import LoanDecisionModel

PREDICTION_COLUMN = "risk_score"


def monitored_columns(params: dict) -> tuple[list[str], list[str]]:
    """The model's own inputs, split into numeric and categorical."""
    groups = params["features"]["groups"]
    numerical = (
        list(groups["skewed_amount"])
        + list(groups["numeric"])
        + list(groups["numeric_with_indicator"])
        + list(groups["structural_missing"])
    )
    categorical = ["grade", "sub_grade"] + list(groups["nominal"]) + list(groups["boolean"])
    return numerical, categorical


def _definition(params: dict) -> tuple[DataDefinition, list[str]]:
    numerical, categorical = monitored_columns(params)
    columns = numerical + categorical
    definition = DataDefinition(
        numerical_columns=numerical + [PREDICTION_COLUMN],
        categorical_columns=categorical,
    )
    return definition, columns


def _report(params: dict) -> Report:
    cfg = params["monitoring"]["drift"]
    definition, columns = _definition(params)
    return Report(
        [
            DataDriftPreset(
                columns=columns + [PREDICTION_COLUMN],
                num_method=cfg["num_method"],
                cat_method=cfg["cat_method"],
                threshold=cfg["threshold"],
            ),
            DataSummaryPreset(),
        ],
        include_tests=False,
    )


def _drift_values(snapshot) -> dict[str, float]:
    """Pull the per-column drift statistics out of an Evidently snapshot."""
    values = {}
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("ValueDrift"):
            values[config["column"]] = float(metric["value"])
    return values


def _quality(current: pl.DataFrame, reference: pl.DataFrame, params: dict) -> dict:
    """Exact data-quality deltas: missingness jumps and unseen categories."""
    numerical, categorical = monitored_columns(params)
    columns = numerical + categorical

    def missing_share(frame: pl.DataFrame) -> dict[str, float]:
        row = frame.select(pl.col(c).is_null().mean().alias(c) for c in columns).row(0)
        return dict(zip(columns, row, strict=True))

    current_missing, reference_missing = missing_share(current), missing_share(reference)
    increase = {
        c: delta
        for c in columns
        if (delta := round(current_missing[c] - reference_missing[c], 4)) > 0
    }

    new_categories = {}
    for column in categorical:
        seen = set(reference[column].unique().to_list())
        unseen = set(current[column].unique().to_list()) - seen - {None}
        if unseen:
            new_categories[column] = sorted(str(v) for v in unseen)
    return {"missing_increase": increase, "new_categories": new_categories}


def summarize_batch(
    current: pl.DataFrame,
    reference: pl.DataFrame,
    params: dict,
    html_path=None,
) -> BatchSummary:
    """Compare one month against the reference and reduce it to a summary."""
    definition, columns = _definition(params)
    threshold = params["monitoring"]["drift"]["threshold"]
    keep = columns + [PREDICTION_COLUMN]

    snapshot = _report(params).run(
        Dataset.from_pandas(current.select(keep).to_pandas(), data_definition=definition),
        Dataset.from_pandas(reference.select(keep).to_pandas(), data_definition=definition),
    )
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    values = _drift_values(snapshot)
    # The share counts model inputs only. The prediction is tracked separately:
    # it is an output, and mixing it in would dilute a signal we care about on
    # its own.
    drifted = sorted(c for c in columns if values.get(c, 0.0) > threshold)
    quality = _quality(current, reference, params)

    return BatchSummary(
        batch=current["batch"][0],
        loans=current.height,
        drift_share=round(len(drifted) / len(columns), 4),
        prediction_drift=round(values.get(PREDICTION_COLUMN, 0.0), 4),
        drifted_columns=drifted,
        fund_rate=round(float((current["decision"] == "fund").mean()), 4),
        mean_probability=round(float(current["probability_of_default"].mean()), 4),
        outcome_known=round(float(current["outcome_known"].mean()), 4),
        **quality,
    )


def run() -> dict:
    """Score every month against the reference and apply the trigger."""
    params = load_params()
    bundle = LoanDecisionModel.load()
    reference = reference_frame(bundle, params)

    batch_dir = path_of("monitoring_dir")
    reports_dir = path_of("monitoring_reports_dir")
    files = sorted(batch_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly batches in {batch_dir}; run the batches stage first")

    keep_html = set(params["monitoring"]["html_months"])
    summaries = []
    for path in files:
        current = pl.read_parquet(path)
        html_path = reports_dir / "html" / f"{path.stem}.html" if path.stem in keep_html else None
        summary = summarize_batch(current, reference, params, html_path=html_path)
        summaries.append(summary)
        print(
            f"{summary.batch}  {summary.loans:>6,} loans  "
            f"drift_share={summary.drift_share:.2f}  "
            f"prediction_drift={summary.prediction_drift:.3f}  "
            f"fund_rate={summary.fund_rate:.2f}  "
            f"labeled={summary.outcome_known:.0%}"
        )

    alerts = evaluate(summaries, params["monitoring"]["trigger"])
    payload = {
        "reference": {
            "split": params["monitoring"]["reference_split"],
            "loans": reference.height,
        },
        "config": params["monitoring"]["trigger"] | params["monitoring"]["drift"],
        "batches": [asdict(s) for s in summaries],
        "alerts": [asdict(a) for a in alerts],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "drift.json").write_text(json.dumps(payload, indent=2) + "\n")
    _figure(summaries, params)

    print(f"\n{len(alerts)} alert(s):")
    for alert in alerts:
        print(f"  [{alert.kind}] {alert.batch}: {alert.reason}")
    return payload


def _figure(summaries: list[BatchSummary], params: dict) -> None:
    """Three years of monitoring on one chart, small enough to keep in git.

    The fourth panel is the point of the exercise: the share of loans whose
    outcome is known collapses towards the present, which is why none of the
    other three panels can be an accuracy measurement.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = params["monitoring"]["trigger"]
    months = [s.batch for s in summaries]
    ticks = [i for i, m in enumerate(months) if m.endswith(("-01", "-07"))]

    panels = [
        ("share of features drifted", [s.drift_share for s in summaries], cfg["drift_share"]),
        ("prediction drift", [s.prediction_drift for s in summaries], cfg["prediction_drift"]),
        ("share of loans funded", [s.fund_rate for s in summaries], None),
        ("share with a known outcome", [s.outcome_known for s in summaries], None),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for axis, (title, values, limit) in zip(axes, panels, strict=True):
        axis.plot(months, values, "o-", color="#2a9d8f", markersize=3)
        if limit is not None:
            axis.axhline(limit, ls="--", color="#e76f51", label=f"trigger at {limit}")
            axis.legend(loc="lower right")
        axis.set_ylabel(title, fontsize=9)
        axis.set_ylim(0, max(max(values) * 1.15, (limit or 0) * 1.3))
        axis.grid(alpha=0.25)

    axes[0].set_title("Monitoring 2016-2018: the model was trained on 2007-2013")
    axes[-1].set_xticks(ticks, [months[i] for i in ticks], rotation=45, ha="right")
    plt.tight_layout()

    figures = path_of("figures_dir")
    figures.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures / "drift.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
