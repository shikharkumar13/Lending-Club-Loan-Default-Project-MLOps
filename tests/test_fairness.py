"""Phase 8: the group audit behind MODEL_CARD.md.

Built on handmade frames with a planted disparity, so the logic is checked
without loading the bundle or the test year.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lending_club.fairness import by_group, disparity


def frame(groups: dict[str, tuple[int, float, float]]) -> pl.DataFrame:
    """{group: (n, share funded, default rate among funded)} -> a scored frame."""
    rows = {"group": [], "funded": [], "risk_score": [], "target": []}
    for name, (n, fund_rate, default_if_funded) in groups.items():
        funded = [i < round(n * fund_rate) for i in range(n)]
        n_funded = sum(funded)
        bad = round(n_funded * default_if_funded)
        rows["group"] += [name] * n
        rows["funded"] += funded
        rows["risk_score"] += [1.0 - fund_rate] * n
        rows["target"] += [1 if (f and i < bad) else 0 for i, f in enumerate(funded)]
    return pl.DataFrame(rows)


def test_group_table_reports_funding_and_realized_error():
    table = by_group(frame({"A": (2_000, 0.80, 0.05)}), "group", min_count=100)
    row = table.to_dicts()[0]
    assert row["n"] == 2_000
    assert row["fund_rate"] == 0.8
    assert row["n_funded"] == 1_600
    assert np.isclose(row["default_if_funded"], 0.05)


def test_small_groups_are_dropped():
    """A rate measured on a handful of loans is noise, not a disparity."""
    table = by_group(frame({"big": (2_000, 0.7, 0.1), "tiny": (40, 0.1, 0.9)}), "group", 1_000)
    assert table["group"].to_list() == ["big"]


def test_the_summary_names_the_extremes():
    table = by_group(
        frame({"low": (2_000, 0.40, 0.14), "high": (2_000, 0.80, 0.06)}), "group", 1_000
    )
    summary = disparity(table, "group", 800)
    assert summary["fund_rate_min"]["group"] == "low"
    assert summary["fund_rate_max"]["group"] == "high"
    assert np.isclose(summary["fund_rate_spread"], 0.40)


def test_equal_error_across_groups_is_visible_as_a_zero_spread():
    """The point of the audit: same bar everywhere means same realized error."""
    table = by_group(frame({"a": (2_000, 0.50, 0.10), "b": (2_000, 0.90, 0.10)}), "group", 1_000)
    summary = disparity(table, "group", 800)
    assert np.isclose(summary["default_if_funded_min"], summary["default_if_funded_max"])


def test_scores_that_track_outcomes_correlate_perfectly():
    """A funding gap fully explained by a real risk gap is not a fairness problem."""
    table = by_group(
        frame({"risky": (2_000, 0.30, 0.20), "safe": (2_000, 0.90, 0.05)}), "group", 1_000
    )
    # risky has the higher mean score AND the higher default rate.
    assert disparity(table, "group", 800)["score_vs_outcome_correlation"] == 1.0
