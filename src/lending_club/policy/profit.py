"""Turning a probability into a funding decision (D-009).

A classifier answers "how likely is default?". An investor needs "should I put
$10,000 into this loan?". The bridge is expected value:

    E[return] = (1 - p) x expected_gain_if_repaid - p x loss_if_default

Both sides are estimated from TRAINING loans only, the same discipline applied
to every other learned statistic (D-016). The realized-profit columns used here
come from finished loans and are never features (D-008).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ProfitModel:
    """Economics of a loan, estimated on training data.

    lgd:           share of the funded amount lost when a loan defaults
    prepay_factor: repaid loans return less than the contract promises because
                   borrowers pay off early; this is the measured shortfall
    """

    lgd: float
    prepay_factor: float

    def expected_gain(self, frame: pl.DataFrame) -> np.ndarray:
        return contractual_gain(frame) * self.prepay_factor

    def expected_return(self, frame: pl.DataFrame, p_default: np.ndarray) -> np.ndarray:
        """Dollars expected from funding each loan, at the listed amount."""
        gain = self.expected_gain(frame)
        loss = self.lgd * frame["funded_amnt"].to_numpy()
        return (1 - p_default) * gain - p_default * loss

    def breakeven_probability(self, frame: pl.DataFrame) -> np.ndarray:
        """The default probability at which a loan is exactly worth funding.

        Useful as intuition: a loan whose predicted probability is below its
        breakeven point has positive expected value.
        """
        gain = self.expected_gain(frame)
        loss = self.lgd * frame["funded_amnt"].to_numpy()
        return gain / (gain + loss)


def contractual_gain(frame: pl.DataFrame) -> np.ndarray:
    """Interest the loan would pay if every scheduled payment were made."""
    return (
        frame["installment"].to_numpy() * frame["term_months"].to_numpy()
        - frame["funded_amnt"].to_numpy()
    )


def fit_profit_model(train: pl.DataFrame) -> ProfitModel:
    """Measure loss given default and the prepayment shortfall on training loans."""
    defaulted = train.filter(pl.col("target") == 1)
    repaid = train.filter(pl.col("target") == 0)

    # realized_profit is negative for a default, so -profit / funded is the loss share.
    lgd = float(
        (-defaulted["realized_profit"].to_numpy() / defaulted["funded_amnt"].to_numpy()).mean()
    )
    # Repaid loans earn less than the contract: borrowers refinance or pay early.
    prepay_factor = float((repaid["realized_profit"].to_numpy() / contractual_gain(repaid)).mean())
    return ProfitModel(lgd=lgd, prepay_factor=prepay_factor)


def evaluate_policy(frame: pl.DataFrame, funded: np.ndarray) -> dict[str, float]:
    """What an investor actually earns by funding the selected loans.

    Uses realized outcomes, so this is money, not a proxy metric.
    """
    invested = float(frame["funded_amnt"].to_numpy()[funded].sum())
    profit = float(frame["realized_profit"].to_numpy()[funded].sum())
    targets = frame["target"].to_numpy()[funded]
    return {
        "loans_funded": int(funded.sum()),
        "share_funded": float(funded.mean()),
        "capital_invested": invested,
        "total_profit": profit,
        # The headline number: cents earned per dollar invested over ~3 years.
        "return_per_dollar": profit / invested if invested else 0.0,
        "default_rate_funded": float(targets.mean()) if len(targets) else 0.0,
    }


def threshold_curve(
    frame: pl.DataFrame, p_default: np.ndarray, grid: np.ndarray
) -> list[dict[str, float]]:
    """Profit of the rule "fund when predicted probability < t", for each t."""
    return [{"threshold": float(t)} | evaluate_policy(frame, p_default < t) for t in grid]


def threshold_for_share(p_default: np.ndarray, share: float) -> float:
    """The probability cut-off that funds exactly `share` of the loans.

    Used to compare strategies at equal selectivity: otherwise a policy can
    look better simply by being pickier.
    """
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share}")
    return float(np.quantile(p_default, share))


def choose_threshold(
    frame: pl.DataFrame,
    p_default: np.ndarray,
    grid: np.ndarray,
    objective: str = "return_per_dollar",
    min_share_funded: float = 0.10,
) -> tuple[float, list[dict[str, float]]]:
    """Pick the threshold that maximizes `objective` on the VALIDATION year.

    min_share_funded rules out degenerate strategies: funding the 50 safest
    loans in the country would look wonderful per dollar but is not a usable
    strategy, and its measured return would be mostly luck.
    """
    curve = threshold_curve(frame, p_default, grid)
    usable = [row for row in curve if row["share_funded"] >= min_share_funded]
    if not usable:
        raise ValueError("no threshold funds the minimum share of loans")
    best = max(usable, key=lambda row: row[objective])
    return best["threshold"], curve
