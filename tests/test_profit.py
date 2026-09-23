"""Tests for the profit model and the funding policy (D-009)."""

import numpy as np
import polars as pl
import pytest

from lending_club.policy.profit import (
    ProfitModel,
    choose_threshold,
    contractual_gain,
    evaluate_policy,
    fit_profit_model,
    threshold_for_share,
)


def loans(n: int = 4, **overrides) -> pl.DataFrame:
    data = {
        "funded_amnt": [10_000.0] * n,
        "installment": [330.0] * n,
        "term_months": [36] * n,
        "target": [0] * n,
        "realized_profit": [1_500.0] * n,
    }
    data.update(overrides)
    return pl.DataFrame(data)


def test_contractual_gain_is_total_interest():
    # 36 payments of $330 on a $10,000 loan = $11,880 repaid, so $1,880 interest.
    assert contractual_gain(loans(1))[0] == pytest.approx(1_880.0)


def test_profit_model_measures_lgd_and_prepayment():
    frame = loans(
        2,
        target=[1, 0],
        realized_profit=[-4_000.0, 940.0],  # 40% loss; half the interest
    )
    model = fit_profit_model(frame)
    assert model.lgd == pytest.approx(0.4)
    assert model.prepay_factor == pytest.approx(0.5)


def test_expected_return_switches_sign_at_breakeven():
    """Below its breakeven probability a loan is worth funding; above it, not."""
    model = ProfitModel(lgd=0.365, prepay_factor=0.8)
    frame = loans(1)
    breakeven = model.breakeven_probability(frame)[0]

    assert model.expected_return(frame, np.array([breakeven - 0.05]))[0] > 0
    assert model.expected_return(frame, np.array([breakeven + 0.05]))[0] < 0
    assert model.expected_return(frame, np.array([breakeven]))[0] == pytest.approx(0, abs=1e-6)


def test_a_smaller_loss_makes_riskier_loans_worth_funding():
    """The measured LGD of 0.365 (not the assumed 0.65) raises the breakeven
    probability, so the policy funds more loans (D-032)."""
    frame = loans(1)
    optimistic = ProfitModel(lgd=0.365, prepay_factor=1.0).breakeven_probability(frame)[0]
    pessimistic = ProfitModel(lgd=0.650, prepay_factor=1.0).breakeven_probability(frame)[0]
    assert optimistic > pessimistic


def test_evaluate_policy_uses_realized_money():
    frame = loans(3, target=[0, 1, 0], realized_profit=[1_500.0, -6_000.0, 1_500.0])
    all_loans = evaluate_policy(frame, np.array([True, True, True]))
    assert all_loans["total_profit"] == pytest.approx(-3_000.0)
    assert all_loans["return_per_dollar"] == pytest.approx(-0.1)
    assert all_loans["default_rate_funded"] == pytest.approx(1 / 3)

    skip_default = evaluate_policy(frame, np.array([True, False, True]))
    assert skip_default["total_profit"] == pytest.approx(3_000.0)
    assert skip_default["share_funded"] == pytest.approx(2 / 3)


def test_choose_threshold_rejects_degenerate_strategies():
    """Funding two loans might look great per dollar, but it is not a strategy."""
    frame = loans(10, target=[0] * 9 + [1], realized_profit=[1_500.0] * 9 + [-6_000.0])
    p = np.linspace(0.01, 0.99, 10)
    grid = np.arange(0.05, 1.0, 0.05)

    threshold, curve = choose_threshold(frame, p, grid, min_share_funded=0.5)
    chosen = next(row for row in curve if row["threshold"] == threshold)
    assert chosen["share_funded"] >= 0.5

    with pytest.raises(ValueError, match="minimum share"):
        choose_threshold(frame, p, grid, min_share_funded=1.5)


def test_threshold_for_share_funds_that_share():
    """Comparing policies at equal selectivity needs an exact share (D-048)."""
    p = np.linspace(0, 1, 101)
    cutoff = threshold_for_share(p, 0.25)
    assert (p < cutoff).mean() == pytest.approx(0.25, abs=0.02)


def test_threshold_for_share_rejects_impossible_shares():
    with pytest.raises(ValueError, match="share must be"):
        threshold_for_share(np.linspace(0, 1, 10), 1.5)
