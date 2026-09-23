"""Tests for the champion/challenger gate (D-060)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import pytest

from lending_club.models.promote import should_promote


@dataclass
class FakeBundle:
    """A bundle stand-in that funds a fixed set of loans."""

    funded: np.ndarray

    def decide(self, applications: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"decision": np.where(self.funded, "fund", "decline")})


@pytest.fixture
def loans() -> pl.DataFrame:
    # Two good loans, two bad ones.
    return pl.DataFrame(
        {
            "funded_amnt": [10_000.0] * 4,
            "realized_profit": [1_500.0, 1_500.0, -6_000.0, -6_000.0],
            "target": [0, 0, 1, 1],
            "term_months": [36] * 4,
        }
    )


@pytest.fixture
def params(monkeypatch):
    """Bypass feature selection: the fakes ignore the frame contents."""
    monkeypatch.setattr(
        "lending_club.models.promote.prepare_for_decision",
        lambda frame, params: pd.DataFrame(index=range(frame.height)),
    )
    return {"promotion": {"margin": 0.0005}}


def test_first_model_is_always_promoted(loans, params):
    candidate = FakeBundle(np.array([True, True, False, False]))
    decision = should_promote(candidate, None, loans, params)
    assert decision.promote
    assert "no model in production" in decision.reason


def test_better_model_is_promoted(loans, params):
    champion = FakeBundle(np.array([True, True, True, True]))  # funds everything
    candidate = FakeBundle(np.array([True, True, False, False]))  # avoids the defaults
    decision = should_promote(candidate, champion, loans, params)
    assert decision.promote
    assert decision.candidate_return > decision.champion_return


def test_worse_model_is_rejected(loans, params):
    champion = FakeBundle(np.array([True, True, False, False]))
    candidate = FakeBundle(np.array([False, False, True, True]))  # funds only defaults
    assert not should_promote(candidate, champion, loans, params).promote


def test_marginally_better_model_is_rejected(loans, params):
    """Churn protection: swapping production for noise is not worth the risk."""
    champion = FakeBundle(np.array([True, True, False, False]))
    candidate = FakeBundle(np.array([True, True, False, False]))  # identical
    decision = should_promote(candidate, champion, loans, params, margin=0.0005)
    assert not decision.promote
    assert "below the 0.0005 margin" in decision.reason


def test_a_model_that_funds_nothing_scores_zero(loans, params):
    champion = FakeBundle(np.array([True, True, False, False]))
    candidate = FakeBundle(np.zeros(4, dtype=bool))
    decision = should_promote(candidate, champion, loans, params)
    assert not decision.promote
    assert decision.candidate_return == 0.0
