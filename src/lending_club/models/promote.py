"""Champion vs. challenger: should the new model replace the live one?

Retraining on a schedule is only safe with a gate. Without one, a bad training
run quietly reaches production — the pipeline succeeded, every task was green,
and the model is worse.

Two rules make the gate trustworthy:

1. **Judge on money, not on log loss.** A model can improve its likelihood and
   still pick a worse portfolio.
2. **Judge on validation data, never on the test year.** The test year is
   scored once, for the report. A gate that consulted it every retrain would
   erode it into just another tuning set (D-061).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from lending_club.policy.profit import evaluate_policy
from lending_club.serving.bundle import LoanDecisionModel, prepare_for_decision


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    candidate_return: float
    champion_return: float | None
    margin: float
    reason: str


def policy_return(bundle: LoanDecisionModel, frame: pl.DataFrame, params: dict) -> float:
    """Return per dollar this bundle's funding policy would have earned."""
    decisions = bundle.decide(prepare_for_decision(frame, params))
    funded = (decisions["decision"] == "fund").to_numpy()
    if not funded.any():
        return 0.0
    return evaluate_policy(frame, funded)["return_per_dollar"]


def should_promote(
    candidate: LoanDecisionModel,
    champion: LoanDecisionModel | None,
    frame: pl.DataFrame,
    params: dict,
    margin: float = 0.0005,
) -> PromotionDecision:
    """Promote only on a clear improvement, measured in return per dollar.

    The margin stops churn: swapping the production model for a 0.00001
    improvement is noise, and every swap carries real risk.
    """
    candidate_return = policy_return(candidate, frame, params)

    if champion is None:
        return PromotionDecision(
            promote=True,
            candidate_return=candidate_return,
            champion_return=None,
            margin=margin,
            reason="no model in production yet",
        )

    champion_return = policy_return(champion, frame, params)
    improvement = candidate_return - champion_return
    promote = improvement > margin
    return PromotionDecision(
        promote=promote,
        candidate_return=candidate_return,
        champion_return=champion_return,
        margin=margin,
        reason=(
            f"candidate {candidate_return:+.5f} vs champion {champion_return:+.5f} "
            f"({improvement:+.5f}, {'above' if promote else 'below'} the {margin} margin)"
        ),
    )
