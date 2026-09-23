"""One artifact that contains the whole decision, not just the model.

A served model is more than an estimator. Reproducing a funding decision needs
four things that were fitted or chosen at different stages:

    preprocessing + model   (trained on 2007-2013)
    calibrator              (fitted on 2014 H1)
    funding threshold       (tuned on 2014 H2)
    profit economics        (LGD and prepayment, measured on training loans)

Shipping them separately is how production drifts out of sync with the report:
someone updates the model and forgets the threshold. Bundling them means the
service either has a complete, consistent decision or it has nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from lending_club.config import load_params, path_of
from lending_club.features.pipeline import feature_columns
from lending_club.policy.profit import ProfitModel

BUNDLE_FILENAME = "loan_decision_model.joblib"


@dataclass
class BundleMetadata:
    """Everything a reviewer needs to know about how this artifact was made."""

    model_family: str
    threshold: float
    calibrated_risk_at_threshold: float
    lgd: float
    prepay_factor: float
    feature_columns: list[str]
    # What each categorical column was actually trained on, read back off the
    # fitted encoder. The API checks its request schema against this at startup,
    # so an accepted-but-never-trained category cannot reach the model (D-072).
    categorical_vocabulary: dict[str, list[str]]
    train_window: dict[str, str]
    cv_log_loss: float
    test_return_per_dollar: float
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema_version: int = 1


@dataclass
class LoanDecisionModel:
    """Scores a loan application and answers: fund it, or not?"""

    pipeline: Any  # preprocessing + estimator
    calibrator: Any  # isotonic map from score to probability
    profit: ProfitModel
    metadata: BundleMetadata

    def decide(self, applications: pd.DataFrame) -> pd.DataFrame:
        features = applications[self.metadata.feature_columns]

        # The funding rule uses the raw score: it is continuous, so the policy
        # degrades smoothly (D-053).
        score = self.pipeline.predict_proba(features)[:, 1]
        # The probability is what a human should read, and what the expected
        # value calculation needs.
        probability = self.calibrator.predict_proba(features)[:, 1]

        amount = applications["loan_amnt"].to_numpy(dtype=float)
        term = applications["term_months"].to_numpy(dtype=float)
        installment = applications["installment"].to_numpy(dtype=float)

        gain = (installment * term - amount) * self.profit.prepay_factor
        loss = self.profit.lgd * amount
        expected_return = (1 - probability) * gain - probability * loss

        # Two conditions, not one (D-074). The threshold is what binds in
        # practice — at these interest rates almost every loan has positive
        # expected value — but without the second condition the service can
        # return "fund" next to an expected return of -$13,066, which is what
        # it did for 10 of the 283,026 test loans. Those are corrupt listings
        # whose stated installment contradicts the amount and rate; the API now
        # rejects them at the schema (D-073), and this floor means the two
        # numbers in the response can never contradict each other, whatever
        # reaches the model.
        above_threshold = score >= self.metadata.threshold
        negative_value = expected_return <= 0

        return pd.DataFrame(
            {
                "risk_score": score,
                "probability_of_default": probability,
                "expected_return_usd": expected_return,
                "decision": np.where(above_threshold | negative_value, "decline", "fund"),
                "decision_basis": np.select(
                    [above_threshold, negative_value],
                    ["score_above_threshold", "negative_expected_return"],
                    default="score_below_threshold",
                ),
            }
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(asdict(self.metadata), indent=2) + "\n")
        return path

    @staticmethod
    def load(path: Path | None = None) -> LoanDecisionModel:
        return joblib.load(path or default_bundle_path())


def prepare_for_decision(frame, params: dict) -> pd.DataFrame:
    """polars processed frame -> the pandas frame `decide()` expects.

    `decide` needs the model inputs plus `term_months`, which is not a feature
    (it is always 36 in v1) but is required for the interest calculation.
    """
    from lending_club.features.pipeline import to_model_frame

    applications = to_model_frame(frame, params)
    applications["term_months"] = frame["term_months"].to_numpy()
    return applications


def categorical_vocabulary(pipeline, params: dict) -> dict[str, list[str]]:
    """The categories the fitted one-hot encoder actually saw, per column.

    Read off the artifact rather than recomputed from the training data: the
    point is to describe *this* model, so the two cannot disagree.
    """
    columns = params["features"]["groups"]["nominal"]
    try:
        encoder = (
            pipeline.named_steps["preprocess"].named_steps["columns"].named_transformers_["nominal"]
        )
    except (AttributeError, KeyError):  # e.g. the preprocessing-free baseline
        return {}
    categories = encoder.named_steps["encode"].categories_
    return {
        column: sorted(str(v) for v in values)
        for column, values in zip(columns, categories, strict=True)
    }


def default_bundle_path() -> Path:
    return path_of("bundle_dir") / BUNDLE_FILENAME


def build_bundle(family: str = "lightgbm") -> LoanDecisionModel:
    """Assemble the bundle from the artifacts the earlier stages produced."""
    params = load_params()
    metrics = json.loads((path_of("reports_dir") / "metrics.json").read_text())
    cv = json.loads((path_of("reports_dir") / "cv_results.json").read_text())

    result = metrics["results"][family]
    selected = next(row for row in cv["selected"] if row["family"] == family)

    pipeline = joblib.load(path_of("models_dir") / f"{family}.joblib")
    bundle = LoanDecisionModel(
        pipeline=pipeline,
        calibrator=joblib.load(path_of("calibrators_dir") / f"{family}.joblib"),
        profit=ProfitModel(
            lgd=metrics["profit_model"]["lgd"],
            prepay_factor=metrics["profit_model"]["prepay_factor"],
        ),
        metadata=BundleMetadata(
            model_family=family,
            threshold=result["threshold"],
            calibrated_risk_at_threshold=result["calibrated_risk_at_threshold"],
            lgd=metrics["profit_model"]["lgd"],
            prepay_factor=metrics["profit_model"]["prepay_factor"],
            feature_columns=feature_columns(params),
            categorical_vocabulary=categorical_vocabulary(pipeline, params),
            train_window=params["split"]["train"],
            cv_log_loss=selected["cv"]["log_loss"],
            test_return_per_dollar=result["policies"]["threshold"]["return_per_dollar"],
        ),
    )
    return bundle


def package() -> Path:
    bundle = build_bundle()
    destination = bundle.save(default_bundle_path())
    print(
        f"packaged {bundle.metadata.model_family}: threshold={bundle.metadata.threshold:.3f} "
        f"({bundle.metadata.calibrated_risk_at_threshold:.1%} risk) -> {destination}"
    )
    return destination


# No `if __name__ == "__main__"` here on purpose: running this file as a script
# would define LoanDecisionModel in the `__main__` module, and joblib would
# record that name inside the artifact. Nothing else could then load it —
# "Can't get attribute 'LoanDecisionModel' on module '__main__'". The entry
# point lives in serving/__main__.py, which imports the class from here, so the
# pickle records `lending_club.serving.bundle.LoanDecisionModel` (D-058).
