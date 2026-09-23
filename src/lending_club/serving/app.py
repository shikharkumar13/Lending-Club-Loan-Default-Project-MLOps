"""FastAPI service: score a listing and return a funding decision.

The service loads the bundle once at startup (D-057). Loading per request would
add ~100ms of deserialization to every call, and worse, could serve two
different model versions at the same time during a deploy.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from lending_club.serving.bundle import LoanDecisionModel
from lending_club.serving.schema import LoanApplication, LoanDecision, to_model_frame

MAX_BATCH = 1_000
state: dict[str, LoanDecisionModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = LoanDecisionModel.load()
    yield
    state.clear()


app = FastAPI(
    title="Lending Club funding decisions",
    description=(
        "Given a loan listing, returns the probability of default, the expected "
        "return in dollars, and whether to fund it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _decide(applications: list[LoanApplication]) -> list[LoanDecision]:
    bundle = state["bundle"]
    decisions = bundle.decide(to_model_frame(applications))
    return [
        LoanDecision(
            decision=row.decision,
            probability_of_default=round(float(row.probability_of_default), 6),
            expected_return_usd=round(float(row.expected_return_usd), 2),
            risk_score=round(float(row.risk_score), 6),
            threshold=bundle.metadata.threshold,
            model_version=bundle.metadata.created_at,
        )
        for row in decisions.itertuples()
    ]


@app.get("/health")
def health() -> dict:
    """Liveness plus which model is actually loaded — a deploy check needs both."""
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_family": bundle.metadata.model_family,
        "model_version": bundle.metadata.created_at,
    }


@app.get("/model")
def model_metadata() -> dict:
    """Full provenance: threshold, economics, training window, offline scores."""
    return state["bundle"].metadata.__dict__


@app.post("/predict", response_model=LoanDecision)
def predict(application: LoanApplication) -> LoanDecision:
    return _decide([application])[0]


@app.post("/predict/batch", response_model=list[LoanDecision])
def predict_batch(
    applications: Annotated[list[LoanApplication], Body(min_length=1)],
) -> list[LoanDecision]:
    if len(applications) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH} applications per request")
    return _decide(applications)
