"""FastAPI service: score a listing and return a funding decision.

The service loads the bundle once at startup (D-057). Loading per request would
add ~100ms of deserialization to every call, and worse, could serve two
different model versions at the same time during a deploy.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, get_args

from fastapi import Body, FastAPI, HTTPException, Request

from lending_club.config import load_params
from lending_club.serving.bundle import LoanDecisionModel
from lending_club.serving.schema import (
    HomeOwnership,
    LoanApplication,
    LoanDecision,
    Purpose,
    VerificationStatus,
    to_model_frame,
)

logger = logging.getLogger(__name__)
state: dict = {}

# Declared in the request schema as Literals so they appear in the OpenAPI
# docs; checked against the loaded model at startup so the two can never drift
# apart silently (D-072).
SCHEMA_VOCABULARY = {
    "home_ownership": HomeOwnership,
    "verification_status": VerificationStatus,
    "purpose": Purpose,
}


def check_vocabulary(bundle: LoanDecisionModel) -> dict[str, list[str]]:
    """Categories the request schema accepts that this model never saw (D-072).

    Reported, not fatal. The encoder maps an unseen level to "infrequent",
    which is deliberate and tested (D-018), and the drift monitor reports it —
    so this is a known-quality caveat, not corruption. Refusing to start would
    turn a rare missing category in one retraining window into an outage, which
    is a worse failure than the one it prevents. The gap is logged at startup
    and exposed on /model so it cannot go unnoticed.
    """
    known = bundle.metadata.categorical_vocabulary
    if not known:  # bundles built before this field existed
        return {}
    gaps = {}
    for column, literal in SCHEMA_VOCABULARY.items():
        untrained = sorted(set(get_args(literal)) - set(known.get(column, [])))
        if untrained:
            gaps[column] = untrained
            logger.warning(
                "request schema accepts %s for %r, but the model never saw them; "
                "they will be encoded as 'infrequent'",
                untrained,
                column,
            )
    return gaps


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = LoanDecisionModel.load()
    state["vocabulary_gaps"] = check_vocabulary(bundle)
    state["bundle"] = bundle
    state["max_batch"] = load_params()["serving"]["max_batch"]
    yield
    state.clear()


def _bundle() -> LoanDecisionModel:
    """The loaded model, or a 503 — never a KeyError (D-075).

    503 tells a load balancer "not ready, retry"; the 500 a KeyError produces
    tells it "broken, page someone". Getting that wrong turns a startup race
    into an incident.
    """
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


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
    bundle = _bundle()
    decisions = bundle.decide(to_model_frame(applications))
    return [
        LoanDecision(
            decision=row.decision,
            decision_basis=row.decision_basis,
            probability_of_default=round(float(row.probability_of_default), 6),
            expected_return_usd=round(float(row.expected_return_usd), 2),
            risk_score=round(float(row.risk_score), 6),
            threshold=bundle.metadata.threshold,
            model_version=bundle.metadata.created_at,
        )
        for row in decisions.itertuples()
    ]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One structured line per request (m3).

    A credit decision service with no request log cannot answer "what did we
    tell this applicant, and when". Deliberately does not log the request body:
    these payloads are personal financial data, and the reason codes a reviewer
    would need are in the response, not the input.
    """
    started = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %s in %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get("/health")
def health() -> dict:
    """Liveness plus which model is actually loaded — a deploy check needs both."""
    bundle = _bundle()
    return {
        "status": "ok",
        "model_family": bundle.metadata.model_family,
        "model_version": bundle.metadata.created_at,
    }


@app.get("/model")
def model_metadata() -> dict:
    """Full provenance: threshold, economics, training window, offline scores."""
    return _bundle().metadata.__dict__ | {"vocabulary_gaps": state.get("vocabulary_gaps", {})}


@app.post("/predict", response_model=LoanDecision)
def predict(application: LoanApplication) -> LoanDecision:
    return _decide([application])[0]


@app.post("/predict/batch", response_model=list[LoanDecision])
def predict_batch(
    applications: Annotated[list[LoanApplication], Body(min_length=1)],
) -> list[LoanDecision]:
    max_batch = state.get("max_batch", 1_000)
    if len(applications) > max_batch:
        raise HTTPException(status_code=413, detail=f"at most {max_batch} applications per request")
    return _decide(applications)
