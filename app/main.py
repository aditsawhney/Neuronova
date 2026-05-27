"""
main.py — NeuroNova FastAPI backend

Loads the saved model once at startup, exposes a single /predict endpoint.

Run from the project root:
    uv run uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import (
    FeatureContributions,
    PredictRequest,
    PredictResponse,
    probability_to_risk,
)

MODELS_DIR = Path("models")

# Model is loaded once at startup and reused across all requests.
# Storing it in a dict rather than a global avoids issues with hot reload.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run train.py first.")
    state["model"] = joblib.load(model_path)
    state["meta"] = joblib.load(MODELS_DIR / "meta.joblib")
    print(f"Loaded model: {state['meta']['model_name']}")
    yield
    state.clear()


app = FastAPI(
    title="NeuroNova",
    description="ADHD screening tool — returns referral recommendation, not a diagnosis.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the React dev server during development.
# Tighten this to the deployed frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": state.get("meta", {}).get("model_name")}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Accepts pre-assessment information and returns a risk level with context.

    This is a referral recommendation, not a diagnosis. The probability
    output reflects model confidence given the provided inputs only.
    """
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a single-row DataFrame matching the training feature names exactly
    row = request.to_model_input()
    X = pd.DataFrame([row])
    print(X.dtypes)
    print(X)

    try:
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    risk_level = probability_to_risk(prob)

    # Adolescent warning — model was trained mostly on children 7–13.
    # Behaviour scores were normed on younger children, so 14+ predictions
    # should be treated with additional caution.
    warning = None
    if request.age >= 14:
        warning = (
            "This tool has reduced accuracy for adolescents aged 14 and over. "
            "The behaviour rating scales used were primarily normed on children "
            "aged 7–13. Clinical judgement is especially important for this age group."
        )

    contributions = FeatureContributions(
        inattentive_score=request.inattentive,
        hyper_impulsive_score=request.hyper_impulsive,
        age=request.age,
        gender=request.gender,
    )

    return PredictResponse(
        probability=round(prob, 4),
        risk_level=risk_level,
        warning=warning,
        feature_contributions=contributions,
    )