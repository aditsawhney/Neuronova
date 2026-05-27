"""
schema.py — request and response models for the /predict endpoint

Input validation is handled by Pydantic — invalid values never reach the model.
Risk level thresholds are set conservatively based on the calibration curve:
mid-range probabilities (0.35–0.65) are noisy, so borderline is wide.
"""

from pydantic import BaseModel, Field, field_validator


# ── helpers ────────────────────────────────────────────────────────────────

def _raw_to_tscore(raw: int) -> float:
    """
    Linear map from 0–12 raw item sum → 9–90 T-score range.

    The ADHD-200 dataset stores Conners Rating Scale T-scores (range 9–90,
    population mean ~50, clinical threshold ~65). Our form collects raw item
    sums (4 questions × 0–3 = 0–12). This maps form inputs onto the scale
    the model was trained on.
    """
    return 9.0 + (raw / 12.0) * (90.0 - 9.0)


# ── input ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    age: float = Field(..., ge=4, le=18, description="Child's age in years")
    gender: str = Field(..., description="Male or Female")
    handedness: str = Field(..., description="Left, Right, or Mixed")
    inattentive: int = Field(..., ge=0, le=12, description="Sum of 4 inattention items (0–12)")
    hyper_impulsive: int = Field(..., ge=0, le=12, description="Sum of 4 hyperactivity items (0–12)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("handedness")
    @classmethod
    def validate_handedness(cls, v: str) -> str:
        allowed = {"Left", "Right", "Mixed"}
        if v not in allowed:
            raise ValueError(f"handedness must be one of {allowed}")
        return v

    def to_model_input(self) -> dict:
        """
        Converts validated request fields into the format the model pipeline
        expects — matching the exact column names from training.

        Gender encoding: Male → 0.0, Female → 1.0 (matches dataset coding).
        Handedness stays as a string — the pipeline's OneHotEncoder handles it.
        Behaviour scores are mapped from raw 0–12 to T-score 9–90.
        """
        return {
            "Gender": 0.0 if self.gender == "Male" else 1.0,
            "Age": self.age,
            "Handedness": self.handedness,
            "Inattentive": _raw_to_tscore(self.inattentive),
            "Hyper/Impulsive": _raw_to_tscore(self.hyper_impulsive),
        }


# ── risk level ─────────────────────────────────────────────────────────────

# Thresholds informed by the calibration curve.
# High-probability predictions (>0.65) track the diagonal well.
# Mid-range (0.35–0.65) is noisy — treated as borderline.
LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65


def probability_to_risk(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "low"
    if prob > HIGH_THRESHOLD:
        return "high"
    return "borderline"


# ── output ─────────────────────────────────────────────────────────────────

class FeatureContributions(BaseModel):
    inattentive_score: int
    inattentive_max: int = 12
    hyper_impulsive_score: int
    hyper_impulsive_max: int = 12
    age: float
    gender: str


class PredictResponse(BaseModel):
    probability: float
    risk_level: str  # "low" | "borderline" | "high"
    warning: str | None
    feature_contributions: FeatureContributions

    model_config = {"json_schema_extra": {
        "example": {
            "probability": 0.72,
            "risk_level": "high",
            "warning": None,
            "feature_contributions": {
                "inattentive_score": 10,
                "inattentive_max": 12,
                "hyper_impulsive_score": 7,
                "hyper_impulsive_max": 12,
                "age": 9.5,
                "gender": "Female",
            }
        }
    }}