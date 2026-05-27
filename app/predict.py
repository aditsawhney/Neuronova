import numpy as np
import pandas as pd
import shap
from app.schema import FeatureContribution, PredictionResponse

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65

FEATURE_DISPLAY_NAMES = {
    "Age": "Age",
    "Gender": "Sex",
    "Inattentive": "Inattention score",
    "Hyper/Impulsive": "Hyperactivity/impulsivity score",
    "Handedness_Left": "Left-handed",
    "Handedness_Mixed": "Mixed-handed",
    "Handedness_Right": "Right-handed",
}


def _form_to_tscore(form_sum: int) -> float:
    """Map 0–12 Likert sum to approximate Conners T-score range (30–80)."""
    return 30.0 + (form_sum / 12.0) * 50.0


def _probability_to_risk(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "low"
    elif prob > HIGH_THRESHOLD:
        return "high"
    return "borderline"


def _extract_shap_for_adhd(shap_values, n_features: int) -> np.ndarray:
    """
    TreeExplainer returns different shapes depending on shap version and model type:
      - list of 2 arrays (one per class): take index 1 (ADHD)
      - single 3-d array (n_samples, n_features, n_classes): take [..., 1]
      - single 2-d array (n_samples, n_features): already the positive class
    Always returns a 1-d array of length n_features for the single input row.
    """
    if isinstance(shap_values, list):
        arr = np.array(shap_values[1])  # ADHD class
    else:
        arr = np.array(shap_values)

    arr = arr.squeeze()  # collapse any size-1 dimensions

    if arr.ndim == 2 and arr.shape[-1] == 2:
        # (n_features, n_classes) after squeeze on 1-sample input
        arr = arr[:, 1]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]

    assert arr.ndim == 1 and len(arr) == n_features, (
        f"Unexpected SHAP shape after normalisation: {arr.shape}"
    )
    return arr


def run_prediction(raw_input: dict, model, meta: dict) -> PredictionResponse:
    df = pd.DataFrame([{
        "Age": raw_input["age"],
        "Gender": float(raw_input["gender"]),
        "Handedness": raw_input["handedness"],
        "Inattentive": _form_to_tscore(raw_input["inattentive"]),
        "Hyper/Impulsive": _form_to_tscore(raw_input["hyper_impulsive"]),
    }])

    prob = float(model.predict_proba(df)[0][1])
    risk = _probability_to_risk(prob)

    shap_values_list = []

    for cc in model.calibrated_classifiers_:
        inner_pipeline = cc.estimator
        preprocessor = inner_pipeline.named_steps["preprocessor"]
        clf = inner_pipeline.named_steps["clf"]

        X_transformed = preprocessor.transform(df)
        n_features = X_transformed.shape[1]

        explainer = shap.TreeExplainer(clf)
        raw_sv = explainer.shap_values(X_transformed)
        sv = _extract_shap_for_adhd(raw_sv, n_features)
        shap_values_list.append(sv)

    mean_shap = np.mean(shap_values_list, axis=0)

    preprocessor_0 = model.calibrated_classifiers_[0].estimator.named_steps["preprocessor"]
    feature_names = [
        n.split("__")[-1]
        for n in preprocessor_0.get_feature_names_out()
    ]

    contributions = [
        FeatureContribution(
            feature=FEATURE_DISPLAY_NAMES.get(name, name),
            effect=round(float(np.float64(val).item()), 4),
        )
        for name, val in zip(feature_names, mean_shap)
    ]
    contributions.sort(key=lambda c: abs(c.effect), reverse=True)

    return PredictionResponse(
        probability=round(prob, 4),
        risk_level=risk,
        contributions=contributions,
    )