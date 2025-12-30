from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sex_classifier_ratios.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

app = FastAPI(title="Skull Sex Estimation API (Ratios)")

class PredictRequest(BaseModel):
    measurements: dict

class PredictResponse(BaseModel):
    sex: str
    male_probability: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = [req.measurements.get(c, np.nan) for c in feature_cols]
    x = np.array([row], dtype=float)
    proba_m = float(model.predict_proba(x)[0, 1])
    sex = "M" if proba_m >= 0.5 else "F"
    return PredictResponse(sex=sex, male_probability=proba_m)
