from pathlib import Path
from typing import Dict, Any, List
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "fraud_scaler.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {e}")

MODEL_NAME = metadata.get("model_name", "Unknown")
MODEL_VERSION = metadata.get("model_version", "1.0.0")
THRESHOLD = float(metadata.get("threshold", 0.5))
FEATURES = metadata.get("features", [])
TARGET = metadata.get("target", "Class")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="REST API for AI-powered credit card fraud detection using the trained machine learning model.",
    version=MODEL_VERSION,
)


class PredictionRequest(BaseModel):
    Time: float = Field(..., description="Transaction time")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")


def get_risk_level(probability: float) -> str:
    if probability < 0.10:
        return "LOW"
    elif probability < 0.30:
        return "MEDIUM"
    elif probability < 0.70:
        return "HIGH"
    else:
        return "CRITICAL"


def get_decision(probability: float, prediction: int) -> str:
    if prediction == 1:
        if probability >= 0.70:
            return "BLOCK / INVESTIGATE"
        return "REVIEW"
    return "ALLOW"


def prepare_dataframe(transaction: Dict[str, Any]) -> pd.DataFrame:
    if not FEATURES:
        raise ValueError("No features found in model_metadata.json")

    missing_features = [
        feature for feature in FEATURES
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    data = {
        feature: [float(transaction[feature])]
        for feature in FEATURES
    }

    df = pd.DataFrame(data)

    scale_columns = [
        column for column in ["Time", "Amount"]
        if column in df.columns
    ]

    if scale_columns:
        df[scale_columns] = scaler.transform(df[scale_columns])

    return df


def make_prediction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    df = prepare_dataframe(transaction)
    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= THRESHOLD)
    classification = "FRAUD" if prediction == 1 else "NORMAL"
    risk_level = get_risk_level(probability)
    decision = get_decision(probability, prediction)

    return {
        "fraud_probability": round(probability, 6),
        "fraud_probability_percent": round(probability * 100, 2),
        "prediction": prediction,
        "classification": classification,
        "risk_level": risk_level,
        "decision": decision,
        "threshold": THRESHOLD,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "metadata_loaded": metadata is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "target": TARGET,
        "smote_used": metadata.get("smote_used", False),
        "validation": metadata.get("validation", "Not specified"),
    }


@app.post("/predict")
def predict(transaction: PredictionRequest):
    try:
        return make_prediction(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(transactions: List[PredictionRequest]):
    """Vectorized batch prediction for large CSV uploads.

    The model is called once per internal chunk instead of once per row,
    which is dramatically faster for large datasets.
    """
    if not transactions:
        raise HTTPException(status_code=400, detail="Batch must contain at least one transaction.")

    # Streamlit sends manageable chunks. This limit also protects the API
    # from accidentally receiving an enormous JSON payload.
    MAX_API_BATCH = 5000
    if len(transactions) > MAX_API_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size cannot exceed {MAX_API_BATCH} transactions. Send smaller chunks.",
        )

    try:
        rows = [transaction.model_dump() for transaction in transactions]
        frame = pd.DataFrame(rows)

        missing_features = [feature for feature in FEATURES if feature not in frame.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        frame = frame[FEATURES].apply(pd.to_numeric, errors="raise")

        scale_columns = [column for column in ["Time", "Amount"] if column in frame.columns]
        if scale_columns:
            frame[scale_columns] = scaler.transform(frame[scale_columns])

        # Keep model memory predictable while still using vectorized inference.
        INTERNAL_BATCH = 2000
        probabilities = []
        for start in range(0, len(frame), INTERNAL_BATCH):
            chunk = frame.iloc[start:start + INTERNAL_BATCH]
            chunk_probabilities = model.predict_proba(chunk)[:, 1]
            probabilities.extend(chunk_probabilities.tolist())

        predictions = []
        for probability in probabilities:
            probability = float(probability)
            prediction = int(probability >= THRESHOLD)
            predictions.append({
                "fraud_probability": round(probability, 6),
                "fraud_probability_percent": round(probability * 100, 2),
                "prediction": prediction,
                "classification": "FRAUD" if prediction == 1 else "NORMAL",
                "risk_level": get_risk_level(probability),
                "decision": get_decision(probability, prediction),
                "threshold": THRESHOLD,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            })

        fraud_count = sum(item["prediction"] for item in predictions)

        return {
            "count": len(predictions),
            "fraud_count": fraud_count,
            "normal_count": len(predictions) - fraud_count,
            "threshold": THRESHOLD,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predictions": predictions,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/example")
def example_transaction():
    example = {}
    for feature in FEATURES:
        if feature == "Time":
            example[feature] = 100000.0
        elif feature == "Amount":
            example[feature] = 100.0
        else:
            example[feature] = 0.0

    return {
        "message": "Example transaction",
        "request_body": example,
    }
