import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load(MODEL_DIR / "fraud_model.pkl")
scaler = joblib.load(MODEL_DIR / "fraud_scaler.pkl")

with open(MODEL_DIR / "model_metadata.json", "r") as f:
    metadata = json.load(f)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection")
st.subheader("Machine Learning Fraud Risk Prediction")

# -----------------------------
# Model information
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", metadata["model_name"])

with col2:
    st.metric("Version", metadata["model_version"])

with col3:
    st.metric("Threshold", f'{metadata["threshold"]:.2f}')

st.divider()

st.info(
    "Model loaded successfully. "
    "The prediction interface will be connected to the trained "
    "transaction features next."
)
