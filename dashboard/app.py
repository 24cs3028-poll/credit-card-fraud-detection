 import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        MODEL_DIR / "fraud_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "fraud_scaler.pkl"
    )

    with open(
        MODEL_DIR / "model_metadata.json",
        "r"
    ) as f:
        metadata = json.load(f)

    return model, scaler, metadata


model, scaler, metadata = load_artifacts()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Credit Card Fraud Detection")
st.caption(
    "AI-powered transaction risk analysis platform"
)

st.divider()


# ============================================================
# SIDEBAR — MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        f"**Model:** {metadata['model_name']}"
    )

    st.write(
        f"**Version:** {metadata['model_version']}"
    )

    st.write(
        f"**Decision Threshold:** "
        f"{metadata['threshold']:.2f}"
    )

    st.write(
        "**SMOTE:** Not Used"
    )

    st.write(
        "**Validation:** "
        "Chronological 70/15/15"
    )

    st.divider()

    st.info(
        "This dashboard is an educational "
        "fraud-detection prototype."
    )


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🤖 Model",
        metadata["model_name"]
    )

with col2:
    st.metric(
        "🎯 Threshold",
        f"{metadata['threshold']:.2f}"
    )

with col3:
    st.metric(
        "📊 Features",
        len(metadata["features"])
    )


st.divider()


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.header("🔍 Analyze Transaction")

st.write(
    "Enter transaction features below to estimate "
    "the probability of fraud."
)


# ------------------------------------------------------------
# Basic transaction information
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    transaction_time = st.number_input(
        "Transaction Time",
        value=0.0,
        format="%.4f"
    )

with col2:
    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )


# ------------------------------------------------------------
# PCA features
# ------------------------------------------------------------

st.subheader("Anonymized Transaction Features")

st.caption(
    "V1–V28 are anonymized PCA-transformed features "
    "from the original dataset."
)

feature_values = {}

columns = st.columns(4)

for i in range(1, 29):

    with columns[(i - 1) % 4]:

        feature_values[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f",
            key=f"V{i}"
        )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Analyze Transaction",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Create transaction dataframe
    # --------------------------------------------------------

    transaction = {
        "Time": transaction_time,
        **feature_values,
        "Amount": transaction_amount
    }

    transaction_df = pd.DataFrame(
        [transaction]
    )

    # Ensure exact feature order
    transaction_df = transaction_df[
        metadata["features"]
    ]

    # --------------------------------------------------------
    # Scale Time and Amount
    # --------------------------------------------------------

    transaction_df[
        ["Time", "Amount"]
    ] = scaler.transform(
        transaction_df[
            ["Time", "Amount"]
        ]
    )

    # --------------------------------------------------------
    # Fraud probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        transaction_df
    )[0, 1]

    threshold = float(
        metadata["threshold"]
    )

    prediction = (
        probability >= threshold
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if probability < 0.30:

        risk_level = "LOW"
        decision = "APPROVE"

    elif probability < 0.70:

        risk_level = "MEDIUM"
        decision = "ADDITIONAL CHECK"

    elif probability < 0.90:

        risk_level = "HIGH"
        decision = "MANUAL REVIEW"

    else:

        risk_level = "CRITICAL"
        decision = "BLOCK / INVESTIGATE"


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with col3:

        st.metric(
            "Decision",
            decision
        )

    with col4:

        st.metric(
            "Model Threshold",
            f"{threshold:.2f}"
        )


    # --------------------------------------------------------
    # Decision message
    # --------------------------------------------------------

    if prediction:

        st.error(
            f"🚨 FRAUD DETECTED\n\n"
            f"Fraud probability: "
            f"{probability * 100:.2f}%"
        )

    else:

        st.success(
            f"✅ TRANSACTION CLASSIFIED AS NORMAL\n\n"
            f"Fraud probability: "
            f"{probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # Probability bar
    # --------------------------------------------------------

    st.subheader(
        "Fraud Probability"
    )

    st.progress(
        min(probability, 1.0)
    )


    # --------------------------------------------------------
    # Transaction summary
    # --------------------------------------------------------

    st.subheader(
        "Transaction Summary"
    )

    summary = pd.DataFrame(
        {
            "Parameter": [
                "Transaction Amount",
                "Transaction Time",
                "Fraud Probability",
                "Risk Level",
                "Decision",
                "Model"
            ],
            "Value": [
                f"${transaction_amount:,.2f}",
                f"{transaction_time:.4f}",
                f"{probability * 100:.2f}%",
                risk_level,
                decision,
                metadata["model_name"]
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Research Project"
)
