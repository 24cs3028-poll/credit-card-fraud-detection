import streamlit as st
import joblib
import json
import pandas as pd
import sqlite3
import math
import requests
from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR

DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "fraud_predictions.db"

# FastAPI backend
API_URL = "http://127.0.0.1:8000"


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
# MODEL SETTINGS
# ============================================================

MODEL_NAME = metadata.get(
    "model_name",
    "XGBoost"
)

MODEL_VERSION = metadata.get(
    "model_version",
    "1.0.0"
)

THRESHOLD = float(
    metadata.get(
        "threshold",
        0.5
    )
)

FEATURES = metadata.get(
    "features",
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

TARGET_COLUMN = metadata.get(
    "target",
    "Class"
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "transaction_time_input" not in st.session_state:
    st.session_state["transaction_time_input"] = 0.0


if "transaction_amount_input" not in st.session_state:
    st.session_state["transaction_amount_input"] = 0.0


for i in range(1, 29):

    if f"V{i}_input" not in st.session_state:

        st.session_state[
            f"V{i}_input"
        ] = 0.0


if "actual_class" not in st.session_state:
    st.session_state["actual_class"] = None


if "dataset_test_mode" not in st.session_state:
    st.session_state["dataset_test_mode"] = False


if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None


if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection


def initialize_database():

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            source TEXT NOT NULL,

            transaction_time REAL,

            amount REAL,

            fraud_probability REAL,

            prediction INTEGER,

            classification TEXT,

            risk_level TEXT,

            decision TEXT,

            model_name TEXT,

            model_version TEXT,

            threshold REAL

        )
        """
    )

    connection.commit()

    connection.close()


def log_prediction(
    source,
    transaction_time,
    amount,
    fraud_probability,
    prediction,
    classification,
    risk_level,
    decision
):

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (

            timestamp,
            source,
            transaction_time,
            amount,
            fraud_probability,
            prediction,
            classification,
            risk_level,
            decision,
            model_name,
            model_version,
            threshold

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(
                timespec="seconds"
            ),

            source,

            float(transaction_time),

            float(amount),

            float(fraud_probability),

            int(prediction),

            classification,

            risk_level,

            decision,

            MODEL_NAME,

            MODEL_VERSION,

            THRESHOLD
        )
    )

    connection.commit()

    connection.close()


def log_batch_predictions(results):

    connection = get_db_connection()

    cursor = connection.cursor()

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    rows = []

    for _, row in results.iterrows():

        rows.append(
            (
                timestamp,

                "Batch CSV",

                float(
                    row["Time"]
                )
                if "Time" in row
                else 0.0,

                float(
                    row["Amount"]
                )
                if "Amount" in row
                else 0.0,

                float(
                    row["Fraud_Probability"]
                ),

                int(
                    row["Prediction"]
                ),

                str(
                    row["Classification"]
                ),

                str(
                    row["Risk_Level"]
                ),

                str(
                    row["Decision"]
                ),

                MODEL_NAME,

                MODEL_VERSION,

                THRESHOLD
            )
        )

    cursor.executemany(
        """
        INSERT INTO predictions (

            timestamp,
            source,
            transaction_time,
            amount,
            fraud_probability,
            prediction,
            classification,
            risk_level,
            decision,
            model_name,
            model_version,
            threshold

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows
    )

    connection.commit()

    connection.close()


def load_prediction_history():

    connection = get_db_connection()

    try:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """,
            connection
        )

    finally:

        connection.close()

    return df


initialize_database()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_using_api(transaction_data):
    """Send a single transaction to the FastAPI backend."""
    response = requests.post(
        f"{API_URL}/predict",
        json=transaction_data,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def predict_batch_using_api(transactions, chunk_size=2000, progress_callback=None):
    """Send a large dataset to FastAPI in optimized chunks.

    FastAPI performs vectorized model inference, while Streamlit keeps each
    HTTP request small enough to be reliable.
    """
    all_predictions = []
    total = len(transactions)

    for start in range(0, total, chunk_size):
        chunk = transactions[start:start + chunk_size]

        response = requests.post(
            f"{API_URL}/predict/batch",
            json=chunk,
            timeout=300,
        )
        response.raise_for_status()

        chunk_result = response.json()
        chunk_predictions = chunk_result.get("predictions", [])
        all_predictions.extend(chunk_predictions)

        if progress_callback:
            progress_callback(min(start + len(chunk), total), total)

    if len(all_predictions) != total:
        raise ValueError(
            f"FastAPI returned {len(all_predictions)} predictions for {total} transactions."
        )

    return {
        "predictions": all_predictions,
        "count": len(all_predictions),
    }


def check_api_health():
    """Check whether the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as e:
        return False, str(e)


def get_risk_level(probability):

    probability = float(probability)

    if probability < 0.10:

        return "LOW"

    elif probability < 0.30:

        return "MEDIUM"

    elif probability < 0.70:

        return "HIGH"

    else:

        return "CRITICAL"


def get_decision(
    probability,
    threshold
):

    probability = float(probability)

    if probability >= threshold:

        if probability >= 0.70:

            return "BLOCK / INVESTIGATE"

        return "MANUAL REVIEW"

    if probability >= 0.30:

        return "ADDITIONAL CHECK"

    return "APPROVE"


def prepare_prediction_dataframe(
    dataframe
):

    prediction_df = dataframe[
        FEATURES
    ].copy()

    prediction_df = prediction_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if prediction_df.isnull().any().any():

        missing_locations = (
            prediction_df
            .isnull()
            .sum()
        )

        invalid_columns = (
            missing_locations[
                missing_locations > 0
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Invalid or missing numeric values "
            "in: "
            + ", ".join(
                invalid_columns
            )
        )

    # --------------------------------------------------------
    # SCALE ONLY TIME AND AMOUNT
    # --------------------------------------------------------

    prediction_df[
        ["Time", "Amount"]
    ] = scaler.transform(
        prediction_df[
            ["Time", "Amount"]
        ]
    )

    return prediction_df


def predict_dataframe(
    dataframe
):

    prediction_df = (
        prepare_prediction_dataframe(
            dataframe
        )
    )

    probabilities = (
        model.predict_proba(
            prediction_df
        )[:, 1]
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return (
        probabilities,
        predictions
    )


# ============================================================
# PSI FUNCTION
# ============================================================

def calculate_psi(
    reference_series,
    current_series,
    bins=10
):

    reference = pd.to_numeric(
        reference_series,
        errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current_series,
        errors="coerce"
    ).dropna()

    if len(reference) == 0:
        return 0.0

    if len(current) == 0:
        return 0.0

    # Avoid duplicate bin edges
    quantiles = reference.quantile(
        [
            i / bins
            for i in range(
                0,
                bins + 1
            )
        ]
    )

    edges = (
        quantiles
        .drop_duplicates()
        .tolist()
    )

    if len(edges) < 3:

        return 0.0

    reference_binned = pd.cut(
        reference,
        bins=edges,
        include_lowest=True
    )

    current_binned = pd.cut(
        current,
        bins=edges,
        include_lowest=True
    )

    reference_distribution = (
        reference_binned
        .value_counts(
            normalize=True,
            sort=False
        )
        .fillna(0)
    )

    current_distribution = (
        current_binned
        .value_counts(
            normalize=True,
            sort=False
        )
        .fillna(0)
    )

    current_distribution = (
        current_distribution
        .reindex(
            reference_distribution.index,
            fill_value=0
        )
    )

    psi_value = 0.0

    for ref_pct, cur_pct in zip(
        reference_distribution,
        current_distribution
    ):

        ref_pct = max(
            float(ref_pct),
            0.0001
        )

        cur_pct = max(
            float(cur_pct),
            0.0001
        )

        psi_value += (
            cur_pct - ref_pct
        ) * math.log(
            cur_pct / ref_pct
        )

    return float(
        psi_value
    )


def interpret_psi(psi):

    if psi < 0.10:

        return (
            "STABLE",
            "No significant distribution shift."
        )

    elif psi < 0.25:

        return (
            "WARNING",
            "Moderate distribution shift detected."
        )

    else:

        return (
            "CRITICAL",
            "Significant distribution drift detected."
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Credit Card Fraud Detection"
)

st.caption(
    "AI-powered transaction risk analysis platform"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Model Information"
    )

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Version:** {MODEL_VERSION}"
    )

    st.write(
        f"**Decision Threshold:** "
        f"{THRESHOLD:.4f}"
    )

    st.write(
        "**SMOTE:** Not Used"
    )

    st.write(
        "**Validation:** "
        "Chronological 70/15/15"
    )

    st.divider()

    api_ok, api_status = check_api_health()
    if api_ok:
        st.success("FastAPI: Connected")
    else:
        st.error("FastAPI: Disconnected")

    st.write(
        f"**API:** {API_URL}"
    )

    st.write(
        f"**Database:** "
        f"{DATABASE_PATH.name}"
    )

    st.info(
        "Educational fraud-detection "
        "prototype."
    )


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🤖 Model",
        MODEL_NAME
    )


with col2:

    st.metric(
        "🎯 Threshold",
        f"{THRESHOLD:.2f}"
    )


with col3:

    st.metric(
        "📊 Features",
        len(FEATURES)
    )


with col4:

    st.metric(
        "🗄️ Database",
        "SQLite"
    )


st.divider()


# ============================================================
# DATASET TESTING
# ============================================================

st.header(
    "🧪 Test Using Dataset"
)

st.write(
    "Upload creditcard.csv to automatically "
    "load a real normal or fraudulent transaction."
)


uploaded_file = st.file_uploader(
    "Upload creditcard.csv",
    type=["csv"],
    key="dataset_csv"
)


if uploaded_file is not None:

    try:

        test_df = pd.read_csv(
            uploaded_file
        )

        required_columns = (
            ["Time"]
            + [
                f"V{i}"
                for i in range(1, 29)
            ]
            + ["Amount", "Class"]
        )

        missing_columns = [
            col
            for col in required_columns
            if col not in test_df.columns
        ]

        if missing_columns:

            st.error(
                "CSV is missing required "
                "columns: "
                + ", ".join(
                    missing_columns
                )
            )

        else:

            st.success(
                f"Dataset loaded successfully: "
                f"{len(test_df):,} transactions"
            )

            test_type = st.radio(
                "Choose transaction type",
                [
                    "Normal (Class 0)",
                    "Fraud (Class 1)"
                ],
                horizontal=True
            )

            selected_class = (
                0
                if test_type ==
                "Normal (Class 0)"
                else 1
            )

            filtered_df = test_df[
                test_df["Class"]
                == selected_class
            ]

            if len(filtered_df) == 0:

                st.warning(
                    f"No transactions with "
                    f"Class = {selected_class} found."
                )

            else:

                st.write(
                    f"Available transactions: "
                    f"**{len(filtered_df):,}**"
                )

                max_index = (
                    len(filtered_df) - 1
                )

                sample_number = (
                    st.number_input(
                        "Select transaction number",
                        min_value=0,
                        max_value=max_index,
                        value=0,
                        step=1,
                        key="dataset_sample_number"
                    )
                )

                selected_row = (
                    filtered_df.iloc[
                        int(sample_number)
                    ]
                )

                st.write(
                    f"Selected dataset row: "
                    f"**{selected_row.name}**"
                )

                actual_class_display = (
                    "NORMAL"
                    if int(
                        selected_row["Class"]
                    ) == 0
                    else "FRAUD"
                )

                st.write(
                    f"Actual Class: "
                    f"**{actual_class_display} "
                    f"(Class "
                    f"{int(selected_row['Class'])})**"
                )

                if st.button(
                    "📥 Load Transaction Into Form",
                    use_container_width=True
                ):

                    st.session_state[
                        "transaction_time_input"
                    ] = float(
                        selected_row["Time"]
                    )

                    st.session_state[
                        "transaction_amount_input"
                    ] = float(
                        selected_row["Amount"]
                    )

                    for i in range(1, 29):

                        st.session_state[
                            f"V{i}_input"
                        ] = float(
                            selected_row[
                                f"V{i}"
                            ]
                        )

                    st.session_state[
                        "actual_class"
                    ] = int(
                        selected_row["Class"]
                    )

                    st.session_state[
                        "dataset_test_mode"
                    ] = True

                    st.success(
                        "✅ Transaction loaded into the form."
                    )

                    st.rerun()

    except Exception as e:

        st.error(
            f"Unable to read CSV: {e}"
        )


st.divider()


# ============================================================
# SINGLE TRANSACTION INPUT
# ============================================================

st.header(
    "🔍 Analyze Transaction"
)

st.write(
    "Enter transaction features below to "
    "estimate the probability of fraud."
)


# ============================================================
# TIME AND AMOUNT
# ============================================================

col1, col2 = st.columns(2)


with col1:

    transaction_time = st.number_input(
        "Transaction Time",
        value=st.session_state[
            "transaction_time_input"
        ],
        format="%.4f",
        key="transaction_time_input"
    )


with col2:

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=st.session_state[
            "transaction_amount_input"
        ],
        format="%.2f",
        key="transaction_amount_input"
    )


# ============================================================
# V1-V28
# ============================================================

st.subheader(
    "Anonymized Transaction Features"
)

st.caption(
    "V1–V28 are anonymized PCA-transformed features."
)


feature_values = {}

columns = st.columns(4)


for i in range(1, 29):

    with columns[
        (i - 1) % 4
    ]:

        feature_values[
            f"V{i}"
        ] = st.number_input(
            f"V{i}",
            value=st.session_state[
                f"V{i}_input"
            ],
            format="%.6f",
            key=f"V{i}_input"
        )


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

st.divider()


predict_button = st.button(
    "🚀 Analyze Transaction",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE TRANSACTION
        # ----------------------------------------------------

        transaction = {
            "Time": transaction_time,
            **feature_values,
            "Amount": transaction_amount
        }

        # ----------------------------------------------------
        # PREDICTION THROUGH FASTAPI
        # ----------------------------------------------------

        api_result = predict_using_api(transaction)

        probability = float(
            api_result["fraud_probability"]
        )

        prediction = int(
            api_result["prediction"]
        )

        classification = api_result["classification"]
        risk_level = api_result["risk_level"]
        decision = api_result["decision"]

        # Use the backend threshold returned by FastAPI.
        api_threshold = float(
            api_result.get("threshold", THRESHOLD)
        )

        # ----------------------------------------------------
        # SAVE SESSION RESULT
        # ----------------------------------------------------

        st.session_state[
            "last_prediction"
        ] = {
            "probability": probability,
            "prediction": prediction,
            "classification": classification,
            "risk_level": risk_level,
            "decision": decision
        }

        # ----------------------------------------------------
        # LOG PREDICTION
        # ----------------------------------------------------

        log_prediction(
            source="Single Transaction",

            transaction_time=(
                transaction_time
            ),

            amount=(
                transaction_amount
            ),

            fraud_probability=(
                probability
            ),

            prediction=(
                prediction
            ),

            classification=(
                classification
            ),

            risk_level=(
                risk_level
            ),

            decision=(
                decision
            )
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.divider()

        st.header(
            "📊 Prediction Result"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

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
                "Threshold",
                f"{THRESHOLD:.2f}"
            )

        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                f"🚨 FRAUD DETECTED\n\n"
                f"Fraud probability: "
                f"{probability * 100:.2f}%\n\n"
                f"Model classification: **FRAUD**"
            )

        else:

            st.success(
                f"✅ TRANSACTION CLASSIFIED AS NORMAL\n\n"
                f"Fraud probability: "
                f"{probability * 100:.2f}%\n\n"
                f"Model classification: **NORMAL**"
            )

        # ----------------------------------------------------
        # ACTUAL DATASET CLASS
        # ----------------------------------------------------

        if (
            st.session_state[
                "dataset_test_mode"
            ]
            and
            st.session_state[
                "actual_class"
            ] is not None
        ):

            actual_class = (
                st.session_state[
                    "actual_class"
                ]
            )

            actual_label = (
                "FRAUD"
                if actual_class == 1
                else "NORMAL"
            )

            st.info(
                f"📌 Dataset actual class: "
                f"**{actual_label} "
                f"(Class {actual_class})**"
            )

            if prediction == actual_class:

                st.success(
                    "✅ Model prediction matches "
                    "the dataset label."
                )

            else:

                st.warning(
                    "⚠️ Model prediction does not "
                    "match the dataset label."
                )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.subheader(
            "Fraud Probability"
        )

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )

        # ----------------------------------------------------
        # TRANSACTION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Transaction Summary"
        )

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Transaction Amount",
                    "Transaction Time",
                    "Fraud Probability",
                    "Model Classification",
                    "Risk Level",
                    "Decision",
                    "Model",
                    "Decision Threshold"
                ],

                "Value": [
                    f"${transaction_amount:,.2f}",
                    f"{transaction_time:.4f}",
                    f"{probability * 100:.2f}%",
                    classification,
                    risk_level,
                    decision,
                    MODEL_NAME,
                    f"{THRESHOLD:.2f}"
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Prediction has been recorded in the "
            "SQLite audit database."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ============================================================
# BATCH CSV PREDICTION
# ============================================================

st.divider()

st.header(
    "📁 Batch Transaction Analysis"
)

st.write(
    "Upload a CSV containing multiple transactions "
    "to generate fraud predictions automatically."
)

st.caption(
    "Batch predictions are processed through the FastAPI backend."
)


batch_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    key="batch_csv"
)


if batch_file is not None:

    try:

        batch_df = pd.read_csv(
            batch_file
        )

        required_features = FEATURES

        missing_features = [
            col
            for col in required_features
            if col not in batch_df.columns
        ]

        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(
                    missing_features
                )
            )

        else:

            st.success(
                f"Loaded "
                f"{len(batch_df):,} transactions."
            )

            # ------------------------------------------------
            # PREPARE DATA
            # ------------------------------------------------

            prediction_df = batch_df[
                required_features
            ].copy()

            prediction_df = (
                prediction_df.apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            missing_values = (
                prediction_df
                .isnull()
                .sum()
                .sum()
            )

            if missing_values > 0:

                invalid_columns = (
                    prediction_df
                    .columns[
                        prediction_df
                        .isnull()
                        .any()
                    ]
                    .tolist()
                )

                st.error(
                    f"Dataset contains "
                    f"{missing_values} missing "
                    f"or invalid values."
                )

                st.write(
                    "Affected columns:",
                    invalid_columns
                )

            else:

                # ------------------------------------------------
                # MODEL PREDICTION THROUGH FASTAPI
                # ------------------------------------------------

                transactions_payload = (
                    prediction_df
                    .to_dict(orient="records")
                )

                progress_bar = st.progress(0)
                progress_text = st.empty()

                def update_batch_progress(done, total):
                    progress_bar.progress(done / total if total else 1.0)
                    progress_text.caption(
                        f"FastAPI prediction progress: {done:,} / {total:,}"
                    )

                api_batch_response = predict_batch_using_api(
                    transactions_payload,
                    chunk_size=2000,
                    progress_callback=update_batch_progress,
                )

                progress_bar.progress(1.0)
                progress_text.caption(
                    f"FastAPI prediction complete: {len(transactions_payload):,} / {len(transactions_payload):,}"
                )

                api_predictions = (
                    api_batch_response["predictions"]
                )

                results = batch_df.copy()

                results[
                    "Fraud_Probability"
                ] = [
                    item["fraud_probability"]
                    for item in api_predictions
                ]

                results[
                    "Prediction"
                ] = [
                    item["prediction"]
                    for item in api_predictions
                ]

                results[
                    "Classification"
                ] = [
                    item["classification"]
                    for item in api_predictions
                ]

                results[
                    "Risk_Level"
                ] = [
                    item["risk_level"]
                    for item in api_predictions
                ]

                results[
                    "Decision"
                ] = [
                    item["decision"]
                    for item in api_predictions
                ]

                # ------------------------------------------------
                # SAVE RESULTS IN SESSION
                # ------------------------------------------------

                st.session_state[
                    "batch_results"
                ] = results

                # ------------------------------------------------
                # KPI CALCULATIONS
                # ------------------------------------------------

                total_transactions = (
                    len(results)
                )

                fraud_count = int(
                    results[
                        "Prediction"
                    ].sum()
                )

                normal_count = (
                    total_transactions
                    - fraud_count
                )

                fraud_rate = (
                    fraud_count
                    / total_transactions
                    * 100
                    if total_transactions > 0
                    else 0
                )

                average_probability = (
                    results[
                        "Fraud_Probability"
                    ].mean()
                    * 100
                )

                # ------------------------------------------------
                # KPI CARDS
                # ------------------------------------------------

                st.subheader(
                    "📈 Batch Analysis Summary"
                )

                col1, col2, col3, col4, col5 = (
                    st.columns(5)
                )

                with col1:

                    st.metric(
                        "Transactions",
                        f"{total_transactions:,}"
                    )

                with col2:

                    st.metric(
                        "Fraud Detected",
                        f"{fraud_count:,}"
                    )

                with col3:

                    st.metric(
                        "Normal",
                        f"{normal_count:,}"
                    )

                with col4:

                    st.metric(
                        "Fraud Rate",
                        f"{fraud_rate:.2f}%"
                    )

                with col5:

                    st.metric(
                        "Avg Probability",
                        f"{average_probability:.2f}%"
                    )

                # ------------------------------------------------
                # RISK DISTRIBUTION
                # ------------------------------------------------

                st.subheader(
                    "⚠️ Risk Distribution"
                )

                risk_counts = (
                    results[
                        "Risk_Level"
                    ]
                    .value_counts()
                    .rename_axis(
                        "Risk Level"
                    )
                    .reset_index(
                        name="Transactions"
                    )
                )

                st.dataframe(
                    risk_counts,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    risk_counts.set_index(
                        "Risk Level"
                    )
                )

                # ------------------------------------------------
                # DECISION DISTRIBUTION
                # ------------------------------------------------

                st.subheader(
                    "🎯 Business Decision Distribution"
                )

                decision_counts = (
                    results[
                        "Decision"
                    ]
                    .value_counts()
                    .rename_axis(
                        "Decision"
                    )
                    .reset_index(
                        name="Transactions"
                    )
                )

                st.dataframe(
                    decision_counts,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    decision_counts.set_index(
                        "Decision"
                    )
                )

                # ------------------------------------------------
                # RESULTS TABLE
                # ------------------------------------------------

                st.subheader(
                    "📊 Prediction Results"
                )

                display_columns = [
                    "Time",
                    "Amount",
                    "Fraud_Probability",
                    "Classification",
                    "Risk_Level",
                    "Decision"
                ]

                available_display_columns = [
                    col
                    for col in display_columns
                    if col in results.columns
                ]

                display_df = results[
                    available_display_columns
                ].copy()

                display_df[
                    "Fraud_Probability"
                ] = (
                    display_df[
                        "Fraud_Probability"
                    ]
                    * 100
                ).round(2)

                display_df = (
                    display_df.rename(
                        columns={
                            "Fraud_Probability":
                                "Fraud Probability (%)",

                            "Risk_Level":
                                "Risk Level"
                        }
                    )
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # SAVE BATCH TO DATABASE
                # ------------------------------------------------

                st.subheader(
                    "🗄️ Audit Logging"
                )

                if st.button(
                    "💾 Save Batch Predictions to Audit Log",
                    use_container_width=True
                ):

                    try:

                        log_batch_predictions(
                            results
                        )

                        st.success(
                            f"Successfully saved "
                            f"{len(results):,} predictions "
                            f"to SQLite."
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to save batch "
                            f"predictions: {e}"
                        )

                # ------------------------------------------------
                # DOWNLOAD RESULTS
                # ------------------------------------------------

                st.subheader(
                    "⬇️ Export Results"
                )

                csv_output = (
                    results
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    label=(
                        "⬇️ Download Prediction Results"
                    ),

                    data=csv_output,

                    file_name=(
                        "fraud_predictions.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True
                )

    except Exception as e:

        st.error(
            f"Batch prediction failed: {e}"
        )


# ============================================================
# DASHBOARD ANALYTICS
# ============================================================

st.divider()

st.header(
    "📊 Dashboard Analytics"
)

history_df = load_prediction_history()


if history_df.empty:

    st.info(
        "No prediction history available yet. "
        "Run predictions to populate the dashboard."
    )

else:

    # --------------------------------------------------------
    # OVERALL KPIs
    # --------------------------------------------------------

    total_logged = len(
        history_df
    )

    fraud_logged = int(
        history_df[
            "prediction"
        ].sum()
    )

    normal_logged = (
        total_logged
        - fraud_logged
    )

    average_probability = (
        history_df[
            "fraud_probability"
        ]
        .mean()
        * 100
    )

    logged_fraud_rate = (
        fraud_logged
        / total_logged
        * 100
        if total_logged > 0
        else 0
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Total Predictions",
            f"{total_logged:,}"
        )

    with col2:

        st.metric(
            "Fraud Predictions",
            f"{fraud_logged:,}"
        )

    with col3:

        st.metric(
            "Normal Predictions",
            f"{normal_logged:,}"
        )

    with col4:

        st.metric(
            "Fraud Rate",
            f"{logged_fraud_rate:.2f}%"
        )

    with col5:

        st.metric(
            "Avg Fraud Probability",
            f"{average_probability:.2f}%"
        )

    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Prediction Distribution"
        )

        prediction_chart = (
            history_df[
                "classification"
            ]
            .value_counts()
            .rename_axis(
                "Classification"
            )
            .to_frame(
                "Transactions"
            )
        )

        st.bar_chart(
            prediction_chart
        )

    with col2:

        st.subheader(
            "Risk Distribution"
        )

        risk_chart = (
            history_df[
                "risk_level"
            ]
            .value_counts()
            .rename_axis(
                "Risk Level"
            )
            .to_frame(
                "Transactions"
            )
        )

        st.bar_chart(
            risk_chart
        )

    # --------------------------------------------------------
    # DECISION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Business Decision Distribution"
    )

    decision_chart = (
        history_df[
            "decision"
        ]
        .value_counts()
        .rename_axis(
            "Decision"
        )
        .to_frame(
            "Transactions"
        )
    )

    st.bar_chart(
        decision_chart
    )

    # --------------------------------------------------------
    # FRAUD PROBABILITY TREND
    # --------------------------------------------------------

    st.subheader(
        "Fraud Probability Trend"
    )

    trend_df = history_df.copy()

    trend_df[
        "timestamp"
    ] = pd.to_datetime(
        trend_df[
            "timestamp"
        ],
        errors="coerce"
    )

    trend_df = (
        trend_df
        .dropna(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
    )

    if not trend_df.empty:

        trend_series = (
            trend_df[
                [
                    "timestamp",
                    "fraud_probability"
                ]
            ]
            .set_index(
                "timestamp"
            )
        )

        st.line_chart(
            trend_series
        )

    # --------------------------------------------------------
    # RECENT PREDICTIONS
    # --------------------------------------------------------

    st.subheader(
        "🕒 Recent Prediction Activity"
    )

    recent_df = history_df.head(
        20
    ).copy()

    recent_display = recent_df[
        [
            "timestamp",
            "source",
            "amount",
            "fraud_probability",
            "classification",
            "risk_level",
            "decision"
        ]
    ].copy()

    recent_display[
        "fraud_probability"
    ] = (
        recent_display[
            "fraud_probability"
        ]
        * 100
    ).round(2)

    recent_display = (
        recent_display.rename(
            columns={
                "timestamp":
                    "Timestamp",

                "source":
                    "Source",

                "amount":
                    "Amount",

                "fraud_probability":
                    "Fraud Probability (%)",

                "classification":
                    "Classification",

                "risk_level":
                    "Risk Level",

                "decision":
                    "Decision"
            }
        )
    )

    st.dataframe(
        recent_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD AUDIT LOG
    # --------------------------------------------------------

    audit_csv = (
        history_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Full Audit Log",
        data=audit_csv,
        file_name="fraud_prediction_audit_log.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# DRIFT MONITORING
# ============================================================

st.divider()

st.header(
    "📡 Model Monitoring & Drift Detection"
)

st.write(
    "Upload a reference dataset and a current "
    "dataset to calculate Population Stability Index "
    "(PSI) for important transaction features."
)

st.info(
    "PSI interpretation: "
    "< 0.10 = Stable | "
    "0.10–0.25 = Warning | "
    "≥ 0.25 = Critical"
)


col1, col2 = st.columns(2)


with col1:

    reference_file = st.file_uploader(
        "📁 Reference Dataset",
        type=["csv"],
        key="reference_dataset"
    )


with col2:

    current_file = st.file_uploader(
        "📁 Current Dataset",
        type=["csv"],
        key="current_dataset"
    )


if (
    reference_file is not None
    and current_file is not None
):

    try:

        reference_df = pd.read_csv(
            reference_file
        )

        current_df = pd.read_csv(
            current_file
        )

        monitoring_features = [
            "Time",
            "Amount"
        ]

        monitoring_features += [
            f"V{i}"
            for i in range(1, 29)
        ]

        available_features = [
            feature
            for feature in monitoring_features
            if (
                feature in reference_df.columns
                and
                feature in current_df.columns
            )
        ]

        if not available_features:

            st.error(
                "No common monitoring features "
                "were found in the two datasets."
            )

        else:

            psi_results = []

            for feature in available_features:

                psi_value = calculate_psi(
                    reference_df[feature],
                    current_df[feature]
                )

                status, explanation = (
                    interpret_psi(
                        psi_value
                    )
                )

                psi_results.append(
                    {
                        "Feature": feature,
                        "PSI": psi_value,
                        "Status": status,
                        "Interpretation":
                            explanation
                    }
                )

            psi_df = pd.DataFrame(
                psi_results
            )

            psi_df[
                "PSI"
            ] = psi_df[
                "PSI"
            ].round(4)

            # ------------------------------------------------
            # PSI SUMMARY
            # ------------------------------------------------

            critical_count = int(
                (
                    psi_df["Status"]
                    == "CRITICAL"
                ).sum()
            )

            warning_count = int(
                (
                    psi_df["Status"]
                    == "WARNING"
                ).sum()
            )

            stable_count = int(
                (
                    psi_df["Status"]
                    == "STABLE"
                ).sum()
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Stable Features",
                    stable_count
                )

            with col2:

                st.metric(
                    "Warning Features",
                    warning_count
                )

            with col3:

                st.metric(
                    "Critical Features",
                    critical_count
                )

            # ------------------------------------------------
            # OVERALL STATUS
            # ------------------------------------------------

            if critical_count > 0:

                st.error(
                    "🚨 CRITICAL DRIFT DETECTED"
                )

            elif warning_count > 0:

                st.warning(
                    "⚠️ MODERATE DRIFT DETECTED"
                )

            else:

                st.success(
                    "✅ NO SIGNIFICANT DRIFT DETECTED"
                )

            # ------------------------------------------------
            # PSI TABLE
            # ------------------------------------------------

            st.subheader(
                "PSI Feature Report"
            )

            st.dataframe(
                psi_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # PSI CHART
            # ------------------------------------------------

            st.subheader(
                "PSI by Feature"
            )

            psi_chart = (
                psi_df[
                    [
                        "Feature",
                        "PSI"
                    ]
                ]
                .set_index(
                    "Feature"
                )
            )

            st.bar_chart(
                psi_chart
            )

            # ------------------------------------------------
            # HIGH DRIFT FEATURES
            # ------------------------------------------------

            high_drift = psi_df[
                psi_df["PSI"] >= 0.25
            ].sort_values(
                "PSI",
                ascending=False
            )

            if not high_drift.empty:

                st.subheader(
                    "🚨 Features Requiring Attention"
                )

                st.dataframe(
                    high_drift,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as e:

        st.error(
            f"Drift monitoring failed: {e}"
        )


# ============================================================
# DATABASE INFORMATION
# ============================================================

st.divider()

st.header(
    "🗄️ Prediction Audit Database"
)

try:

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    database_count = cursor.fetchone()[0]

    connection.close()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Stored Predictions",
            f"{database_count:,}"
        )

    with col2:

        st.write(
            "**Database file:**"
        )

        st.code(
            str(DATABASE_PATH)
        )

except Exception as e:

    st.error(
        f"Database status unavailable: {e}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Research Project | "
    "Random Forest | Streamlit | FastAPI | SQLite"
)