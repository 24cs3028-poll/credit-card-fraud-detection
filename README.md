💳 Credit Card Fraud Detection — ML & API Platform





An end-to-end machine-learning application for detecting potentially fraudulent credit-card transactions using a Random Forest classifier, FastAPI, Streamlit, batch prediction, SQLite audit logging, and PSI-based drift monitoring.

Project status: Educational / portfolio-ready fraud-detection prototype. It demonstrates a complete ML-to-application workflow but should not be represented as a production banking or payment-security system without additional security, reliability, compliance, monitoring, and validation work.

📌 Project Overview

Credit-card fraud detection is a highly imbalanced classification problem in which fraudulent transactions represent a very small fraction of total transactions.

This project provides a complete workflow for:

Individual transaction fraud analysis

Large-scale CSV batch prediction

Machine-learning based fraud probability estimation

Risk-level classification

Decision recommendations such as ALLOW, REVIEW, and BLOCK

REST API inference through FastAPI

Interactive analysis through Streamlit

SQLite prediction/audit logging

Population Stability Index (PSI) monitoring

Model and threshold information

API documentation through Swagger/OpenAPI

The application demonstrates how a trained fraud-detection model can be integrated into a usable ML application rather than remaining only inside a notebook.

✨ Key Features

🤖 Machine Learning

Random Forest fraud classification model

30 transaction features

Fraud probability estimation

Configurable classification threshold

Serialized model and scaler artifacts

Model metadata and version information

Chronological 70/15/15 validation split

Imbalanced fraud-detection workflow

🔎 Individual Transaction Analysis

The Streamlit dashboard supports single-transaction analysis with:

Fraud probability

Fraud / normal classification

Risk-level classification

Decision recommendation

Model information

Classification threshold

API-backed prediction

📦 Batch Transaction Analysis

The application supports CSV-based batch prediction.

Large datasets can be processed in chunks to reduce memory/API request pressure and provide progress feedback.

CSV Dataset
    ↓
Chunked Upload
    ↓
FastAPI Prediction
    ↓
Random Forest
    ↓
Fraud Probabilities
    ↓
Risk Classification
    ↓
Prediction Results

⚡ FastAPI Backend

The machine-learning inference layer is exposed through REST endpoints for health checks, model information, examples, single prediction, and batch prediction.

📊 Streamlit Dashboard

The dashboard provides a user-friendly interface for:

Individual transaction prediction

Batch prediction

Prediction results

Analytics

Model information

Risk interpretation

Drift monitoring

🗄️ SQLite Audit Database

Prediction activity can be recorded in a local SQLite database for later analysis and audit-style review.

📈 PSI Drift Monitoring

Population Stability Index (PSI) monitoring helps identify changes in transaction-data distributions.

PSI Value

Interpretation

< 0.10

🟢 Stable

0.10 – < 0.25

🟡 Warning

≥ 0.25

🔴 Critical

🏗️ System Architecture

flowchart LR
    A[Transaction Data] --> B[Streamlit Dashboard]
    B --> C[FastAPI]
    C --> D[Random Forest Model]
    D --> E[Fraud Probability]
    E --> F[Risk Classification]
    F --> G[ALLOW / REVIEW / BLOCK]
    B --> H[(SQLite Database)]
    B --> I[PSI Drift Monitoring]
    C --> J[Batch Prediction]
    J --> D

Architecture Components

Component

Responsibility

Streamlit

Interactive dashboard and user interface

FastAPI

ML inference API

Random Forest

Fraud classification

Scikit-learn

Machine-learning framework

Joblib

Model/scaler serialization

SQLite

Local prediction/audit storage

PSI

Distribution/drift monitoring

Pandas / NumPy

Data processing

Uvicorn

FastAPI application server

🔄 Machine Learning Workflow

flowchart LR
    A[Transaction Dataset] --> B[Preprocessing]
    B --> C[Feature Preparation]
    C --> D[Chronological Split]
    D --> E[Random Forest Training]
    E --> F[Model Evaluation]
    F --> G[Threshold Selection]
    G --> H[Model Serialization]
    H --> I[FastAPI]
    I --> J[Streamlit]

End-to-End Flow

Load transaction data.

Separate features and target.

Prepare the 30 model features.

Perform chronological train/validation/test splitting.

Train the Random Forest classifier.

Evaluate the model.

Apply the configured fraud threshold.

Save model artifacts.

Load artifacts in FastAPI.

Expose prediction endpoints.

Connect Streamlit to the API.

Store prediction information in SQLite.

Monitor distribution changes using PSI.

🧠 Model Information

Property

Value

Model

Random Forest

Model Version

1.0.0

Target

Class

Features

30

Feature Range

Time, V1–V28, Amount

Fraud Threshold

0.14

SMOTE

Not used

Validation Strategy

Chronological 70/15/15 split

Model Artifact

fraud_model.pkl

Scaler Artifact

fraud_scaler.pkl

Metadata

model_metadata.json

Feature Set

Time
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20
V21, V22, V23, V24, V25, V26, V27, V28
Amount

🚦 Risk Classification

Fraud Probability

Risk Level

< 0.10

🟢 LOW

0.10 – < 0.30

🟡 MEDIUM

0.30 – < 0.70

🟠 HIGH

≥ 0.70

🔴 CRITICAL

The fraud classification and probability can be converted into a decision recommendation such as:

ALLOW
REVIEW
BLOCK

The exact business decision logic should be validated and tuned for the intended real-world use case before deployment.

📊 Model Performance

Important: Add the actual graphs produced by your notebook/model evaluation here. Do not use fabricated accuracy, precision, recall, F1, ROC-AUC, or other performance values.

Confusion Matrix

<img width="648" height="651" alt="image" src="https://github.com/user-attachments/assets/71cb4e1a-3bcc-4fd9-9a10-d0c63ea82799" />



ROC-AUC Curve

<img width="892" height="604" alt="image" src="https://github.com/user-attachments/assets/5c51bbbb-1781-49ff-8974-c4f29a362e1d" />


Precision-Recall Curve

<img width="917" height="602" alt="image" src="https://github.com/user-attachments/assets/492c7351-9c7e-4989-9d0c-795bf1f57641" />



Feature Importance

<img width="569" height="346" alt="image" src="https://github.com/user-attachments/assets/752f9152-25a0-4ba9-b692-9d50b3010217" />


Performance Interpretation

For fraud detection, accuracy alone can be misleading because the dataset is highly imbalanced.

Useful evaluation measures include:

Precision

Recall

F1-score

ROC-AUC

PR-AUC

Confusion matrix

False-positive rate

False-negative rate

The operating threshold should be selected according to the business cost of missed fraud versus unnecessary transaction review.

🖥️ Application Screenshots

Dashboard

<img width="1283" height="420" alt="image" src="https://github.com/user-attachments/assets/98eef775-48d6-4836-9870-ad1eb4f39b55" />



Individual Transaction Prediction


<img width="1214" height="582" alt="image" src="https://github.com/user-attachments/assets/704e8855-048f-4be9-88c0-83313208b529" />
<img width="1209" height="469" alt="image" src="https://github.com/user-attachments/assets/98d1d551-97cb-41e3-bb19-61b8a8d32df2" />




Batch Analysis

<img width="1218" height="466" alt="image" src="https://github.com/user-attachments/assets/98930e92-ee51-4057-bc47-69360e1c80e6" />



Dashboard Analytics

<img width="1526" height="476" alt="image" src="https://github.com/user-attachments/assets/4d13155a-1e2c-49e5-9d51-39ac2903f448" />

<img width="1532" height="302" alt="image" src="https://github.com/user-attachments/assets/808c5d0b-ba90-4a1c-acdd-a2b9f74a0190" />



Drift Monitoring

<img width="1589" height="389" alt="image" src="https://github.com/user-attachments/assets/aeb7ee1c-7009-4d60-86bd-1c25af7fb8a2" />




FastAPI Swagger Documentation

<img width="1267" height="901" alt="image" src="https://github.com/user-attachments/assets/288eb6ba-38cc-4578-a7b9-c75cd2a5b53a" />
<img width="1265" height="828" alt="image" src="https://github.com/user-attachments/assets/e2a04d3e-36f6-4a8f-a844-a286ee96f3d8" />



🔎 Individual Prediction

A single transaction can be submitted through the Streamlit dashboard.

Fraud Probability
        ↓
Fraud / Normal
        ↓
Risk Level
        ↓
Decision

Example presentation:

Fraud Probability: 0.03
Classification: NORMAL
Risk Level: LOW
Decision: ALLOW

The exact values depend on the submitted transaction.

📦 Batch Prediction

The application supports CSV-based batch prediction for large transaction datasets.

Large datasets are processed in chunks rather than sending the entire file as one API request. This helps reduce HTTP request size, memory pressure, API timeout risk, and dashboard freezing.

The interface can also display processing progress while the prediction job is running.

Expected Dataset Structure

The model expects:

Time
V1 ... V28
Amount

If a Class column is present, it can be used for evaluation/analysis where supported.

⚡ FastAPI API

The FastAPI backend provides the machine-learning inference layer.

Available Endpoints

Method

Endpoint

Description

GET

/

API information

GET

/health

Backend health check

GET

/model-info

Model metadata

GET

/example

Example transaction

POST

/predict

Single transaction prediction

POST

/predict/batch

Batch transaction prediction

Interactive Swagger Documentation

When FastAPI is running:

http://127.0.0.1:8000/docs

📡 API Example

The exact request schema should be checked through Swagger because the endpoint expects the model's required feature set.

Request Structure

{
  "Time": 0,
  "V1": -1.36,
  "V2": -0.07,
  "V3": 2.53,
  "V4": 1.38,
  "V5": -0.34,
  "V6": 0.46,
  "V7": -0.02,
  "V8": -0.12,
  "V9": 0.12,
  "V10": -0.09,
  "V11": 0.42,
  "V12": -0.38,
  "V13": 0.27,
  "V14": -0.51,
  "V15": 0.18,
  "V16": 0.05,
  "V17": -0.21,
  "V18": 0.10,
  "V19": 0.03,
  "V20": -0.06,
  "V21": 0.01,
  "V22": 0.02,
  "V23": -0.04,
  "V24": 0.03,
  "V25": 0.02,
  "V26": -0.01,
  "V27": 0.00,
  "V28": 0.01,
  "Amount": 149.62
}

Response Structure

{
  "prediction": 0,
  "fraud_probability": 0.03,
  "risk_level": "LOW",
  "decision": "ALLOW"
}

The exact response fields and values should always be verified against the currently deployed FastAPI implementation.

🗄️ Prediction Audit Trail

Prediction information can be recorded in:

database/fraud_predictions.db

Typical information can include:

Prediction result

Fraud probability

Risk level

Decision

Timestamp

For production, SQLite would generally be replaced with a managed relational database appropriate for the expected workload.

📈 Model & Data Drift Monitoring

The application includes PSI-based monitoring to identify changes in transaction-data distributions.

PSI < 0.10
    ↓
Stable

0.10 ≤ PSI < 0.25
    ↓
Warning

PSI ≥ 0.25
    ↓
Critical

Why Drift Monitoring Matters

A fraud model can become less effective when transaction behavior changes.

Potential causes include:

New transaction patterns

Changes in customer behavior

Changes in merchant behavior

New fraud strategies

Data pipeline changes

Feature distribution changes

PSI can act as an early-warning signal for investigation and potential model retraining.

🧪 Quick Demo

1. Clone the Repository

git clone https://github.com/24cs3028-poll/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Create/Activate the Virtual Environment

Windows PowerShell

python -m venv venv
.\venv\Scripts\Activate.ps1

3. Install Dependencies

pip install -r requirements.txt

4. Start FastAPI

Open PowerShell 1:

.\venv\Scripts\python.exe -m uvicorn api.main:app --reload

FastAPI:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs

5. Start Streamlit

Open PowerShell 2:

.\venv\Scripts\python.exe -m streamlit run dashboard\app.py

Dashboard:

http://localhost:8501

🐳 Docker

The repository includes a Dockerfile for containerized deployment.

Before using Docker for a complete deployment, verify the Dockerfile's configured entrypoint and whether the intended deployment requires FastAPI, Streamlit, or both services.

Example:

docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection

Container ports and startup commands should match the current Dockerfile.

📁 Project Structure

credit-card-fraud-detection/
│
├── api/
│   └── main.py
│
├── dashboard/
│   ├── app.py
│   └── app_backup.py
│
├── database/
│   └── fraud_predictions.db
│
├── screenshots/
│   ├── dashboard.png
│   ├── prediction-result.png
│   ├── batch-analysis.png
│   ├── analytics.png
│   ├── drift-monitoring.png
│   ├── api-swagger.png
│   ├── confusion_matrix.png
│   ├── roc_auc.png
│   ├── precision_recall.png
│   └── feature_importance.png
│
├── fraud_model.pkl
├── fraud_scaler.pkl
├── model_metadata.json
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
│
└── Credit_Card_Fraud_Detection_Production_Ready (1).ipynb

🔐 Security & Production Considerations

This repository is an educational/portfolio implementation.

A real financial fraud-detection platform would require additional controls, including:

HTTPS/TLS

Authentication

Authorization / RBAC

API rate limiting

Secure secrets management

Input validation

Secure model artifact storage

Production database

Encryption

Audit logging

Monitoring and alerting

High availability

Load testing

Disaster recovery

Model versioning

Automated retraining

Data privacy controls

Regulatory and compliance review

Human review workflows

False-positive management

No production banking or payment-security guarantees should be inferred from this project.

⚠️ Limitations

The current project is a demonstration of an end-to-end fraud-detection workflow.

It is not a complete production banking system.

Potential limitations include:

Local SQLite database

Local model artifacts

No built-in authentication layer

No enterprise secrets-management system

No streaming transaction infrastructure

No guaranteed high availability

No production-scale distributed inference

No automated model retraining pipeline

No complete regulatory/compliance implementation

Limited explainability compared with dedicated XAI systems

Model performance depends on the underlying training data

Threshold selection requires business-specific validation

🚀 Future Improvements

Streaming Fraud Detection

Integrate real-time transaction streams using technologies such as:

Kafka / Redis / Cloud Streaming

Production Database

Replace local SQLite with:

PostgreSQL / Managed SQL Database

Model Explainability

Add SHAP or other explainability techniques to show why a transaction was considered suspicious.

Authentication & Authorization

Add:

JWT / OAuth2 / RBAC

for controlled API and dashboard access.

Automated Retraining

New Data
   ↓
Validation
   ↓
Training
   ↓
Evaluation
   ↓
Approval
   ↓
Model Deployment

Advanced Monitoring

Add:

Data-quality monitoring

Model-performance monitoring

Drift alerts

Prediction-volume monitoring

Latency monitoring

Error-rate monitoring

CI/CD

Introduce automated:

Testing

Linting

Model validation

Docker builds

Deployment

Scalable Deployment

Load Balancer
      ↓
FastAPI Instances
      ↓
Model Serving
      ↓
Production Database
      ↓
Monitoring / Alerting

🧰 Technology Stack

Technology

Purpose

Python

Core programming language

Pandas

Data processing

NumPy

Numerical operations

Scikit-learn

Machine learning

Random Forest

Fraud classification

Joblib

Model serialization

FastAPI

REST API

Uvicorn

API server

Streamlit

Interactive dashboard

SQLite

Local audit/prediction database

Docker

Containerization

Git

Version control

GitHub

Source-code hosting

Mermaid

Architecture/workflow diagrams

🧭 Example User Workflow

                    ┌──────────────────────┐
                    │  Transaction Input   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     FastAPI API      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Random Forest Model │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             Fraud Probability       Classification
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │    Risk Level        │
                    │ LOW / MEDIUM / HIGH  │
                    │ / CRITICAL           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ ALLOW / REVIEW /     │
                    │ BLOCK                │
                    └──────────────────────┘

📦 Model Artifacts

fraud_model.pkl
fraud_scaler.pkl
model_metadata.json

fraud_model.pkl

Serialized Random Forest model.

fraud_scaler.pkl

Serialized preprocessing/scaling artifact used by the inference pipeline.

model_metadata.json

Stores model-related metadata such as:

Model name

Version

Feature information

Target

Threshold

Validation information

🧪 Reproducibility

For consistent local execution:

Use the provided requirements.txt.

Use the provided model artifacts.

Keep the expected feature names unchanged.

Run FastAPI before using API-backed dashboard prediction.

Ensure model and scaler files are available in the expected project location.

📝 Dataset Notes

The application expects the model's required transaction features:

Time, V1, V2, ..., V28, Amount

If your dataset contains additional columns, remove or appropriately handle them according to the application's batch-processing implementation.

Never upload confidential, personally identifiable, or regulated financial information to a public repository.

🎯 Project Objectives

This project demonstrates practical skills in:

Fraud classification

Imbalanced classification awareness

Model training and serialization

API-based model serving

Interactive ML dashboards

Batch inference

Database logging

Model/data monitoring

Threshold-based decision systems

Software integration

Docker/containerization

Git/GitHub project management

📚 Educational Purpose

This project is intended for:

Machine-learning portfolio demonstration

Academic projects

Learning ML deployment

API integration practice

Streamlit application development

Fraud-detection experimentation

It should not be treated as financial, banking, security, legal, or compliance advice.

👤 Author

24cs3028-poll

GitHub:

https://github.com/24cs3028-poll/credit-card-fraud-detection

⭐ Acknowledgements

This project uses the Python data-science and machine-learning ecosystem, including:

Pandas

NumPy

Scikit-learn

FastAPI

Streamlit

SQLite

Joblib

Docker

📄 License

This project is intended for educational and portfolio purposes.

If you plan to distribute or commercially deploy the project, add an appropriate license and review the licensing requirements of all included datasets, libraries, models, and third-party components.
