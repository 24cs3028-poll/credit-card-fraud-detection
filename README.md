# 💳 Credit Card Fraud Detection — ML & API Platform

An end-to-end machine learning application for detecting potentially fraudulent
credit-card transactions using a trained **Random Forest classifier**.

The project combines machine learning, REST APIs, interactive dashboards,
batch prediction, audit logging, and model monitoring into a single
application.

---

## 🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem
where fraudulent transactions represent only a small fraction of total
transactions.

This project provides an application for:

- 🔍 Individual transaction fraud analysis
- 📁 Large-scale batch transaction prediction
- 🤖 Machine-learning based fraud classification
- ⚡ FastAPI REST prediction service
- 📊 Interactive Streamlit dashboard
- 🗄️ SQLite prediction audit database
- 📡 Population Stability Index (PSI) drift monitoring
- 📈 Prediction analytics and risk analysis
- 🐳 Docker-based deployment support

The application is designed as an **educational and portfolio-oriented
fraud detection prototype** demonstrating how a machine-learning model can
be integrated into a complete software application.

---

# ✨ Key Features

## 🤖 Machine Learning

- Random Forest fraud classification model
- 30 transaction features
- Fraud probability estimation
- Configurable classification threshold
- Model metadata management
- Serialized model and scaler artifacts

## 🔍 Individual Transaction Analysis

Analyze a single transaction and receive:

- Fraud probability
- Fraud / Normal classification
- Risk level
- Decision recommendation
- Model information
- Classification threshold

### Risk levels

| Fraud Probability | Risk Level |
|---:|---|
| `< 0.10` | 🟢 LOW |
| `0.10 – < 0.30` | 🟡 MEDIUM |
| `0.30 – < 0.70` | 🟠 HIGH |
| `≥ 0.70` | 🔴 CRITICAL |

---

## 📁 Batch Transaction Analysis

The application supports CSV-based batch prediction.

Large datasets are processed through the **FastAPI backend in chunks** to
avoid oversized HTTP requests and improve reliability.

The system can process tens of thousands of transactions while displaying
prediction progress in the Streamlit interface.

---

## ⚡ FastAPI Backend

The machine-learning model is exposed through a REST API.

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Backend health check |
| `GET` | `/model-info` | Model metadata |
| `GET` | `/example` | Example transaction |
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch transaction prediction |

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
🖥️ Streamlit Dashboard

The Streamlit application provides an interactive interface for analysts
and users.

Dashboard modules
🧪 Test Using Dataset

Upload a dataset to evaluate the trained model against known transaction
records.

🔍 Analyze Transaction

Enter transaction information manually and receive an immediate prediction.

📁 Batch Transaction Analysis

Upload a CSV containing multiple transactions and process them through the
FastAPI backend.

📊 Dashboard Analytics

The analytics dashboard provides:

Total predictions
Fraud predictions
Normal predictions
Fraud rate
Average fraud probability
Prediction distribution
Risk-level distribution
Decision distribution
Prediction trends
Recent prediction activity
📡 Model Monitoring

Monitor changes in transaction distributions using PSI-based drift detection.

🗄️ Prediction Audit Database

Prediction results are stored in SQLite for:

Auditing
Historical analysis
Dashboard statistics
Prediction tracking
CSV export
🏗️ System Architecture
🔄 Machine Learning Workflow
📊 Model Information
Property	Value
Model	Random Forest
Model Version	1.0.0
Target	Class
Features	30
Classification Threshold	0.14
SMOTE	Not used
Validation	Chronological 70/15/15 split
Input Features

The model uses:

Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount

The target variable is:

Class

where:

0 = Normal transaction
1 = Fraudulent transaction
📈 Model Performance

Performance metrics should be reported using the actual results produced
during model evaluation.

Recommended visualizations for the project include:

ROC-AUC curve
Precision-Recall curve
Confusion matrix
Feature importance
Fraud vs. normal distribution
Prediction probability distribution

Example section:

## 📈 Model Performance

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

### ROC-AUC Curve

![ROC-AUC Curve](screenshots/roc_auc.png)

### Precision-Recall Curve

![Precision Recall Curve](screenshots/precision_recall.png)

### Feature Importance

![Feature Importance](screenshots/feature_importance.png)

Add these images only after generating them from the actual model results.
Do not use fabricated performance values.

📸 Application Screenshots

Recommended screenshots for the GitHub repository:

screenshots/
├── dashboard.png
├── prediction-result.png
├── batch-analysis.png
├── analytics.png
├── drift-monitoring.png
└── api-swagger.png

Then display them in the README:

Dashboard

Transaction Prediction

Batch Analysis

Analytics

Model Monitoring

FastAPI Documentation

📡 Model Monitoring & Drift Detection

The application supports Population Stability Index (PSI) based monitoring.

PSI compares the distribution of a reference dataset with a current dataset.

PSI interpretation
PSI	Interpretation
< 0.10	🟢 Stable
0.10 – < 0.25	🟡 Warning
≥ 0.25	🔴 Critical

This provides a simple mechanism for identifying potential changes in
transaction data distributions.

🗄️ Prediction Audit Database

Prediction results are stored in a local SQLite database.

Database:

database/fraud_predictions.db

The audit records can contain information such as:

Timestamp
Transaction time
Amount
Fraud probability
Prediction
Classification
Risk level
Decision
Model name
Model version
Threshold
Prediction source

The dashboard can also provide downloadable prediction records.

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
├── screenshots/
│   ├── dashboard.png
│   ├── prediction-result.png
│   ├── batch-analysis.png
│   ├── analytics.png
│   ├── drift-monitoring.png
│   └── api-swagger.png
│
├── Credit_Card_Fraud_Detection_Production_Ready (1).ipynb
│
├── fraud_model.pkl
├── fraud_scaler.pkl
├── model_metadata.json
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation
1. Clone the repository
git clone https://github.com/24cs3028-poll/credit-card-fraud-detection.git
cd credit-card-fraud-detection
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment
Windows PowerShell
.\venv\Scripts\Activate.ps1
Windows Command Prompt
venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
▶️ Running the Application

The application uses two services:

Streamlit Dashboard
        ↓
FastAPI Backend
        ↓
Random Forest Model
5. Start FastAPI

Open the first terminal:

python -m uvicorn api.main:app --reload

The API will run at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
6. Start Streamlit

Open another terminal:

python -m streamlit run dashboard/app.py

The dashboard will open in your browser.

The Streamlit sidebar should display:

FastAPI: Connected
API: http://127.0.0.1:8000
🧪 Testing the API
Health Check

Open:

http://127.0.0.1:8000/health

Expected response:

{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "metadata_loaded": true
}
Model Information
http://127.0.0.1:8000/model-info
Example Prediction

The API provides an example transaction through:

http://127.0.0.1:8000/example

The returned transaction can be submitted to:

POST /predict
🐳 Docker

Build the Docker image:

docker build -t fraud-detection .

Run the container:

docker run -p 8501:8501 fraud-detection
🔐 Security Considerations

For a real production deployment, additional security controls should be
implemented.

Recommended improvements include:

API authentication
HTTPS/TLS
Secure secrets management
Input validation
Rate limiting
Role-based access control
Database access controls
Secure model artifact storage
Logging and monitoring
Dependency vulnerability scanning
⚠️ Limitations

This project is an educational and portfolio-oriented fraud detection
prototype.

It should not be considered a complete production banking or financial
fraud prevention system.

Important real-world considerations include:

Continuous model retraining
Real-time transaction streams
Strong authentication and authorization
Secure secret management
Model explainability
Production-grade database infrastructure
Monitoring and alerting
Data privacy and regulatory compliance
Load testing
High-availability deployment
Model governance
False-positive management
Human review workflows
🔮 Future Improvements

Potential future improvements include:

Real-time transaction streaming
Cloud deployment
Kafka / Redis integration
Automated model retraining
SHAP-based model explanations
Authentication and role-based access
PostgreSQL database
Automated CI/CD
Advanced model monitoring
Alert notifications for high-risk transactions
Model version management
Automated drift-triggered retraining
Container orchestration
Horizontal API scaling
🧰 Technology Stack
Technology	Purpose
Python	Core programming language
Pandas	Data processing
NumPy	Numerical computation
Scikit-learn	Machine learning
Random Forest	Fraud classification
Joblib	Model serialization
FastAPI	REST API
Uvicorn	API server
Streamlit	Interactive dashboard
SQLite	Prediction audit database
Docker	Containerization
Git	Version control
GitHub	Source-code hosting
📋 Example Workflow
1. User opens Streamlit dashboard
              ↓
2. User enters transaction information
              ↓
3. Streamlit sends request to FastAPI
              ↓
4. FastAPI validates the input
              ↓
5. Random Forest generates fraud probability
              ↓
6. Classification threshold is applied
              ↓
7. Risk level and decision are generated
              ↓
8. Result is displayed in Streamlit
              ↓
9. Prediction is recorded in SQLite
              ↓
10. Dashboard analytics are updated
📦 Model Artifacts

The repository contains the trained model artifacts:

fraud_model.pkl
fraud_scaler.pkl
model_metadata.json

The metadata file contains information about:

Model name
Model version
Features
Target variable
Classification threshold
Validation approach
SMOTE usage
🎓 Educational Purpose

This project demonstrates how a machine-learning model can move beyond a
notebook into a complete application architecture.

It covers:

Machine Learning
      ↓
Model Serialization
      ↓
REST API
      ↓
Interactive Dashboard
      ↓
Batch Processing
      ↓
Database Logging
      ↓
Monitoring
      ↓
Containerization
      ↓
Version Control
👤 Author

24cs3028-poll

GitHub:

https://github.com/24cs3028-poll

Repository:

https://github.com/24cs3028-poll/credit-card-fraud-detection

📄 License

This project is intended for educational and portfolio purposes.

Add an appropriate open-source license if you plan to distribute the project
for reuse.

⭐ Acknowledgements

The project uses the commonly used credit-card fraud detection dataset
containing anonymized transaction features.

The project was developed as an end-to-end demonstration of machine-learning
model deployment, API integration, dashboard development, batch processing,
and model monitoring.
