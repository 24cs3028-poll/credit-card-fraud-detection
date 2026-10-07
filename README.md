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
