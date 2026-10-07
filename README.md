# Credit Card Fraud Detection — Production-Ready ML Platform

An end-to-end machine learning system for detecting fraudulent credit card transactions using time-aware validation, multiple ML models, optimized decision thresholds, and a production-oriented software architecture.

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent a very small proportion of total transactions.

This project develops a complete fraud detection pipeline:

- Data preprocessing
- Chronological train/validation/test splitting
- Multiple machine learning models
- Model comparison using PR-AUC and ROC-AUC
- Validation-based threshold optimization
- Final evaluation on an untouched test set
- Model serialization
- FastAPI backend
- Streamlit dashboard
- Prediction logging and monitoring

## Machine Learning Models

The project evaluates:

- Logistic Regression — LBFGS
- Logistic Regression — SAGA
- Random Forest
- XGBoost
- CatBoost

The final model is selected using validation performance.

## Imbalanced Classification Strategy

SMOTE is **not used** in this project.

The project instead uses:

- Time-aware data splitting
- Model comparison
- Precision-Recall evaluation
- Validation-based threshold optimization
- Recall/precision trade-off analysis

## Data Splitting

The dataset is divided chronologically:

```text
70% → Training
15% → Validation
15% → Test
