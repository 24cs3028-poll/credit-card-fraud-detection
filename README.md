# Credit Card Fraud Detection Platform

Advanced company-presentation prototype with chronological validation, cost-sensitive learning, Logistic Regression (LBFGS/SAGA), Random Forest, XGBoost, CatBoost, neural-network optimizer comparison (Adam/RMSprop/SGD), PR-AUC, threshold optimization, SHAP, risk scoring, FastAPI, SQLite logging, Streamlit monitoring and Docker.

## Run API
uvicorn api.main:app --reload

## Run dashboard
streamlit run dashboard/app.py

## Architecture
Dataset -> preprocessing -> ML -> threshold -> risk engine -> FastAPI -> SQLite -> Streamlit

This is an educational prototype. Production financial deployment requires security, authentication, encryption, governance, compliance, drift monitoring and extensive validation.
