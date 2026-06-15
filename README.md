# infrastructure-failure-prediction
Full-stack machine learning platform that predicts server failures from infrastructure telemetry data using React, FastAPI, and XGBoost. Provides real-time failure risk analysis and explainable insights through an interactive dashboard.
# Predictive Server Failure System

A full-stack machine learning platform that predicts server failures using infrastructure telemetry data.

## Features

- Real-time failure prediction
- React dashboard
- FastAPI backend
- XGBoost classification model
- Infrastructure telemetry analysis
- Explainable AI with feature importance

## Tech Stack

### Frontend
- React
- Axios
- Vite

### Backend
- FastAPI
- Pandas
- XGBoost
- Scikit-Learn

## Telemetry Inputs

- CPU Utilization
- Memory Utilization
- Disk Usage
- Temperature
- Network Throughput

## Example Prediction

| Metric | Value |
|----------|----------|
| CPU | 95% |
| Memory | 90% |
| Temperature | 88°C |

Predicted Failure Probability: **97.12%**

## Run Backend

```bash
cd backend

source ../venv/bin/activate

python -m uvicorn api:app --reload
```

## Run Frontend

```bash
cd frontend

npm install

npm run dev
```

## Resume Description

Developed a full-stack predictive maintenance platform using React, FastAPI, and XGBoost to analyze infrastructure telemetry and forecast server failures through REST APIs and interactive dashboards.