Predictive Server Failure System

A machine learning app that estimates the probability of a server or infrastructure failure from live telemetry — CPU load, memory usage, disk usage, temperature, and network traffic. Feed it a snapshot of system stats and it returns a failure probability in real time.

Built as a full-stack project: a React dashboard on the frontend, a FastAPI service on the backend, and an XGBoost model doing the actual prediction.

Demo


Frontend: https://infrastructure-failure-prediction-1.onrender.com/
Backend API docs (Swagger): https://infrastructure-failure-prediction-2.onrender.com/docs


What it does


Takes telemetry input and predicts failure probability
Shows which features are driving each prediction (feature importance)
Interactive dashboard for entering values and viewing results
REST API with auto-generated Swagger docs


Stack

Frontend — React, Axios, CSS
Backend — FastAPI, Uvicorn
ML — XGBoost, scikit-learn, pandas, joblib
Deployment — Render

How it fits together

React dashboard → Axios request → FastAPI → XGBoost model → prediction → JSON response → back to dashboard

Running it locally

Clone the repo:

bashgit clone https://github.com/YOUR_USERNAME/infrastructure-failure-prediction.git

Install Python dependencies:

bashpip install -r requirements.txt

Start the backend:

bashcd backend
uvicorn api:app --reload

Start the frontend:

bashcd frontend
npm install
npm run dev


Frontend: http://localhost:5173
Backend: http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs


Author

Rimmi