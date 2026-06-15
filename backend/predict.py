# src/predict.py

import joblib
import pandas as pd

model = joblib.load("models/failure_model.pkl")

sample = pd.DataFrame([{
    "cpu":95,
    "memory":90,
    "disk":70,
    "temperature":88,
    "network":500
}])

prob = model.predict_proba(sample)[0][1]

print(f"Failure Probability: {prob*100:.2f}%")