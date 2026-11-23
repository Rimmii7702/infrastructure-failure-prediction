from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "failure_model.pkl"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame([{
    "cpu":95,
    "memory":90,
    "disk":70,
    "temperature":88,
    "network":500
}])

prob = model.predict_proba(sample)[0][1]

print(f"Failure Probability: {prob*100:.2f}%")
