from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/failure_model.pkl")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):

    sample = pd.DataFrame([{
        "cpu": data["cpu"],
        "memory": data["memory"],
        "disk": data["disk"],
        "temperature": data["temperature"],
        "network": data["network"]
    }])

    probability = model.predict_proba(sample)[0][1]

    return {
        "failure_probability": round(float(probability * 100),2)
    }