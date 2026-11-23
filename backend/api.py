from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

MODEL_PATH = Path(__file__).resolve().parent / "models" / "failure_model.pkl"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    cpu: FiniteFloat = Field(ge=0, le=100)
    memory: FiniteFloat = Field(ge=0, le=100)
    disk: FiniteFloat = Field(ge=0, le=100)
    temperature: FiniteFloat = Field(ge=20, le=95)
    network: FiniteFloat = Field(ge=0, le=1000)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError):
    # Rejected values such as infinity cannot be serialized in a JSON response.
    errors = jsonable_encoder(exc.errors(), exclude={"input"})
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(data: PredictionRequest):
    sample = pd.DataFrame([data.model_dump()])
    probability = model.predict_proba(sample)[0][1]

    return {"failure_probability": round(float(probability * 100), 2)}
