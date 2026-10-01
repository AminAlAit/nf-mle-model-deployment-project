from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import FEATURE_COLUMNS, MODEL_PATH


app = FastAPI(title="NYC Taxi Trip Duration API", version="1.0.0")
# load the saved pipeline only when a prediction is first requested
MODEL = None


class TripRequest(BaseModel):
    trip_distance: float = Field(gt=0, le=100)
    passenger_count: int = Field(ge=1, le=6)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_weekday: int = Field(ge=0, le=6)
    PULocationID: int = Field(gt=0)
    DOLocationID: int = Field(gt=0)


@app.get("/health")
def health_check() -> dict[str, str]:
    if Path(MODEL_PATH).is_file() or MODEL is not None:
        return {"status": "ready"}
    return {"status": "model_not_trained"}


@app.post("/predict")
def predict(request: TripRequest) -> dict[str, float]:
    global MODEL
    if MODEL is None:
        if not Path(MODEL_PATH).is_file():
            raise HTTPException(status_code=503, detail="Train the model before requesting predictions")
        MODEL = joblib.load(MODEL_PATH)

    features = pd.DataFrame([request.model_dump()], columns=FEATURE_COLUMNS)
    prediction = float(MODEL.predict(features)[0])
    return {"predicted_duration_minutes": prediction}