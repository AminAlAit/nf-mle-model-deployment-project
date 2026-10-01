from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "taxi_duration_pipeline.joblib"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"

NUMERIC_FEATURES = ["trip_distance", "passenger_count", "pickup_hour", "pickup_weekday"]
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "duration_minutes"


def prepare_training_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
    }
    missing_columns = required_columns.difference(raw_data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    data = raw_data[list(required_columns)].copy()
    # derive the target and calendar features from trip timestamps
    data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"], errors="coerce")
    data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"], errors="coerce")
    data["duration_minutes"] = (
        data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    data["pickup_hour"] = data["tpep_pickup_datetime"].dt.hour
    data["pickup_weekday"] = data["tpep_pickup_datetime"].dt.dayofweek

    data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    # discard implausible values so data errors do not dominate training
    valid_rows = (
        data[TARGET_COLUMN].between(1, 180)
        & data["trip_distance"].between(0.1, 100)
        & data["passenger_count"].between(1, 6)
    )
    prepared = data.loc[valid_rows, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    if prepared.empty:
        raise ValueError("No valid taxi trips remain after cleaning")

    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].astype("int64")

    return prepared.reset_index(drop=True)


def build_model(random_state: int = 42) -> Pipeline:
    # treat zone identifiers as categories and preserve numeric measurements
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("locations", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", forest)])