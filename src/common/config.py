import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    PREFECT_API_URL: str = os.getenv("PREFECT_API_URL", "http://localhost:4200")
    TRAIN_DATASET_URI: str = os.getenv("TRAIN_DATASET_URI", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet")
    TEST_DATASET_URI: str = os.getenv("TEST_DATASET_URI", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet")


def get_settings() -> Settings:
    return Settings()