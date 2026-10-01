from pathlib import Path
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.pipeline import (
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_PATH,
    TARGET_COLUMN,
    build_model,
    prepare_training_data,
)


def train_model(
    raw_data: pd.DataFrame,
    sample_size: int = 100_000,
    random_state: int = 42,
    model_path: Path = MODEL_PATH,
    tracking_uri: str | None = None,
) -> dict[str, float | int | str]:
    if sample_size < 20:
        raise ValueError("sample_size must be at least 20")

    # cap the first run so training stays practical on a personal computer
    prepared = prepare_training_data(raw_data)
    if len(prepared) > sample_size:
        prepared = prepared.sample(n=sample_size, random_state=random_state)

    features = prepared[FEATURE_COLUMNS]
    target = prepared[TARGET_COLUMN]
    train_features, validation_features, train_target, validation_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    model = build_model(random_state=random_state)
    model.fit(train_features, train_target)
    predictions = model.predict(validation_features)
    rmse = mean_squared_error(validation_target, predictions) ** 0.5

    model_path.parent.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # keep experiment tracking in a local sqlite database unless a test provides a temporary store
    tracking_db = Path(__file__).resolve().parents[1] / "mlruns" / "mlflow.db"
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    local_tracking_uri = tracking_uri or f"sqlite:///{tracking_db.as_posix()}"
    mlflow.set_tracking_uri(local_tracking_uri)
    mlflow.set_experiment("nyc-yellow-taxi-duration")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 18,
                "min_samples_leaf": 2,
                "training_rows": len(train_features),
                "validation_rows": len(validation_features),
                "random_state": random_state,
            }
        )
        mlflow.log_metric("rmse_minutes", rmse)
        mlflow.sklearn.log_model(
            model,
            name="taxi_duration_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    import joblib

    joblib.dump(model, model_path)
    return {
        "rmse_minutes": float(rmse),
        "training_rows": len(train_features),
        "validation_rows": len(validation_features),
        "run_id": run.info.run_id,
        "model_path": str(model_path),
    }