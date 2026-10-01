from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from fastapi.testclient import TestClient

import app.main as api
from src.pipeline import FEATURE_COLUMNS, TARGET_COLUMN, build_model, prepare_training_data
from src.train import train_model


def make_test_trips(row_count: int = 60) -> pd.DataFrame:
    pickup_times = pd.date_range("2025-01-01 08:00:00", periods=row_count, freq="h")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup_times,
            "tpep_dropoff_datetime": pickup_times + pd.to_timedelta(10, unit="m"),
            "trip_distance": np.linspace(1, 12, row_count),
            "passenger_count": np.ones(row_count, dtype=int),
            "PULocationID": np.where(np.arange(row_count) % 2, 161, 162),
            "DOLocationID": np.where(np.arange(row_count) % 3, 236, 237),
        }
    )


def test_cleaning_rejects_invalid_trips() -> None:
    # combine valid and invalid examples to check the cleaning boundary
    raw = make_test_trips(4)
    raw.loc[1, "tpep_dropoff_datetime"] = raw.loc[1, "tpep_pickup_datetime"] - timedelta(minutes=2)
    raw.loc[2, "trip_distance"] = 0
    raw.loc[3, "tpep_pickup_datetime"] = pd.NaT

    cleaned = prepare_training_data(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, TARGET_COLUMN] == 10
    assert cleaned.loc[0, "pickup_hour"] == 8


def test_cleaning_reports_missing_columns() -> None:
    try:
        prepare_training_data(pd.DataFrame({"trip_distance": [1.0]}))
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("expected missing input columns to be rejected")


def test_model_fits_and_returns_finite_predictions() -> None:
    cleaned = prepare_training_data(make_test_trips())
    model = build_model().fit(cleaned[FEATURE_COLUMNS], cleaned[TARGET_COLUMN])

    predictions = model.predict(cleaned[FEATURE_COLUMNS].head(5))

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_training_logs_metrics_and_saves_model(tmp_path) -> None:
    model_path = tmp_path / "artifacts" / "taxi_model.joblib"
    tracking_path = f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}"

    result = train_model(
        make_test_trips(),
        sample_size=60,
        random_state=17,
        model_path=model_path,
        tracking_uri=tracking_path,
    )

    saved_model = joblib.load(model_path)
    cleaned = prepare_training_data(make_test_trips())
    predictions = saved_model.predict(cleaned[FEATURE_COLUMNS].head(3))
    assert result["training_rows"] == 48
    assert result["validation_rows"] == 12
    assert np.isfinite(result["rmse_minutes"])
    assert len(result["run_id"]) > 0
    assert np.isfinite(predictions).all()


def test_api_rejects_bad_input_and_predicts(monkeypatch) -> None:
    client = TestClient(api.app)
    monkeypatch.setattr(api, "MODEL_PATH", api.MODEL_PATH.with_name("missing-test-model.joblib"))
    monkeypatch.setattr(api, "MODEL", None)

    payload = {
        "trip_distance": 2.5,
        "passenger_count": 1,
        "pickup_hour": 9,
        "pickup_weekday": 2,
        "PULocationID": 161,
        "DOLocationID": 236,
    }
    assert client.post("/predict", json=payload).status_code == 503

    cleaned = prepare_training_data(make_test_trips())
    monkeypatch.setattr(
        api,
        "MODEL",
        build_model().fit(cleaned[FEATURE_COLUMNS], cleaned[TARGET_COLUMN]),
    )
    prediction_response = client.post("/predict", json=payload)
    invalid_response = client.post("/predict", json={**payload, "pickup_hour": 25})

    assert prediction_response.status_code == 200
    assert prediction_response.json()["predicted_duration_minutes"] > 0
    assert invalid_response.status_code == 422