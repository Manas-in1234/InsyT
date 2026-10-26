from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.ml_service import train_regressor

client = TestClient(app)


def mixed_data(rows: int = 80) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    return pd.DataFrame({"numeric signal": x, "segment": np.where(x % 2, "odd", "even"),
                         "target score": 4 * x + np.where(x % 2, 15, -10)})


def test_random_forest_regression_numeric_and_categorical_pipeline():
    result = train_regressor(mixed_data(), "target score", ["numeric signal", "segment"], .2, 30)
    assert result["configuration"]["model"] == "RandomForestRegressor"
    assert result["counts"]["training_rows"] == 64
    assert result["counts"]["test_rows"] == 16
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["rmse"] >= 0
    assert result["metrics"]["r2"] is not None
    assert len(result["predictions"]) == 16
    assert {"row", "actual", "predicted", "absolute_error"} <= result["predictions"][0].keys()
    assert result["feature_importance"]
    assert "segment" in [x["feature"] for x in result["feature_importance"]]
    assert result["predictions_csv"].startswith("row,actual,predicted,absolute_error")


def test_missing_feature_and_target_rows_are_handled():
    df = mixed_data(60)
    df.loc[[2, 5], "numeric signal"] = np.nan
    df.loc[[8, 12], "segment"] = None
    df.loc[[1, 4, 9], "target score"] = np.nan
    result = train_regressor(df, "target score", ["numeric signal", "segment"], .2, 20)
    assert result["counts"]["target_rows_removed"] == 3
    assert result["counts"]["usable_rows"] == 57
    assert result["counts"]["training_rows"] + result["counts"]["test_rows"] == 57


def test_invalid_target_and_feature_inputs():
    df = mixed_data()
    with pytest.raises(ValueError, match="target column"):
        train_regressor(df, "missing", ["numeric signal"])
    with pytest.raises(ValueError, match="cannot also be an input"):
        train_regressor(df, "target score", ["numeric signal", "target score"])
    with pytest.raises(ValueError, match="duplicates the target"):
        train_regressor(df.assign(copy=df["target score"]), "target score", ["copy"])
    with pytest.raises(ValueError, match="Duplicate input features"):
        train_regressor(df, "target score", ["numeric signal", "numeric signal"])
    with pytest.raises(ValueError, match="do not exist"):
        train_regressor(df, "target score", ["bad feature"])


def test_insufficient_rows_constant_target_and_invalid_parameters():
    with pytest.raises(ValueError, match="At least 10"):
        train_regressor(mixed_data(9), "target score", ["numeric signal"])
    constant = mixed_data(40)
    constant["target score"] = 3
    with pytest.raises(ValueError, match="constant"):
        train_regressor(constant, "target score", ["numeric signal"])
    with pytest.raises(ValueError, match="Test proportion"):
        train_regressor(mixed_data(), "target score", ["numeric signal"], .05)
    with pytest.raises(ValueError, match="Number of trees"):
        train_regressor(mixed_data(), "target score", ["numeric signal"], .2, 501)


def test_insufficient_features_and_test_target_variation():
    df = mixed_data(40)
    df["constant feature"] = 1
    with pytest.raises(ValueError, match="constant"):
        train_regressor(df, "target score", ["constant feature"])
    test_targets = [0] * 10 + [1] * 10
    test_targets[17] = 0  # random_state=42 selects rows 0 and 17 for this test split
    constant_test = pd.DataFrame({"x": range(20), "target": test_targets})
    result = train_regressor(constant_test, "target", ["x"], .1, 20)
    assert result["metrics"]["r2"] is None
    assert any("R² is omitted" in warning for warning in result["warnings"])


def test_ml_api_success_and_errors():
    stream = io.BytesIO()
    mixed_data(80).to_csv(stream, index=False)
    response = client.post("/api/ml/train", data={"target_column": "target score", "feature_columns": '["numeric signal", "segment"]', "test_size": "0.2", "n_estimators": "20"}, files={"file": ("model.csv", stream.getvalue(), "text/csv")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["counts"]["usable_rows"] == 80
    assert body["prediction_count"] == body["counts"]["test_rows"]
    assert "Predictive intelligence" in body["report"]
    assert "causation" in body["report"].lower()
    invalid = client.post("/api/ml/train", data={"target_column": "missing", "feature_columns": '["numeric signal"]'}, files={"file": ("model.csv", stream.getvalue(), "text/csv")})
    assert invalid.status_code == 400


def test_existing_statistical_endpoint_remains_available():
    response = client.post("/api/analyze", files={"file": ("data.csv", b"value,group\n1,a\n2,b\n3,a\n", "text/csv")})
    assert response.status_code == 200
    assert "kpis" in response.json() and "charts" in response.json()
