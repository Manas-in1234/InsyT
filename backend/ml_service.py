"""Leakage-conscious Random Forest regression for uploaded tables."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_regressor(df: pd.DataFrame, target: str, features: list[str], test_size: float = .2,
                    n_estimators: int = 100, filename: str = "dataset") -> dict[str, Any]:
    if not isinstance(target, str) or target not in df.columns:
        raise ValueError("Choose a target column that exists in the dataset.")
    if not isinstance(features, list) or not features:
        raise ValueError("Select at least one input feature.")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate input features are not allowed.")
    if target in features:
        raise ValueError("The target column cannot also be an input feature.")
    unknown = [col for col in features if not isinstance(col, str) or col not in df.columns]
    if unknown:
        raise ValueError("One or more selected input features do not exist in the dataset.")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not .1 <= float(test_size) <= .4:
        raise ValueError("Test proportion must be between 0.10 and 0.40.")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or not 10 <= n_estimators <= 500:
        raise ValueError("Number of trees must be an integer between 10 and 500.")

    target_values = pd.to_numeric(df[target], errors="coerce")
    usable_target = target_values.notna() & np.isfinite(target_values)
    target_rows_removed = int((~usable_target).sum())
    working = df.loc[usable_target, features].copy().reset_index(drop=True)
    y = target_values.loc[usable_target].astype(float).reset_index(drop=True)
    original_rows = np.flatnonzero(usable_target.to_numpy()) + 2  # account for header row
    if len(y) < 10:
        raise ValueError("At least 10 rows with a numeric target are required for training and evaluation.")
    if y.nunique() < 2:
        raise ValueError("The target is constant; regression evaluation is not meaningful.")
    test_count = int(np.ceil(len(y) * float(test_size)))
    train_count = len(y) - test_count
    if test_count < 2 or train_count < 5:
        raise ValueError("The selected split leaves too few training or test observations; provide more usable rows or choose a smaller test proportion.")

    numeric_features, categorical_features = [], []
    for col in features:
        series = working[col]
        candidate_numeric = pd.to_numeric(series, errors="coerce")
        target_numeric = pd.to_numeric(df[target].loc[usable_target], errors="coerce").reset_index(drop=True)
        if series.astype("string").fillna("<NA>").reset_index(drop=True).equals(df[target].loc[usable_target].astype("string").fillna("<NA>").reset_index(drop=True)) or (
            candidate_numeric.notna().sum() == target_numeric.notna().sum() and candidate_numeric.notna().all() and
            target_numeric.notna().all() and np.array_equal(candidate_numeric.to_numpy(), target_numeric.to_numpy())
        ):
            raise ValueError(f"Feature '{col}' duplicates the target values and would leak the answer.")
        converted = pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_numeric_dtype(series) or (series.notna().sum() > 0 and converted.notna().sum() >= series.notna().sum() * .8):
            working[col] = converted
            if converted.nunique(dropna=True) > 1:
                numeric_features.append(col)
        else:
            # String conversion gives OneHotEncoder a stable representation for mixed labels/dates.
            working[col] = series.astype(object).where(series.notna(), np.nan)
            if series.nunique(dropna=True) > 1:
                categorical_features.append(col)
    usable_features = numeric_features + categorical_features
    if not usable_features:
        raise ValueError("Selected input features are empty, missing, or constant. Choose at least one varying feature.")
    working = working[usable_features]

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=float(test_size), random_state=42)
    transformers = []
    if numeric_features:
        transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="(missing)")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=2, max_categories=500)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=1, min_samples_leaf=1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(working.iloc[train_idx], y.iloc[train_idx])
    predicted = pipeline.predict(working.iloc[test_idx])
    actual = y.iloc[test_idx].to_numpy()
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 and np.unique(actual).size >= 2 else None

    model_importance = pipeline.named_steps["model"].feature_importances_
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    grouped: dict[str, float] = {col: 0.0 for col in usable_features}
    for name, importance in zip(transformed_names, model_importance):
        clean = str(name).split("__", 1)[-1]
        owner = next((col for col in sorted(usable_features, key=len, reverse=True) if clean == col or clean.startswith(f"{col}_")), None)
        if owner is not None:
            grouped[owner] += float(importance)
    importance = [{"feature": key, "importance": round(value, 6)} for key, value in grouped.items()]
    importance.sort(key=lambda item: item["importance"], reverse=True)

    predictions = []
    all_rows = []
    for position, true_value, predicted_value in zip(test_idx, actual, predicted):
        item = {"row": int(original_rows[position]), "actual": float(true_value), "predicted": float(predicted_value),
                "absolute_error": float(abs(true_value - predicted_value))}
        all_rows.append(item)
    all_rows.sort(key=lambda item: item["row"])
    predictions = all_rows[:250]
    csv_stream = io.StringIO()
    writer = csv.DictWriter(csv_stream, fieldnames=["row", "actual", "predicted", "absolute_error"])
    writer.writeheader()
    writer.writerows(all_rows)

    excluded_features = [col for col in features if col not in usable_features]
    warnings = ["This is ordinary regression using a reproducible random split, not a future forecast.",
                "Feature importance describes model reliance, not causation.",
                "Performance is measured only on the held-out test rows and may vary with other samples."]
    if excluded_features:
        warnings.append(f"Constant or entirely missing input fields were excluded: {', '.join(excluded_features)}.")
    date_candidates = [col for col in df.columns if any(token in col.lower() for token in ("date", "time", "month", "year"))]
    if date_candidates:
        warnings.append("Date-like columns exist. A random split can mix earlier and later records; use chronological evaluation for forecasting.")
    if len(y) < 50:
        warnings.append("A small dataset can make held-out metrics unstable.")
    if len(test_idx) < 10:
        warnings.append("The test set is small; R² and other metrics may be unreliable.")
    if r2 is None:
        warnings.append("R² is omitted because test targets do not vary enough to make it meaningful.")
    if r2 is not None and r2 < 0:
        warnings.append("Negative test R² means this model performed worse than predicting the test-set mean for these rows.")
    return {"filename": filename, "target": target, "features": usable_features,
            "excluded_constant_features": excluded_features,
            "configuration": {"model": "RandomForestRegressor", "n_estimators": n_estimators, "test_size": float(test_size), "random_state": 42, "split": "random"},
            "counts": {"usable_rows": len(y), "target_rows_removed": target_rows_removed, "training_rows": len(train_idx), "test_rows": len(test_idx)},
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2}, "feature_importance": importance,
            "predictions": predictions, "prediction_count": len(all_rows), "predictions_csv": csv_stream.getvalue(),
            "warnings": warnings}


def parse_features(value: str) -> list[str]:
    try:
        features = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Features must be provided as a JSON array of column names.") from exc
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ValueError("Features must be provided as a JSON array of column names.")
    return features
