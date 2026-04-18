"""
Machine Learning module for forecasting asset returns and confidence scores.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def load_and_prep_data(filepath: str = "data/features.parquet"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")

    df = pd.read_parquet(filepath)
    df["date"] = pd.to_datetime(df["date"])

    target_col = "fwd_21d_ret"
    meta_cols = ["permno", "date", "ticker", "sector"]
    drop_cols = meta_cols + ([target_col] if target_col in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    missing_pct = df[feature_cols].isnull().mean() * 100
    bad_cols = missing_pct[missing_pct > 30].index.tolist()
    if bad_cols:
        feature_cols = [c for c in feature_cols if c not in bad_cols]

    if target_col in df.columns:
        df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    else:
        df_clean = df.dropna(subset=feature_cols).copy()

    for col in feature_cols + [target_col]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna(subset=feature_cols + [target_col])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    return df_clean, feature_cols, target_col


def time_series_split(df: pd.DataFrame, split_date: str = "2024-01-01"):
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test


def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    X_train = X_train.astype(float)
    y_train = pd.to_numeric(y_train, errors="coerce").astype(float)

    rf_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=5,
        random_state=42,
        n_jobs=1,
    )
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model


def generate_ml_views(df: pd.DataFrame, feature_cols: list[str], rf_model, xgb_model):
    latest_data = df.sort_values("date").groupby("ticker").tail(1).copy()
    X_latest = latest_data[feature_cols].astype(float)
    X_latest_np = X_latest.to_numpy()

    rf_preds = rf_model.predict(X_latest)
    xgb_preds = xgb_model.predict(X_latest)
    ensemble_preds = (rf_preds + xgb_preds) / 2.0
    annualized_returns = ensemble_preds * 12

    tree_preds = np.array([tree.predict(X_latest_np) for tree in rf_model.estimators_])
    rf_std = np.std(tree_preds, axis=0)
    confidence = np.clip(1.0 - (rf_std * 10), 0.1, 0.9)

    return pd.DataFrame(
        {
            "ticker": latest_data["ticker"].values,
            "return": annualized_returns.astype(float),
            "confidence": confidence.astype(float),
        }
    ).reset_index(drop=True)
