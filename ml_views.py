"""Machine learning views generator for Black-Litterman."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES_PATH = Path('data/features.parquet')
TARGET_COL = 'fwd_21d_ret'
META_COLS = ['permno', 'date', 'ticker', 'sector']


class MLViewsError(RuntimeError):
    pass


def load_and_prep_data(filepath: str | Path = FEATURES_PATH):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'Could not find {path.resolve()}')

    df = pd.read_parquet(path).copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)

    drop_cols = META_COLS + ([TARGET_COL] if TARGET_COL in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    missing_pct = df[feature_cols].isnull().mean() * 100
    bad_cols = missing_pct[missing_pct > 30].index.tolist()
    if bad_cols:
        feature_cols = [c for c in feature_cols if c not in bad_cols]

    required = feature_cols + [TARGET_COL]
    df_clean = df.dropna(subset=required).copy()
    if df_clean.empty:
        raise MLViewsError('No rows remained after dropping missing values for features and target.')

    return df_clean, feature_cols, TARGET_COL


def time_series_split(df: pd.DataFrame, split_date: str = '2024-01-01'):
    split_ts = pd.Timestamp(split_date)
    train = df[df['date'] < split_ts].copy()
    test = df[df['date'] >= split_ts].copy()
    if train.empty or test.empty:
        raise MLViewsError('Train or test split is empty. Adjust split_date.')
    return train, test


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def generate_ml_views(df: pd.DataFrame, feature_cols: list[str], model) -> pd.DataFrame:
    latest_data = df.sort_values('date').groupby('ticker').tail(1).copy()
    X_latest = latest_data[feature_cols]
    X_latest_np = X_latest.to_numpy()

    preds = model.predict(X_latest)
    annualized_returns = preds * 12

    tree_preds = np.array([tree.predict(X_latest_np) for tree in model.estimators_])
    rf_std = np.std(tree_preds, axis=0)
    confidence = np.clip(1.0 - (rf_std * 10), 0.1, 0.9)

    return pd.DataFrame(
        {
            'ticker': latest_data['ticker'].values,
            'return': annualized_returns,
            'confidence': confidence,
        }
    ).reset_index(drop=True)


def get_ml_views(filepath: str | Path = FEATURES_PATH, split_date: str = '2024-01-01') -> pd.DataFrame:
    df, feature_cols, target_col = load_and_prep_data(filepath)
    train_df, _ = time_series_split(df, split_date=split_date)
    model = train_model(train_df[feature_cols], train_df[target_col])
    return generate_ml_views(df, feature_cols, model)
