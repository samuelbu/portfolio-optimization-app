"""
ml_views.py
===========
Machine Learning module for forecasting asset returns and volatility.

Connects with:
  - data_pipeline.py    → feature engineered parquet file
  - black_litterman.py  → outputs predicted returns + confidences
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# =============================================================================
# 1. LOAD AND PREP DATA
# =============================================================================

def load_and_prep_data(filepath="data/features.parquet"):
    """
    Loads the engineered features, handles missing values smartly, 
    and prepares the data for time-series modeling.
    """
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Make sure Member A has generated it!")

    df = pd.read_parquet(filepath)

    target_col = "fwd_21d_ret"
    meta_cols = ["permno", "date", "ticker", "sector"]
    
    drop_cols = meta_cols + ([target_col] if target_col in df.columns else [])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # --- NEW: Smart Missing Value Handling ---
    print("\n--- Missing Data Check ---")
    missing_pct = df[feature_cols].isnull().mean() * 100
    
    # Identify completely broken/mostly empty columns
    bad_cols = missing_pct[missing_pct > 30].index.tolist()
    if bad_cols:
        print(f"Dropping features with >30% missing data: {bad_cols}")
        feature_cols = [c for c in feature_cols if c not in bad_cols]
        
    # Print what minor missing data is left
    remaining_missing = df[feature_cols].isnull().mean() * 100
    missing_to_show = remaining_missing[remaining_missing > 0]
    if not missing_to_show.empty:
        print("Minor missing data in remaining features (%):")
        print(missing_to_show.round(1))
    print("--------------------------\n")

    # Now we safely drop rows with missing values
    if target_col in df.columns:
        df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    else:
        df_clean = df.dropna(subset=feature_cols).copy()

    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    print(f"Successfully loaded {len(df_clean):,} clean observations.")
    return df_clean, feature_cols, target_col

# =============================================================================
# 2. TIME-SERIES SPLIT
# =============================================================================

def time_series_split(df, split_date="2024-01-01"):
    """
    Splits the data chronologically.
    Train: Before split_date
    Test: On or after split_date
    """
    print(f"Splitting data at {split_date}...")
    
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    
    print(f"Train size: {len(train):,} | Test size: {len(test):,}")
    return train, test

# --- Quick Test ---
if __name__ == "__main__":
    df, features, target = load_and_prep_data()
    train_df, test_df = time_series_split(df)

    # =============================================================================
# 3. TRAIN MODELS
# =============================================================================

def train_models(X_train, y_train):
    """
    Trains Random Forest and XGBoost models on the historical features.
    """
    print("\n--- Training Models ---")
    
    print("Training Random Forest (this might take a few seconds)...")
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=5, 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    print("Models trained successfully!")
    return rf_model, xgb_model

# =============================================================================
# 4. GENERATE BL VIEWS
# =============================================================================

def generate_ml_views(df, feature_cols, rf_model, xgb_model):
    """
    Takes the most recent data for each stock, predicts forward returns,
    annualizes them, and calculates a confidence score for Black-Litterman.
    """
    print("\n--- Generating Views for Black-Litterman ---")
    
    # Isolate the single most recent day of data for each stock
    latest_data = df.sort_values('date').groupby('ticker').tail(1).copy()
    
    # We must pass the raw numpy array to avoid feature name warnings
    X_latest = latest_data[feature_cols] 
    
    # 1. Predict 21-day returns
    rf_preds = rf_model.predict(X_latest)
    xgb_preds = xgb_model.predict(X_latest)
    
    # 2. Ensemble (Average the two models for robustness)
    ensemble_preds = (rf_preds + xgb_preds) / 2.0
    
    # 3. Annualize the returns (21 days * 12 = ~252 trading days)
    annualized_returns = ensemble_preds * 12
    
    # 4. Calculate Confidence Score (using RF tree variance)
    # Get predictions from every individual tree in the forest
    tree_preds = np.array([tree.predict(X_latest) for tree in rf_model.estimators_])
    rf_std = np.std(tree_preds, axis=0)
    
    # Lower standard deviation = higher agreement = higher confidence.
    # We scale it to be roughly between 0.1 (low confidence) and 0.9 (high confidence).
    confidence = np.clip(1.0 - (rf_std * 10), 0.1, 0.9)
    
    # 5. Format exactly as Member C's black_litterman.py expects
    ml_views = pd.DataFrame({
        "ticker": latest_data["ticker"].values,
        "return": annualized_returns,
        "confidence": confidence
    })
    
    return ml_views.reset_index(drop=True)

# =============================================================================
# EXECUTE PIPELINE
# =============================================================================

if __name__ == "__main__":
    # 1. Load Data
    df, features, target = load_and_prep_data()
    
    # 2. Split Data
    train_df, test_df = time_series_split(df)
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    # 3. Train Models
    rf_model, xgb_model = train_models(X_train, y_train)
    
    # 4. Generate the final output
    final_views = generate_ml_views(df, features, rf_model, xgb_model)
    
    print("\n=== FINAL ML VIEWS FOR OPTIMIZATION ===")
    print(final_views.head(10))