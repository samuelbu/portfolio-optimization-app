"""
black_litterman.py
==================
Portfolio optimization module using the Black-Litterman model.

Designed for a Streamlit financial analytics app.

This script connects with:
  - market_data.py  → prices DataFrame
  - ml_views.py     → predicted returns + confidences

Authors: Akbar Wibowo, Danish Azmi and Samuel Buelvas 
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
import ml_views

# 1. RISK PROFILE
# ──────────────────────────────────────────────

# Maps a questionnaire score (1–10) to a risk aversion coefficient δ.
# Low score = conservative investor → high δ (penalizes variance more)
# High score = aggressive investor  → low δ (chases returns)
RISK_AVERSION_MAP = {
    1:  4.0,
    2:  3.5,
    3:  3.0,
    4:  2.5,
    5:  2.0,
    6:  1.75,
    7:  1.5,
    8:  1.25,
    9:  1.0,
    10: 0.75,
}

# Maximum portfolio volatility (annualized) allowed per risk score
MAX_VOLATILITY_MAP = {
    1:  0.08,   # 8%  — very conservative
    2:  0.10,
    3:  0.12,
    4:  0.14,
    5:  0.16,   # 16% — moderate
    6:  0.18,
    7:  0.20,
    8:  0.22,
    9:  0.25,
    10: 0.30,   # 30% — aggressive
}

RISK_LABELS = {
    1:  "Ultra Conservative",
    2:  "Very Conservative",
    3:  "Conservative",
    4:  "Moderately Conservative",
    5:  "Moderate",
    6:  "Moderately Aggressive",
    7:  "Aggressive",
    8:  "Very Aggressive",
    9:  "Speculative",
    10: "Ultra Speculative",
}


# placeholder function to define the questions to determine risk profile

def score_risk_questionnaire(answers: dict) -> int:
    """
    Score a risk questionnaire and return a risk score from 1 to 10.

    Parameters
    ----------
    answers : dict
        Keys are question IDs (str), values are selected option scores (int 1–5).
        Expected questions:
            q1: Investment horizon
            q2: Reaction to a 20% portfolio drop
            q3: Income stability
            q4: Investment experience
            q5: Primary investment goal

    Returns
    -------
    int
        Risk score from 1 (most conservative) to 10 (most aggressive).

    Example
    -------
    >>> answers = {"q1": 4, "q2": 3, "q3": 5, "q4": 2, "q5": 4}
    >>> score = score_risk_questionnaire(answers)
    """
    expected_questions = ["q1", "q2", "q3", "q4", "q5"]
    for q in expected_questions:
        if q not in answers:
            raise ValueError(f"Missing answer for question: {q}")
        if answers[q] not in range(1, 6):
            raise ValueError(f"Answer to {q} must be between 1 and 5.")

    total = sum(answers[q] for q in expected_questions)   # range: 5–25

    # Scale from [5, 25] to [1, 10]
    score = round(1 + (total - 5) * (9 / 20))
    return int(np.clip(score, 1, 10))


# 2. MARKET DATA PREPROCESSING
# ──────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute periodic returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns = asset tickers, rows = dates (ascending), values = adjusted close prices.
    method : str
        'log'    → log returns  (default, preferred for BL)
        'simple' → simple returns

    Returns
    -------
    pd.DataFrame
        Returns DataFrame (same shape minus first row).
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        returns = prices.pct_change().dropna()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return returns


def compute_covariance_matrix(returns: pd.DataFrame, annualize: bool = True,
                               trading_days: int = 252) -> pd.DataFrame:
    """
    Compute the annualized covariance matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns from compute_returns().
    annualize : bool
        If True, multiply by trading_days (default 252).
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    pd.DataFrame
        Covariance matrix (n_assets × n_assets).
    """
    cov = returns.cov()
    if annualize:
        cov = cov * trading_days
    return cov


def compute_market_weights(prices: pd.DataFrame,
                            market_caps: Optional[pd.Series] = None) -> pd.Series:
    """
    Compute market-cap weights for the asset universe.
    If market caps are not available, falls back to equal weights.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices DataFrame (used only to get ticker names if market_caps is None).
    market_caps : pd.Series, optional
        Market capitalizations indexed by ticker. If None, equal weights are used.

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.
    """
    tickers = prices.columns.tolist()

    if market_caps is None:
        print("⚠️  No market caps provided — using equal weights as proxy.")
        weights = pd.Series(1.0 / len(tickers), index=tickers)
    else:
        missing = set(tickers) - set(market_caps.index)
        if missing:
            raise ValueError(f"Market caps missing for: {missing}")
        weights = market_caps[tickers] / market_caps[tickers].sum()

    return weights


# 3. BLACK-LITTERMAN MODEL
# ──────────────────────────────────────────────

def compute_implied_returns(cov_matrix: pd.DataFrame, market_weights: pd.Series,
                             risk_aversion: float) -> pd.Series:
    """
    Compute the CAPM-implied equilibrium excess returns (Π).

    Formula: Π = δ × Σ × w_mkt

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    market_weights : pd.Series
        Market-cap weights for each asset.
    risk_aversion : float
        Risk aversion coefficient δ (from RISK_AVERSION_MAP).

    Returns
    -------
    pd.Series
        Implied excess returns Π indexed by ticker.
    """
    tickers = cov_matrix.columns.tolist()
    w = market_weights[tickers].values
    sigma = cov_matrix.values
    pi = risk_aversion * sigma @ w
    return pd.Series(pi, index=tickers)


def build_views(ml_predictions: pd.DataFrame,
                assets: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ML model predictions into Black-Litterman view matrices.

    Each prediction is an ABSOLUTE view: "Asset X will return Y% annually."

    Parameters
    ----------
    ml_predictions : pd.DataFrame
        Must contain columns:
          'ticker'     : asset ticker (must be in `assets`)
          'return'     : predicted annualized return (e.g. 0.12 for 12%)
          'confidence' : model confidence, float in (0, 1]
                         (1.0 = very confident, 0.1 = uncertain)
    assets : list
        Ordered list of tickers in the optimization universe.

    Returns
    -------
    P : np.ndarray  (k × n)  — pick matrix (which assets each view refers to)
    Q : np.ndarray  (k,)     — view return values
    Omega : np.ndarray (k × k) — diagonal view uncertainty matrix

    Notes
    -----
    Omega is computed as: ω_i = (1 - confidence_i) / confidence_i × variance_of_view
    A simpler proxy used here: ω_i = (1 / confidence_i - 1) × 0.01
    """
    valid_predictions = ml_predictions[ml_predictions["ticker"].isin(assets)].copy()

    if valid_predictions.empty:
        raise ValueError("None of the ML predictions match assets in the universe.")

    k = len(valid_predictions)
    n = len(assets)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, (_, row) in enumerate(valid_predictions.iterrows()):
        j = assets.index(row["ticker"])
        P[i, j] = 1.0
        Q[i] = row["return"]
        # Uncertainty inversely proportional to confidence
        omega_diag[i] = (1.0 / row["confidence"] - 1) * 0.01

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def black_litterman(cov_matrix: pd.DataFrame,
                     implied_returns: pd.Series,
                     P: np.ndarray,
                     Q: np.ndarray,
                     Omega: np.ndarray,
                     tau: float = 0.05) -> pd.Series:
    """
    Apply the Black-Litterman formula to compute posterior expected returns.

    Formula:
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualized covariance matrix (n × n).
    implied_returns : pd.Series
        CAPM implied excess returns Π (n,).
    P : np.ndarray
        Pick matrix (k × n).
    Q : np.ndarray
        View returns (k,).
    Omega : np.ndarray
        View uncertainty matrix (k × k).
    tau : float
        Scalar expressing uncertainty in the prior (typically 0.01–0.10).

    Returns
    -------
    pd.Series
        Posterior (Black-Litterman) expected returns, indexed by ticker.
    """
    tickers = cov_matrix.columns.tolist()
    sigma = cov_matrix.values
    pi = implied_returns[tickers].values

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)

    # Posterior precision (inverse covariance of the estimate)
    M = tau_sigma_inv + P.T @ omega_inv @ P

    # Posterior mean
    mu_bl = np.linalg.inv(M) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    return pd.Series(mu_bl, index=tickers)


# 4. PORTFOLIO OPTIMIZATION
# ──────────────────────────────────────────────

def optimize_portfolio(expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        risk_aversion: float,
                        max_volatility: float,
                        min_weight: float = 0.0,
                        max_weight: float = 0.40) -> pd.Series:
    """
    Maximize the risk-adjusted return (Sharpe-like utility) subject to constraints.

    Objective: maximize  μ'w - (δ/2) × w'Σw

    Constraints:
        - Weights sum to 1 (fully invested)
        - Portfolio volatility ≤ max_volatility (risk profile cap)
        - min_weight ≤ wᵢ ≤ max_weight (no extreme concentration)
        - No short selling (wᵢ ≥ 0) by default

    Parameters
    ----------
    expected_returns : pd.Series
        Black-Litterman posterior expected returns.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    risk_aversion : float
        δ from RISK_AVERSION_MAP.
    max_volatility : float
        Max allowed annualized portfolio volatility from MAX_VOLATILITY_MAP.
    min_weight : float
        Minimum weight per asset (default 0 = no shorting).
    max_weight : float
        Maximum weight per asset (default 40% = concentration limit).

    Returns
    -------
    pd.Series
        Optimized portfolio weights indexed by ticker.
    """
    tickers = expected_returns.index.tolist()
    n = len(tickers)
    mu = expected_returns.values
    sigma = cov_matrix.loc[tickers, tickers].values

    def neg_utility(w):
        port_return = mu @ w
        port_variance = w @ sigma @ w
        return -(port_return - (risk_aversion / 2) * port_variance)

    def portfolio_volatility(w):
        return np.sqrt(w @ sigma @ w)

    constraints = [
        {"type": "eq",  "fun": lambda w: np.sum(w) - 1},          # fully invested
        {"type": "ineq","fun": lambda w: max_volatility - portfolio_volatility(w)},  # vol cap
    ]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n  # start from equal weights

    result = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        print(f"⚠️  Optimizer warning: {result.message}. Returning equal weights.")
        return pd.Series(np.ones(n) / n, index=tickers)

    weights = pd.Series(result.x, index=tickers)
    weights = weights.clip(lower=0).round(6)
    weights /= weights.sum()  # re-normalize after clipping
    return weights


# 5. PORTFOLIO METRICS
# ──────────────────────────────────────────────

def compute_portfolio_metrics(weights: pd.Series,
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               risk_free_rate: float = 0.045) -> dict:
    """
    Compute key portfolio performance metrics for display in Streamlit.

    Parameters
    ----------
    weights : pd.Series
        Optimized portfolio weights.
    expected_returns : pd.Series
        Black-Litterman posterior expected returns.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.
    risk_free_rate : float
        Current risk-free rate (default ~4.5%, approximate US T-bill rate 2025).

    Returns
    -------
    dict with keys:
        expected_return  : float — annualized portfolio return
        volatility       : float — annualized portfolio std deviation
        sharpe_ratio     : float — (return - rf) / volatility
        weights_df       : pd.DataFrame — formatted weights table
    """
    tickers = weights.index.tolist()
    w = weights.values
    mu = expected_returns[tickers].values
    sigma = cov_matrix.loc[tickers, tickers].values

    port_return = float(mu @ w)
    port_vol = float(np.sqrt(w @ sigma @ w))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight (%)": (weights.values * 100).round(2),
    }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

    return {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "weights_df": weights_df,
    }


def dollar_allocation(weights: pd.Series, capital: float,
                       prices: pd.Series) -> pd.DataFrame:
    """
    Convert portfolio weights into number of shares given available capital.

    Parameters
    ----------
    weights : pd.Series
        Optimized portfolio weights.
    capital : float
        Total capital to invest (in USD).
    prices : pd.Series
        Latest prices indexed by ticker.

    Returns
    -------
    pd.DataFrame with columns: Ticker, Weight(%), Allocation($), Shares, Price
    """
    tickers = weights.index.tolist()
    allocations = weights * capital
    shares = (allocations / prices[tickers]).apply(np.floor)  # whole shares only
    actual_alloc = shares * prices[tickers]
    cash_remaining = capital - actual_alloc.sum()

    df = pd.DataFrame({
        "Ticker": tickers,
        "Weight (%)": (weights.values * 100).round(2),
        "Allocation ($)": actual_alloc.round(2).values,
        "Shares": shares.values.astype(int),
        "Price ($)": prices[tickers].round(2).values,
    }).sort_values("Allocation ($)", ascending=False).reset_index(drop=True)

    return df, round(cash_remaining, 2)


# Idea of data pipeline to retrieve results
# 6. MAIN PIPELINE
# ──────────────────────────────────────────────

def run_optimization_pipeline(
    prices: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    risk_score: int,
    capital: float,
    market_caps: Optional[pd.Series] = None,
    tau: float = 0.05,
    risk_free_rate: float = 0.045,
) -> dict:
    """
    Full Black-Litterman optimization pipeline — single entry point for Streamlit.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices (rows=dates, columns=tickers). From your teammate.
    ml_predictions : pd.DataFrame
        Columns: ['ticker', 'return', 'confidence']. From your ML teammate.
    risk_score : int
        Risk score 1–10 from score_risk_questionnaire().
    capital : float
        Total capital to invest (USD).
    market_caps : pd.Series, optional
        Market caps by ticker. If None, equal weights are used as prior.
    tau : float
        Black-Litterman uncertainty scalar (default 0.05).
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation.

    Returns
    -------
    dict with keys:
        weights          : pd.Series — optimized weights
        metrics          : dict      — return, vol, sharpe
        allocation_df    : pd.DataFrame — shares to buy
        cash_remaining   : float     — unallocated capital (due to integer shares)
        risk_label       : str       — e.g. "Moderate"
        bl_returns       : pd.Series — Black-Litterman posterior returns
    """
    # --- Validate inputs ---
    if risk_score not in range(1, 11):
        raise ValueError("risk_score must be an integer from 1 to 10.")
    if capital <= 0:
        raise ValueError("capital must be positive.")

    assets = prices.columns.tolist()
    risk_aversion = RISK_AVERSION_MAP[risk_score]
    max_vol = MAX_VOLATILITY_MAP[risk_score]

    # --- Step 1: Market data ---
    returns = compute_returns(prices, method="log")
    cov_matrix = compute_covariance_matrix(returns, annualize=True)
    market_weights = compute_market_weights(prices, market_caps)

    # --- Step 2: Equilibrium returns ---
    implied_returns = compute_implied_returns(cov_matrix, market_weights, risk_aversion)

    # --- Step 3: Build views from ML predictions ---
    P, Q, Omega = build_views(ml_predictions, assets)

    # --- Step 4: Black-Litterman posterior ---
    bl_returns = black_litterman(cov_matrix, implied_returns, P, Q, Omega, tau)

    # --- Step 5: Optimize ---
    weights = optimize_portfolio(bl_returns, cov_matrix, risk_aversion, max_vol)

    # --- Step 6: Metrics ---
    metrics = compute_portfolio_metrics(weights, bl_returns, cov_matrix, risk_free_rate)

    # --- Step 7: Dollar allocation ---
    latest_prices = prices.iloc[-1]
    allocation_df, cash_remaining = dollar_allocation(weights, capital, latest_prices)

    return {
        "weights": weights,
        "metrics": metrics,
        "allocation_df": allocation_df,
        "cash_remaining": cash_remaining,
        "risk_label": RISK_LABELS[risk_score],
        "bl_returns": bl_returns,
        "implied_returns": implied_returns,
        "cov_matrix": cov_matrix,
    }


# ──────────────────────────────────────────────
# 7. QUICK TEST (run with: python black_litterman.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Full Pipeline Integration Test ===\n")

    # 1. Get Simulated Prices
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM"]
    import yfinance as yf
    
    # Download prices (adding progress=False cleans up the terminal output)
    prices = yf.download(TICKERS, start="2022-01-01", end="2024-12-31", auto_adjust=True, progress=False)["Close"]
    
    # --- NEW: Smart Data Cleaning ---
    # First, drop any columns (tickers) that completely failed to download
    prices = prices.dropna(axis=1, how="all")
    # Next, drop any rows (days) with missing data for the surviving tickers
    prices = prices.dropna()
    
    # Track which tickers actually survived the download
    successful_tickers = prices.columns.tolist()
    print(f"--> Successfully downloaded data for: {successful_tickers}")

    # 2. GENERATE REAL ML PREDICTIONS
    print("--> Running ML Pipeline...")
    df, features, target = ml_views.load_and_prep_data()
    train_df, test_df = ml_views.time_series_split(df)
    
    rf_model, xgb_model = ml_views.train_models(train_df[features], train_df[target])
    real_ml_predictions = ml_views.generate_ml_views(df, features, rf_model, xgb_model)
    
    # Only keep predictions for tickers that successfully downloaded
    real_ml_predictions = real_ml_predictions[real_ml_predictions['ticker'].isin(successful_tickers)]

    # 3. Simulated User Risk Questionnaire
    answers = {"q1": 4, "q2": 3, "q3": 4, "q4": 3, "q5": 4}
    risk_score = score_risk_questionnaire(answers)
    print(f"\n--> User Risk Score: {risk_score}/10 — {RISK_LABELS[risk_score]}")

    # 4. RUN THE OPTIMIZER 
    print("\n--> Running Black-Litterman Optimization...")
    result = run_optimization_pipeline(
        prices=prices,
        ml_predictions=real_ml_predictions, 
        risk_score=risk_score,
        capital=10_000,
    )

    print("\n=== FINAL OPTIMAL WEIGHTS ===")
    print(result)

    # Display results
    print(f"\n📊 Portfolio Metrics:")
    print(f"  Expected Return : {result['metrics']['expected_return']:.2%}")
    print(f"  Volatility      : {result['metrics']['volatility']:.2%}")
    print(f"  Sharpe Ratio    : {result['metrics']['sharpe_ratio']:.2f}")

    print(f"\n📋 Allocation (${10_000:,.0f}):")
    print(result["allocation_df"].to_string(index=False))
    print(f"\n💵 Cash Remaining: ${result['cash_remaining']:,.2f}")
