"""Portfolio optimization module using the Black-Litterman model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Optional

RISK_AVERSION_MAP = {
    1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5, 5: 2.0,
    6: 1.75, 7: 1.5, 8: 1.25, 9: 1.0, 10: 0.75,
}

MARKET_RISK_AVERSION = 2.5

MAX_VOLATILITY_MAP = {
    1: 0.08, 2: 0.10, 3: 0.12, 4: 0.14, 5: 0.16,
    6: 0.18, 7: 0.20, 8: 0.22, 9: 0.25, 10: 0.30,
}

RISK_LABELS = {
    1: "Ultra Conservative", 2: "Very Conservative", 3: "Conservative",
    4: "Moderately Conservative", 5: "Moderate", 6: "Moderately Aggressive",
    7: "Aggressive", 8: "Very Aggressive", 9: "Speculative", 10: "Ultra Speculative",
}


def score_risk_questionnaire(answers: dict[str, int]) -> int:
    expected_questions = ["q1", "q2", "q3", "q4", "q5"]
    for question in expected_questions:
        if question not in answers:
            raise ValueError(f"Missing answer for question: {question}")
        if answers[question] not in range(1, 6):
            raise ValueError(f"Answer to {question} must be between 1 and 5.")

    total_score = sum(answers[question] for question in expected_questions)
    risk_score = round(1 + (total_score - 5) * (9 / 20))
    return max(1, min(10, int(risk_score)))


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    prices = prices.astype(float)
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        returns = prices.pct_change().dropna()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return returns.astype(float)


def compute_covariance_matrix(returns: pd.DataFrame, annualize: bool = True, trading_days: int = 252) -> pd.DataFrame:
    clean_returns = returns.astype(float).dropna(how="all")
    if clean_returns.empty:
        raise ValueError("returns is empty after dropping missing values.")

    lw = LedoitWolf().fit(clean_returns.fillna(0.0).values)
    cov = pd.DataFrame(lw.covariance_, index=clean_returns.columns, columns=clean_returns.columns)
    if annualize:
        cov = cov * trading_days
    return cov.astype(float)


def compute_market_weights(prices: pd.DataFrame, market_caps: Optional[pd.Series] = None) -> pd.Series:
    tickers = prices.columns.tolist()
    if market_caps is None:
        weights = pd.Series(1.0 / len(tickers), index=tickers)
    else:
        missing = set(tickers) - set(market_caps.index)
        if missing:
            raise ValueError(f"Market caps missing for: {missing}")
        weights = market_caps[tickers].astype(float) / market_caps[tickers].astype(float).sum()
    return weights.astype(float)


def compute_implied_returns(cov_matrix: pd.DataFrame, market_weights: pd.Series, risk_aversion: float) -> pd.Series:
    tickers = cov_matrix.columns.tolist()
    w = market_weights[tickers].astype(float).values
    sigma = cov_matrix.astype(float).values
    pi = risk_aversion * sigma @ w
    return pd.Series(pi, index=tickers, dtype=float)


def build_views(ml_predictions: pd.DataFrame, assets: list[str]):
    valid_predictions = ml_predictions[ml_predictions["ticker"].isin(assets)].copy()
    if valid_predictions.empty:
        raise ValueError("None of the ML predictions match assets in the universe.")

    k, n = len(valid_predictions), len(assets)
    P = np.zeros((k, n), dtype=float)
    Q = np.zeros(k, dtype=float)
    omega_diag = np.zeros(k, dtype=float)

    for i, (_, row) in enumerate(valid_predictions.iterrows()):
        j = assets.index(row["ticker"])
        P[i, j] = 1.0
        Q[i] = float(row["return"])
        confidence = float(row["confidence"])
        omega_diag[i] = max((1.0 / confidence - 1.0) * 0.01, 1e-6)

    return P, Q, np.diag(omega_diag)


def prepare_black_litterman_views(
    ml_predictions: pd.DataFrame,
    implied_returns: pd.Series,
    assets: list[str],
    max_views: int = 8,
    view_blend: float = 0.35,
    confidence_floor: float = 0.15,
    confidence_cap: float = 0.75,
) -> pd.DataFrame:
    valid = ml_predictions[ml_predictions["ticker"].isin(assets)].copy()
    if valid.empty:
        raise ValueError("None of the ML predictions match assets in the universe.")

    valid["prior_return"] = valid["ticker"].map(implied_returns).astype(float)
    valid["return"] = valid["return"].astype(float)
    valid["confidence"] = valid["confidence"].astype(float)
    valid["view_gap"] = valid["return"] - valid["prior_return"]

    half = max(1, max_views // 2)
    positive = valid.nlargest(half, "view_gap")
    negative = valid.nsmallest(half, "view_gap")
    selected = pd.concat([positive, negative], ignore_index=True).drop_duplicates("ticker")

    if selected.empty:
        selected = valid.nlargest(min(max_views, len(valid)), "return")

    selected["return"] = selected["prior_return"] + view_blend * selected["view_gap"]
    selected["confidence"] = selected["confidence"].clip(lower=confidence_floor, upper=confidence_cap)
    return selected[["ticker", "return", "confidence"]].reset_index(drop=True)


def black_litterman(cov_matrix: pd.DataFrame, implied_returns: pd.Series, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray, tau: float = 0.05) -> pd.Series:
    tickers = cov_matrix.columns.tolist()
    sigma = cov_matrix.astype(float).values
    pi = implied_returns[tickers].astype(float).values

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(Omega)
    M = tau_sigma_inv + P.T @ omega_inv @ P
    mu_bl = np.linalg.pinv(M) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    return pd.Series(mu_bl, index=tickers, dtype=float)


def minimum_variance_portfolio(
    cov_matrix: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 0.20,
) -> pd.Series:
    tickers = cov_matrix.columns.tolist()
    n = len(tickers)
    sigma = cov_matrix.loc[tickers, tickers].astype(float).values

    def portfolio_variance(w):
        w = np.asarray(w, dtype=float)
        return float(w @ sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.full(n, 1.0 / n, dtype=float)

    result = minimize(
        portfolio_variance,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        return pd.Series(np.ones(n) / n, index=tickers, dtype=float)

    weights = pd.Series(result.x, index=tickers, dtype=float).clip(lower=0)
    weights /= weights.sum()
    return weights


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float,
    max_volatility: float,
    benchmark_weights: Optional[pd.Series] = None,
    anchor_strength: float = 8.0,
    min_weight: float = 0.0,
    max_weight: float = 0.20,
) -> pd.Series:
    tickers = expected_returns.index.tolist()
    n = len(tickers)
    mu = expected_returns.astype(float).values
    sigma = cov_matrix.loc[tickers, tickers].astype(float).values
    benchmark = (
        benchmark_weights.reindex(tickers).fillna(0.0).astype(float).values
        if benchmark_weights is not None
        else np.full(n, 1.0 / n, dtype=float)
    )

    min_var_weights = minimum_variance_portfolio(
        cov_matrix.loc[tickers, tickers],
        min_weight=min_weight,
        max_weight=max_weight,
    )
    min_var_vol = float(np.sqrt(max(min_var_weights.values @ sigma @ min_var_weights.values, 0.0)))
    effective_max_vol = max(float(max_volatility), min_var_vol)

    def neg_utility(w):
        w = np.asarray(w, dtype=float)
        port_return = float(mu @ w)
        port_variance = float(w @ sigma @ w)
        diversification_penalty = anchor_strength * float(np.sum((w - benchmark) ** 2))
        return -(port_return - (risk_aversion / 2.0) * port_variance - diversification_penalty)

    def portfolio_volatility(w):
        w = np.asarray(w, dtype=float)
        return float(np.sqrt(max(w @ sigma @ w, 0)))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: effective_max_vol - portfolio_volatility(w)},
    ]
    bounds = [(min_weight, max_weight)] * n
    w0 = benchmark.copy()

    result = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        return min_var_weights

    weights = pd.Series(result.x, index=tickers, dtype=float).clip(lower=0)
    weights /= weights.sum()
    return weights


def compute_portfolio_metrics(weights: pd.Series, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.045) -> dict:
    tickers = weights.index.tolist()
    w = weights.astype(float).values
    mu = expected_returns[tickers].astype(float).values
    sigma = cov_matrix.loc[tickers, tickers].astype(float).values

    port_return = float(mu @ w)
    port_vol = float(np.sqrt(max(w @ sigma @ w, 0)))
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


def dollar_allocation(weights: pd.Series, capital: float, prices: pd.Series):
    tickers = weights.index.tolist()
    aligned_prices = prices[tickers].astype(float)
    allocations = weights.astype(float) * float(capital)
    shares = np.floor(allocations / aligned_prices).astype(int)
    actual_alloc = shares * aligned_prices
    cash_remaining = float(capital - actual_alloc.sum())

    df = pd.DataFrame({
        "Ticker": tickers,
        "Weight (%)": (weights.values * 100).round(2),
        "Allocation ($)": actual_alloc.round(2).values,
        "Shares": shares.values,
        "Price ($)": aligned_prices.round(2).values,
    }).sort_values("Allocation ($)", ascending=False).reset_index(drop=True)
    return df, round(cash_remaining, 2)


def run_optimization_pipeline(
    prices: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    risk_score: int,
    capital: float,
    market_caps: Optional[pd.Series] = None,
    returns: Optional[pd.DataFrame] = None,
    tau: float = 0.02,
    risk_free_rate: float = 0.045,
) -> dict:
    if risk_score not in range(1, 11):
        raise ValueError("risk_score must be an integer from 1 to 10.")
    if capital <= 0:
        raise ValueError("capital must be positive.")

    assets = prices.columns.tolist()
    risk_aversion = RISK_AVERSION_MAP[risk_score]
    max_vol = MAX_VOLATILITY_MAP[risk_score]

    return_matrix = returns if returns is not None else compute_returns(prices, method="log")
    return_matrix = return_matrix.reindex(columns=assets)
    cov_matrix = compute_covariance_matrix(return_matrix, annualize=True)
    market_weights = compute_market_weights(prices, market_caps)
    min_var_weights = minimum_variance_portfolio(cov_matrix)
    min_feasible_volatility = float(
        np.sqrt(max(min_var_weights.values @ cov_matrix.loc[assets, assets].values @ min_var_weights.values, 0.0))
    )
    effective_max_volatility = max(max_vol, min_feasible_volatility)
    implied_returns = compute_implied_returns(cov_matrix, market_weights, MARKET_RISK_AVERSION)
    prepared_views = prepare_black_litterman_views(ml_predictions, implied_returns, assets)
    P, Q, Omega = build_views(prepared_views, assets)
    bl_returns = black_litterman(cov_matrix, implied_returns, P, Q, Omega, tau)
    weights = optimize_portfolio(
        bl_returns,
        cov_matrix,
        risk_aversion,
        max_vol,
        benchmark_weights=market_weights,
    )
    metrics = compute_portfolio_metrics(weights, bl_returns, cov_matrix, risk_free_rate)
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
        "views_used": prepared_views,
        "cov_matrix": cov_matrix,
        "target_volatility": max_vol,
        "effective_max_volatility": effective_max_volatility,
        "min_feasible_volatility": min_feasible_volatility,
    }
