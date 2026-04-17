"""Portfolio optimization module using the Black-Litterman model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RISK_AVERSION_MAP = {
    1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5, 5: 2.0,
    6: 1.75, 7: 1.5, 8: 1.25, 9: 1.0, 10: 0.75,
}

MAX_VOLATILITY_MAP = {
    1: 0.08, 2: 0.10, 3: 0.12, 4: 0.14, 5: 0.16,
    6: 0.18, 7: 0.20, 8: 0.22, 9: 0.25, 10: 0.30,
}

RISK_LABELS = {
    1: 'Ultra Conservative', 2: 'Very Conservative', 3: 'Conservative',
    4: 'Moderately Conservative', 5: 'Moderate', 6: 'Moderately Aggressive',
    7: 'Aggressive', 8: 'Very Aggressive', 9: 'Speculative', 10: 'Ultra Speculative',
}


def score_risk_questionnaire(answers: dict) -> int:
    expected_questions = ['q1', 'q2', 'q3', 'q4', 'q5']
    for q in expected_questions:
        if q not in answers:
            raise ValueError(f'Missing answer for question: {q}')
        if answers[q] not in range(1, 6):
            raise ValueError(f'Answer to {q} must be between 1 and 5.')
    total = sum(answers[q] for q in expected_questions)
    score = round(1 + (total - 5) * (9 / 20))
    return int(np.clip(score, 1, 10))


def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    if method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    elif method == 'simple':
        returns = prices.pct_change().dropna()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return returns


def compute_covariance_matrix(returns: pd.DataFrame, annualize: bool = True, trading_days: int = 252) -> pd.DataFrame:
    cov = returns.cov()
    if annualize:
        cov = cov * trading_days
    return cov


def compute_market_weights(prices: pd.DataFrame, market_caps: Optional[pd.Series] = None) -> pd.Series:
    tickers = prices.columns.tolist()
    if market_caps is None:
        return pd.Series(1.0 / len(tickers), index=tickers)
    missing = set(tickers) - set(market_caps.index)
    if missing:
        raise ValueError(f'Market caps missing for: {missing}')
    weights = market_caps[tickers] / market_caps[tickers].sum()
    return weights


def compute_implied_returns(cov_matrix: pd.DataFrame, market_weights: pd.Series, risk_aversion: float) -> pd.Series:
    tickers = cov_matrix.columns.tolist()
    w = market_weights[tickers].values
    sigma = cov_matrix.values
    pi = risk_aversion * sigma @ w
    return pd.Series(pi, index=tickers)


def build_views(ml_predictions: pd.DataFrame, assets: list[str]):
    valid_predictions = ml_predictions[ml_predictions['ticker'].isin(assets)].copy()
    if valid_predictions.empty:
        raise ValueError('None of the ML predictions match assets in the universe.')

    k = len(valid_predictions)
    n = len(assets)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, (_, row) in enumerate(valid_predictions.iterrows()):
        j = assets.index(row['ticker'])
        P[i, j] = 1.0
        Q[i] = row['return']
        omega_diag[i] = (1.0 / max(row['confidence'], 1e-6) - 1) * 0.01

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def black_litterman(cov_matrix: pd.DataFrame, implied_returns: pd.Series, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray, tau: float = 0.05) -> pd.Series:
    tickers = cov_matrix.columns.tolist()
    sigma = cov_matrix.values
    pi = implied_returns[tickers].values

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)
    M = tau_sigma_inv + P.T @ omega_inv @ P
    mu_bl = np.linalg.inv(M) @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    return pd.Series(mu_bl, index=tickers)


def optimize_portfolio(expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_aversion: float, max_volatility: float, min_weight: float = 0.0, max_weight: float = 0.25) -> pd.Series:
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
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: max_volatility - portfolio_volatility(w)},
    ]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_utility,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000},
    )

    if not result.success:
        return pd.Series(np.ones(n) / n, index=tickers)

    weights = pd.Series(result.x, index=tickers)
    weights = weights.clip(lower=0).round(8)
    weights /= weights.sum()
    return weights


def compute_portfolio_metrics(weights: pd.Series, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.045) -> dict:
    tickers = weights.index.tolist()
    w = weights.values
    mu = expected_returns[tickers].values
    sigma = cov_matrix.loc[tickers, tickers].values

    port_return = float(mu @ w)
    port_vol = float(np.sqrt(w @ sigma @ w))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight (%)': (weights.values * 100).round(2),
    }).sort_values('Weight (%)', ascending=False).reset_index(drop=True)

    return {
        'expected_return': port_return,
        'volatility': port_vol,
        'sharpe_ratio': sharpe,
        'weights_df': weights_df,
    }


def dollar_allocation(weights: pd.Series, capital: float, prices: pd.Series):
    tickers = weights.index.tolist()
    allocations = weights * capital
    shares = (allocations / prices[tickers]).apply(np.floor)
    actual_alloc = shares * prices[tickers]
    cash_remaining = capital - actual_alloc.sum()

    df = pd.DataFrame({
        'Ticker': tickers,
        'Weight (%)': (weights.values * 100).round(2),
        'Allocation ($)': actual_alloc.round(2).values,
        'Shares': shares.values.astype(int),
        'Price ($)': prices[tickers].round(2).values,
    }).sort_values('Allocation ($)', ascending=False).reset_index(drop=True)

    return df, round(cash_remaining, 2)


def run_optimization_pipeline(prices: pd.DataFrame, ml_predictions: pd.DataFrame, risk_score: int, capital: float, market_caps: Optional[pd.Series] = None, tau: float = 0.05, risk_free_rate: float = 0.045) -> dict:
    if risk_score not in range(1, 11):
        raise ValueError('risk_score must be an integer from 1 to 10.')
    if capital <= 0:
        raise ValueError('capital must be positive.')

    assets = prices.columns.tolist()
    risk_aversion = RISK_AVERSION_MAP[risk_score]
    max_vol = MAX_VOLATILITY_MAP[risk_score]

    returns = compute_returns(prices, method='log')
    cov_matrix = compute_covariance_matrix(returns, annualize=True)
    market_weights = compute_market_weights(prices, market_caps)
    implied_returns = compute_implied_returns(cov_matrix, market_weights, risk_aversion)
    P, Q, Omega = build_views(ml_predictions, assets)
    bl_returns = black_litterman(cov_matrix, implied_returns, P, Q, Omega, tau)
    weights = optimize_portfolio(bl_returns, cov_matrix, risk_aversion, max_vol)
    metrics = compute_portfolio_metrics(weights, bl_returns, cov_matrix, risk_free_rate)
    latest_prices = prices.iloc[-1]
    allocation_df, cash_remaining = dollar_allocation(weights, capital, latest_prices)

    return {
        'weights': weights,
        'metrics': metrics,
        'allocation_df': allocation_df,
        'cash_remaining': cash_remaining,
        'risk_label': RISK_LABELS[risk_score],
        'bl_returns': bl_returns,
        'implied_returns': implied_returns,
        'cov_matrix': cov_matrix,
    }
