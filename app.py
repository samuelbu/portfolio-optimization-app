from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import build_prices_and_market_caps, load_features
from src import ml_views
from src.black_litterman import (
    MAX_VOLATILITY_MAP,
    RISK_LABELS,
    compute_portfolio_metrics,
    run_optimization_pipeline,
    score_risk_questionnaire,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features.parquet"
RISK_FREE_RATE = 0.045
RANDOM_PORTFOLIO_COUNT = 350

st.set_page_config(
    page_title="Smart Portfolio Builder",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

QUESTIONNAIRE = [
    (
        "q1",
        "Investment Horizon",
        [
            ("Less than 1 year", 1),
            ("1-3 years", 2),
            ("3-5 years", 3),
            ("5-10 years", 4),
            ("More than 10 years", 5),
        ],
    ),
    (
        "q2",
        "Reaction to Loss",
        [
            ("Sell everything immediately", 1),
            ("Sell part of it", 2),
            ("Do nothing and wait", 3),
            ("Buy a little more", 4),
            ("Invest significantly more", 5),
        ],
    ),
    (
        "q3",
        "Income Stability",
        [
            ("Very unstable", 1),
            ("Somewhat unstable", 2),
            ("Stable", 3),
            ("Very stable", 4),
            ("Extremely stable", 5),
        ],
    ),
    (
        "q4",
        "Investment Experience",
        [
            ("No experience", 1),
            ("Basic understanding", 2),
            ("Some experience", 3),
            ("Experienced investor", 4),
            ("Very experienced", 5),
        ],
    ),
    (
        "q5",
        "Investment Goal",
        [
            ("Preserve my money", 1),
            ("Generate some income", 2),
            ("Balanced growth and stability", 3),
            ("Grow my wealth", 4),
            ("Maximize returns with high risk", 5),
        ],
    ),
]

RISK_EXPLANATIONS = {
    1: "Your portfolio will focus strongly on capital preservation and limiting volatility.",
    2: "Your portfolio will lean defensive, aiming for stability with only modest growth exposure.",
    3: "Your portfolio will favor steady investing with limited risk and measured upside.",
    4: "Your portfolio will balance caution with some room for long-term growth.",
    5: "Your portfolio will target a middle ground between stability and growth.",
    6: "Your portfolio will lean toward growth while still keeping risk in a reasonable range.",
    7: "Your portfolio will prioritize growth while accepting moderate fluctuations.",
    8: "Your portfolio will pursue stronger growth and tolerate larger market swings.",
    9: "Your portfolio will seek high upside and accept significant volatility along the way.",
    10: "Your portfolio will maximize growth potential and tolerate very high risk and sharp fluctuations.",
}

COMPANY_NAMES = {
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "AMZN": "Amazon",
    "BAC": "Bank of America",
    "BRK-B": "Berkshire Hathaway",
    "CVX": "Chevron",
    "GS": "Goldman Sachs",
    "HD": "Home Depot",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "MA": "Mastercard",
    "MCD": "McDonald's",
    "MRK": "Merck",
    "MSFT": "Microsoft",
    "NKE": "Nike",
    "ORCL": "Oracle",
    "PSX": "Phillips 66",
    "SBUX": "Starbucks",
    "SLB": "SLB",
    "TSLA": "Tesla",
    "UNH": "UnitedHealth Group",
    "XOM": "Exxon Mobil",
}

METRIC_HELP = {
    "expected_return": "This is the model's estimate of how much the portfolio could grow in a typical year. It is not guaranteed, but it helps compare more defensive and more growth-oriented portfolios.",
    "volatility": "This shows how much the portfolio may move up and down over time. Higher volatility means a bumpier ride and larger swings in value.",
    "sharpe_ratio": "This compares expected reward to expected risk. Higher values generally mean the portfolio is offering more expected return for each unit of risk taken.",
    "risk_profile": "This is the investor profile created from your questionnaire. Higher scores aim for more growth and accept more risk.",
    "cash_left": "This is the part of your budget that stays uninvested after rounding down to whole shares.",
}


@st.cache_data
def load_base_data():
    df = load_features(DATA_PATH)
    prices, market_caps, sectors = build_prices_and_market_caps(df)
    return df, prices, market_caps, sectors


@st.cache_resource
def train_and_predict():
    df_clean, feature_cols, target_col = ml_views.load_and_prep_data(DATA_PATH)
    train_df, _ = ml_views.time_series_split(df_clean, split_date="2024-01-01")
    rf_model, xgb_model = ml_views.train_models(train_df[feature_cols], train_df[target_col])
    ml_pred = ml_views.generate_ml_views(df_clean, feature_cols, rf_model, xgb_model)
    return df_clean, ml_pred


def add_company_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker, ticker)


def latest_company_snapshot(raw_df: pd.DataFrame) -> pd.DataFrame:
    snapshot = (
        raw_df.sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "sector", "prc", "mkt_cap", "pe_ratio", "rolling_beta"]]
        .copy()
    )
    snapshot["Company"] = snapshot["ticker"].map(add_company_name)
    return snapshot.rename(
        columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "prc": "Latest Price ($)",
            "mkt_cap": "Market Cap ($)",
            "pe_ratio": "P/E Ratio",
            "rolling_beta": "Rolling Beta",
        }
    )


def build_portfolio_snapshot(
    raw_df: pd.DataFrame,
    weights: pd.Series,
    allocation_df: pd.DataFrame,
    ml_pred: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    holdings = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values})
    holdings["Weight (%)"] = holdings["Weight"] * 100
    holdings["Company"] = holdings["Ticker"].map(add_company_name)
    holdings["Target Allocation ($)"] = holdings["Weight"] * capital

    latest_snapshot = latest_company_snapshot(raw_df)
    ml_summary = ml_pred.rename(
        columns={
            "ticker": "Ticker",
            "return": "Predicted Return",
            "confidence": "Confidence",
        }
    ).copy()

    snapshot = holdings.merge(latest_snapshot, on=["Ticker", "Company"], how="left")
    snapshot = snapshot.merge(allocation_df, on="Ticker", how="left", suffixes=("", "_trade"))
    snapshot = snapshot.merge(ml_summary, on="Ticker", how="left")

    numeric_cols = [
        "Weight",
        "Weight (%)",
        "Target Allocation ($)",
        "Latest Price ($)",
        "Market Cap ($)",
        "P/E Ratio",
        "Rolling Beta",
        "Allocation ($)",
        "Shares",
        "Price ($)",
        "Predicted Return",
        "Confidence",
    ]
    for col in numeric_cols:
        if col in snapshot.columns:
            snapshot[col] = pd.to_numeric(snapshot[col], errors="coerce")

    snapshot["Sector"] = snapshot["Sector"].fillna("Unknown")
    snapshot["Allocation ($)"] = snapshot["Allocation ($)"].fillna(0.0)
    snapshot["Shares"] = snapshot["Shares"].fillna(0).astype(int)
    snapshot["Price ($)"] = snapshot["Price ($)"].fillna(snapshot["Latest Price ($)"])
    return snapshot.sort_values("Weight (%)", ascending=False).reset_index(drop=True)


def build_sector_exposure(weights: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"ticker": weights.index, "weight": weights.values})
    tmp["sector"] = tmp["ticker"].map(sectors.to_dict()).fillna("Unknown")
    return tmp.groupby("sector", as_index=False)["weight"].sum().sort_values("weight", ascending=False)


def build_sector_commentary(sector_df: pd.DataFrame) -> str:
    top_sector = sector_df.iloc[0]
    top_weight = top_sector["weight"] * 100
    if top_weight >= 35:
        return f"This portfolio is concentrated in {top_sector['sector']} ({top_weight:.1f}%)."
    if top_weight >= 25:
        return f"This portfolio leans heavily toward {top_sector['sector']} ({top_weight:.1f}%)."
    return f"This portfolio is fairly diversified, with its largest sector exposure in {top_sector['sector']} ({top_weight:.1f}%)."


def build_display_snapshot(portfolio_snapshot: pd.DataFrame) -> pd.DataFrame:
    display = portfolio_snapshot[portfolio_snapshot["Weight (%)"] > 0.05].copy()
    if display.empty:
        display = portfolio_snapshot.head(10).copy()
    return display.reset_index(drop=True)


def build_top_holdings_chart(portfolio_snapshot: pd.DataFrame) -> go.Figure:
    top = portfolio_snapshot.head(10).copy()
    top["Holding"] = top["Company"] + " (" + top["Ticker"] + ")"
    fig = px.bar(
        top,
        x="Holding",
        y="Weight (%)",
        color="Sector",
        text="Weight (%)",
        title="Top 10 holdings by portfolio weight",
        hover_data={
            "Weight (%)": ":.2f",
            "Target Allocation ($)": ":,.0f",
            "Predicted Return": ":.1%",
            "Confidence": ":.0%",
            "Holding": False,
        },
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="")
    return fig


def build_allocation_pie(portfolio_snapshot: pd.DataFrame) -> go.Figure:
    pie_df = portfolio_snapshot.copy()
    pie_df["Holding"] = pie_df["Company"] + " (" + pie_df["Ticker"] + ")"
    pie_df = pie_df[pie_df["Weight (%)"] >= 1.0].copy()

    other_weight = max(0.0, 100 - pie_df["Weight (%)"].sum())
    if other_weight > 0.01:
        pie_df = pd.concat(
            [
                pie_df,
                pd.DataFrame(
                    {
                        "Holding": ["Other"],
                        "Weight (%)": [other_weight],
                    }
                ),
            ],
            ignore_index=True,
        )

    fig = px.pie(
        pie_df,
        values="Weight (%)",
        names="Holding",
        title="Allocation mix",
        hole=0.35,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_sector_chart(sector_df: pd.DataFrame) -> go.Figure:
    chart_df = sector_df.assign(weight_pct=sector_df["weight"] * 100)
    fig = px.bar(
        chart_df,
        x="sector",
        y="weight_pct",
        text="weight_pct",
        title="Sector exposure",
        labels={"sector": "Sector", "weight_pct": "Weight (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_risk_gauge(volatility: float, risk_score: int) -> go.Figure:
    threshold = MAX_VOLATILITY_MAP[risk_score] * 100
    max_axis = max(35.0, threshold * 1.25)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=volatility * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Expected volatility"},
            gauge={
                "axis": {"range": [0, max_axis]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, max_axis * 0.4], "color": "#d8f3dc"},
                    {"range": [max_axis * 0.4, max_axis * 0.7], "color": "#ffe8a1"},
                    {"range": [max_axis * 0.7, max_axis], "color": "#f8c7c7"},
                ],
                "threshold": {
                    "line": {"color": "#d62728", "width": 4},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def build_random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = RANDOM_PORTFOLIO_COUNT,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mu = expected_returns.astype(float).values
    sigma = cov_matrix.loc[expected_returns.index, expected_returns.index].astype(float).values
    weights = rng.dirichlet(np.ones(len(expected_returns)), size=n_portfolios)
    returns = weights @ mu
    volatility = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights))
    sharpe = np.divide(
        returns - risk_free_rate,
        volatility,
        out=np.zeros_like(returns),
        where=volatility > 0,
    )
    return pd.DataFrame(
        {
            "Return": returns * 100,
            "Volatility": volatility * 100,
            "Sharpe": sharpe,
        }
    )


def build_risk_return_chart(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    optimized_weights: pd.Series,
    equal_weights: pd.Series,
) -> go.Figure:
    cloud = build_random_portfolios(expected_returns, cov_matrix)
    fig = px.scatter(
        cloud,
        x="Volatility",
        y="Return",
        color="Sharpe",
        color_continuous_scale="Viridis",
        title="Risk vs return",
        labels={"Volatility": "Expected volatility (%)", "Return": "Expected return (%)"},
        opacity=0.45,
    )

    optimized_metrics = compute_portfolio_metrics(
        optimized_weights,
        expected_returns,
        cov_matrix,
        risk_free_rate=RISK_FREE_RATE,
    )
    equal_metrics = compute_portfolio_metrics(
        equal_weights,
        expected_returns,
        cov_matrix,
        risk_free_rate=RISK_FREE_RATE,
    )

    points = [
        ("Optimized portfolio", optimized_metrics, "#0b6e4f"),
        ("Equal-weight portfolio", equal_metrics, "#d97706"),
    ]

    for name, metrics, color in points:
        fig.add_trace(
            go.Scatter(
                x=[metrics["volatility"] * 100],
                y=[metrics["expected_return"] * 100],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(size=14, color=color, line=dict(width=2, color="white")),
                hovertemplate=f"{name}<br>Return: %{{y:.1f}}%<br>Volatility: %{{x:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_market_benchmark(raw_df: pd.DataFrame) -> pd.Series:
    benchmark = (
        raw_df[["date", "mkt_rf", "rf"]]
        .drop_duplicates()
        .dropna()
        .sort_values("date")
        .set_index("date")
    )
    return (benchmark["mkt_rf"] + benchmark["rf"]).rename("Market benchmark")


def build_asset_return_panel(raw_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    panel = (
        raw_df.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
        .sort_index()
        .reindex(columns=tickers)
    )
    return panel.dropna(how="all").fillna(0.0)


def build_performance_frame(
    optimized_weights: pd.Series,
    equal_weights: pd.Series,
    raw_df: pd.DataFrame,
    capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_returns = build_asset_return_panel(raw_df, optimized_weights.index.tolist())
    optimized_returns = (asset_returns @ optimized_weights.reindex(asset_returns.columns).fillna(0.0)).rename("Optimized portfolio")
    equal_returns = (asset_returns @ equal_weights.reindex(asset_returns.columns).fillna(0.0)).rename("Equal-weight portfolio")
    benchmark_returns = build_market_benchmark(raw_df)

    return_frame = pd.concat([optimized_returns, equal_returns, benchmark_returns], axis=1).dropna()
    value_frame = (1 + return_frame).cumprod() * capital
    return return_frame, value_frame


def build_performance_chart(value_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {
        "Optimized portfolio": "#0b6e4f",
        "Equal-weight portfolio": "#d97706",
        "Market benchmark": "#1f77b4",
    }

    for column in value_frame.columns:
        fig.add_trace(
            go.Scatter(
                x=value_frame.index,
                y=value_frame[column],
                mode="lines",
                name=column,
                line=dict(width=3 if column == "Optimized portfolio" else 2, color=colors.get(column)),
            )
        )

    fig.update_layout(
        title="Historical growth of your starting capital",
        xaxis_title="Date",
        yaxis_title="Portfolio value ($)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_drawdown_chart(value_frame: pd.DataFrame) -> go.Figure:
    drawdown = value_frame.div(value_frame.cummax()).sub(1.0) * 100
    fig = go.Figure()
    colors = {
        "Optimized portfolio": "#0b6e4f",
        "Equal-weight portfolio": "#d97706",
        "Market benchmark": "#1f77b4",
    }

    for column in drawdown.columns:
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[column],
                mode="lines",
                name=column,
                line=dict(width=2, color=colors.get(column)),
            )
        )

    fig.update_layout(
        title="Historical drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_comparison_table(
    optimized_metrics: dict,
    equal_metrics: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Portfolio": "Optimized portfolio",
                "Expected Return (%)": optimized_metrics["expected_return"] * 100,
                "Volatility (%)": optimized_metrics["volatility"] * 100,
                "Sharpe Ratio": optimized_metrics["sharpe_ratio"],
            },
            {
                "Portfolio": "Equal-weight portfolio",
                "Expected Return (%)": equal_metrics["expected_return"] * 100,
                "Volatility (%)": equal_metrics["volatility"] * 100,
                "Sharpe Ratio": equal_metrics["sharpe_ratio"],
            },
        ]
    )


def build_scenario_analysis(portfolio_snapshot: pd.DataFrame, capital: float) -> dict:
    beta_series = portfolio_snapshot["Rolling Beta"].fillna(1.0)
    weights = portfolio_snapshot["Weight"].clip(lower=0.0)
    weighted_beta = np.average(beta_series, weights=weights)
    market_drop = -0.10
    portfolio_drop = weighted_beta * market_drop
    return {
        "market_drop": market_drop,
        "portfolio_drop": portfolio_drop,
        "dollar_impact": capital * portfolio_drop,
        "beta": weighted_beta,
    }


def build_why_portfolio_points(
    portfolio_snapshot: pd.DataFrame,
    sector_df: pd.DataFrame,
    risk_label: str,
) -> list[str]:
    top_holdings = portfolio_snapshot.head(3)
    top_holdings_text = ", ".join(
        f"{row['Company']} ({row['Weight (%)']:.1f}%)" for _, row in top_holdings.iterrows()
    )

    conviction_df = portfolio_snapshot.sort_values(["Predicted Return", "Confidence"], ascending=False).head(3)
    conviction_text = ", ".join(
        f"{row['Company']} ({row['Predicted Return']:.1%})" for _, row in conviction_df.iterrows() if pd.notna(row["Predicted Return"])
    )

    top_sector = sector_df.iloc[0]
    diversification_text = (
        f"The portfolio spreads across {sector_df['sector'].nunique()} sectors, with {top_sector['sector']} as the largest exposure at {top_sector['weight'] * 100:.1f}%."
    )

    points = [f"Top holdings: {top_holdings_text}."]
    if conviction_text:
        points.append(
            f"The model sees the strongest return potential among your holdings in {conviction_text}, which helps explain why they earned meaningful weights."
        )
    points.append(f"{diversification_text} That mix is designed to fit a {risk_label.lower()} investor.")
    return points


def build_ml_insights_table(portfolio_snapshot: pd.DataFrame) -> pd.DataFrame:
    insights = portfolio_snapshot[
        ["Company", "Ticker", "Weight (%)", "Predicted Return", "Confidence"]
    ].copy()
    insights["Predicted Return (%)"] = insights["Predicted Return"] * 100
    insights["Confidence (%)"] = insights["Confidence"] * 100
    insights = insights.sort_values(["Weight (%)", "Predicted Return"], ascending=False)
    insights = insights.drop(columns=["Predicted Return", "Confidence"])
    return insights.reset_index(drop=True)


def render_risk_questionnaire(prices: pd.DataFrame) -> tuple[int | None, str | None, float, bool]:
    with st.sidebar:
        st.header("Step 1: Your preferences")
        st.caption("Answer all five questions. The portfolio refreshes automatically as you change answers or budget.")

        answers: dict[str, int] = {}
        answered = 0
        for key, label, options in QUESTIONNAIRE:
            labels = [option_label for option_label, _ in options]
            selected = st.radio(label, options=labels, index=None, key=key)
            if selected is not None:
                answered += 1
                answers[key] = dict(options)[selected]

        st.progress(answered / len(QUESTIONNAIRE), text=f"{answered} of {len(QUESTIONNAIRE)} questions answered")

        capital = float(
            st.number_input(
                "Budget (USD)",
                min_value=1000,
                max_value=10000000,
                value=10000,
                step=500,
                key="capital",
            )
        )
        show_details = st.checkbox("Show technical details", value=False, key="show_details")

        if answered < len(QUESTIONNAIRE):
            st.info("Finish the questionnaire to see your portfolio.")
            return None, None, capital, show_details

        risk_score = score_risk_questionnaire(answers)
        risk_label = RISK_LABELS[risk_score]
        st.success(f"{risk_label} ({risk_score}/10)")
        st.caption(RISK_EXPLANATIONS[risk_score])

        st.divider()
        st.header("Portfolio setup")
        st.write(f"Budget: **${capital:,.0f}**")
        st.write(f"Stock universe available: **{prices.shape[1]}**")
        st.write(f"Data window: **{prices.index.min().date()}** to **{prices.index.max().date()}**")

    return risk_score, risk_label, capital, show_details


def render_portfolio_summary(metrics: dict, risk_score: int, risk_label: str, cash_remaining: float):
    st.subheader("Portfolio Summary")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected annual return", f"{metrics['expected_return']:.1%}", help=METRIC_HELP["expected_return"])
        c2.metric("Expected volatility", f"{metrics['volatility']:.1%}", help=METRIC_HELP["volatility"])
        c3.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}", help=METRIC_HELP["sharpe_ratio"])
        c4.metric("Risk profile", f"{risk_label} ({risk_score}/10)", help=METRIC_HELP["risk_profile"])
        st.caption(f"Cash left after rounding to whole shares: ${cash_remaining:,.0f}.")


def render_why_portfolio(points: list[str]):
    st.subheader("Why this portfolio?")
    st.caption("A simple explanation of why these holdings fit your answers and the model's outlook.")
    for point in points:
        st.write(f"- {point}")


def main():
    st.title("Smart Portfolio Builder")
    st.caption("A guided portfolio experience powered by ML return signals, Black-Litterman optimization, and historical market data.")

    with st.spinner("Loading market and feature data..."):
        raw_df, prices, market_caps, sectors = load_base_data()
        _, ml_pred = train_and_predict()

    asset_returns = build_asset_return_panel(raw_df, prices.columns.tolist())

    st.markdown(
        "Answer the questionnaire in the left panel to generate a portfolio that matches your risk comfort, "
        "shows what you would own, and explains how the recommendation compares with simpler alternatives."
    )

    risk_score, risk_label, capital, show_details = render_risk_questionnaire(prices)
    if risk_score is None or risk_label is None:
        st.stop()

    with st.spinner("Optimizing your portfolio..."):
        result = run_optimization_pipeline(
            prices=prices,
            ml_predictions=ml_pred,
            risk_score=risk_score,
            capital=capital,
            market_caps=market_caps,
            returns=asset_returns,
            risk_free_rate=RISK_FREE_RATE,
        )

    weights = result["weights"]
    metrics = result["metrics"]
    equal_weights = pd.Series(1.0 / len(weights), index=weights.index)
    equal_metrics = compute_portfolio_metrics(
        equal_weights,
        result["bl_returns"],
        result["cov_matrix"],
        risk_free_rate=RISK_FREE_RATE,
    )

    portfolio_snapshot = build_portfolio_snapshot(
        raw_df=raw_df,
        weights=weights,
        allocation_df=result["allocation_df"],
        ml_pred=ml_pred,
        capital=capital,
    )
    display_snapshot = build_display_snapshot(portfolio_snapshot)
    sector_df = build_sector_exposure(weights, sectors)
    performance_returns, performance_values = build_performance_frame(
        optimized_weights=weights,
        equal_weights=equal_weights,
        raw_df=raw_df,
        capital=capital,
    )
    comparison_df = build_comparison_table(metrics, equal_metrics)
    scenario = build_scenario_analysis(portfolio_snapshot, capital)
    why_points = build_why_portfolio_points(display_snapshot, sector_df, risk_label)
    ml_insights_df = build_ml_insights_table(display_snapshot)

    render_portfolio_summary(metrics, risk_score, risk_label, result["cash_remaining"])
    if result["target_volatility"] < result["min_feasible_volatility"]:
        st.caption(
            f"Note: In this stock-only universe, the lowest achievable volatility is about "
            f"{result['min_feasible_volatility']:.1%}. For this profile, the app is showing the "
            f"minimum-volatility stock portfolio rather than an unattainable lower-risk target."
        )

    st.subheader("Portfolio Composition")
    st.caption("See where your money is expected to go and which stocks matter most.")
    comp_left, comp_right = st.columns((1.1, 0.9))
    with comp_left:
        st.plotly_chart(build_top_holdings_chart(display_snapshot), use_container_width=True)
    with comp_right:
        st.plotly_chart(build_allocation_pie(display_snapshot), use_container_width=True)

    st.plotly_chart(build_sector_chart(sector_df), use_container_width=True)
    st.caption(build_sector_commentary(sector_df))

    st.subheader("Risk and Performance")
    st.caption("These visuals show how the portfolio balances opportunity, downside, and historical behavior.")

    risk_left, risk_right = st.columns((1.4, 0.8))
    with risk_left:
        st.plotly_chart(
            build_risk_return_chart(
                expected_returns=result["bl_returns"],
                cov_matrix=result["cov_matrix"],
                optimized_weights=weights,
                equal_weights=equal_weights,
            ),
            use_container_width=True,
        )
    with risk_right:
        st.plotly_chart(build_risk_gauge(metrics["volatility"], risk_score), use_container_width=True)
        st.caption(
            f"The red marker shows the top end of the volatility range typically associated with a {risk_label.lower()} profile."
        )

        with st.container(border=True):
            st.markdown("**Scenario analysis**")
            st.write(
                f"If the broad market drops **10%**, this portfolio is expected to move by roughly **{scenario['portfolio_drop']:.1%}** "
                f"based on its weighted beta of **{scenario['beta']:.2f}**."
            )
            st.caption(f"Estimated impact on a ${capital:,.0f} portfolio: ${scenario['dollar_impact']:,.0f}.")

    st.plotly_chart(build_performance_chart(performance_values), use_container_width=True)
    st.caption(
        "Historical simulation applies today's portfolio weights to past daily returns. "
        "The benchmark line uses the market return series available in the dataset as a broad-market proxy."
    )
    st.caption(
        "Disclosure: Past performance does not guarantee future results. This chart is illustrative only "
        "and is meant to show how the current portfolio mix would have behaved in past market conditions, "
        "not to predict future returns."
    )

    compare_left, compare_right = st.columns((1.1, 0.9))
    with compare_left:
        st.plotly_chart(build_drawdown_chart(performance_values), use_container_width=True)
    with compare_right:
        st.markdown("**Optimized vs equal-weight**")
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Expected Return (%)": st.column_config.NumberColumn("Expected Return (%)", format="%.2f%%"),
                "Volatility (%)": st.column_config.NumberColumn("Volatility (%)", format="%.2f%%"),
                "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
            },
        )

    render_why_portfolio(why_points)

    st.subheader("Allocation Table")
    st.caption("Click any column header to sort. Download the table if you want to review it outside the app.")
    allocation_display = portfolio_snapshot[
        ["Company", "Ticker", "Sector", "Weight (%)", "Allocation ($)", "Shares", "Price ($)"]
    ].copy()
    allocation_display = allocation_display[allocation_display["Allocation ($)"] > 0].reset_index(drop=True)
    st.dataframe(
        allocation_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", format="%.2f%%"),
            "Allocation ($)": st.column_config.NumberColumn("Dollar Allocation", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("Shares", format="%d"),
            "Price ($)": st.column_config.NumberColumn("Price", format="$%.2f"),
        },
    )

    csv = allocation_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download allocation as CSV",
        data=csv,
        file_name="portfolio_allocation.csv",
        mime="text/csv",
    )

    st.subheader("ML Insights")
    st.caption("Higher confidence means the model's tree-based predictions agree more closely for that stock.")
    st.dataframe(
        ml_insights_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", format="%.2f%%"),
            "Predicted Return (%)": st.column_config.NumberColumn("Predicted Return (%)", format="%.2f%%"),
            "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.0f%%"),
        },
    )

    if show_details:
        with st.expander("Technical details", expanded=False):
            st.write("**Black-Litterman posterior returns**")
            st.dataframe(
                result["bl_returns"].sort_values(ascending=False).rename("posterior_return"),
                use_container_width=True,
            )

            st.write("**Views used in the optimization**")
            views_used_display = result["views_used"].copy()
            views_used_display["View Return (%)"] = views_used_display["return"] * 100
            views_used_display["Confidence (%)"] = views_used_display["confidence"] * 100
            views_used_display = views_used_display.drop(columns=["return", "confidence"])
            st.dataframe(
                views_used_display.sort_values("View Return (%)", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "View Return (%)": st.column_config.NumberColumn("View Return (%)", format="%.2f%%"),
                    "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.0f%%"),
                },
            )

            st.write("**Full ML prediction table**")
            full_ml_df = ml_pred.sort_values("return", ascending=False).copy()
            full_ml_df.insert(1, "company", full_ml_df["ticker"].map(add_company_name))
            full_ml_df["Predicted Return (%)"] = full_ml_df["return"] * 100
            full_ml_df["Confidence (%)"] = full_ml_df["confidence"] * 100
            full_ml_display = full_ml_df.drop(columns=["return", "confidence"])
            st.dataframe(
                full_ml_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Predicted Return (%)": st.column_config.NumberColumn("Predicted Return (%)", format="%.2f%%"),
                    "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.0f%%"),
                },
            )

            st.write("**Historical return summary**")
            realized_stats = pd.DataFrame(
                {
                    "Series": performance_returns.columns,
                    "Annualized Return (%)": [
                        ((1 + performance_returns[col]).prod() ** (252 / len(performance_returns[col])) - 1) * 100
                        for col in performance_returns.columns
                    ],
                    "Annualized Volatility (%)": [
                        performance_returns[col].std() * np.sqrt(252) * 100
                        for col in performance_returns.columns
                    ],
                }
            )
            st.dataframe(
                realized_stats,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Annualized Return (%)": st.column_config.NumberColumn("Annualized Return (%)", format="%.2f%%"),
                    "Annualized Volatility (%)": st.column_config.NumberColumn("Annualized Volatility (%)", format="%.2f%%"),
                },
            )


if __name__ == "__main__":
    main()
