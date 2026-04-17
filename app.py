from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_features, build_prices_and_market_caps
import ml_views
from black_litterman import run_optimization_pipeline, RISK_LABELS

st.set_page_config(
    page_title="Smart Portfolio Builder",
    page_icon="📈",
    layout="wide",
)

RISK_PRESETS = {
    "Conservative": 3,
    "Moderate": 5,
    "Growth": 8,
}


@st.cache_data
def load_base_data():
    df = load_features("data/features.parquet")
    prices, market_caps, sectors = build_prices_and_market_caps(df)
    return df, prices, market_caps, sectors


@st.cache_resource
def train_and_predict():
    df_clean, feature_cols, target_col = ml_views.load_and_prep_data("data/features.parquet")
    train_df, _ = ml_views.time_series_split(df_clean, split_date="2024-01-01")
    rf_model, xgb_model = ml_views.train_models(train_df[feature_cols], train_df[target_col])
    ml_pred = ml_views.generate_ml_views(df_clean, feature_cols, rf_model, xgb_model)
    return df_clean, ml_pred


def build_sector_exposure(weights: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"ticker": weights.index, "weight": weights.values})
    tmp["sector"] = tmp["ticker"].map(sectors.to_dict())
    return tmp.groupby("sector", as_index=False)["weight"].sum().sort_values("weight", ascending=False)


def build_growth_chart(prices: pd.DataFrame, weights: pd.Series) -> go.Figure:
    top = weights.sort_values(ascending=False).head(5).index.tolist()
    norm = prices[top] / prices[top].iloc[0]
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
    fig.update_layout(
        title="Normalized price history of top holdings",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend_title="Ticker",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def main():
    st.title("📈 Smart Portfolio Builder")
    st.caption("A fully automated Black-Litterman portfolio app powered by ML signals and S&P 500 stock data.")

    with st.spinner("Loading market and feature data..."):
        raw_df, prices, market_caps, sectors = load_base_data()
        _, ml_pred = train_and_predict()

    st.markdown(
        "This app builds a stock portfolio from the available S&P 500-style universe in your dataset. "
        "It uses historical market data, machine learning return views, and Black-Litterman optimization to recommend a portfolio based on the user's risk profile and budget."
    )

    with st.sidebar:
        st.header("Your inputs")
        profile = st.radio("Investment style", options=list(RISK_PRESETS.keys()), index=1)
        risk_score = RISK_PRESETS[profile]
        capital = st.number_input("Budget (USD)", min_value=1000, max_value=10000000, value=10000, step=500)
        show_details = st.checkbox("Show technical details", value=False)
        run = st.button("Build my portfolio", type="primary", use_container_width=True)

        st.divider()
        st.write("**Model profile**")
        st.write(f"Risk score: **{risk_score}/10**")
        st.write(f"Mapped label: **{RISK_LABELS[risk_score]}**")
        st.write(f"Stock universe available: **{prices.shape[1]}**")
        st.write(f"Data window: **{prices.index.min().date()}** to **{prices.index.max().date()}**")

    if not run:
        st.info("Choose your profile and budget, then click **Build my portfolio**.")
        st.stop()

    with st.spinner("Optimizing your portfolio..."):
        result = run_optimization_pipeline(
            prices=prices,
            ml_predictions=ml_pred,
            risk_score=risk_score,
            capital=float(capital),
            market_caps=market_caps,
        )

    weights = result["weights"]
    metrics = result["metrics"]
    allocation_df = result["allocation_df"]
    sector_df = build_sector_exposure(weights, sectors)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected annual return", f"{metrics['expected_return']:.1%}")
    c2.metric("Expected volatility", f"{metrics['volatility']:.1%}")
    c3.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}")
    c4.metric("Cash left", f"${result['cash_remaining']:,.0f}")

    st.success(f"Portfolio built for a **{result['risk_label']}** investor with a **${capital:,.0f}** budget.")

    left, right = st.columns((1.1, 0.9))

    with left:
        top_holdings = metrics["weights_df"].query("`Weight (%)` > 0").head(10)
        fig_bar = px.bar(
            top_holdings,
            x="Ticker",
            y="Weight (%)",
            title="Top portfolio holdings",
            text="Weight (%)",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        pie_df = metrics["weights_df"].query("`Weight (%)` > 0").head(8).copy()
        other = max(0.0, 100 - pie_df["Weight (%)"].sum())
        if other > 0.01:
            pie_df.loc[len(pie_df)] = ["Other", other]
        fig_pie = px.pie(pie_df, values="Weight (%)", names="Ticker", title="Portfolio mix")
        fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig_sector = px.bar(
            sector_df.assign(weight_pct=sector_df["weight"] * 100),
            x="sector",
            y="weight_pct",
            title="Sector exposure",
            labels={"weight_pct": "Weight (%)", "sector": "Sector"},
        )
        fig_sector.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_sector, use_container_width=True)

    with c6:
        st.plotly_chart(build_growth_chart(prices, weights), use_container_width=True)

    st.subheader("Recommended allocation")
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    csv = allocation_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download allocation as CSV",
        data=csv,
        file_name="portfolio_allocation.csv",
        mime="text/csv",
    )

    if show_details:
        st.subheader("Technical details")
        c7, c8 = st.columns(2)
        with c7:
            st.write("**Black-Litterman posterior returns**")
            st.dataframe(result["bl_returns"].sort_values(ascending=False).rename("posterior_return"))
        with c8:
            st.write("**ML views used as inputs**")
            st.dataframe(ml_pred.sort_values("return", ascending=False), use_container_width=True, hide_index=True)

        st.write("**Universe snapshot**")
        latest_snapshot = raw_df.sort_values("date").groupby("ticker").tail(1)[["ticker", "sector", "prc", "mkt_cap", "pe_ratio"]]
        st.dataframe(latest_snapshot.sort_values("mkt_cap", ascending=False), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
