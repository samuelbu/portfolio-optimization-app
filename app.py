from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_features, build_prices_and_market_caps
from src import ml_views
from src.black_litterman import run_optimization_pipeline, RISK_LABELS, score_risk_questionnaire

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features.parquet"

st.set_page_config(
    page_title="Smart Portfolio Builder",
    page_icon="📈",
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


def build_sector_exposure(weights: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame({"ticker": weights.index, "weight": weights.values})
    tmp["sector"] = tmp["ticker"].map(sectors.to_dict())
    return tmp.groupby("sector", as_index=False)["weight"].sum().sort_values("weight", ascending=False)


def build_growth_chart(prices: pd.DataFrame, weights: pd.Series) -> go.Figure:
    top = weights.sort_values(ascending=False).head(5).index.tolist()
    norm = prices[top] / prices[top].iloc[0]
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[col],
                mode="lines",
                name=f"{COMPANY_NAMES.get(col, col)} ({col})",
            )
        )
    fig.update_layout(
        title="Normalized price history of top holdings",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend_title="Company",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_risk_questionnaire() -> tuple[int | None, str | None]:
    with st.sidebar:
        st.header("Step 1: Your preferences")
        st.caption("Answer these questions to match the portfolio to your comfort with risk.")

        answered = sum(1 for key, _, _ in QUESTIONNAIRE if st.session_state.get(key) is not None)
        st.progress(answered / len(QUESTIONNAIRE), text=f"{answered} of {len(QUESTIONNAIRE)} questions answered")

        answers: dict[str, int] = {}
        with st.form("risk_questionnaire"):
            for key, label, options in QUESTIONNAIRE:
                labels = [option_label for option_label, _ in options]
                selected = st.radio(
                    label,
                    options=labels,
                    index=None,
                    key=key,
                )
                if selected is not None:
                    answers[key] = dict(options)[selected]

            capital = st.number_input("Budget (USD)", min_value=1000, max_value=10000000, value=10000, step=500)
            show_details = st.checkbox("Show technical details", value=False)
            submitted = st.form_submit_button("Build my portfolio", type="primary", use_container_width=True)

    if not submitted:
        st.info("Use the left panel to answer the questionnaire, then click **Build my portfolio**.")
        return None, None

    missing = [label for key, label, _ in QUESTIONNAIRE if key not in answers]
    if missing:
        st.sidebar.warning("Please answer all five questions before continuing.")
        return None, None

    risk_score = score_risk_questionnaire(answers)
    risk_label = RISK_LABELS[risk_score]
    explanation = RISK_EXPLANATIONS[risk_score]

    with st.container(border=True):
        st.success(f"Your profile is **{risk_label} ({risk_score}/10)**.")
        st.write(explanation)

    st.session_state["capital"] = float(capital)
    st.session_state["show_details"] = show_details
    return risk_score, risk_label


def add_company_names(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    enriched = df.copy()
    enriched["Company"] = enriched[ticker_col].map(lambda ticker: COMPANY_NAMES.get(ticker, ticker))
    if ticker_col == "Ticker":
        columns = ["Company", "Ticker"] + [col for col in enriched.columns if col not in {"Company", "Ticker"}]
        enriched = enriched[columns]
    return enriched


def build_company_snapshot(raw_df: pd.DataFrame, weights_df: pd.DataFrame, ml_pred: pd.DataFrame) -> pd.DataFrame:
    latest_snapshot = (
        raw_df.sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "sector", "prc", "mkt_cap", "pe_ratio"]]
        .copy()
    )
    latest_snapshot["Company"] = latest_snapshot["ticker"].map(lambda ticker: COMPANY_NAMES.get(ticker, ticker))
    latest_snapshot = latest_snapshot.rename(
        columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "prc": "Latest Price ($)",
            "mkt_cap": "Market Cap ($)",
            "pe_ratio": "P/E Ratio",
        }
    )

    ml_summary = ml_pred.rename(
        columns={
            "ticker": "Ticker",
            "return": "Expected Return",
            "confidence": "Model Confidence",
        }
    ).copy()

    snapshot = weights_df.merge(latest_snapshot, on=["Ticker", "Company"], how="left")
    snapshot = snapshot.merge(ml_summary, on="Ticker", how="left")
    snapshot["Allocation ($)"] = snapshot["Weight (%)"] / 100 * float(st.session_state["capital"])
    snapshot["Market Cap ($)"] = pd.to_numeric(snapshot["Market Cap ($)"], errors="coerce")
    snapshot["P/E Ratio"] = pd.to_numeric(snapshot["P/E Ratio"], errors="coerce")
    snapshot["Latest Price ($)"] = pd.to_numeric(snapshot["Latest Price ($)"], errors="coerce")
    snapshot["Expected Return"] = pd.to_numeric(snapshot["Expected Return"], errors="coerce")
    snapshot["Model Confidence"] = pd.to_numeric(snapshot["Model Confidence"], errors="coerce")
    return snapshot.sort_values("Weight (%)", ascending=False).reset_index(drop=True)


def build_allocation_chart(company_snapshot: pd.DataFrame) -> go.Figure:
    top = company_snapshot.head(10).copy()
    top["Holding"] = top["Company"] + " (" + top["Ticker"] + ")"
    fig = px.bar(
        top,
        x="Holding",
        y="Allocation ($)",
        color="Sector",
        title="Where your money goes",
        hover_data={
            "Weight (%)": ":.2f",
            "Allocation ($)": ":,.0f",
            "Latest Price ($)": ":.2f",
            "Expected Return": ":.1%",
            "Holding": False,
        },
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="")
    return fig


def build_company_map(company_snapshot: pd.DataFrame) -> go.Figure:
    plot_df = company_snapshot.head(12).copy()
    plot_df["Holding"] = plot_df["Company"] + " (" + plot_df["Ticker"] + ")"
    fig = px.scatter(
        plot_df,
        x="P/E Ratio",
        y="Expected Return",
        size="Weight (%)",
        color="Sector",
        hover_name="Holding",
        hover_data={
            "Latest Price ($)": ":.2f",
            "Market Cap ($)": ":,.0f",
            "Model Confidence": ":.0%",
            "Weight (%)": ":.2f",
        },
        title="Company map: valuation vs expected growth",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="P/E ratio",
        yaxis_title="Expected annual return",
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

    risk_score, risk_label = render_risk_questionnaire()
    if risk_score is None or risk_label is None:
        st.stop()

    capital = float(st.session_state["capital"])
    show_details = bool(st.session_state["show_details"])

    with st.sidebar:
        st.divider()
        st.header("Portfolio setup")
        st.write(f"Risk score: **{risk_score}/10**")
        st.write(f"Risk profile: **{risk_label}**")
        st.write(f"Budget: **${capital:,.0f}**")
        st.write(f"Stock universe available: **{prices.shape[1]}**")
        st.write(f"Data window: **{prices.index.min().date()}** to **{prices.index.max().date()}**")

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
    allocation_df = add_company_names(result["allocation_df"])
    sector_df = build_sector_exposure(weights, sectors)
    weights_df = add_company_names(metrics["weights_df"])
    company_snapshot = build_company_snapshot(raw_df, weights_df, ml_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected annual return", f"{metrics['expected_return']:.1%}", help=METRIC_HELP["expected_return"])
    c2.metric("Expected volatility", f"{metrics['volatility']:.1%}", help=METRIC_HELP["volatility"])
    c3.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}", help=METRIC_HELP["sharpe_ratio"])
    c4.metric("Cash left", f"${result['cash_remaining']:,.0f}", help=METRIC_HELP["cash_left"])

    st.success(f"Portfolio built for a **{result['risk_label']}** investor with a **${capital:,.0f}** budget.")
    st.caption("Tip: hover over the small question-mark icons on the key metrics for a simple explanation.")

    with st.expander("What these results mean", expanded=False):
        st.write(
            "The portfolio combines company-level signals, your risk profile, and diversification rules. "
            "The charts below help you see where your money goes, how concentrated the portfolio is, and what kinds of businesses you would be buying."
        )

    left, right = st.columns((1.1, 0.9))

    with left:
        top_holdings = weights_df.query("`Weight (%)` > 0").head(10).copy()
        top_holdings["Holding"] = top_holdings["Company"] + " (" + top_holdings["Ticker"] + ")"
        fig_bar = px.bar(
            top_holdings,
            x="Holding",
            y="Weight (%)",
            title="Top portfolio holdings",
            text="Weight (%)",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        pie_df = weights_df.query("`Weight (%)` > 0").head(8).copy()
        pie_df["Holding"] = pie_df["Company"] + " (" + pie_df["Ticker"] + ")"
        other = max(0.0, 100 - pie_df["Weight (%)"].sum())
        if other > 0.01:
            pie_df.loc[len(pie_df)] = {"Company": "Other", "Ticker": "Other", "Weight (%)": other, "Holding": "Other"}
        fig_pie = px.pie(pie_df, values="Weight (%)", names="Holding", title="Portfolio mix")
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

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(build_allocation_chart(company_snapshot), use_container_width=True)
    with c8:
        st.plotly_chart(build_company_map(company_snapshot), use_container_width=True)

    st.subheader("What you're buying")
    st.caption("These are the main companies in your recommended portfolio, along with simple business and valuation context.")
    company_display = company_snapshot.head(10).copy()
    company_display["Expected Return"] = company_display["Expected Return"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    company_display["Model Confidence"] = company_display["Model Confidence"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
    company_display["Weight (%)"] = company_display["Weight (%)"].map(lambda x: f"{x:.2f}%")
    company_display["Latest Price ($)"] = company_display["Latest Price ($)"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    company_display["Allocation ($)"] = company_display["Allocation ($)"].map(lambda x: f"${x:,.0f}")
    company_display["Market Cap ($)"] = company_display["Market Cap ($)"].map(lambda x: f"${x/1e9:,.1f}B" if pd.notna(x) else "N/A")
    company_display["P/E Ratio"] = company_display["P/E Ratio"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    st.dataframe(
        company_display[
            [
                "Company",
                "Ticker",
                "Sector",
                "Weight (%)",
                "Allocation ($)",
                "Latest Price ($)",
                "Market Cap ($)",
                "P/E Ratio",
                "Expected Return",
                "Model Confidence",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

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
        c9, c10 = st.columns(2)
        with c9:
            st.write("**Black-Litterman posterior returns**")
            st.dataframe(result["bl_returns"].sort_values(ascending=False).rename("posterior_return"))
        with c10:
            st.write("**ML views used as inputs**")
            ml_views_df = ml_pred.sort_values("return", ascending=False).copy()
            ml_views_df.insert(1, "company", ml_views_df["ticker"].map(lambda ticker: COMPANY_NAMES.get(ticker, ticker)))
            st.dataframe(ml_views_df, use_container_width=True, hide_index=True)

        st.write("**Universe snapshot**")
        latest_snapshot = raw_df.sort_values("date").groupby("ticker").tail(1)[["ticker", "sector", "prc", "mkt_cap", "pe_ratio"]]
        latest_snapshot.insert(1, "company", latest_snapshot["ticker"].map(lambda ticker: COMPANY_NAMES.get(ticker, ticker)))
        st.dataframe(latest_snapshot.sort_values("mkt_cap", ascending=False), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
