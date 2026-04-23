from __future__ import annotations

import base64
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
LOGO_PATH = BASE_DIR / "Logo.png"
RISK_FREE_RATE = 0.045
RANDOM_PORTFOLIO_COUNT = 350

BRAND_COLORS = {
    "bg": "#F5F9FF",
    "panel": "#FFFFFF",
    "panel_alt": "#EEF5FF",
    "text": "#10213A",
    "muted": "#5F7391",
    "teal": "#18CDBE",
    "blue": "#2488FF",
    "blue_soft": "#6EB8FF",
    "slate": "#7187A4",
    "rose": "#FF6B86",
}
CHART_COLOR_SEQUENCE = [
    BRAND_COLORS["teal"],
    "#F97316",
    BRAND_COLORS["blue"],
    "#7C3AED",
    "#EF4444",
    "#22C55E",
    "#EAB308",
    "#EC4899",
    "#14B8A6",
    "#6366F1",
]
PLOTLY_FONT_FAMILY = '"Segoe UI Variable Display", Aptos, "Trebuchet MS", sans-serif'

st.set_page_config(
    page_title="AllocateIQ",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else ":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
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

def load_logo_base64() -> str | None:
    if not LOGO_PATH.exists():
        return None
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


def inject_brand_theme():
    css = """
    <style>
    :root {
        --brand-bg: __BG__;
        --brand-panel: __PANEL__;
        --brand-panel-alt: __PANEL_ALT__;
        --brand-text: __TEXT__;
        --brand-muted: __MUTED__;
        --brand-teal: __TEAL__;
        --brand-blue: __BLUE__;
        --brand-blue-soft: __BLUE_SOFT__;
        --brand-slate: __SLATE__;
        --brand-rose: __ROSE__;
        --brand-border: rgba(36, 136, 255, 0.12);
        --brand-glow: 0 20px 45px rgba(18, 39, 74, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(36, 136, 255, 0.12), transparent 24%),
            radial-gradient(circle at 92% 6%, rgba(24, 205, 190, 0.12), transparent 22%),
            linear-gradient(180deg, #f8fbff 0%, #f3f8ff 44%, #eef5ff 100%);
        color: var(--brand-text);
        font-family: "Segoe UI Variable Display", Aptos, "Trebuchet MS", sans-serif;
    }

    .stApp .main .block-container {
        max-width: 1480px;
        padding-top: 2.2rem;
        padding-right: 3rem;
        padding-bottom: 4rem;
        padding-left: 3.6rem;
    }

    .stApp [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(244, 249, 255, 0.98) 100%);
        border-right: 1px solid rgba(36, 136, 255, 0.12);
        box-shadow: 10px 0 30px rgba(18, 39, 74, 0.04);
    }

    .stApp [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
        padding-right: 1.05rem;
        padding-left: 1.05rem;
    }

    .stApp [data-testid="stSidebarCollapseButton"],
    .stApp [data-testid="collapsedControl"] {
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(36, 136, 255, 0.14);
        box-shadow: 0 10px 20px rgba(18, 39, 74, 0.08);
    }

    .stApp [data-testid="stSidebarCollapseButton"]:hover,
    .stApp [data-testid="collapsedControl"]:hover {
        background: rgba(240, 246, 255, 1);
    }

    .stApp [data-testid="stSidebar"] * {
        color: var(--brand-text);
    }

    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: var(--brand-text);
        font-family: Bahnschrift, "Segoe UI Variable Display", Aptos, sans-serif;
        letter-spacing: -0.03em;
    }

    .stApp p,
    .stApp label,
    .stApp .stCaption,
    .stApp small {
        color: #5b6f8c;
        line-height: 1.6;
    }

    .stApp [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(247, 251, 255, 0.98) 100%);
        border: 1px solid rgba(36, 136, 255, 0.12);
        border-radius: 20px;
        padding: 1.1rem 1.15rem;
        box-shadow: var(--brand-glow);
    }

    .stApp [data-testid="stMetricLabel"] {
        color: var(--brand-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .stApp [data-testid="stMetricValue"] {
        color: var(--brand-text);
    }

    .stApp [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(246, 250, 255, 0.98) 100%);
        border: 1px solid rgba(36, 136, 255, 0.12);
        border-radius: 22px;
        box-shadow: var(--brand-glow);
        padding: 0.35rem 0.5rem 0.7rem;
    }

    .stApp [data-testid="stDataFrame"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(246, 250, 255, 0.98) 100%);
        border: 1px solid rgba(36, 136, 255, 0.12);
        border-radius: 20px;
        overflow: hidden;
    }

    .stApp .stButton > button,
    .stApp .stDownloadButton > button {
        background: linear-gradient(90deg, var(--brand-teal) 0%, var(--brand-blue) 100%);
        color: #ffffff;
        border: 0;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 12px 26px rgba(36, 136, 255, 0.2);
        transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
    }

    .stApp .stButton > button:hover,
    .stApp .stDownloadButton > button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(36, 136, 255, 0.24);
    }

    .stApp .stNumberInput input,
    .stApp .stTextInput input,
    .stApp [data-baseweb="select"] > div,
    .stApp .stTextArea textarea {
        background: rgba(255, 255, 255, 0.96);
        color: var(--brand-text);
        border: 1px solid rgba(36, 136, 255, 0.14);
        border-radius: 14px;
    }

    .stApp [data-baseweb="radio"] > div {
        gap: 0.4rem;
    }

    .stApp [data-baseweb="radio"] label {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(36, 136, 255, 0.12);
        border-radius: 14px;
        padding: 0.55rem 0.75rem;
    }

    .stApp [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--brand-teal) 0%, var(--brand-blue) 100%);
    }

    .stApp [data-testid="stSidebar"] [data-testid="stImage"] {
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.45rem;
        border-radius: 22px;
        border: 1px solid rgba(36, 136, 255, 0.14);
        background: linear-gradient(145deg, #081022 0%, #102447 60%, #0b1a34 100%);
        box-shadow: 0 16px 34px rgba(18, 39, 74, 0.12);
    }

    .brand-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.8rem 2rem;
        margin: 0.25rem 0 1.1rem;
        border-radius: 28px;
        border: 1px solid rgba(36, 136, 255, 0.12);
        background: linear-gradient(135deg, rgba(234, 244, 255, 0.96) 0%, rgba(255, 255, 255, 0.98) 52%, rgba(236, 249, 248, 0.96) 100%);
        box-shadow: 0 24px 50px rgba(18, 39, 74, 0.08);
    }

    .brand-hero--compact {
        gap: 0.9rem;
        padding: 1rem 1.35rem;
        margin-bottom: 0.75rem;
        border-radius: 22px;
    }

    .brand-hero::before {
        content: "";
        position: absolute;
        inset: -30% auto auto -5%;
        width: 340px;
        height: 340px;
        background: radial-gradient(circle, rgba(24, 205, 190, 0.12) 0%, rgba(24, 205, 190, 0.0) 70%);
        pointer-events: none;
    }

    .brand-hero::after {
        content: "";
        position: absolute;
        right: -90px;
        bottom: -120px;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(36, 136, 255, 0.14) 0%, rgba(36, 136, 255, 0.0) 72%);
        pointer-events: none;
    }

    .brand-hero__copy,
    .brand-hero__visual {
        position: relative;
        z-index: 1;
    }

    .brand-hero__copy {
        flex: 1 1 420px;
        max-width: 680px;
    }

    .brand-hero__kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.8rem;
        margin-bottom: 1rem;
        border-radius: 999px;
        border: 1px solid rgba(24, 205, 190, 0.18);
        background: rgba(24, 205, 190, 0.08);
        color: #0f7c73;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .brand-hero__copy h1 {
        margin: 0;
        font-size: clamp(2.3rem, 5vw, 3.8rem);
        line-height: 0.95;
        color: #0f203a;
    }

    .brand-hero__copy p {
        max-width: 56rem;
        margin: 1rem 0 0;
        color: #5c6f8a;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .brand-hero__pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.25rem;
    }

    .brand-hero__pills span {
        padding: 0.5rem 0.95rem;
        border-radius: 999px;
        border: 1px solid rgba(36, 136, 255, 0.12);
        background: rgba(255, 255, 255, 0.82);
        color: #27496d;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .brand-hero__visual {
        flex: 0 1 460px;
        margin: 0 auto;
        text-align: center;
        padding: 1rem 1.2rem;
        border-radius: 24px;
        border: 1px solid rgba(36, 136, 255, 0.14);
        background: linear-gradient(145deg, #081022 0%, #102447 60%, #0b1a34 100%);
        box-shadow: 0 18px 42px rgba(18, 39, 74, 0.14);
    }

    .brand-hero__visual img {
        width: min(100%, 520px);
        filter: drop-shadow(0 18px 36px rgba(0, 0, 0, 0.24));
    }

    .brand-hero--compact .brand-hero__copy {
        max-width: 560px;
    }

    .brand-hero--compact .brand-hero__kicker {
        margin-bottom: 0.45rem;
        padding: 0.28rem 0.7rem;
        font-size: 0.72rem;
    }

    .brand-hero--compact .brand-hero__copy h1 {
        font-size: clamp(1.55rem, 3vw, 2.2rem);
        line-height: 1.02;
    }

    .brand-hero--compact .brand-hero__copy p,
    .brand-hero--compact .brand-hero__pills {
        display: none;
    }

    .brand-hero--compact .brand-hero__visual {
        flex: 0 1 250px;
    }

    .brand-hero--compact .brand-hero__visual img {
        width: min(100%, 250px);
    }

    .brand-note {
        margin: 0.35rem 0 1.6rem;
        padding: 1rem 1.15rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(36, 136, 255, 0.12);
        color: #52657f;
        box-shadow: var(--brand-glow);
    }

    @media (max-width: 900px) {
        .stApp .main .block-container {
            padding-top: 1.35rem;
            padding-right: 1rem;
            padding-left: 1rem;
        }

        .brand-hero {
            padding: 1.35rem 1.2rem;
        }

        .brand-hero__visual {
            order: -1;
            flex-basis: 100%;
        }

        .brand-hero__copy h1 {
            font-size: clamp(2rem, 10vw, 3rem);
        }

        .brand-hero--compact .brand-hero__visual {
            flex-basis: 100%;
        }

        .brand-hero--compact .brand-hero__visual img {
            width: min(100%, 220px);
        }
    }
    </style>
    """

    replacements = {
        "__BG__": BRAND_COLORS["bg"],
        "__PANEL__": BRAND_COLORS["panel"],
        "__PANEL_ALT__": BRAND_COLORS["panel_alt"],
        "__TEXT__": BRAND_COLORS["text"],
        "__MUTED__": BRAND_COLORS["muted"],
        "__TEAL__": BRAND_COLORS["teal"],
        "__BLUE__": BRAND_COLORS["blue"],
        "__BLUE_SOFT__": BRAND_COLORS["blue_soft"],
        "__SLATE__": BRAND_COLORS["slate"],
        "__ROSE__": BRAND_COLORS["rose"],
    }
    for token, value in replacements.items():
        css = css.replace(token, value)

    st.markdown(css, unsafe_allow_html=True)


def render_app_hero(*, compact: bool = False):
    logo_base64 = load_logo_base64()
    logo_markup = ""
    if logo_base64:
        logo_markup = f'<img src="data:image/png;base64,{logo_base64}" alt="AllocateIQ logo">'
    hero_class = "brand-hero brand-hero--compact" if compact else "brand-hero"

    st.markdown(
        f"""
        <section class="{hero_class}">
            <div class="brand-hero__copy">
                <div class="brand-hero__kicker">AllocateIQ</div>
                <h1>Smart Portfolio Builder</h1>
                <p>
                    A sleek portfolio intelligence workspace for turning your risk profile into an optimized equity allocation
                    with explainable signals, historical context, and a sharper market view.
                </p>
                <div class="brand-hero__pills">
                    <span>Allocate smarter</span>
                    <span>Optimize risk</span>
                    <span>Explain decisions</span>
                </div>
            </div>
            <div class="brand-hero__visual">
                {logo_markup}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def questionnaire_is_complete() -> bool:
    return all(st.session_state.get(key) is not None for key, _, _ in QUESTIONNAIRE)


def apply_brand_chart_layout(
    fig: go.Figure,
    *,
    height: int,
    margin: dict | None = None,
) -> go.Figure:
    title_text = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    layout_updates = dict(
        template="plotly_white",
        height=height,
        margin=margin or dict(l=28, r=24, t=120, b=32),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#F7FAFE",
        font=dict(color=BRAND_COLORS["text"], family=PLOTLY_FONT_FAMILY),
        colorway=CHART_COLOR_SEQUENCE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#4E6481"),
        ),
        legend_title_text="",
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color=BRAND_COLORS["text"],
            bordercolor="rgba(36, 136, 255, 0.18)",
        ),
        coloraxis_colorbar=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            outlinecolor="rgba(36, 136, 255, 0.18)",
            tickfont=dict(color="#4E6481"),
            title_font=dict(color=BRAND_COLORS["text"]),
        ),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    if title_text:
        layout_updates["title"] = dict(
            text=title_text,
            font=dict(size=20, color=BRAND_COLORS["text"]),
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(b=18),
        )

    fig.update_layout(**layout_updates)
    if not title_text:
        # Plotly leaves an empty title object when update_layout(title=None) is used,
        # which can render as "undefined" in indicator charts.
        fig.layout.title = None
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(153, 173, 199, 0.22)",
        linecolor="rgba(153, 173, 199, 0.34)",
        tickfont=dict(color="#52657F"),
        title_font=dict(color="#314867"),
        zeroline=False,
        showline=True,
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(153, 173, 199, 0.22)",
        linecolor="rgba(153, 173, 199, 0.34)",
        tickfont=dict(color="#52657F"),
        title_font=dict(color="#314867"),
        zeroline=False,
        showline=True,
        automargin=True,
    )
    return fig


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
    y_max = top["Weight (%)"].max() * 1.22 if not top.empty else 5
    fig = px.bar(
        top,
        x="Holding",
        y="Weight (%)",
        color="Holding",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
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
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
        marker_line=dict(color="rgba(255, 255, 255, 0.85)", width=1.0),
    )
    fig.update_layout(xaxis_title="", showlegend=False)
    fig.update_xaxes(tickangle=-32)
    fig.update_yaxes(range=[0, y_max])
    return apply_brand_chart_layout(fig, height=420, margin=dict(l=28, r=22, t=90, b=92))


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
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
    )
    fig.update_traces(
        textfont_color="#FFFFFF",
        marker=dict(line=dict(color="#FFFFFF", width=1.6)),
        hovertemplate="%{label}<br>%{value:.1f}% of portfolio<extra></extra>",
    )
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(color="#4E6481"),
        )
    )
    fig = apply_brand_chart_layout(fig, height=420, margin=dict(l=20, r=120, t=90, b=24))
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(color="#4E6481"),
        )
    )
    return fig


def build_sector_chart(sector_df: pd.DataFrame) -> go.Figure:
    chart_df = sector_df.assign(weight_pct=sector_df["weight"] * 100)
    y_max = chart_df["weight_pct"].max() * 1.2 if not chart_df.empty else 5
    fig = px.bar(
        chart_df,
        x="sector",
        y="weight_pct",
        color="sector",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        text="weight_pct",
        title="Sector exposure",
        labels={"sector": "Sector", "weight_pct": "Weight (%)"},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
        marker_line=dict(color="rgba(255, 255, 255, 0.85)", width=1.0),
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-15)
    fig.update_yaxes(range=[0, y_max])
    return apply_brand_chart_layout(fig, height=380, margin=dict(l=28, r=22, t=90, b=70))


def build_risk_gauge(volatility: float, risk_score: int) -> go.Figure:
    threshold = MAX_VOLATILITY_MAP[risk_score] * 100
    max_axis = max(35.0, threshold * 1.25)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=volatility * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"color": BRAND_COLORS["text"]}},
            title={"text": "Expected volatility", "font": {"size": 20, "color": BRAND_COLORS["text"]}},
            gauge={
                "axis": {"range": [0, max_axis], "tickcolor": "#7B90AA"},
                "bar": {"color": BRAND_COLORS["blue"]},
                "bgcolor": "rgba(255, 255, 255, 0)",
                "bordercolor": "rgba(36, 136, 255, 0.16)",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, max_axis * 0.4], "color": "rgba(34, 197, 94, 0.16)"},
                    {"range": [max_axis * 0.4, max_axis * 0.7], "color": "rgba(234, 179, 8, 0.18)"},
                    {"range": [max_axis * 0.7, max_axis], "color": "rgba(239, 68, 68, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": BRAND_COLORS["rose"], "width": 4},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    return apply_brand_chart_layout(fig, height=320, margin=dict(l=24, r=24, t=84, b=18))


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
        color_continuous_scale=[
            [0.0, "#FFD166"],
            [0.45, "#F97316"],
            [1.0, "#7C3AED"],
        ],
        title="Risk vs return",
        labels={"Volatility": "Expected volatility (%)", "Return": "Expected return (%)"},
        opacity=0.45,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0, color="rgba(255,255,255,0)")))

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
        ("Optimized portfolio", optimized_metrics, BRAND_COLORS["teal"]),
        ("Equal-weight portfolio", equal_metrics, "#F97316"),
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
                marker=dict(size=14, color=color, line=dict(width=2, color=BRAND_COLORS["panel_alt"])),
                textfont=dict(color=BRAND_COLORS["text"]),
                hovertemplate=f"{name}<br>Return: %{{y:.1f}}%<br>Volatility: %{{x:.1f}}%<extra></extra>",
            )
        )

    return apply_brand_chart_layout(fig, height=420, margin=dict(l=28, r=24, t=110, b=30))


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
        "Optimized portfolio": BRAND_COLORS["teal"],
        "Equal-weight portfolio": "#F97316",
        "Market benchmark": "#7C3AED",
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
    )
    return apply_brand_chart_layout(fig, height=420, margin=dict(l=28, r=24, t=100, b=30))


def build_drawdown_chart(value_frame: pd.DataFrame) -> go.Figure:
    drawdown = value_frame.div(value_frame.cummax()).sub(1.0) * 100
    fig = go.Figure()
    colors = {
        "Optimized portfolio": BRAND_COLORS["teal"],
        "Equal-weight portfolio": "#F97316",
        "Market benchmark": "#7C3AED",
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
    )
    return apply_brand_chart_layout(fig, height=360, margin=dict(l=28, r=24, t=100, b=30))


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
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        st.caption("Use the arrow at the top of the sidebar to collapse or reopen this control panel anytime.")
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
    inject_brand_theme()
    render_app_hero(compact=questionnaire_is_complete())

    with st.spinner("Loading market and feature data..."):
        raw_df, prices, market_caps, sectors = load_base_data()
        _, ml_pred = train_and_predict()

    asset_returns = build_asset_return_panel(raw_df, prices.columns.tolist())

    st.markdown(
        """
        <div class="brand-note">
            Answer the questionnaire in the left panel to generate a portfolio that matches your risk comfort,
            shows what you would own, and explains how the recommendation compares with simpler alternatives.
        </div>
        """,
        unsafe_allow_html=True,
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
