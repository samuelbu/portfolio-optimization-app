from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq


def load_features(filepath: str = "data/features.parquet") -> pd.DataFrame:
    """Load feature parquet safely across environments."""
    table = pq.read_table(filepath)
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_prices_and_market_caps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build adjusted price panel, latest market caps, and sector mapping from feature data."""
    prices = (
        df.pivot(index="date", columns="ticker", values="prc")
        .sort_index()
        .dropna(axis=1, how="all")
    )
    prices = prices.ffill().dropna(how="all")

    latest_rows = (
        df.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")
    )

    market_caps = latest_rows["mkt_cap"].astype(float)
    sectors = latest_rows["sector"].astype(str)

    common = [c for c in prices.columns if c in market_caps.index]
    return prices[common].astype(float), market_caps.loc[common], sectors.loc[common]
