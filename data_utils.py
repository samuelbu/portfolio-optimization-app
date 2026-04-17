from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path('data/features.parquet')


def load_features(filepath: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the engineered parquet dataset and apply light standardization."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'Could not find dataset at {path.resolve()}')

    df = pd.read_parquet(path).copy()
    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].astype(str)
    df['sector'] = df['sector'].astype(str)
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)

    # WRDS CRSP prices can be negative by convention for bid/ask midpoint flags.
    if 'prc' in df.columns:
        df['prc'] = pd.to_numeric(df['prc'], errors='coerce').abs()

    return df


def get_available_tickers(df: pd.DataFrame) -> list[str]:
    return sorted(df['ticker'].dropna().unique().tolist())


def latest_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ['ticker', 'sector', 'mkt_cap', 'prc', 'date'] if c in df.columns]
    meta = (
        df.sort_values('date')
          .groupby('ticker', as_index=False)
          .tail(1)[cols]
          .sort_values('ticker')
          .reset_index(drop=True)
    )
    return meta


def build_price_matrix(df: pd.DataFrame, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Pivot the parquet into a date x ticker adjusted price-like matrix."""
    work = df.copy()
    if tickers is not None:
        tickers = list(tickers)
        work = work[work['ticker'].isin(tickers)].copy()

    prices = (
        work.pivot_table(index='date', columns='ticker', values='prc', aggfunc='last')
            .sort_index()
            .dropna(axis=1, how='all')
    )

    # Keep only assets with a complete history after forward/back filling short gaps.
    prices = prices.ffill().bfill().dropna(axis=1, how='any')
    return prices


def latest_market_caps(df: pd.DataFrame, tickers: Iterable[str] | None = None) -> pd.Series:
    meta = latest_metadata(df).set_index('ticker')
    if tickers is not None:
        meta = meta.loc[list(tickers)]
    if 'mkt_cap' not in meta.columns:
        raise ValueError('mkt_cap column not found in features data.')
    return pd.to_numeric(meta['mkt_cap'], errors='coerce').fillna(0.0)


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        'n_rows': int(len(df)),
        'n_tickers': int(df['ticker'].nunique()),
        'start_date': df['date'].min().date().isoformat(),
        'end_date': df['date'].max().date().isoformat(),
    }
