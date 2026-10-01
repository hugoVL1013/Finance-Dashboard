"""Market-data tools the agents can call.

Agents never see or invent raw numbers: every figure they quote comes from one of
these functions. Pure computation (``compute_price_stats``) is separated from data
fetching (``fetch_history``) so it can be unit-tested without the network, and so the
price source can be swapped (yfinance today, a fallback provider later).
"""
from __future__ import annotations

import re
import time
from typing import Callable

import numpy as np
import pandas as pd

from .agent import Tool

TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")
MAX_TICKERS = 12
TRADING_DAYS = 252

FetchFn = Callable[[str, str], pd.Series]


def _clean_tickers(tickers: list[str]) -> list[str]:
    cleaned = []
    for t in tickers[:MAX_TICKERS]:
        t = str(t).strip().upper()
        if not TICKER_RE.match(t):
            raise ValueError(f"invalid ticker '{t}'")
        cleaned.append(t)
    if not cleaned:
        raise ValueError("no tickers given")
    return cleaned


def fetch_history(ticker: str, period: str = "2y", retries: int = 3) -> pd.Series:
    """Daily total-return-adjusted closes from Yahoo Finance via yfinance, with
    exponential-backoff retries (Yahoo rate-limits routinely)."""
    import yfinance as yf  # lazy: keeps the module importable without yfinance
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            close = hist["Close"].dropna() if hist is not None and "Close" in hist else None
            if close is not None and not close.empty:
                close.index = pd.to_datetime(close.index).tz_localize(None)
                return close.astype(float)
            last_err = ValueError("empty history")
        except Exception as err:  # noqa: BLE001 - yfinance raises a wide range
            last_err = err
        time.sleep(1.0 * 2 ** attempt)
    raise RuntimeError(f"no price data for {ticker}: {last_err}")


def compute_price_stats(close: pd.Series) -> dict:
    """Return/volatility/drawdown summary from a series of adjusted closes."""
    close = close.dropna()
    if len(close) < 30:
        raise ValueError(f"only {len(close)} observations; need at least 30")
    rets = close.pct_change().dropna()

    def trailing(days: int):
        return round(float(close.iloc[-1] / close.iloc[-1 - days] - 1), 4) if len(close) > days else None

    window = close.iloc[-TRADING_DAYS:]
    drawdown = window / window.cummax() - 1
    return {
        "last_date": str(close.index[-1].date()),
        "last_close": round(float(close.iloc[-1]), 4),
        "ret_1m": trailing(21),
        "ret_3m": trailing(63),
        "ret_6m": trailing(126),
        "ret_1y": trailing(252),
        "ann_vol_60d": round(float(rets.iloc[-60:].std() * np.sqrt(TRADING_DAYS)), 4),
        "max_drawdown_1y": round(float(drawdown.min()), 4),
        "pct_below_1y_high": round(float(close.iloc[-1] / window.max() - 1), 4),
        "n_obs": int(len(close)),
    }


def compute_correlations(closes: dict[str, pd.Series], lookback_days: int = 252) -> dict:
    df = pd.DataFrame(closes).dropna().pct_change().dropna().iloc[-lookback_days:]
    corr = df.corr().round(3)
    return {"lookback_days": int(len(df)), "matrix": corr.to_dict()}


def make_market_tools(fetch: FetchFn = fetch_history) -> list[Tool]:
    """Build the market-data tools around a price-fetch function."""

    def get_price_stats(tickers: list[str]) -> dict:
        out = {}
        for t in _clean_tickers(tickers):
            try:
                out[t] = compute_price_stats(fetch(t, "2y"))
            except Exception as err:  # noqa: BLE001 - report per ticker, keep the rest
                out[t] = {"error": str(err)}
        return out

    def get_correlations(tickers: list[str], lookback_days: int = 252) -> dict:
        closes = {t: fetch(t, "2y") for t in _clean_tickers(tickers)}
        if len(closes) < 2:
            raise ValueError("need at least 2 tickers")
        return compute_correlations(closes, int(lookback_days))

    def get_etf_info(ticker: str) -> dict:
        import yfinance as yf
        t = _clean_tickers([ticker])[0]
        info = yf.Ticker(t).info or {}
        keys = ["longName", "quoteType", "category", "fundFamily", "totalAssets",
                "netExpenseRatio", "annualReportExpenseRatio", "averageVolume",
                "yield", "currency", "exchange"]
        return {k: info.get(k) for k in keys if info.get(k) is not None} or {"error": "no metadata"}

    ticker_list = {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TICKERS,
                   "description": "Uppercase tickers, e.g. ['EEM', 'INDA']"}
    return [
        Tool("get_price_stats",
             "Trailing returns (1m/3m/6m/1y), 60d annualised volatility, 1y max drawdown "
             "and distance below the 1y high for each ticker, from adjusted closes.",
             {"type": "object", "properties": {"tickers": ticker_list},
              "required": ["tickers"], "additionalProperties": False},
             get_price_stats),
        Tool("get_correlations",
             "Pairwise correlation matrix of daily returns over a lookback window.",
             {"type": "object",
              "properties": {"tickers": ticker_list,
                             "lookback_days": {"type": "integer", "minimum": 30, "maximum": 504}},
              "required": ["tickers"], "additionalProperties": False},
             get_correlations),
        Tool("get_etf_info",
             "Fund metadata for one ETF: name, category, AUM, expense ratio, average volume, "
             "yield. Fields may be missing; Yahoo is sometimes rate-limited.",
             {"type": "object", "properties": {"ticker": {"type": "string"}},
              "required": ["ticker"], "additionalProperties": False},
             get_etf_info),
    ]
