"""Build the panel: one row per (ticker, weekend) -> data/processed/panel.parquet.

Target
    gap           next session open / last session close - 1

Predictors (all returns vs Friday close, all strictly before Monday open)
    token_wknd    token price at futures reopen (Sun 18:00)      <- the hypothesis
    token_premkt  token price just before Monday open
    fut_sun       futures: Friday 17:00 -> first hour after Sunday reopen
    fut_premkt    futures: Friday 17:00 -> just before Monday open (strict baseline)
    btc_wknd      BTC over the same window as token_wknd (crypto sentiment control)
Liquidity
    n_trades, vol_usd   token trades in the token-only window

Look-ahead guard: yfinance labels hourly bars by START time, so a bar is only
usable once ts + 1h <= t.
"""

import numpy as np
import pandas as pd

from weekend_tokens.config import FUTURES, MIN_WEEKEND_TRADES, PROCESSED, RAW, TICKERS, TZ

VWAP_WINDOW = pd.Timedelta(hours=1)
STALE_LIMIT = pd.Timedelta(hours=6)


def token_price_at(trades: pd.DataFrame, t: pd.Timestamp) -> float:
    """VWAP of the last hour before t; fall back to last trade within STALE_LIMIT."""
    win = trades[(trades["ts"] > t - VWAP_WINDOW) & (trades["ts"] <= t)]
    if len(win):
        return float(np.average(win["price"], weights=win["size_usd"].clip(lower=1e-9)))
    prior = trades[(trades["ts"] <= t) & (trades["ts"] > t - STALE_LIMIT)]
    return float(prior["price"].iloc[-1]) if len(prior) else np.nan


def bar_price_at(bars: pd.DataFrame, t: pd.Timestamp) -> float:
    """Close of the last hourly bar that has fully finished by t."""
    done = bars[bars["ts_end"] <= t]
    return float(done["price"].iloc[-1]) if len(done) else np.nan


def bar_price_after(bars: pd.DataFrame, t: pd.Timestamp) -> float:
    """Close of the first full bar starting at/after t (first hour after reopen)."""
    nxt = bars[bars["ts"] >= t]
    return float(nxt["price"].iloc[0]) if len(nxt) else np.nan


def _prep_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price"]).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
    df["ts_end"] = df["ts"] + pd.Timedelta(hours=1)
    return df.sort_values("ts")


def build_panel(
    windows: pd.DataFrame,
    stocks: pd.DataFrame,
    tokens: pd.DataFrame,
    futures: pd.DataFrame,
    btc: pd.DataFrame,
) -> pd.DataFrame:
    stocks = stocks.assign(date=pd.to_datetime(stocks["date"]).dt.date)
    px = stocks.set_index(["ticker", "date"])
    futures = _prep_bars(futures)
    btc = _prep_bars(btc)
    tokens = tokens.sort_values("ts")

    rows = []
    for ticker in TICKERS:
        tk = tokens[tokens["ticker"] == ticker]
        fb = futures[futures["symbol"] == FUTURES[ticker]]
        for w in windows.itertuples():
            try:
                fri_close = px.loc[(ticker, w.weekend), "close"]
                mon_open = px.loc[(ticker, w.next_session), "open"]
            except KeyError:
                continue
            fri_17 = w.fri_close + pd.Timedelta(hours=1)
            pre_open = w.mon_open - pd.Timedelta(minutes=1)
            fut_fri = bar_price_at(fb, fri_17)
            btc_fri = bar_price_at(btc, w.fri_close)
            only = tk[(tk["ts"] > w.token_only) & (tk["ts"] <= w.futures_open)]

            rows.append(
                {
                    "ticker": ticker,
                    "weekend": w.weekend,
                    "long_weekend": w.long_weekend,
                    "gap": mon_open / fri_close - 1,
                    "token_wknd": token_price_at(tk, w.futures_open) / fri_close - 1,
                    "token_premkt": token_price_at(tk, pre_open) / fri_close - 1,
                    "fut_sun": bar_price_after(fb, w.futures_open) / fut_fri - 1,
                    "fut_premkt": bar_price_at(fb, pre_open) / fut_fri - 1,
                    "btc_wknd": bar_price_at(btc, w.futures_open) / btc_fri - 1,
                    "n_trades": len(only),
                    "vol_usd": float(only["size_usd"].sum()),
                }
            )
    panel = pd.DataFrame(rows)
    panel["liquid"] = panel["n_trades"] >= MIN_WEEKEND_TRADES
    return panel


def main() -> None:
    panel = build_panel(
        pd.read_parquet(PROCESSED / "windows.parquet"),
        pd.read_parquet(RAW / "stocks_daily.parquet"),
        pd.read_parquet(PROCESSED / "token_trades.parquet"),
        pd.read_parquet(RAW / "futures_1h.parquet"),
        pd.read_parquet(RAW / "btc_1h.parquet"),
    )
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    usable = panel.dropna(subset=["gap", "token_wknd", "fut_sun"]).query("liquid")
    print(f"panel rows: {len(panel)}  usable (liquid, complete): {len(usable)}")


if __name__ == "__main__":
    main()
