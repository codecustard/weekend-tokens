"""Synthetic data with a KNOWN answer, to check the pipeline end to end offline.

    news ~ N(0, 1.5%) per weekend
    gap        = news + noise
    fut_sun    = 0.6 * news + noise   (futures catch part of it at reopen)
    token_wknd = TOKEN_SIGNAL * news + noise

TOKEN_SIGNAL = 0  -> the test should NOT reject H0 (tokens add nothing)
TOKEN_SIGNAL > 0  -> the test should reject H0

Writes the same files as collect.py:  python -m weekend_tokens.synthetic [signal]
"""

import sys

import numpy as np
import pandas as pd

from weekend_tokens.config import FUTURES, RAW, TICKERS, TZ

N_WEEKS = 60


def generate(token_signal: float = 0.5, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-07-07", periods=N_WEEKS * 5)
    fridays = days[days.dayofweek == 4][:-1]
    news = pd.Series(rng.normal(0, 0.015, len(fridays)), index=fridays)

    stock_rows, token_rows = [], []
    fut_shock = {f: 0.6 * n + rng.normal(0, 0.004) for f, n in news.items()}

    for i, ticker in enumerate(TICKERS):
        close = 100.0 * (i + 1)
        for d in days:
            prev_fri = d - pd.Timedelta(days=3)
            if d.dayofweek == 0 and prev_fri in news.index:
                gap = news[prev_fri] + rng.normal(0, 0.003)
                tok = token_signal * news[prev_fri] + rng.normal(0, 0.006)
                fri_ts = pd.Timestamp(prev_fri.date(), tz=TZ) + pd.Timedelta(hours=20)
                n = rng.integers(10, 80)
                ts = fri_ts + pd.to_timedelta(np.sort(rng.uniform(0, 46, n)), unit="h")
                drift = np.linspace(0, 1, n)  # price walks toward the weekend level
                token_rows.append(
                    pd.DataFrame(
                        {
                            "ts": ts.tz_convert("UTC"),
                            "ticker": ticker,
                            "price": close * (1 + tok * drift + rng.normal(0, 0.001, n)),
                            "size_usd": rng.lognormal(6, 1, n),
                            "venue": "synthetic",
                        }
                    )
                )
            else:
                gap = rng.normal(0, 0.003)
            open_ = close * (1 + gap)
            close = open_ * (1 + rng.normal(0, 0.01))
            stock_rows.append({"date": d, "ticker": ticker, "open": open_, "close": close})

    # hourly futures: flat-ish during the week, closed Fri 17:00 -> Sun 18:00, jump at reopen
    hours = pd.date_range(
        pd.Timestamp(days[0].date(), tz=TZ), pd.Timestamp(days[-1].date(), tz=TZ), freq="1h"
    )
    open_mask = ~(
        ((hours.dayofweek == 4) & (hours.hour >= 17))
        | (hours.dayofweek == 5)
        | ((hours.dayofweek == 6) & (hours.hour < 18))
    )
    hours = hours[open_mask]
    fut_rows = []
    for sym in sorted(set(FUTURES.values())):
        level, prices = 5000.0, []
        for h in hours:
            if h.dayofweek == 6 and h.hour == 18:
                fri = (h - pd.Timedelta(days=2)).normalize().tz_localize(None)
                level *= 1 + fut_shock.get(fri, 0.0)
            level *= 1 + rng.normal(0, 0.0005)
            prices.append(level)
        fut_rows.append(pd.DataFrame({"ts": hours.tz_convert("UTC"), "symbol": sym,
                                      "price": prices}))

    all_hours = pd.date_range(hours[0], hours[-1], freq="1h")
    btc = pd.DataFrame(
        {"ts": all_hours.tz_convert("UTC"),
         "price": 60000 * np.exp(np.cumsum(rng.normal(0, 0.003, len(all_hours))))}
    )
    return {
        "stocks_daily": pd.DataFrame(stock_rows),
        "futures_1h": pd.concat(fut_rows, ignore_index=True),
        "btc_1h": btc,
        "token_trades": pd.concat(token_rows, ignore_index=True),
    }


def main() -> None:
    signal = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    for name, df in generate(signal).items():
        df.to_parquet(RAW / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows")
    print(f"synthetic data written (token_signal={signal})")


if __name__ == "__main__":
    main()
