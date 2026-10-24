"""Define weekend windows and clean token trades -> data/processed.

A "weekend" = a break between two stock sessions that contains a Saturday
(normal weekends plus long weekends from Friday/Monday holidays).

Key timestamps per weekend (all ET):
    fri_close      last session close, 16:00
    token_only     last session date 20:00   -- overnight/extended venues close
    futures_open   Sunday 18:00              -- index futures reopen
    mon_open       next session open, 09:30
"""

import pandas as pd

from weekend_tokens.config import (
    MAX_ABS_PREMIUM,
    MONDAY_OPEN,
    PROCESSED,
    RAW,
    TZ,
)


def _at(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return pd.Timestamp(day.date(), tz=TZ) + pd.Timedelta(hours=h, minutes=m)


def build_windows(stocks: pd.DataFrame) -> pd.DataFrame:
    """One row per weekend, from the trading calendar implied by stock data."""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(stocks["date"]).dt.normalize().unique()))
    rows = []
    for d0, d1 in zip(days[:-1], days[1:]):
        span = pd.date_range(d0 + pd.Timedelta(days=1), d1 - pd.Timedelta(days=1))
        sats = [d for d in span if d.dayofweek == 5]
        suns = [d for d in span if d.dayofweek == 6]
        if not sats or not suns:
            continue  # mid-week holiday, not a weekend
        rows.append(
            {
                "weekend": d0.date(),
                "next_session": d1.date(),
                "fri_close": _at(d0, "16:00"),
                "token_only": _at(d0, "20:00"),
                "futures_open": _at(suns[0], "18:00"),
                "mon_open": _at(d1, MONDAY_OPEN),
                "long_weekend": (d1 - d0).days > 3,
            }
        )
    return pd.DataFrame(rows)


def clean_tokens(tokens: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """Drop bad ticks: non-positive prices, and prints far from the last stock close."""
    t = tokens.dropna(subset=["price"]).query("price > 0").copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True).dt.tz_convert(TZ).astype(f"datetime64[ns, {TZ}]")
    t = t.sort_values("ts")

    # reference = most recent stock close before each trade
    s = stocks[["date", "ticker", "close"]].copy()
    s["ref_ts"] = (
        pd.to_datetime(s["date"]).dt.tz_localize(TZ) + pd.Timedelta(hours=16)
    ).astype(f"datetime64[ns, {TZ}]")
    s = s.sort_values("ref_ts")
    t = pd.merge_asof(
        t, s[["ref_ts", "ticker", "close"]], left_on="ts", right_on="ref_ts",
        by="ticker", direction="backward",
    )
    t["premium"] = t["price"] / t["close"] - 1
    before = len(t)
    t = t[t["premium"].abs() <= MAX_ABS_PREMIUM]
    print(f"token trades kept {len(t):,}/{before:,} after premium filter")
    return t.drop(columns=["ref_ts"])


def main() -> None:
    stocks = pd.read_parquet(RAW / "stocks_daily.parquet")
    windows = build_windows(stocks)
    windows.to_parquet(PROCESSED / "windows.parquet", index=False)
    print(f"weekends: {len(windows)}")

    tokens = pd.read_parquet(RAW / "token_trades.parquet")
    clean_tokens(tokens, stocks).to_parquet(PROCESSED / "token_trades.parquet", index=False)


if __name__ == "__main__":
    main()
