"""Pull raw data -> data/raw.

Real:    daily stock OHLC, hourly futures, BTC (yfinance)
Stubbed: xStocks token trades (Dune on-chain + Kraken CEX) -- see TODOs

Token trade schema (every source must produce this):
    ts        datetime64[ns, UTC]
    ticker    underlying ticker, e.g. "TSLA"
    price     float, USD per share-equivalent
    size_usd  float
    venue     str, e.g. "solana_dex", "kraken"
"""

import os

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from weekend_tokens.config import FUTURES, RAW, START, TICKERS

load_dotenv()

TOKEN_COLS = ["ts", "ticker", "price", "size_usd", "venue"]


def pull_stocks() -> pd.DataFrame:
    """Daily open/close for underlyings. Open = 09:30 auction, close = 16:00."""
    df = yf.download(list(TICKERS), start=START, interval="1d", auto_adjust=False, progress=False)
    out = (
        df[["Open", "Close"]]
        .stack(level=1, future_stack=True)
        .rename_axis(["date", "ticker"])
        .reset_index()
        .rename(columns={"Open": "open", "Close": "close"})
    )
    out.to_parquet(RAW / "stocks_daily.parquet", index=False)
    return out


def pull_futures() -> pd.DataFrame:
    """Hourly futures bars. yfinance caps 1h history at ~730 days, enough here."""
    syms = sorted(set(FUTURES.values()))
    df = yf.download(syms, start=START, interval="1h", progress=False)
    out = (
        df[["Close"]]
        .stack(level=1, future_stack=True)
        .rename_axis(["ts", "symbol"])
        .reset_index()
        .rename(columns={"Close": "price"})
    )
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out.to_parquet(RAW / "futures_1h.parquet", index=False)
    return out


def pull_btc() -> pd.DataFrame:
    """BTC hourly, a crypto-sentiment control for weekend token moves."""
    df = yf.download("BTC-USD", start=START, interval="1h", progress=False)
    out = df[["Close"]].droplevel(1, axis=1).rename(columns={"Close": "price"}).reset_index()
    out = out.rename(columns={out.columns[0]: "ts"})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out.to_parquet(RAW / "btc_1h.parquet", index=False)
    return out


def pull_xstocks_dune() -> pd.DataFrame:
    """On-chain xStocks swaps on Solana via a saved Dune query.

    TODO:
      1. Write a Dune query over Solana DEX trades (dex_solana.trades) filtered to the
         xStocks mint addresses for TICKERS. Confirm mints from xstocks.com / Solscan.
      2. Price each swap in USD (token_bought_amount vs USDC/SOL leg; amount_usd column).
      3. Save it, put the id in .env as DUNE_XSTOCKS_QUERY_ID.
      4. Map mint -> underlying ticker below.
    """
    key, qid = os.getenv("DUNE_API_KEY"), os.getenv("DUNE_XSTOCKS_QUERY_ID")
    if not key or not qid:
        print("skip dune: set DUNE_API_KEY and DUNE_XSTOCKS_QUERY_ID in .env")
        return pd.DataFrame(columns=TOKEN_COLS)

    url = f"https://api.dune.com/api/v1/query/{qid}/results/csv"
    r = requests.get(url, headers={"X-Dune-API-Key": key}, timeout=120)
    r.raise_for_status()
    raw = pd.read_csv(pd.io.common.StringIO(r.text))

    # TODO: adjust column names to your query's output
    mint_to_ticker: dict[str, str] = {}  # e.g. {"XsDoVfqe...": "TSLA"}
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(raw["block_time"], utc=True),
            "ticker": raw["token_mint"].map(mint_to_ticker),
            "price": raw["price_usd"],
            "size_usd": raw["amount_usd"],
            "venue": "solana_dex",
        }
    ).dropna(subset=["ticker"])
    return out


def pull_xstocks_kraken() -> pd.DataFrame:
    """Kraken public trades for xStocks pairs (cross-check vs on-chain).

    TODO:
      1. Confirm pair names via https://api.kraken.com/0/public/AssetPairs (e.g. "TSLAxUSD").
      2. Page /0/public/Trades with `since` from START; respect rate limits.
      3. Kraken may not serve xStocks to US IPs -- run from where it's available or skip.
    """
    print("skip kraken: not implemented (see TODO)")
    return pd.DataFrame(columns=TOKEN_COLS)


def pull_tokens() -> pd.DataFrame:
    parts = [p for p in (pull_xstocks_dune(), pull_xstocks_kraken()) if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TOKEN_COLS)
    out.to_parquet(RAW / "token_trades.parquet", index=False)
    print(f"token trades: {len(out):,}")
    return out


def main() -> None:
    print(f"stocks: {len(pull_stocks()):,} rows")
    print(f"futures: {len(pull_futures()):,} rows")
    print(f"btc: {len(pull_btc()):,} rows")
    pull_tokens()


if __name__ == "__main__":
    main()
