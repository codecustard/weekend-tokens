"""Paths, tickers, and the weekend window definition."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

for p in (RAW, PROCESSED, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

# underlying ticker -> xStocks token symbol
TICKERS = {
    "TSLA": "TSLAx",
    "NVDA": "NVDAx",
    "SPY": "SPYx",
    "QQQ": "QQQx",
    "AAPL": "AAPLx",
}

# futures proxy per underlying (index futures, ~23h/day Sun-Fri)
FUTURES = {"TSLA": "NQ=F", "NVDA": "NQ=F", "QQQ": "NQ=F", "AAPL": "NQ=F", "SPY": "ES=F"}

START = "2025-07-01"  # xStocks launched mid-2025

TZ = "America/New_York"
# Weekend window where tokens are the only venue trading:
#   Friday 20:00 ET (overnight ATS / extended hours close) -> Sunday 18:00 ET (futures reopen)
TOKEN_ONLY_START = ("Fri", "20:00")
FUTURES_REOPEN = ("Sun", "18:00")
MONDAY_OPEN = "09:30"

# cleaning thresholds
MIN_WEEKEND_TRADES = 20       # drop (ticker, weekend) with fewer token trades
MAX_ABS_PREMIUM = 0.15        # drop prints >15% from Friday close (bad ticks)
