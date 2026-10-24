import pandas as pd

from weekend_tokens.clean import build_windows, clean_tokens
from weekend_tokens.features import bar_price_at, build_panel
from weekend_tokens.hypothesis import SPECS, MAIN, fit
from weekend_tokens.synthetic import generate


def _panel(signal: float) -> pd.DataFrame:
    d = generate(signal, seed=1)
    w = build_windows(d["stocks_daily"])
    t = clean_tokens(d["token_trades"], d["stocks_daily"])
    p = build_panel(w, d["stocks_daily"], t, d["futures_1h"], d["btc_1h"])
    return p.dropna(subset=["gap", "token_wknd", "fut_sun"]).query("liquid")


def test_windows_are_weekends():
    days = pd.bdate_range("2026-01-05", "2026-01-30")
    stocks = pd.DataFrame({"date": days, "ticker": "X", "open": 1.0, "close": 1.0})
    w = build_windows(stocks)
    assert len(w) == 3
    assert all(pd.Timestamp(d).dayofweek == 4 for d in w["weekend"])
    assert all(w["futures_open"].dt.dayofweek == 6)


def test_no_lookahead_on_hourly_bars():
    tz = "America/New_York"
    bars = pd.DataFrame({"ts": pd.date_range("2026-01-12 08:00", periods=3, freq="1h", tz=tz),
                         "price": [1.0, 2.0, 3.0]})
    bars["ts_end"] = bars["ts"] + pd.Timedelta(hours=1)
    # at 09:30 only the 08:00-09:00 bar is finished
    assert bar_price_at(bars, pd.Timestamp("2026-01-12 09:30", tz=tz)) == 1.0


def test_detects_planted_signal():
    r = fit(_panel(0.5), SPECS[MAIN])
    assert r.pvalues["token_wknd"] < 0.05


def test_null_mostly_not_rejected():
    rejections = 0
    for seed in range(5):
        d = generate(0.0, seed=seed)
        w = build_windows(d["stocks_daily"])
        t = clean_tokens(d["token_trades"], d["stocks_daily"])
        p = build_panel(w, d["stocks_daily"], t, d["futures_1h"], d["btc_1h"])
        p = p.dropna(subset=["gap", "token_wknd", "fut_sun"]).query("liquid")
        rejections += fit(p, SPECS[MAIN]).pvalues["token_wknd"] < 0.05
    assert rejections <= 2
