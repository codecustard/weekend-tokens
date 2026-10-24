# weekend-tokens

**Do weekend prices of tokenized stocks predict Monday's open, beyond what index futures already tell you?**

> **TL;DR** _(fill in after running the test)_
> - Result: _reject / fail to reject H0 (p = …)_
> - Effect: _a 1% weekend token move adds … to the predicted Monday gap, after futures_
> - Out of sample: _token-aware model beats futures-only by … / does not_

---

## 1. Question

Tokenized stocks (here: [xStocks](https://xstocks.com) on Solana) trade 24/7 while the underlying
market is closed. Between **Friday 20:00 ET** and **Sunday 18:00 ET**, when index futures reopen,
tokens are essentially the only venue trading US equity exposure.

Industry reports claim weekend token moves foreshadow Monday gaps (Kaiko: same direction in 20 of 25
large swings; Binance Research: tokens captured a median 92% of Monday's move over 7 weekends).
Neither tests against futures, the obvious competitor. That's the gap this project fills.

## 2. Hypotheses

Written before looking at results.

| | Statement |
|---|---|
| **H0** | Weekend token return adds no predictive power for the Monday gap beyond Sunday futures (b1 = 0) |
| **H1** | Weekend token return carries information futures don't (b1 ≠ 0) |

```
gap = a + b1 · token_wknd + b2 · fut_sun + e
```

- `gap`: Monday open / Friday close − 1
- `token_wknd`: token price at Sunday 18:00 ET (1h VWAP) / Friday close − 1
- `fut_sun`: index futures, Friday 17:00 → first hour after Sunday reopen

Pooled OLS across tickers, standard errors **clustered by weekend** (every ticker reacts to the
same weekend news, so rows within a weekend aren't independent). Significance level 5%.

## 3. Data

| Source | What | Notes |
|---|---|---|
| Dune (Solana DEX trades) | xStocks swaps: TSLAx, NVDAx, SPYx, QQQx, AAPLx | on-chain, primary |
| Kraken | xStocks trades | CEX cross-check, _TODO_ |
| yfinance | daily stock open/close, hourly ES/NQ futures, hourly BTC | |

Sample: _N weekends × M tickers_, from mid-2025 (xStocks launch).

**Cleaning:** drop token prints >15% from the last close (bad ticks), drop weekends with fewer
than 20 token trades in the token-only window, use 1h VWAP instead of last trade, and only use
futures bars that have fully closed before each timestamp (no look-ahead).

## 4. EDA

_Add figures from `reports/figures/` and a sentence on what each shows._

- `04_trades_by_hour.png`: where token volume actually lives
- `01_token_vs_gap.png`: raw relationship
- `02_futures_vs_gap.png`: the baseline to beat

## 5. Results

### Hypothesis test

_Paste the spec table from `reports/test_results.txt`._

`03_partial_token.png` is the main test drawn as a picture: Monday gap vs token move, both with
the futures-explained part removed. A flat cloud means tokens add nothing.

### Robustness

_By ticker, by gap size (≥1% vs <1%), excluding long weekends, with a BTC control, and against
strict premarket futures._

### Out-of-sample model (optional)

Walk-forward by weekend: futures-only OLS vs token + futures OLS vs LightGBM.
_Paste `reports/model_results.txt`._

## 6. Limitations

- **Small sample.** Tokenized stocks are recent; a few dozen weekends pooled across tickers.
- **Liquidity.** Weekend token prices may reflect market-maker skew rather than informed trading.
- **Futures are an index proxy.** Single-stock news (e.g. TSLA-specific) isn't in NQ.
- **No transaction costs.** A statistical edge isn't a tradable one.
- _Add what you find._

## 7. Conclusion

_2–3 sentences._

---

## Reproduce

```bash
uv sync
cp .env.example .env      # add Dune key + query id
make all                  # collect -> clean -> features -> test -> eda -> model
make demo                 # offline: synthetic data with a planted effect
make pytest
```

```
src/weekend_tokens/
  collect.py      raw pulls (stocks, futures, BTC real; token sources TODO)
  clean.py        weekend windows + bad-tick filter
  features.py     one row per (ticker, weekend)
  hypothesis.py   H0 test + robustness -> reports/test_results.txt
  eda.py          figures -> reports/figures/
  model.py        walk-forward comparison -> reports/model_results.txt
  synthetic.py    fake data with a known answer, for testing the pipeline
notebooks/        thin walkthroughs that call src/
```

The synthetic data validates the method: with a planted token effect the test rejects H0
(p < 0.001), and with no effect it doesn't (p = 0.16).
