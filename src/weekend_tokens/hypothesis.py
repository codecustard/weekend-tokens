"""Hypothesis test -> reports/test_results.txt

H0: weekend token return adds no predictive power for the Monday gap
    beyond index futures.  gap = a + b1*token_wknd + b2*fut_sun + e,  H0: b1 = 0

Pooled OLS across tickers with standard errors clustered by weekend
(all tickers react to the same weekend news, so rows within a weekend aren't independent).
"""

import io

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from weekend_tokens.config import PROCESSED, REPORTS

SPECS = {
    "1 token only": "gap ~ token_wknd",
    "2 futures only (baseline)": "gap ~ fut_sun",
    "3 token + futures  <- main test": "gap ~ token_wknd + fut_sun",
    "4 + btc control": "gap ~ token_wknd + fut_sun + btc_wknd",
    "5 strict: + premarket futures": "gap ~ token_wknd + fut_premkt",
}
MAIN = "3 token + futures  <- main test"
BIG_GAP = 0.01


def load_usable() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    need = ["gap", "token_wknd", "fut_sun", "fut_premkt", "btc_wknd"]
    return p.dropna(subset=need).query("liquid").reset_index(drop=True)


def fit(df: pd.DataFrame, formula: str):
    groups = pd.factorize(df["weekend"])[0]
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": groups})


def hit_rate(df: pd.DataFrame, col: str) -> float:
    nz = df[(df["gap"] != 0) & (df[col] != 0)]
    return float((np.sign(nz[col]) == np.sign(nz["gap"])).mean())


def run(df: pd.DataFrame) -> str:
    out = io.StringIO()
    w = lambda s="": print(s, file=out)  # noqa: E731

    w(f"rows: {len(df)}  weekends: {df['weekend'].nunique()}  tickers: {df['ticker'].nunique()}")
    w()
    w(f"{'spec':34} {'b_token':>9} {'p_token':>9} {'adj_R2':>8}")
    for name, f in SPECS.items():
        r = fit(df, f)
        b = r.params.get("token_wknd", np.nan)
        p = r.pvalues.get("token_wknd", np.nan)
        w(f"{name:34} {b:9.3f} {p:9.4f} {r.rsquared_adj:8.3f}")

    main = fit(df, SPECS[MAIN])
    w()
    w("=== main test ===")
    w(main.summary().as_text())
    verdict = "REJECT H0" if main.pvalues["token_wknd"] < 0.05 else "fail to reject H0"
    w(f"\nH0: b_token = 0 -> p = {main.pvalues['token_wknd']:.4f} -> {verdict} at 5%")

    w("\n=== directional hit rate (sign matches Monday gap) ===")
    big = df[df["gap"].abs() >= BIG_GAP]
    w(f"{'':12} {'all':>8} {f'|gap|>={BIG_GAP:.0%}':>12}")
    for col in ("token_wknd", "fut_sun", "fut_premkt"):
        w(f"{col:12} {hit_rate(df, col):8.2%} {hit_rate(big, col):12.2%}")
    w(f"(big-gap n = {len(big)})")

    w("\n=== robustness: main spec by subset ===")
    subsets = {
        "normal weekends": df[~df["long_weekend"]],
        f"|gap| >= {BIG_GAP:.0%}": big,
        f"|gap| <  {BIG_GAP:.0%}": df[df["gap"].abs() < BIG_GAP],
        **{f"ticker {t}": g for t, g in df.groupby("ticker")},
    }
    for name, sub in subsets.items():
        if sub["weekend"].nunique() < 8:
            w(f"{name:18} n={len(sub):4}  (too few weekends)")
            continue
        r = fit(sub, SPECS[MAIN])
        w(f"{name:18} n={len(sub):4}  b={r.params['token_wknd']:.3f}  "
          f"p={r.pvalues['token_wknd']:.4f}")
    return out.getvalue()


def main() -> None:
    text = run(load_usable())
    (REPORTS / "test_results.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
