"""Optional: walk-forward out-of-sample comparison -> reports/model_results.txt

Train on all weekends before W, predict W, roll forward. Models:
    baseline   OLS gap ~ fut_sun
    linear     OLS gap ~ token_wknd + fut_sun
    gbm        LightGBM on token_wknd, fut_sun, btc_wknd, log volume, ticker

A model only "wins" if it beats the futures baseline out of sample.
Small trees + strong regularization: there are few weekends, overfitting is the default.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from weekend_tokens.config import REPORTS
from weekend_tokens.hypothesis import load_usable

MIN_TRAIN_WEEKENDS = 15
GBM_FEATURES = ["token_wknd", "fut_sun", "btc_wknd", "log_vol", "ticker_code"]
GBM_PARAMS = dict(
    n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2,
    min_child_samples=10, reg_lambda=5.0, subsample=0.8, subsample_freq=1, verbose=-1,
)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        log_vol=np.log1p(df["vol_usd"]),
        ticker_code=df["ticker"].astype("category").cat.codes,
    )
    weekends = sorted(df["weekend"].unique())
    preds = []
    for w in weekends[MIN_TRAIN_WEEKENDS:]:
        train, test = df[df["weekend"] < w], df[df["weekend"] == w]
        base = smf.ols("gap ~ fut_sun", train).fit()
        lin = smf.ols("gap ~ token_wknd + fut_sun", train).fit()
        gbm = lgb.LGBMRegressor(**GBM_PARAMS).fit(train[GBM_FEATURES], train["gap"])
        preds.append(
            test[["ticker", "weekend", "gap"]].assign(
                baseline=base.predict(test),
                linear=lin.predict(test),
                gbm=gbm.predict(test[GBM_FEATURES]),
            )
        )
    return pd.concat(preds, ignore_index=True)


def score(p: pd.DataFrame) -> str:
    lines = [f"out-of-sample rows: {len(p)}  weekends: {p['weekend'].nunique()}", ""]
    lines.append(f"{'model':10} {'MAE (bp)':>9} {'dir acc':>8} {'R2 vs base':>11}")
    sse_base = ((p["gap"] - p["baseline"]) ** 2).sum()
    for m in ("baseline", "linear", "gbm"):
        err = p["gap"] - p[m]
        mae = err.abs().mean() * 1e4
        acc = (np.sign(p[m]) == np.sign(p["gap"])).mean()
        r2 = 1 - (err**2).sum() / sse_base
        lines.append(f"{m:10} {mae:9.1f} {acc:8.2%} {r2:11.3f}")
    lines += ["", "R2 vs base > 0 means the model beats futures-only out of sample."]
    return "\n".join(lines)


def main() -> None:
    df = load_usable()
    if df["weekend"].nunique() <= MIN_TRAIN_WEEKENDS:
        print(f"need > {MIN_TRAIN_WEEKENDS} weekends, have {df['weekend'].nunique()}")
        return
    text = score(walk_forward(df))
    (REPORTS / "model_results.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
