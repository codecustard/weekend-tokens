"""EDA figures -> reports/figures/*.png

01 token_vs_gap      weekend token return vs Monday gap (the raw relationship)
02 futures_vs_gap    same for Sunday futures (the baseline to beat)
03 partial_token     gap residual after futures vs token residual after futures
                     (this IS the main test, drawn: a flat cloud means b1 = 0)
04 trades_by_hour    token trade count by hour of week (where does volume live?)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from weekend_tokens.config import FIGURES, PROCESSED
from weekend_tokens.hypothesis import load_usable

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
    }
)


def _pct(ax) -> None:
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")


def scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, fname: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x], df[y], s=28, color=SERIES, alpha=0.6, edgecolor=SURFACE, linewidth=1)
    lim = np.nanmax(np.abs(df[[x, y]].to_numpy())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=1, ls="--", label="y = x")
    b, a = np.polyfit(df[x], df[y], 1)
    xs = np.linspace(-lim, lim, 2)
    ax.plot(xs, a + b * xs, color=INK, lw=2, label=f"fit, slope {b:.2f}")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(-lim, lim), ylim=(-lim, lim))
    ax.legend(frameon=False, loc="upper left")
    _pct(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=150)
    plt.close(fig)


def partial_plot(df: pd.DataFrame) -> None:
    df = df.assign(
        gap_res=smf.ols("gap ~ fut_sun", df).fit().resid,
        tok_res=smf.ols("token_wknd ~ fut_sun", df).fit().resid,
    )
    scatter(
        df, "tok_res", "gap_res",
        "What tokens add beyond futures",
        "token weekend return, unexplained by futures",
        "Monday gap, unexplained by futures",
        "03_partial_token.png",
    )


def trades_by_hour() -> None:
    t = pd.read_parquet(PROCESSED / "token_trades.parquet")
    if t.empty:
        print("skip trades_by_hour: no token trades yet")
        return
    how = t["ts"].dt.dayofweek * 24 + t["ts"].dt.hour
    counts = how.value_counts().reindex(range(168), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axvspan(4 * 24 + 20, 6 * 24 + 18, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.bar(counts.index, counts.values, width=0.85, color=SERIES, zorder=2)
    ax.annotate("token-only window", xy=(5 * 24 + 7, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points", color=INK_2, ha="center", va="bottom")
    ax.set_xticks(range(12, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set(title="Token trades by hour of week (ET)", ylabel="trades")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_trades_by_hour.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_usable()
    print(f"usable rows: {len(df)}")
    scatter(df, "token_wknd", "gap", "Weekend token move vs Monday gap",
            "token return, Fri close to Sun 18:00 ET", "Monday open gap", "01_token_vs_gap.png")
    scatter(df, "fut_sun", "gap", "Sunday futures vs Monday gap",
            "futures return, Fri 17:00 to first hour after reopen", "Monday open gap",
            "02_futures_vs_gap.png")
    partial_plot(df)
    trades_by_hour()
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
