"""
main.py
Runs the full pipeline: data -> features -> regime detection -> optimization
-> walk-forward backtest -> results, and saves all plots + a results table.

Usage:
    python main.py                 # downloads fresh data via yfinance
    python main.py --use-cache     # loads data_cache.csv instead (offline runs)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data_loader
from src.backtest import run_walk_forward_backtest, summarize_results

OUT_DIR = "outputs"
REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}


def plot_regimes(prices_aligned, regime_labels, out_path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(prices_aligned.index, prices_aligned["stocks"], color="black", lw=1.1)
    ax.set_ylabel("Stocks (proxy) price")
    ax.set_title("Detected Market Regimes (out-of-sample, walk-forward)")

    labels = regime_labels.reindex(prices_aligned.index)
    start = labels.index[0]
    current = labels.iloc[0]
    for i in range(1, len(labels)):
        if labels.iloc[i] != current or i == len(labels) - 1:
            end = labels.index[i]
            ax.axvspan(start, end, color=REGIME_COLORS.get(current, "grey"), alpha=0.25)
            start = labels.index[i]
            current = labels.iloc[i]

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_equity_curves(results, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    curves = {
        "HMM Regime-Switching": results["strategy_returns"],
        "Static 60/40": results["static_60_40_returns"],
        "Equal-Weight": results["equal_weight_returns"],
    }
    for name, r in curves.items():
        eq = (1 + r).cumprod()
        axes[0].plot(eq.index, eq.values, label=name, lw=1.4)
    axes[0].set_title("Out-of-Sample Equity Curves")
    axes[0].legend()
    axes[0].set_ylabel("Growth of 1")

    strat_eq = (1 + results["strategy_returns"]).cumprod()
    dd = strat_eq / strat_eq.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    axes[1].set_ylabel("Strategy Drawdown")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(use_cache=False):
    os.makedirs(OUT_DIR, exist_ok=True)

    if use_cache:
        prices, vix = data_loader.load_cache()
    else:
        prices, vix = data_loader.download_prices()
        data_loader.save_cache(prices, vix)

    results = run_walk_forward_backtest(prices, vix)

    plot_regimes(results["prices_aligned"], results["regime_labels"], f"{OUT_DIR}/regime_overlay.png")
    plot_equity_curves(results, f"{OUT_DIR}/equity_curves.png")

    summary = summarize_results(results)
    summary.to_csv(f"{OUT_DIR}/performance_summary.csv")

    trans_df = pd.DataFrame(
        results["transition_matrix"],
        index=[f"from_state_{i}" for i in range(results["transition_matrix"].shape[0])],
        columns=[f"to_state_{i}" for i in range(results["transition_matrix"].shape[0])],
    )
    trans_df.to_csv(f"{OUT_DIR}/transition_matrix_last_fold.csv")

    print("\n=== PERFORMANCE SUMMARY (out-of-sample, net of transaction costs) ===")
    print(summary.round(3).to_string())
    print(f"\nPlots + tables saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true", help="load data_cache.csv instead of hitting yfinance")
    args = parser.parse_args()
    main(use_cache=args.use_cache)
