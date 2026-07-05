# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

A regime-switching portfolio strategy that detects the market's hidden state
(Bull / Bear / Crisis) with a Hidden Markov Model and reallocates between
stocks, bonds, and gold using convex optimization — validated with walk-forward
backtesting so the results reflect what would have been knowable in real time.

## Project structure

```
quant_regime_project/
├── main.py                  # Runs the full pipeline end-to-end
├── requirements.txt
├── src/
│   ├── data_loader.py        # Pulls prices + India VIX via yfinance
│   ├── features.py           # Momentum / volatility feature engineering
│   ├── regime_hmm.py         # Gaussian HMM fit + state-to-label mapping
│   ├── portfolio_opt.py      # cvxpy regime-conditional optimizer
│   ├── backtest.py           # Walk-forward harness + transaction costs
│   └── metrics.py            # Sharpe, Sortino, max drawdown, Calmar, turnover
└── outputs/                  # Generated plots + result tables land here
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This downloads ~10 years of daily data via `yfinance`, runs the full
data → features → regime detection → optimization → backtest pipeline, and
writes to `outputs/`:
- `regime_overlay.png` — price chart with detected Bull/Bear/Crisis bands
- `equity_curves.png` — strategy vs. benchmarks, with a drawdown panel
- `performance_summary.csv` — Sharpe/Sortino/Max DD/Calmar/turnover table
- `transition_matrix_last_fold.csv` — regime transition probabilities

If you're offline or yfinance is rate-limited, run once with a live
connection to build a cache, then reuse it:
```bash
python -c "from src import data_loader; p,v = data_loader.download_prices(); data_loader.save_cache(p,v)"
python main.py --use-cache
```

**Note on the bond ticker:** `src/data_loader.py` defaults to a placeholder
for the "bonds" leg. NSE-listed gilt ETFs can have short/patchy Yahoo Finance
history. If the run errors out on insufficient bond history, open
`data_loader.py` and swap in `LICNETFGSEC.NS` or `SETFGILT.NS` (both noted
inline in the file), then re-run.

## Key decisions

**Why 3 regimes?** The project targets an interpretable Bull/Bear/Crisis
framing that maps directly onto the optimizer's three objective functions
(max-Sharpe, balanced, min-variance). More states start splitting "Bear" into
statistically-real-but-not-actionable sub-states without adding portfolio
value; fewer than 3 can't separate ordinary drawdowns from true crises.

**Why these features?** Momentum (21d, 63d) captures direction; realized
volatility (21d, 63d) is the single most reliable crisis signal in practice;
India VIX level and its 5-day change add a market-implied (forward-looking,
but still causal as of "today") fear gauge that price-based features alone
tend to lag. All are computed with rolling/expanding windows only — nothing
in `features.py` peeks forward.

**How lookahead bias is avoided (the hard part of this project):**
1. Feature *construction* only ever uses `.rolling()`/`.pct_change()` — no
   full-sample statistics.
2. Feature *scaling* (z-scoring) is fit fresh inside every walk-forward
   training window and only ever applied — never refit — to that window's
   test segment (`features.expanding_zscore_fit` / `apply_zscore`).
3. The HMM itself is refit from scratch inside each training window
   (`backtest.run_walk_forward_backtest`) — it never sees test-period data
   during fitting.
4. Portfolio weights for day *t* are computed from asset mean/covariance
   estimated over the trailing 252 days *ending before t*
   (`backtest.estimate_mu_sigma`), and that day's realized return is only
   ever applied to the weights decided using information available the
   day before.
5. Every rebalance (a regime change) pays a 5–10 bps transaction cost, so
   the reported Sharpe/Calmar numbers are net of trading friction, not
   inflated by regime flips traded for free.

**Walk-forward setup:** expanding training window (starts at ~3 years of
history), rolling 6-month held-out test windows, refit at each step — see
`backtest.expanding_walk_forward_splits`.

**Benchmarks:** static 60/40 (stocks/bonds) and equal-weight across all three
assets, evaluated over the identical out-of-sample dates as the strategy so
the comparison is apples-to-apples.

## Reproducing results

Everything is seeded (`random_state=42` in the HMM). Given the same date
range and tickers, `python main.py` will reproduce the same regime labels,
weights, and performance table.

## Checklist against the project brief

- [x] HMM-based Bull/Bear/Crisis classifier, no manual labeling (`regime_hmm.py`)
- [x] Regime-conditional cvxpy optimization, different objective per regime (`portfolio_opt.py`)
- [x] Walk-forward validation, HMM refit only on past data each step (`backtest.py`)
- [x] Transaction costs (5–10 bps) explicitly modeled (`TXN_COST_BPS` in `backtest.py`)
- [x] Compared vs static 60/40 and equal-weight on Sharpe, Max DD, Calmar (`metrics.py`, `main.py`)
- [x] Transition probability matrix exported (`transition_matrix_last_fold.csv`)
