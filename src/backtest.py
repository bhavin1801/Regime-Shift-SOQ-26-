"""
backtest.py
Ties everything together:
  1. Expanding walk-forward splits (never trains on future data).
  2. Re-fits the HMM + z-score scaler INSIDE each training window only.
  3. Applies regime-driven cvxpy portfolio optimization on the held-out
     (test) segment, re-optimizing only when the detected regime changes
     (this is also what makes the transaction-cost modeling meaningful).
  4. Shifts weights by one day before applying them to returns, and charges
     a transaction cost whenever the portfolio rebalances.
  5. Benchmarks against a static 60/40 (stocks/bonds) and an equal-weight
     portfolio over the identical out-of-sample period.
"""

import numpy as np
import pandas as pd

from src.features import build_features, expanding_zscore_fit, apply_zscore
from src.regime_hmm import fit_hmm, predict_states, label_states, N_STATES
from src.portfolio_opt import optimal_weights
from src import metrics

TXN_COST_BPS = 7.5   # 5-10 bps per rebalance, per the project spec
LOOKBACK_DAYS = 252  # trailing window used to estimate mu/Sigma for the optimizer


def expanding_walk_forward_splits(n_obs, n_splits=6, min_train_size=750, test_size=126):
    """Yields (train_idx, test_idx) arrays. Expanding train window, fixed-size
    sliding test window. min_train_size ~ 3y, test_size ~ 6mo by default."""
    splits = []
    start_test = min_train_size
    for _ in range(n_splits):
        end_test = start_test + test_size
        if end_test > n_obs:
            break
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, end_test)
        splits.append((train_idx, test_idx))
        start_test = end_test
    return splits


def estimate_mu_sigma(asset_returns: pd.DataFrame, as_of_idx: int, lookback=LOOKBACK_DAYS):
    """Annualized mean & covariance using ONLY data strictly before as_of_idx."""
    start = max(0, as_of_idx - lookback)
    window = asset_returns.iloc[start:as_of_idx]
    mu = window.mean().values * 252
    sigma = window.cov().values * 252
    # Tiny ridge for numerical stability in cvxpy
    sigma = sigma + np.eye(len(mu)) * 1e-6
    return mu, sigma


def run_walk_forward_backtest(prices: pd.DataFrame, vix: pd.Series, asset_cols=("stocks", "bonds", "gold")):
    asset_cols = list(asset_cols)
    asset_returns = np.log(prices[asset_cols]).diff().dropna()
    feat_raw = build_features(prices, vix, market_col="stocks")

    common_idx = feat_raw.index.intersection(asset_returns.index)
    feat_raw = feat_raw.loc[common_idx]
    asset_returns = asset_returns.loc[common_idx]
    prices_aligned = prices.loc[common_idx]

    n_obs = len(feat_raw)
    splits = expanding_walk_forward_splits(n_obs)
    if not splits:
        raise ValueError("Not enough history for the requested walk-forward window sizes.")

    all_regime_labels = pd.Series(index=feat_raw.index, dtype=object)
    strategy_returns = pd.Series(index=feat_raw.index, dtype=float)
    weights_history = pd.DataFrame(index=feat_raw.index, columns=asset_cols, dtype=float)
    last_transition_matrix = None

    for train_idx, test_idx in splits:
        train_feat = feat_raw.iloc[train_idx]
        test_feat = feat_raw.iloc[test_idx]

        # --- Fit scaler and HMM on TRAIN ONLY ---
        mu_scale, sigma_scale = expanding_zscore_fit(train_feat)
        train_scaled = apply_zscore(train_feat, mu_scale, sigma_scale)
        test_scaled = apply_zscore(test_feat, mu_scale, sigma_scale)

        model = fit_hmm(train_scaled.values, n_states=N_STATES)
        train_states = predict_states(model, train_scaled.values)
        state_label_map = label_states(train_scaled, train_states)
        last_transition_matrix = model.transmat_

        test_states = predict_states(model, test_scaled.values)
        test_labels = pd.Series(
            [state_label_map.get(s, "Bear") for s in test_states], index=test_feat.index
        )
        all_regime_labels.loc[test_labels.index] = test_labels.values

        # --- Regime-driven allocation on the held-out segment ---
        current_regime = None
        current_weights = None
        pos_map = {d: i for i, d in enumerate(feat_raw.index)}

        for date in test_feat.index:
            i = pos_map[date]  # global position for lookback slicing
            regime = test_labels.loc[date]
            rebalanced_today = False

            if regime != current_regime or current_weights is None:
                mu, sigma = estimate_mu_sigma(asset_returns, i)
                new_weights = optimal_weights(mu, sigma, regime)
                if current_weights is not None:
                    cost = TXN_COST_BPS / 1e4 * np.abs(new_weights - current_weights).sum() / 2
                else:
                    cost = TXN_COST_BPS / 1e4  # initial allocation also incurs cost
                rebalanced_today = True
                current_weights = new_weights
                current_regime = regime
            else:
                cost = 0.0

            day_return = float(np.dot(current_weights, asset_returns.loc[date, asset_cols].values))
            strategy_returns.loc[date] = day_return - (cost if rebalanced_today else 0.0)
            weights_history.loc[date] = current_weights

    strategy_returns = strategy_returns.dropna()
    weights_history = weights_history.dropna()
    all_regime_labels = all_regime_labels.dropna()

    oos_idx = strategy_returns.index
    bench_returns = asset_returns.loc[oos_idx, asset_cols]

    # Static 60/40 (60% stocks, 40% bonds, 0% gold) -- no rebalancing cost modeled
    static_weights = {"stocks": 0.6, "bonds": 0.4, "gold": 0.0}
    static_w_vec = np.array([static_weights.get(c, 0.0) for c in asset_cols])
    static_returns = bench_returns.values @ static_w_vec
    static_returns = pd.Series(static_returns, index=oos_idx)

    # Equal-weight across all assets, rebalanced daily back to equal weight (no cost modeled)
    eq_w_vec = np.ones(len(asset_cols)) / len(asset_cols)
    eq_returns = bench_returns.values @ eq_w_vec
    eq_returns = pd.Series(eq_returns, index=oos_idx)

    results = {
        "regime_labels": all_regime_labels,
        "strategy_returns": strategy_returns,
        "weights_history": weights_history,
        "static_60_40_returns": static_returns,
        "equal_weight_returns": eq_returns,
        "transition_matrix": last_transition_matrix,
        "prices_aligned": prices_aligned.loc[oos_idx],
    }
    return results


def summarize_results(results: dict) -> pd.DataFrame:
    rows = [
        metrics.performance_summary(
            "HMM Regime-Switching (net of costs)",
            results["strategy_returns"],
            results["weights_history"],
        ),
        metrics.performance_summary("Static 60/40", results["static_60_40_returns"]),
        metrics.performance_summary("Equal-Weight", results["equal_weight_returns"]),
    ]
    return pd.DataFrame(rows).set_index("Strategy")
