"""
metrics.py
Standard performance metrics for comparing strategies.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    denom = downside.std()
    if denom == 0 or np.isnan(denom):
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / denom


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1
    return dd.min()


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    ann_return = (equity_curve.iloc[-1] ** (TRADING_DAYS / len(returns))) - 1
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return ann_return / mdd


def turnover(weights_history: pd.DataFrame) -> float:
    """Average daily one-way turnover: sum(|w_t - w_{t-1}|) / 2, averaged over time."""
    diffs = weights_history.diff().abs().sum(axis=1) / 2
    return diffs.mean()


def performance_summary(name: str, returns: pd.Series, weights_history: pd.DataFrame = None) -> dict:
    equity_curve = (1 + returns).cumprod()
    summary = {
        "Strategy": name,
        "Total Return %": (equity_curve.iloc[-1] - 1) * 100,
        "Ann. Return %": ((equity_curve.iloc[-1] ** (TRADING_DAYS / len(returns))) - 1) * 100,
        "Ann. Vol %": returns.std() * np.sqrt(TRADING_DAYS) * 100,
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown %": max_drawdown(equity_curve) * 100,
        "Calmar": calmar_ratio(returns, equity_curve),
    }
    if weights_history is not None:
        summary["Avg Daily Turnover %"] = turnover(weights_history) * 100
    return summary
