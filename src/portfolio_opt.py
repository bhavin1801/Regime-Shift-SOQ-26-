"""
portfolio_opt.py
Solves for optimal portfolio weights (long-only, fully invested) under
regime-specific objectives using cvxpy.

Regime -> objective mapping:
    Bull   : maximize Sharpe ratio (via max return for a risk budget, approximated
             by maximizing mu^T w - risk_aversion * w^T Sigma w with low risk_aversion)
    Bear   : balanced risk-adjusted objective (moderate risk_aversion)
    Crisis : minimize volatility (pure min-variance, mu term dropped/near-zero weight)
"""

import numpy as np
import cvxpy as cp

REGIME_RISK_AVERSION = {
    "Bull": 1.0,     # tilt toward maximizing return per unit risk
    "Bear": 5.0,     # more conservative, balances return and risk
    "Crisis": 25.0,  # effectively minimum-variance
}

MIN_WEIGHT = 0.0   # long-only
MAX_WEIGHT = 0.7    # cap any single asset to avoid corner solutions


def optimal_weights(mu: np.ndarray, sigma: np.ndarray, regime: str) -> np.ndarray:
    """Solves: maximize mu^T w - gamma * w^T Sigma w s.t. sum(w)=1, 0<=w<=MAX_WEIGHT.

    mu, sigma should be ANNUALIZED expected returns / covariance, estimated
    only from data available up to the current point in time (no lookahead).
    """
    n = len(mu)
    w = cp.Variable(n)
    gamma = REGIME_RISK_AVERSION.get(regime, 5.0)

    if regime == "Crisis":
        # Pure min-variance: ignore expected-return estimates entirely, since
        # they are least reliable exactly when they matter most (crash regimes).
        objective = cp.Minimize(cp.quad_form(w, sigma))
    else:
        objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, sigma))

    constraints = [cp.sum(w) == 1, w >= MIN_WEIGHT, w <= MAX_WEIGHT]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        # Fallback: equal weight if the solver fails for any reason
        return np.ones(n) / n

    weights = np.clip(w.value, 0, None)
    weights = weights / weights.sum()
    return weights
