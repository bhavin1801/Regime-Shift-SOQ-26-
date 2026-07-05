"""
features.py
Builds the feature matrix fed into the HMM regime classifier.

All features here are computed causally (rolling / expanding windows only look
backward), which is necessary but NOT sufficient to avoid lookahead bias -- the
z-scoring/normalization step still has to be refit inside each walk-forward
training window (see backtest.py). Anything computed here is "raw", pre-scaling.
"""

import numpy as np
import pandas as pd

MOM_WINDOWS = [21, 63]      # ~1 month, ~1 quarter
VOL_WINDOWS = [21, 63]


def build_features(prices: pd.DataFrame, vix: pd.Series, market_col="stocks") -> pd.DataFrame:
    """Builds a raw (unscaled) feature DataFrame aligned on prices.index.

    Features:
        mom_{w}   : rolling pct-change momentum of the market asset over window w
        vol_{w}   : rolling annualized volatility of market log-returns over window w
        vix_level : raw India VIX level
        vix_chg_5d: 5-day % change in VIX (captures fear *accelerating*)
    """
    log_ret = np.log(prices[market_col]).diff()

    feat = pd.DataFrame(index=prices.index)
    for w in MOM_WINDOWS:
        feat[f"mom_{w}"] = prices[market_col].pct_change(w)
    for w in VOL_WINDOWS:
        feat[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    feat["vix_level"] = vix
    feat["vix_chg_5d"] = vix.pct_change(5)

    feat = feat.dropna()
    return feat


def expanding_zscore_fit(train_feat: pd.DataFrame):
    """Returns (mean, std) computed ONLY on the training slice."""
    mu = train_feat.mean()
    sigma = train_feat.std().replace(0, 1e-8)
    return mu, sigma


def apply_zscore(feat: pd.DataFrame, mu: pd.Series, sigma: pd.Series) -> pd.DataFrame:
    return (feat - mu) / sigma
