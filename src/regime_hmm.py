"""
regime_hmm.py
Fits a Gaussian HMM to scaled features and maps the arbitrary state indices
(0, 1, 2, ...) to human-readable regime labels (Bull / Bear / Crisis) using
the mean volatility & momentum of each inferred state.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

N_STATES = 3
RANDOM_STATE = 42


def fit_hmm(X: np.ndarray, n_states: int = N_STATES, n_iter: int = 200):
    """Fits a Gaussian HMM with diagonal covariance on scaled feature matrix X."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=RANDOM_STATE,
    )
    model.fit(X)
    return model


def label_states(feat_scaled: pd.DataFrame, states: np.ndarray, vol_col_hint="vol_21") -> dict:
    """Maps state index -> {'Bull','Bear','Crisis'} using average vol & momentum per state.

    Heuristic: the state with highest mean volatility is Crisis. Of the
    remaining two, the one with higher mean momentum is Bull, the other Bear.
    """
    df = feat_scaled.copy()
    df["state"] = states

    vol_cols = [c for c in df.columns if c.startswith("vol_")]
    mom_cols = [c for c in df.columns if c.startswith("mom_")]
    vol_col = vol_cols[0] if vol_cols else vol_col_hint
    mom_col = mom_cols[0] if mom_cols else None

    state_vol = df.groupby("state")[vol_col].mean().sort_values(ascending=False)
    crisis_state = state_vol.index[0]
    remaining = [s for s in state_vol.index if s != crisis_state]

    if mom_col is not None:
        state_mom = df.groupby("state")[mom_col].mean()
        remaining_sorted = sorted(remaining, key=lambda s: state_mom[s], reverse=True)
    else:
        remaining_sorted = remaining

    bull_state, bear_state = remaining_sorted[0], remaining_sorted[1]

    return {crisis_state: "Crisis", bull_state: "Bull", bear_state: "Bear"}


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Viterbi-decodes the most likely state sequence for X."""
    return model.predict(X)
