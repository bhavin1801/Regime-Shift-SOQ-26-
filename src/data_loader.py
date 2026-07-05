"""
data_loader.py
Pulls daily price data for the asset universe + India VIX using yfinance.

Universe (edit TICKERS to change):
    NIFTYBEES.NS - Nifty 50 ETF (proxy for Indian equities / "stocks")
    GOLDBEES.NS  - Gold ETF (proxy for gold)
    LIQUIDBEES.NS or a long bond ETF proxy for "bonds"
    ^INDIAVIX    - India VIX (volatility / fear proxy)

Note: yfinance needs outbound internet access. If you're running this in a
sandboxed environment without internet, use `use_cache=True` and point
`cache_path` at a previously saved CSV (see save_cache()).
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {
    "stocks": "NIFTYBEES.NS",   # Nifty 50 ETF
    "gold": "GOLDBEES.NS",      # Gold ETF
    "bonds": "^NSMIDCP",        # placeholder if a bond ETF isn't available; see NOTE below
}
VIX_TICKER = "^INDIAVIX"

# NOTE on bonds: NSE-listed long-duration bond/gilt ETFs (e.g. LICNETFGSEC.NS,
# SETFGILT.NS) sometimes have short or spotty yfinance history. If your run
# raises "insufficient bond history", swap the "bonds" ticker below for one of:
#   "LICNETFGSEC.NS", "SETFGILT.NS", "SGBFEB.NS"
# and re-run.


def download_prices(start="2015-01-01", end=None, tickers=None, vix_ticker=VIX_TICKER):
    """Downloads adjusted close prices for the asset universe + VIX.

    Returns
    -------
    prices : DataFrame, columns = asset names, adjusted close
    vix    : Series, India VIX level
    """
    tickers = tickers or TICKERS
    all_symbols = list(tickers.values()) + [vix_ticker]

    raw = yf.download(all_symbols, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"].copy()
    close.columns.name = None

    rename_map = {v: k for k, v in tickers.items()}
    prices = close[list(tickers.values())].rename(columns=rename_map)
    vix = close[vix_ticker].rename("VIX")

    # Forward-fill small gaps (holidays that don't line up across exchanges),
    # then drop any rows that are still incomplete.
    prices = prices.ffill().dropna()
    vix = vix.ffill().reindex(prices.index).dropna()

    common_idx = prices.index.intersection(vix.index)
    prices = prices.loc[common_idx]
    vix = vix.loc[common_idx]

    return prices, vix


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def save_cache(prices, vix, path="data_cache.csv"):
    df = prices.copy()
    df["VIX"] = vix
    df.to_csv(path)


def load_cache(path="data_cache.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    vix = df.pop("VIX")
    return df, vix


if __name__ == "__main__":
    prices, vix = download_prices()
    print(prices.tail())
    print(vix.tail())
    save_cache(prices, vix)
