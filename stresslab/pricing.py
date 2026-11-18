"""Black-Scholes prices and greeks for European options on an index."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T), T


def bs_price(S, K, T, r, sigma, kind: str = "put"):
    """Price of a European call or put."""
    d1, d2, T = _d1_d2(S, K, T, r, sigma)
    disc = K * np.exp(-r * T)
    if kind == "call":
        return S * norm.cdf(d1) - disc * norm.cdf(d2)
    if kind == "put":
        return disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def bs_delta(S, K, T, r, sigma, kind: str = "put"):
    d1, _, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    d1, _, T = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (np.asarray(S) * np.asarray(sigma) * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """Sensitivity to a change of 1.00 in volatility (not 1 vol point)."""
    d1, _, T = _d1_d2(S, K, T, r, sigma)
    return np.asarray(S) * norm.pdf(d1) * np.sqrt(T)
