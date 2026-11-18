"""VaR backtesting: exception counts, coverage tests and a rolling engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from . import var as v
from .garch import fit_garch


def kupiec_pof(exceptions, alpha: float = 0.99) -> dict:
    """Kupiec proportion-of-failures test (unconditional coverage)."""
    hits = np.asarray(exceptions, dtype=bool)
    n, x = len(hits), int(hits.sum())
    p, p_hat = 1 - alpha, x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (ll_null - ll_alt)
    return {"n": n, "exceptions": x, "rate": p_hat, "lr": lr, "pvalue": chi2.sf(lr, 1)}


def christoffersen_independence(exceptions) -> dict:
    """Christoffersen test: are exceptions clustered (first-order Markov)?"""
    h = np.asarray(exceptions, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = -2 * (ll_null - ll_alt)
    return {"pi_after_no_hit": pi0, "pi_after_hit": pi1, "lr": lr, "pvalue": chi2.sf(lr, 1)}


def conditional_coverage(exceptions, alpha: float = 0.99) -> dict:
    uc = kupiec_pof(exceptions, alpha)
    ind = christoffersen_independence(exceptions)
    lr = uc["lr"] + ind["lr"]
    return {"lr": lr, "pvalue": chi2.sf(lr, 2)}


def basel_zone(n_exceptions: int) -> str:
    """Basel traffic light for a 99% VaR over 250 days."""
    if n_exceptions <= 4:
        return "green"
    if n_exceptions <= 9:
        return "yellow"
    return "red"


def rolling_backtest(pnl: pd.Series, window: int = 500, alpha: float = 0.99, refit_every: int = 20) -> pd.DataFrame:
    """One-day-ahead VaR forecasts from a rolling window, three methods.

    The GARCH parameters are re-estimated every ``refit_every`` days; in between,
    the variance is updated with the filter, which keeps the run fast.
    """
    values = pnl.to_numpy(dtype=float)
    rows = []
    fit = None
    s2_prev = e_prev = None
    for i, t in enumerate(range(window, len(values))):
        hist = values[t - window:t]
        if fit is None or i % refit_every == 0:
            fit = fit_garch(hist)
            e_prev = hist[-1] - fit.mu
            s2_prev = fit.sigma2[-1]
        else:
            e_prev = values[t - 1] - fit.mu
        s2_prev = fit.forecast_variance(e_prev, s2_prev)
        resid = (hist - fit.mu) / np.sqrt(_rolling_var(fit, hist))
        fhs = -np.quantile(fit.mu + np.sqrt(s2_prev) * resid, 1 - alpha)
        rows.append({
            "date": pnl.index[t],
            "pnl": values[t],
            "var_historical": v.historical_var(hist, alpha),
            "var_normal": v.normal_var(hist, alpha),
            "var_fhs": fhs,
        })
    return pd.DataFrame(rows).set_index("date")


def _rolling_var(fit, hist: np.ndarray) -> np.ndarray:
    e = hist - fit.mu
    s2 = np.empty_like(e)
    s2[0] = e.var()
    for k in range(1, len(e)):
        s2[k] = fit.omega + fit.alpha * e[k - 1] ** 2 + fit.beta * s2[k - 1]
    return s2


def summarize(bt: pd.DataFrame, alpha: float = 0.99) -> pd.DataFrame:
    out = {}
    for col in [c for c in bt.columns if c.startswith("var_")]:
        hits = bt["pnl"] < -bt[col]
        uc, cc = kupiec_pof(hits, alpha), conditional_coverage(hits, alpha)
        last_year = int(hits.iloc[-250:].sum())
        out[col.removeprefix("var_")] = {
            "exceptions": uc["exceptions"],
            "expected": round(len(hits) * (1 - alpha), 1),
            "kupiec_p": round(uc["pvalue"], 3),
            "christoffersen_cc_p": round(cc["pvalue"], 3),
            "last_250d": last_year,
            "basel_zone": basel_zone(last_year),
            "avg_var": round(bt[col].mean()),
        }
    return pd.DataFrame(out).T
