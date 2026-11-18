"""Value-at-Risk and Expected Shortfall.

Convention: VaR and ES are reported as positive numbers (losses), for a
confidence level ``alpha`` such as 0.99.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t

from .garch import fit_garch


def historical_var(pnl, alpha: float = 0.99) -> float:
    return float(-np.quantile(np.asarray(pnl), 1 - alpha))


def historical_es(pnl, alpha: float = 0.99) -> float:
    pnl = np.asarray(pnl)
    cutoff = np.quantile(pnl, 1 - alpha)
    return float(-pnl[pnl <= cutoff].mean())


def normal_var(pnl, alpha: float = 0.99) -> float:
    pnl = np.asarray(pnl)
    return float(-(pnl.mean() + norm.ppf(1 - alpha) * pnl.std(ddof=1)))


def normal_es(pnl, alpha: float = 0.99) -> float:
    pnl = np.asarray(pnl)
    z = norm.ppf(1 - alpha)
    return float(-pnl.mean() + pnl.std(ddof=1) * norm.pdf(z) / (1 - alpha))


def delta_normal_var(grad, cov, alpha: float = 0.99) -> float:
    """VaR of the linearized P&L, g'x, with x ~ N(0, cov)."""
    grad = np.asarray(grad)
    return float(norm.ppf(alpha) * np.sqrt(grad @ np.asarray(cov) @ grad))


def cornish_fisher_var(pnl, alpha: float = 0.99) -> float:
    """Normal VaR corrected for skewness and excess kurtosis of the P&L."""
    pnl = np.asarray(pnl)
    m, s = pnl.mean(), pnl.std(ddof=1)
    z = (pnl - m) / s
    sk, ku = np.mean(z**3), np.mean(z**4) - 3.0
    q = norm.ppf(1 - alpha)
    q_cf = q + (q**2 - 1) * sk / 6 + (q**3 - 3 * q) * ku / 24 - (2 * q**3 - 5 * q) * sk**2 / 36
    return float(-(m + q_cf * s))


def monte_carlo(portfolio, cov, alpha: float = 0.99, n_sims: int = 50_000, dof: float | None = 5.0,
                horizon_days: int = 1, seed: int = 0) -> dict:
    """Full-revaluation Monte Carlo with Gaussian (dof=None) or Student-t shocks."""
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov)
    z = rng.standard_normal((n_sims, cov.shape[0])) @ np.linalg.cholesky(cov).T
    if dof is not None:
        w = rng.chisquare(dof, size=(n_sims, 1)) / dof
        z = z / np.sqrt(w) * np.sqrt((dof - 2) / dof)
    sims = portfolio.pnl(z, horizon_days=horizon_days)
    return {"var": historical_var(sims, alpha), "es": historical_es(sims, alpha), "pnl": sims}


def student_t_quantile(alpha: float, dof: float) -> float:
    """Quantile of a unit-variance Student-t."""
    return float(student_t.ppf(alpha, dof) * np.sqrt((dof - 2) / dof))


def filtered_historical(pnl, alpha: float = 0.99, fit=None) -> dict:
    """Filtered historical simulation: GARCH volatility x empirical residuals.

    The residual quantile comes from history, the scale from today's GARCH
    forecast, so the VaR reacts quickly when volatility picks up.
    """
    pnl = np.asarray(pnl, dtype=float)
    fit = fit or fit_garch(pnl)
    sigma = fit.next_sigma()
    scenarios = fit.mu + sigma * fit.resid
    return {"var": historical_var(scenarios, alpha), "es": historical_es(scenarios, alpha),
            "sigma": sigma, "fit": fit}
