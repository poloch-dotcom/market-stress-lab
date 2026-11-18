"""Stress testing: hypothetical scenarios, historical replay, reverse stress test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .data import FACTORS, rolling_moves

# Illustrative 10-day scenarios. Units: EQ and FX in log-return, IR and CS in bp,
# VOL in vol points. The numbers are mine, chosen to be internally consistent,
# not taken from any regulator.
HYPOTHETICAL_SCENARIOS = {
    "Equity crash, flight to quality": {
        "narrative": "Global sell-off: equities -25%, safe-haven bid on Bunds, spreads blow out.",
        "shocks": {"EQ": -0.25, "IR": -60, "CS": 150, "FX": 0.05, "VOL": 20},
    },
    "Stagflation, rates up": {
        "narrative": "Inflation surprise: yields +150 bp, equities and credit weaken, USD up.",
        "shocks": {"EQ": -0.10, "IR": 150, "CS": 80, "FX": 0.03, "VOL": 8},
    },
    "Credit event": {
        "narrative": "Large corporate default: spreads +250 bp, contagion to equities.",
        "shocks": {"EQ": -0.15, "IR": -30, "CS": 250, "FX": 0.02, "VOL": 12},
    },
    "Volatility spike only": {
        "narrative": "Vol shock with little spot move, the kind that hurts short options.",
        "shocks": {"EQ": -0.03, "IR": -5, "CS": 15, "FX": 0.0, "VOL": 15},
    },
}


def to_vector(shocks: dict) -> np.ndarray:
    return np.array([shocks.get(f, 0.0) for f in FACTORS], dtype=float)


def run_scenarios(portfolio, scenarios: dict = HYPOTHETICAL_SCENARIOS, horizon_days: int = 10) -> pd.DataFrame:
    rows = []
    for name, sc in scenarios.items():
        b = portfolio.breakdown(to_vector(sc["shocks"]), horizon_days).iloc[0]
        rows.append({"scenario": name, **b.to_dict(), "total": b.sum()})
    return pd.DataFrame(rows).set_index("scenario")


def worst_historical_windows(market: pd.DataFrame, portfolio, horizon: int = 10, top: int = 5) -> pd.DataFrame:
    """Replay every ``horizon``-day window of history on today's portfolio and
    keep the ``top`` worst, non-overlapping ones."""
    moves = rolling_moves(market, horizon)
    pnl = pd.Series(portfolio.pnl(moves.to_numpy(), horizon_days=horizon), index=moves.index)
    picked = []
    for date in pnl.sort_values().index:
        if all(abs((date - p).days) > horizon * 1.5 for p in picked):
            picked.append(date)
        if len(picked) == top:
            break
    out = moves.loc[picked].copy()
    out.insert(0, "window_end", out.index.date)
    out["pnl"] = pnl.loc[picked]
    return out.reset_index(drop=True)


def stressed_period(pnl: pd.Series, window: int = 250) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The ``window``-day period where the portfolio P&L was most volatile.

    Same idea as the stressed VaR in Basel 2.5: calibrate on the worst year for
    this book rather than on the whole history.
    """
    vol = pnl.rolling(window).std()
    end = vol.idxmax()
    start = pnl.index[pnl.index.get_loc(end) - window + 1]
    return start, end


@dataclass
class ReverseStressResult:
    shock: np.ndarray
    loss: float
    mahalanobis: float
    tail_probability: float  # P(a random move is at least this far, in Mahalanobis terms)

    def as_series(self) -> pd.Series:
        return pd.Series(self.shock, index=FACTORS)


def _plausibility(x: np.ndarray, cov_inv: np.ndarray) -> tuple[float, float]:
    d = float(np.sqrt(x @ cov_inv @ x))
    return d, float(chi2.sf(d**2, len(x)))


def reverse_stress_linear(grad, cov, loss_target: float) -> np.ndarray:
    """Most likely Gaussian move such that g'x = -loss_target (closed form)."""
    grad, cov = np.asarray(grad), np.asarray(cov)
    return -loss_target * cov @ grad / (grad @ cov @ grad)


def reverse_stress_test(portfolio, cov, loss_target: float, horizon_days: int = 10) -> ReverseStressResult:
    """Find the most plausible shock that produces a loss of at least ``loss_target``.

    Minimise the Mahalanobis distance x' cov^-1 x subject to P&L(x) <= -loss_target,
    with full revaluation. The linear solution is used as the starting point.
    """
    cov = np.asarray(cov)
    cov_inv = np.linalg.inv(cov)
    x0 = reverse_stress_linear(portfolio.sensitivities(), cov, loss_target)
    scale = np.sqrt(np.diag(cov))  # optimise in standardized units for stability

    def objective(u):
        x = u * scale
        return x @ cov_inv @ x

    constraint = {"type": "ineq", "fun": lambda u: -portfolio.pnl(u * scale, horizon_days)[0] - loss_target}
    res = minimize(objective, x0 / scale, constraints=[constraint], method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    x = res.x * scale
    d, p = _plausibility(x, cov_inv)
    return ReverseStressResult(x, float(-portfolio.pnl(x, horizon_days)[0]), d, p)
