"""Market data: a synthetic regime-switching generator and a loader for real data.

Risk factors (daily changes):
    EQ   equity index log-return
    IR   10-year government yield change, in basis points
    CS   corporate credit spread change, in basis points
    FX   EUR/USD log-return (positive = USD appreciates against EUR)
    VOL  change in equity implied volatility, in vol points

The synthetic market is a two-state Markov chain (calm / stress). Each state has
its own volatilities and correlations, and innovations are Student-t. This gives
fat tails and volatility clustering without fitting anything. The implied-vol
factor reacts to equity moves through a leverage term plus a convex term, so a
large sell-off pushes vol up more than proportionally. That non-linearity is
deliberate: it is what the calibration module tries to learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FACTORS = ["EQ", "IR", "CS", "FX", "VOL"]
UNITS = {"EQ": "log-return", "IR": "bp", "CS": "bp", "FX": "log-return", "VOL": "vol pts"}


@dataclass(frozen=True)
class Regime:
    vols: tuple[float, float, float, float]  # EQ, IR, CS, FX (daily)
    corr: np.ndarray  # 4x4 correlation matrix
    vol_noise: float  # idiosyncratic noise on implied vol, in vol points


def _corr(eq_ir: float, eq_cs: float, eq_fx: float, ir_cs: float, ir_fx: float, cs_fx: float) -> np.ndarray:
    return np.array(
        [
            [1.0, eq_ir, eq_cs, eq_fx],
            [eq_ir, 1.0, ir_cs, ir_fx],
            [eq_cs, ir_cs, 1.0, cs_fx],
            [eq_fx, ir_fx, cs_fx, 1.0],
        ]
    )


CALM = Regime(vols=(0.009, 4.0, 1.5, 0.005), corr=_corr(0.30, -0.40, 0.10, -0.10, 0.05, -0.05), vol_noise=0.5)
# In stress, equities and yields fall together (flight to quality), spreads widen
# as equities drop, and the dollar rallies.
STRESS = Regime(vols=(0.024, 8.0, 6.0, 0.009), corr=_corr(0.50, -0.75, -0.30, -0.30, -0.15, 0.25), vol_noise=1.5)


def _student_t(rng: np.random.Generator, corr: np.ndarray, dof: float) -> np.ndarray:
    """One draw of a unit-variance multivariate Student-t."""
    z = np.linalg.cholesky(corr) @ rng.standard_normal(corr.shape[0])
    w = rng.chisquare(dof) / dof
    return z / np.sqrt(w) * np.sqrt((dof - 2) / dof)


def simulate_market(
    n_days: int = 3000,
    seed: int = 42,
    p_stay_calm: float = 0.99,
    p_stay_stress: float = 0.96,
    dof: float = 5.0,
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Simulate daily risk-factor changes.

    Returns a DataFrame indexed by business days with the five factor changes,
    the implied-vol level (``VOL_LEVEL``, in vol points), the equity index level
    (``SPOT``) and the regime label (0 = calm, 1 = stress).
    """
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        stay = p_stay_calm if regimes[t - 1] == 0 else p_stay_stress
        regimes[t] = regimes[t - 1] if rng.random() < stay else 1 - regimes[t - 1]

    out = np.zeros((n_days, 5))
    vol_level = np.zeros(n_days)
    level = 18.0
    for t in range(n_days):
        reg = STRESS if regimes[t] else CALM
        x = _student_t(rng, reg.corr, dof) * np.array(reg.vols)
        eq = x[0]
        # Leverage effect plus a convex response to sell-offs, mean-reverting level.
        d_vol = -45.0 * eq + 2500.0 * max(-eq, 0.0) ** 2 + reg.vol_noise * rng.standard_normal()
        d_vol += 0.02 * (18.0 - level)
        new_level = max(level + d_vol, 8.0)
        d_vol = new_level - level
        level = new_level
        out[t] = [x[0], x[1], x[2], x[3], d_vol]
        vol_level[t] = level

    index = pd.bdate_range(start=start, periods=n_days)
    df = pd.DataFrame(out, index=index, columns=FACTORS)
    df["VOL_LEVEL"] = vol_level
    df["SPOT"] = 4000.0 * np.exp(df["EQ"].cumsum())
    df["regime"] = regimes
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load real market levels and turn them into factor changes.

    Expected columns: ``date``, ``equity`` (index level), ``yield_10y`` (in %),
    ``credit_spread`` (in bp), ``eurusd`` (USD per EUR) and ``implied_vol``
    (in vol points, e.g. VSTOXX or VIX). Possible free sources: ECB Data Portal
    for euro-area yields, FRED for VIX and corporate spreads, Stooq for indices.
    """
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index().dropna()
    df = pd.DataFrame(index=raw.index)
    df["EQ"] = np.log(raw["equity"]).diff()
    df["IR"] = raw["yield_10y"].diff() * 100.0
    df["CS"] = raw["credit_spread"].diff()
    # A USD appreciation is a fall in EUR/USD.
    df["FX"] = -np.log(raw["eurusd"]).diff()
    df["VOL"] = raw["implied_vol"].diff()
    df["VOL_LEVEL"] = raw["implied_vol"]
    df["SPOT"] = raw["equity"]
    return df.dropna()


def rolling_moves(factors: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Overlapping ``horizon``-day factor moves (sums of daily changes)."""
    return factors[FACTORS].rolling(horizon).sum().dropna()
