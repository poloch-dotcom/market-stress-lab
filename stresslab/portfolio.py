"""A small multi-asset trading book, revalued in full for any vector of shocks.

Positions (EUR):
    - long an equity index
    - long a government bond (duration / convexity approximation)
    - long credit, expressed as a CS01 (loss per +1 bp of spread)
    - long USD against EUR
    - short index puts, revalued with Black-Scholes (the non-linear part)

The short puts matter: they make the loss grow faster than linearly in a crash,
and they lose money when implied volatility rises. A delta-normal VaR misses
both effects, which is the point of comparing methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FACTORS
from .pricing import bs_price

TRADING_DAYS = 252


@dataclass
class MarketState:
    spot: float = 4000.0
    implied_vol: float = 0.18  # decimal
    rate: float = 0.02


@dataclass
class Portfolio:
    equity_notional: float = 10e6
    bond_notional: float = 20e6
    bond_duration: float = 7.0
    bond_convexity: float = 60.0
    cs01: float = 6_000.0
    fx_notional: float = 5e6
    short_puts: float = 2_500.0
    put_strike_pct: float = 0.95
    put_maturity: float = 0.5
    state: MarketState = field(default_factory=MarketState)

    @property
    def put_strike(self) -> float:
        return self.put_strike_pct * self.state.spot

    def breakdown(self, shocks, horizon_days: int = 1) -> pd.DataFrame:
        """P&L by position for each row of ``shocks`` (columns in FACTORS order)."""
        x = np.atleast_2d(np.asarray(shocks, dtype=float))
        eq, ir, cs, fx, vol = x.T
        s = self.state

        dy = ir / 1e4
        bond = self.bond_notional * (-self.bond_duration * dy + 0.5 * self.bond_convexity * dy**2)

        put = np.zeros(len(x))
        if self.short_puts:
            K, T = self.put_strike, self.put_maturity
            p0 = bs_price(s.spot, K, T, s.rate, s.implied_vol)
            s1 = s.spot * np.exp(eq)
            sig1 = np.clip(s.implied_vol + vol / 100.0, 0.01, None)
            t1 = max(T - horizon_days / TRADING_DAYS, 1e-6)
            put = -self.short_puts * (bs_price(s1, K, t1, s.rate, sig1) - p0)

        return pd.DataFrame(
            {
                "equity": self.equity_notional * np.expm1(eq),
                "bond": bond,
                "credit": -self.cs01 * cs,
                "fx": self.fx_notional * np.expm1(fx),
                "short_puts": put,
            }
        )

    def pnl(self, shocks, horizon_days: int = 1) -> np.ndarray:
        return self.breakdown(shocks, horizon_days).sum(axis=1).to_numpy()

    def sensitivities(self) -> np.ndarray:
        """First-order P&L sensitivity to each factor (central differences)."""
        bumps = np.array([1e-4, 0.1, 0.1, 1e-4, 0.01])
        grad = np.zeros(len(FACTORS))
        for i, h in enumerate(bumps):
            up, dn = np.zeros(len(FACTORS)), np.zeros(len(FACTORS))
            up[i], dn[i] = h, -h
            grad[i] = (self.pnl(up)[0] - self.pnl(dn)[0]) / (2 * h)
        return grad

    def at_market(self, market: pd.DataFrame) -> "Portfolio":
        """Copy of the portfolio priced at the last spot and implied vol of ``market``."""
        state = MarketState(
            spot=float(market["SPOT"].iloc[-1]),
            implied_vol=float(market["VOL_LEVEL"].iloc[-1]) / 100.0,
            rate=self.state.rate,
        )
        return Portfolio(**{**self.__dict__, "state": state})
