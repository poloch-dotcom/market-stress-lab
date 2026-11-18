"""stresslab: a small market-risk lab for VaR, backtesting and stress testing.

Modules
-------
data         synthetic regime-switching market + CSV loader for real data
pricing      Black-Scholes prices and greeks
portfolio    multi-asset portfolio with full revaluation
garch        GARCH(1,1) estimated by maximum likelihood
var          VaR / Expected Shortfall (historical, parametric, Monte Carlo, FHS)
backtest     Kupiec, Christoffersen, Basel traffic light, rolling backtest
stress       hypothetical scenarios, worst historical windows, reverse stress test
calibration  filling in secondary shocks from a scenario narrative (Gaussian vs ML)
"""

from .data import FACTORS, simulate_market, load_csv
from .portfolio import MarketState, Portfolio

__all__ = ["FACTORS", "simulate_market", "load_csv", "MarketState", "Portfolio"]
__version__ = "0.1.0"
