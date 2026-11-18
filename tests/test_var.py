import numpy as np
from scipy.stats import norm

from stresslab import Portfolio
from stresslab import var as v


def linear_book() -> Portfolio:
    return Portfolio(short_puts=0.0, bond_convexity=0.0)


def test_historical_var_of_normal_sample():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0, 1_000, 400_000)
    # sampling error of a 1% quantile with n = 400k is about 6, allow ~5 standard errors
    assert abs(v.historical_var(pnl, 0.99) - 2326.3) < 30
    assert abs(v.historical_es(pnl, 0.99) - 1_000 * norm.pdf(norm.ppf(0.99)) / 0.01) < 25


def test_es_is_at_least_var():
    rng = np.random.default_rng(2)
    pnl = rng.standard_t(4, 50_000)
    assert v.historical_es(pnl) >= v.historical_var(pnl)


def test_delta_normal_matches_monte_carlo_on_linear_book():
    book = linear_book()
    cov = np.diag([0.01, 5.0, 2.0, 0.006, 1.0]) ** 2
    dn = v.delta_normal_var(book.sensitivities(), cov)
    mc = v.monte_carlo(book, cov, dof=None, n_sims=400_000, seed=3)["var"]
    # expm1 on equity and FX makes the book very slightly non-linear
    assert abs(mc / dn - 1) < 0.02


def test_cornish_fisher_reduces_to_normal_without_skew_or_kurtosis():
    rng = np.random.default_rng(4)
    pnl = rng.normal(0, 1, 1_000_000)
    assert abs(v.cornish_fisher_var(pnl) - v.normal_var(pnl)) < 0.02


def test_short_puts_fatten_the_left_tail():
    rng = np.random.default_rng(5)
    shocks = np.column_stack([rng.normal(0, 0.02, 20_000), np.zeros((20_000, 3)), rng.normal(0, 1, 20_000)])
    with_puts = Portfolio().pnl(shocks)
    without = Portfolio(short_puts=0.0).pnl(shocks)
    skew = lambda x: np.mean(((x - x.mean()) / x.std()) ** 3)
    assert skew(with_puts) < skew(without)
