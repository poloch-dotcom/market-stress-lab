import numpy as np

from stresslab.backtest import basel_zone, christoffersen_independence, kupiec_pof
from stresslab.garch import fit_garch


def test_kupiec_accepts_correct_rate_and_rejects_bad_one():
    rng = np.random.default_rng(0)
    good = rng.random(2_500) < 0.01
    bad = rng.random(2_500) < 0.04
    assert kupiec_pof(good, 0.99)["pvalue"] > 0.05
    assert kupiec_pof(bad, 0.99)["pvalue"] < 0.001


def test_kupiec_handles_zero_exceptions():
    out = kupiec_pof(np.zeros(250, dtype=bool), 0.99)
    assert np.isfinite(out["lr"])


def test_christoffersen_detects_clusters():
    hits = np.zeros(1_000, dtype=bool)
    for start in range(50, 1_000, 200):
        hits[start:start + 3] = True  # exceptions arrive in bunches of three
    assert christoffersen_independence(hits)["pvalue"] < 0.01


def test_basel_zones():
    assert [basel_zone(n) for n in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


def test_garch_recovers_parameters():
    rng = np.random.default_rng(7)
    omega, alpha, beta, n = 0.05, 0.08, 0.90, 6_000
    x, s2 = np.zeros(n), omega / (1 - alpha - beta)
    for t in range(n):
        x[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * x[t] ** 2 + beta * s2
    fit = fit_garch(x)
    assert abs(fit.alpha - alpha) < 0.03
    assert abs(fit.beta - beta) < 0.04
    assert fit.persistence < 1
