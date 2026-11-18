import numpy as np

from stresslab.pricing import bs_delta, bs_price, bs_vega


def test_textbook_call_value():
    # Hull, S=100, K=100, T=1, r=5%, sigma=20%: call = 10.4506
    assert abs(bs_price(100, 100, 1.0, 0.05, 0.2, "call") - 10.4506) < 1e-3


def test_put_call_parity():
    S, K, T, r, s = 4000.0, 3800.0, 0.5, 0.02, 0.25
    c, p = bs_price(S, K, T, r, s, "call"), bs_price(S, K, T, r, s, "put")
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-8


def test_put_delta_matches_finite_difference():
    S, K, T, r, s, h = 4000.0, 3800.0, 0.5, 0.02, 0.25, 0.01
    fd = (bs_price(S + h, K, T, r, s) - bs_price(S - h, K, T, r, s)) / (2 * h)
    assert abs(fd - bs_delta(S, K, T, r, s)) < 1e-6
    fd_vega = (bs_price(S, K, T, r, s + 1e-5) - bs_price(S, K, T, r, s - 1e-5)) / 2e-5
    assert abs(fd_vega - bs_vega(S, K, T, r, s)) < 1e-3
