import numpy as np
import pandas as pd

from stresslab import FACTORS, Portfolio, simulate_market
from stresslab.calibration import ConditionalGaussian, HybridCalibrator
from stresslab.stress import reverse_stress_linear, reverse_stress_test, run_scenarios

COV = np.diag([0.03, 15.0, 10.0, 0.015, 4.0]) ** 2


def test_reverse_stress_matches_closed_form_on_linear_book():
    book = Portfolio(short_puts=0.0, bond_convexity=0.0, equity_notional=10e6, fx_notional=0.0)
    # Small target: the equity leg (expm1) is close to linear, so the two must agree.
    # For large targets they diverge on purpose: a -13% move loses less than 13%.
    numeric = reverse_stress_test(book, COV, 2e5)
    analytic = reverse_stress_linear(book.sensitivities(), COV, 2e5)
    assert abs(numeric.loss - 2e5) < 1e2
    assert np.allclose(numeric.shock, analytic, rtol=0.02, atol=1e-4)


def test_reverse_stress_hits_target_with_options():
    book = Portfolio()
    res = reverse_stress_test(book, COV, 4e6)
    assert abs(res.loss - 4e6) / 4e6 < 1e-3
    assert 0 < res.tail_probability < 1


def test_scenarios_sum_to_total():
    table = run_scenarios(Portfolio())
    parts = table.drop(columns="total").sum(axis=1)
    assert np.allclose(parts, table["total"])


def test_conditional_gaussian_recovers_linear_relation():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"EQ": rng.normal(0, 1, 50_000)})
    Y = pd.DataFrame({"VOL": -3.0 * X["EQ"] + rng.normal(0, 0.5, 50_000)})
    model = ConditionalGaussian().fit(X, Y)
    assert abs(model.beta[0, 0] + 3.0) < 0.02


def test_hybrid_extrapolates_where_trees_cannot():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"EQ": rng.uniform(-0.05, 0.0, 2_000)})
    Y = pd.DataFrame({"VOL": -100.0 * X["EQ"] + rng.normal(0, 0.1, 2_000)})
    far = pd.DataFrame({"EQ": [-0.20]})  # four times the worst move seen in training
    pred = HybridCalibrator().fit(X, Y).predict(far)[0, 0]
    assert pred > 15  # a pure tree model would stay near 5


def test_simulated_market_shape():
    m = simulate_market(n_days=500, seed=1)
    assert list(m.columns[:5]) == FACTORS
    assert set(m["regime"].unique()) <= {0, 1}
    assert (m["VOL_LEVEL"] >= 8.0).all()
