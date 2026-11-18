"""GARCH(1,1) with Gaussian quasi-maximum likelihood, written from scratch.

    x_t = mu + e_t,   e_t = sigma_t * z_t
    sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    sigma2: np.ndarray  # in-sample conditional variances
    resid: np.ndarray  # standardized residuals
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def forecast_variance(self, last_resid: float, last_sigma2: float) -> float:
        return self.omega + self.alpha * last_resid**2 + self.beta * last_sigma2

    def next_sigma(self) -> float:
        e_last = self.resid[-1] * np.sqrt(self.sigma2[-1])
        return float(np.sqrt(self.forecast_variance(e_last, self.sigma2[-1])))


def _filter(e: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    s2 = np.empty_like(e)
    s2[0] = e.var()
    for t in range(1, len(e)):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
    return s2


def _unpack(theta: np.ndarray, var: float) -> tuple[float, float, float]:
    # Map unconstrained parameters to omega > 0, alpha, beta > 0, alpha + beta < 1.
    a, b, w = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2])
    total = 1.0 + a + b
    alpha, beta = a / total, b / total
    omega = w * var * (1 - alpha - beta)
    return omega, alpha, beta


def fit_garch(x) -> GarchFit:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    e = x - mu
    var = e.var()

    def nll(theta):
        omega, alpha, beta = _unpack(theta, var)
        s2 = _filter(e, omega, alpha, beta)
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + e**2 / s2)

    # Start near alpha = 0.08, beta = 0.90, omega matching the sample variance.
    start = np.array([np.log(0.08 / 0.02), np.log(0.90 / 0.02), 0.0])
    res = minimize(nll, start, method="L-BFGS-B")
    omega, alpha, beta = _unpack(res.x, var)
    s2 = _filter(e, omega, alpha, beta)
    return GarchFit(mu, omega, alpha, beta, s2, e / np.sqrt(s2), -res.fun)
