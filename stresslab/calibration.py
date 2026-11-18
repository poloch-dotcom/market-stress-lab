"""Shock calibration: from a scenario narrative to a full set of shocks.

A stress scenario usually starts with a few "core" shocks someone writes down
(equities -25%, spreads +150 bp). The other risk factors still need a value,
and it should be consistent with how markets actually move together in stress.

Four ways to fill them in, compared out of sample:
    zero       leave the secondary factors unshocked (the lazy default)
    gaussian   conditional expectation of a multivariate normal fitted on tail moves
    gbm        gradient boosting, one model per secondary factor
    hybrid     the Gaussian rule plus gradient boosting on its residuals

The Gaussian rule is linear, so it misses a convex response such as implied vol
jumping more than proportionally in a crash. Pure gradient boosting sees the
shape but cannot extrapolate past the training range. The hybrid keeps both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class ConditionalGaussian:
    """E[y | x] = mu_y + S_yx S_xx^-1 (x - mu_x)."""

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "ConditionalGaussian":
        Z = pd.concat([X, Y], axis=1)
        mu, S = Z.mean().to_numpy(), Z.cov().to_numpy()
        k = X.shape[1]
        self.mu_x, self.mu_y = mu[:k], mu[k:]
        self.beta = S[k:, :k] @ np.linalg.inv(S[:k, :k])
        self.columns = list(Y.columns)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.mu_y + (X - self.mu_x) @ self.beta.T


class BoostedCalibrator:
    def __init__(self, **params):
        self.params = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                       "subsample": 0.8, "random_state": 0, **params}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "BoostedCalibrator":
        self.features = list(X.columns)
        self.models = {c: GradientBoostingRegressor(**self.params).fit(X, Y[c]) for c in Y.columns}
        self.columns = list(Y.columns)
        return self

    def predict(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=float)), columns=self.features)
        return np.column_stack([self.models[c].predict(X[self.features]) for c in self.columns])


class HybridCalibrator:
    """Linear model first, then gradient boosting on its residuals.

    Trees cannot extrapolate: beyond the largest move seen in training they
    return a flat prediction, which is exactly where a stress scenario lives.
    Keeping a linear backbone preserves extrapolation; the boosted part only
    corrects the shape inside the range of the data.
    """

    def __init__(self, **params):
        self.params = {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03,
                       "min_samples_leaf": 40, "subsample": 0.8, "random_state": 0, **params}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "HybridCalibrator":
        self.linear = ConditionalGaussian().fit(X, Y)
        resid = Y - self.linear.predict(X)
        self.boost = BoostedCalibrator(**self.params).fit(X, resid)
        self.columns = list(Y.columns)
        return self

    def predict(self, X) -> np.ndarray:
        return self.linear.predict(X) + self.boost.predict(X)


def tail_sample(moves: pd.DataFrame, quantile: float = 0.25, on: str = "EQ") -> pd.DataFrame:
    """Keep the moves where ``on`` is in its lower tail: calibrate on stress, not on calm days."""
    return moves[moves[on] <= moves[on].quantile(quantile)]


def evaluate(moves: pd.DataFrame, core: list[str], secondary: list[str], train_frac: float = 0.7,
             horizon: int = 10, tail_q: float = 0.25, gbm_params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Time-ordered split with a gap of ``horizon`` days (the moves overlap).

    Both models are trained on the tail of the training period and scored on the
    tail of the test period. Returns RMSE by factor and the fitted models.
    """
    split = int(len(moves) * train_frac)
    train, test = moves.iloc[:split - horizon], moves.iloc[split:]
    cut = train["EQ"].quantile(tail_q)
    train_t, test_t = train[train["EQ"] <= cut], test[test["EQ"] <= cut]

    models = {
        "gaussian": ConditionalGaussian().fit(train_t[core], train_t[secondary]),
        "gbm": BoostedCalibrator(**(gbm_params or {})).fit(train_t[core], train_t[secondary]),
        "hybrid": HybridCalibrator().fit(train_t[core], train_t[secondary]),
    }
    truth = test_t[secondary].to_numpy()
    preds = {"zero": np.zeros_like(truth), **{k: m.predict(test_t[core]) for k, m in models.items()}}
    rmse = {k: np.sqrt(((p - truth) ** 2).mean(axis=0)) for k, p in preds.items()}
    table = pd.DataFrame(rmse, index=secondary).T
    return table, {"models": models, "test": test_t, "predictions": preds}


def complete_scenario(core_shocks: dict, model, secondary: list[str]) -> dict:
    """Fill a narrative's missing factors with a fitted calibrator."""
    x = np.array([[core_shocks[c] for c in core_shocks]])
    filled = dict(zip(secondary, model.predict(x)[0]))
    return {**core_shocks, **filled}
