"""
Functions and classes for asset allocation models and portfolio construction methods
"""
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm
import pandas as pd
import numpy as np


# TODO Max Utility
# TODO classes can have a ".from_ts" method to create the object from a time series of returns, this would require an establisehed estimator for mu and sigma

class MaxSharpe:

    def __init__(self, mu, cov, rf=0, allow_shorts=True):
        # TODO Documentation (for when rf is available, returns the tangency portfolio)

        self._assertions(mu, cov)

        # Save attributes
        self.n_assets = mu.shape[0]
        self.asset_names = list(mu.index)
        self.mu = mu
        self.cov = cov
        self.std = pd.Series(data=np.sqrt(np.diag(cov)), index=mu.index)
        self.rf = rf
        self.allow_shorts = allow_shorts

        if allow_shorts:
            # TODO Analytical solution of the max sharpe
            pass
        else:
            # Numerical solution
            self.mu_p, self.sigma_p, self.risky_weights, self.sharpe_p = self._numerical_tangency()

    @staticmethod
    def _assertions(mu, cov):
        # TODO mu and cov types
        cond1 = sorted(mu.index) == sorted(cov.index)
        cond2 = sorted(cov.index) == sorted(cov.columns)
        cond = cond1 and cond2
        assert cond, "elements in the input indexes do not match"

    def _numerical_tangency(self):
        """
        Used when short selling is not allowed, so there is no analytical
        solution
        """

        if self.n_assets == 1:  # one risky asset (analytical)
            mu_p = self.mu.iloc[0]
            sigma_p = self.cov.iloc[0, 0]
            sharpe_p = (mu_p - self.rf) / sigma_p
            weights = pd.Series(
                data={self.mu.index[0]: 1},
                name='Risky Weights',
            )

        else:  # multiple risky assets (optimization)

            # objective function (notice the sign change on the return value)
            def sharpe(x):
                return - self._sharpe(x, self.mu.values, self.cov.values, self.rf, self.n_assets)

            # budget constraint
            constraints = ({'type': 'eq',
                            'fun': lambda w: w.sum() - 1})

            # Create bounds for the weights if short-selling is restricted
            bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Run optimization
            res = minimize(sharpe, w0,
                           method='SLSQP',
                           constraints=constraints,
                           bounds=bounds,
                           options={'ftol': 1e-9, 'disp': False})

            if not res.success:
                raise RuntimeError("Convergence Failed")

            # Compute optimal portfolio parameters
            mu_p = np.sum(res.x * self.mu.values)
            sigma_p = np.sqrt(res.x @ self.cov @ res.x)
            sharpe_p = - sharpe(res.x)
            weights = pd.Series(
                index=self.asset_names,
                data=res.x,
                name='Risky Weights',
            )

        return mu_p, sigma_p, weights, sharpe_p

    @staticmethod
    def _sharpe(w, mu, cov, rf, n):
        er = np.sum(w * mu)

        w = np.reshape(w, (n, 1))
        risk = np.sqrt(w.T @ cov @ w)[0][0]

        sharpe = (er - rf) / risk
        return sharpe
