"""
Functions and classes for asset allocation models and portfolio construction methods
"""
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm
import pandas as pd
import numpy as np


class MaxSharpe:

    def __init__(self, mu, cov, rf=0, allow_shorts=True):
        """
        Returns the portfolio with maximum Sharpe ratio given the expected
        returns, covariance matrix, and risk-free asset.

        All the labels in the input Series and DataFrames must match.

        Parameters
        ----------
        mu : pd.Series
            Expected returns

        cov : pd.DataFrame
            Covariance matrix of the returns

        rf : float, optional
            Risk-free rate. It deafults to zero if nothing is passed, meaning
            that in this case you should pass mu as the excess returns.

        allow_shorts : bool, optional
            If True (default), short selling is allowed and the analytical
            solution is used. If False, short selling is not allowed and a
            numerical optimization is used.
        """

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
            self.mu_p, self.sigma_p, self.risky_weights, self.sharpe_p = self._analytical_tangency()

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

    def _analytical_tangency(self):

        denom = inv(self.cov.values) @ (self.mu.values - self.rf)
        ws = denom / np.sum(denom)
        mu_p = np.sum(ws * self.mu.values)
        sigma_p = np.sqrt(ws.T @ self.cov.values @ ws)
        sharpe_p = (mu_p - self.rf) / sigma_p

        weights = pd.Series(
            index=self.asset_names,
            data=ws,
            name='Risky Weights',
        )

        return mu_p, sigma_p, weights, sharpe_p

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
