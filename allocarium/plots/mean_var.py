class MaxSharpeOLD:

    def plot(self,
             figsize=(10, 7),
             save_path=None,
             title=None,
             assets=True,  # plot individual assets
             gmvp=True,  # plot global min var
             max_sharpe=True,  # Max Sharpe port
             risk_free=True,  # plot rf
             mvf=True,  # MinVar Frontier
             mvfnoss=True,  # MinVar Frontier no short selling
             cal=True,  # Capital Allocation Line
             investor=True,  # Investor's indifference, portfolio, and CE
             ):

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")
        ax = plt.subplot2grid((1, 1), (0, 0))

        # Elements
        if assets:
            ax.scatter(self.sigma, self.mu, label='Assets')

        if gmvp:
            ax.scatter(self.sigma_mv, self.mu_mv, label='Global Minimum Variance')

        if max_sharpe:
            ax.scatter(self.sigma_p, self.mu_p, label='Maximum Sharpe')

        if risk_free:
            ax.scatter(0, self.rf, label='Risk-Free')

        if mvf and self.n_assets != 1:
            mu_mv, sigma_mv = self.min_var_frontier(short_sell=True)
            ax.plot(sigma_mv, mu_mv, marker=None, zorder=-1, label='Minimum Variance Frontier')

        if mvfnoss and self.n_assets != 1:
            mu_mv, sigma_mv = self.min_var_frontier(short_sell=False)
            ax.plot(sigma_mv, mu_mv, marker=None, zorder=-1, label='Minimum Variance Frontier (No Short Selling)')

        if cal:
            if self.rb is None:
                max_sigma = self.sigma.max() + 0.05
                x_values = [0, max_sigma]
                y_values = [self.rf, self.rf + self.sharpe_p * max_sigma]
                plt.plot(x_values, y_values, marker=None, zorder=-1, label='Capital Allocation Line')
            else:
                # If borrowing costs more
                x_cal = [0, self.sigma_p]
                y_cal = [self.rf, self.rf + self.sharpe_p * self.sigma_p]
                plt.plot(x_cal, y_cal, marker=None, zorder=-1, label='Capital Allocation Line (No Borrowing)')

                ax.scatter(self.sigma_b, self.mu_b, label='Maximum Sharpe (Borrowing)')
                ax.scatter(0, self.rb, label='Borrowing Cost')

                max_sigma = self.sigma.max() + 0.05
                x_bor1 = [0, self.sigma_b]
                x_bor2 = [self.sigma_b, max_sigma]
                y_bor1 = [self.rb, self.rb + self.sharpe_b * self.sigma_b]
                y_bor2 = [self.rb + self.sharpe_b * self.sigma_b, self.rb + self.sharpe_b * max_sigma]
                plt.plot(x_bor1, y_bor1, marker=None, zorder=-1, color='grey', ls='--', lw=1, label=None)
                plt.plot(x_bor2, y_bor2, marker=None, zorder=-1, color='grey', label='Capital Allocation Line (Borrowing)')

        if investor and (self.risk_aversion is not None):
            max_sigma = self.sigma_c + 0.02
            x_values = np.arange(0, max_sigma, max_sigma / 100)
            y_values = self.certain_equivalent + 0.5 * self.risk_aversion * (x_values ** 2)
            ax.plot(x_values, y_values, marker=None, zorder=-1, label='Indiference Curve')
            ax.scatter(self.sigma_c, self.mu_c, label="Investor's Portfolio")

        # Adjustments
        ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.legend(loc='best')
        ax.set_xlim((0, self.sigma.max() + 0.05))
        ax.set_xlabel('Risk')
        ax.set_ylabel('Return')
        plt.tight_layout()

        # Save as picture
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def min_var_frontier(self, n_steps=100, short_sell=True):
        # TODO Documentation

        if short_sell:
            # Analytical solution when short-selling is allowed
            E = self.mu.values
            inv_cov = np.linalg.inv(self.cov)

            A = E @ inv_cov @ E
            B = np.ones(self.n_assets) @ inv_cov @ E
            C = np.ones(self.n_assets) @ inv_cov @ np.ones(self.n_assets)

            def min_risk(mu):
                return np.sqrt((C * (mu ** 2) - 2 * B * mu + A) / (A * C - B ** 2))

            min_mu = min(self.mu.min(), self.rf) - 0.05
            max_mu = max(self.mu.max(), self.rf) + 0.05

            mu_range = np.arange(min_mu, max_mu, (max_mu - min_mu) / n_steps)
            sigma_range = np.array(list(map(min_risk, mu_range)))

        else:

            sigma_range = []

            # Objective function
            def risk(x):
                return np.sqrt(x @ self.cov @ x)

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Values for mu to perform the minimization
            mu_range = np.linspace(self.mu.min(), self.mu.max(), n_steps)

            for mu_step in tqdm(mu_range, 'Finding Mininmal variance frontier'):

                # budget and return constraints
                constraints = ({'type': 'eq',
                                'fun': lambda w: w.sum() - 1},
                               {'type': 'eq',
                                'fun': lambda w: sum(w * self.mu) - mu_step})

                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

                # Run optimization
                res = minimize(risk, w0,
                               method='SLSQP',
                               constraints=constraints,
                               bounds=bounds,
                               options={'ftol': 1e-9, 'disp': False})

                if not res.success:
                    raise RuntimeError("Convergence Failed")

                # Compute optimal portfolio parameters
                sigma_step = np.sqrt(res.x @ self.cov @ res.x)

                sigma_range.append(sigma_step)

            sigma_range = np.array(sigma_range)

        return mu_range, sigma_range