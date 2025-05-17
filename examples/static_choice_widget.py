import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt
from allocarium.stats import corr2cov

mu = pd.Series(
    {
        "Asset A": 0.12,
        "Asset B": 0.10,
        "Asset C": 0.08,
    }
)

vol = pd.Series(
    {
        "Asset A": 0.25,
        "Asset B": 0.22,
        "Asset C": 0.20,
    }
)

corr = pd.DataFrame(
    index=vol.index, columns=vol.index,
    data=[
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]
)

cov = corr2cov(corr, vol)


A = mu.T @ la.inv(cov) @ mu
B = mu.T @ la.inv(cov) @ np.ones(mu.shape[0])
C = np.ones(mu.shape[0]) @ la.inv(cov) @ np.ones(mu.shape[0])


def min_std(mu_bar):
    return np.sqrt((C * (mu_bar ** 2) - 2 * B * mu_bar + A) / (A * C - B ** 2))


mu_range = np.linspace(start=mu.min() - 0.02, stop=mu.max() + 0.02, num=50)
sigma_range = np.array(list(map(min_std, mu_range)))


# ===== Chart =====
fig = plt.figure(figsize=(8, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))

ax.scatter(vol, mu, label='Assets')
ax.plot(sigma_range, mu_range, marker=None, zorder=-1, label='Minimum Variance Frontier')

plt.tight_layout()
plt.show()

