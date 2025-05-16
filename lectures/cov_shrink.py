from allocarium.stats import cov_shrinkage, corr_shrinkage
from allocarium.utils import bbg_total_return
import matplotlib.pyplot as plt
import seaborn as sns
import getpass

username = getpass.getuser()

df = bbg_total_return()
rets = df.resample('ME').last().pct_change(1).dropna()
cov = rets.cov()
min_cov = cov.min().min()
max_cov = cov.max().max()


# ============================
# ===== BASIC COVARIANCE =====
# ============================
size = 8
fig = plt.figure(figsize=(size * (4 / 3), size))
fig.suptitle(
    "Basic Covariance Shrinkage",
    fontweight="bold",
)

subplotdicts = [
    {"alpha": 0.0, "pos": (0, 0)},
    {"alpha": 0.3, "pos": (0, 1)},
    {"alpha": 0.6, "pos": (1, 0)},
    {"alpha": 0.9, "pos": (1, 1)},
]
for d in subplotdicts:
    scov = cov_shrinkage(cov, d['alpha'])

    ax = plt.subplot2grid((2, 2), d['pos'])
    ax.set_title(fr"$\alpha={d['alpha']}$")
    sns.heatmap(
        scov,
        vmin=min_cov,
        center=0,
        vmax=max_cov,
        cmap='vlag',
        linewidths=.5,
        linecolor='white',
        cbar_kws={"shrink": .5},
        annot=False,
        # fmt='.1f',
        ax=ax,
    )


plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/shrinkage basic cov.pdf')
plt.show()
plt.close()


# =============================
# ===== BASIC CORRELATION =====
# =============================
size = 8
fig = plt.figure(figsize=(size * (4 / 3), size))
fig.suptitle(
    "Basic Correlation Shrinkage",
    fontweight="bold",
)

subplotdicts = [
    {"alpha": 0.0, "pos": (0, 0)},
    {"alpha": 0.3, "pos": (0, 1)},
    {"alpha": 0.6, "pos": (1, 0)},
    {"alpha": 0.9, "pos": (1, 1)},
]
for d in subplotdicts:
    scov = corr_shrinkage(cov, d['alpha'])

    ax = plt.subplot2grid((2, 2), d['pos'])
    ax.set_title(fr"$\alpha={d['alpha']}$")
    sns.heatmap(
        scov,
        vmin=min_cov,
        center=0,
        vmax=max_cov,
        cmap='vlag',
        linewidths=.5,
        linecolor='white',
        cbar_kws={"shrink": .5},
        annot=False,
        # fmt='.1f',
        ax=ax,
    )


plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/Aulas/Doutorado - International Finance/Research Project/figures/shrinkage basic corr.pdf')
plt.show()
plt.close()
