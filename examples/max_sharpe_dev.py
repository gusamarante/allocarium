from allocarium.utils import mu_cov_example
from allocarium.models import MaxSharpe

mu, cov = mu_cov_example()

ms = MaxSharpe(mu, cov, 0.08, allow_shorts=False)
print(ms.risky_weights)

