"""
hmmlearn==0.2.7
"""

import pandas as pd
import numpy as np
import getpass
from hmmlearn import hmm

n_regimes = 3
username = getpass.getuser()
file_path = f"/Users/{username}/Dropbox/Lectures/MPE - Asset Allocation/2025/Dados BBG AA Course.xlsx"  # TODO add file to repository

# Total Return
idxs = pd.read_excel(file_path, sheet_name='USER', skiprows=4, index_col=0)
idxs = idxs.sort_index()
idxs = idxs.dropna(how='all')

rename_tickers = {
    "SPX Index": "S&P500",  # total return
    "SOFRRATE Index": "SOFR",
    "US0001M Index": "LIBOR",
    "BCOMTR Index": "BCOM",  # Total return
    "SPUSTTTR Index": "US 10y",  # Excess Return
    "ERIXCDIG Index": "CDG IG",  # Excess Return
    "ERINCDHY Index": "CDX HY", # Excess Return
}
idxs = idxs[rename_tickers.keys()].rename(rename_tickers, axis=1)

rf = idxs['SOFR'].fillna(idxs['LIBOR']).dropna()
rf = (1 + rf/100)**(1/261) - 1

rets = pd.DataFrame(index=idxs.index)
rets['S&P500'] = idxs['S&P500'].pct_change(1) - rf
rets['BCOM'] = idxs['BCOM'].pct_change(1) - rf
rets['US 10y'] = idxs['US 10y'].pct_change(1)
# rets['CDG IG'] = idxs['CDG IG'].pct_change(1)
# rets['CDX HY'] = idxs['CDX HY'].pct_change(1)

eri = (1 + rets).cumprod().dropna()
eri = 100 * eri / eri.iloc[0]
eri = eri.resample("ME").last()
er = eri.pct_change(1).dropna()

model = hmm.GaussianHMM(
    n_components=n_regimes,
    covariance_type='full',
    n_iter=500,
)
model.fit(er)

sort_order = np.flip(np.argsort(np.diag(model.transmat_)))
sorted_model = hmm.GaussianHMM(n_components=n_regimes,
                               covariance_type='full')

sorted_model.startprob_ = model.startprob_[sort_order]
sorted_model.transmat_ = pd.DataFrame(model.transmat_).loc[sort_order, sort_order].values
sorted_model.means_ = model.means_[sort_order, :]
sorted_model.covars_ = model.covars_[sort_order, :, :]


print("Transition Matrix")
print(sorted_model.transmat_)
