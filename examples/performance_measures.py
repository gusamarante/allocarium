import pandas as pd
from allocarium.utils import Performance
import getpass


username = getpass.getuser()
file_path = f"/Users/{username}/Dropbox/Aulas/Insper - Asset Allocation/2025/Dados BBG AA Course.xlsx"  # TODO add file to repository


# Total Return
idxs = pd.read_excel(file_path, sheet_name='TOT_RETURN_INDEX_GROSS_DVDS', skiprows=4, index_col=0)
idxs = idxs.sort_index()
idxs = idxs.dropna(how='all')

rename_tickers = {
    "SPX Index": "S&P500",
    "SXXP Index": "EuroStoxx 600",
    "TPX Index": "Topix",
    "IBOV Index": "Ibovespa",
    "SPBDU1BT Index": "US 10y",
    "SPGCCLP Index": "Crude Oil",
    "SPGCGCP Index": "Gold",
    "ERIXCDIG Index": "CDX IG",
    "ERINCDHY Index": "CDX HY",
    "BCOMTR Index": "BCOM",
}
idxs = idxs[rename_tickers.keys()].rename(rename_tickers, axis=1)

perf = Performance(idxs)

print(perf.table)

for col in idxs.columns:
    perf.plot_underwater(col, show_chart=True)

for col in idxs.columns:
    perf.plot_drawdowns(col, show_chart=True)