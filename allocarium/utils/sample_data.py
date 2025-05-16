import getpass
import pandas as pd

# TODO add CSV to repository
USERNAME = getpass.getuser()

def bbg_total_return():

    file_path = f"/Users/{USERNAME}/Dropbox/Aulas/Insper - Asset Allocation/2025/Dados BBG AA Course.xlsx"
    idxs = pd.read_excel(
        file_path,
        sheet_name='TOT_RETURN_INDEX_GROSS_DVDS',
        skiprows=4,
        index_col=0,
    ).sort_index().dropna(how='all')

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
    return idxs

