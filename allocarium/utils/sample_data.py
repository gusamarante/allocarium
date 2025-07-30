import getpass
import numpy as np
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

    rename_tickers = {  # TODO choose better indexes
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


def mu_cov_example(size=3):
    """
    Example of mu and cov matrices to save lines on other codes.
    :return:
    """

    asset_names = ["Asset A", "Asset B", "Asset C"]
    mu = pd.Series(
        data=[0.1, 0.2, 0.15],
        index=asset_names,
    )
    std = pd.Series(
        data=[0.2, 0.25, 0.35],
        index=asset_names,
    )
    corr = pd.DataFrame(
        data=[
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ],
        index=asset_names,
        columns=asset_names,
    )
    cov = pd.DataFrame(
        data=np.diag(std.values) @ corr.values @ np.diag(std.values),
        index=asset_names,
        columns=asset_names,
    )

    mu = mu.iloc[:size]
    cov = cov.iloc[:size, :size]

    return mu, cov
