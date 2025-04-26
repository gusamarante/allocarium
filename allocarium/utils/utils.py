import numpy as np


def rescale_vol(df, target_vol=0.1):
    # TODO Review and better documentation
    """
    Rescale return indexes (total or excess) to have the desired volatitlity.
    :param df: pandas.DataFrame of return indexes.
    :param target_vol: float with the desired volatility.
    :return: pandas.DataFrame with rescaled total return indexes.
    """
    returns = df.pct_change(1, fill_method=None)
    returns_std = returns.std() * np.sqrt(252)
    returns = target_vol * returns / returns_std

    df_adj = (1 + returns).cumprod()
    df_adj = 100 * df_adj / df_adj.fillna(method='bfill').iloc[0]

    return df_adj
