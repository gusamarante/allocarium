import numpy as np
import pandas as pd


# =====================
# ===== UTILITIES =====
# =====================
def cov2corr(cov):
    """
    Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    cov: pandas.DataFrame, numpy.array
        Covariance matrix

    Returns
    -------
    corr: pandas.DataFrame, numpy.array
        Correlation Matrix

    std: pandas.Series, numpy.array
        Vector of standard deviations. The square roots of the main diagonal
        of `cov`
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1  # correct for numerical error
    corr[corr > 1] = 1

    if isinstance(cov, pd.DataFrame):
        std = pd.Series(std, index=cov.index)

    return corr, std


def corr2cov(corr, std):
    """
    Convert a correlation matrix and a vector of standard deviations into a
    covariance matrix

    Parameters
    ----------
    corr: pandas.DataFrame, numpy.array
        Correlation matrix

    std: pandas.Series, numpy.array
        Vector of standard deviations. The square roots of the main diagonal
        of `cov`

    Returns
    -------
    cov: pandas.DataFrame, numpy.array
        Covariance Matrix
    """
    corr_a = np.array(corr)
    std = np.array(std)

    cov = np.diag(std) @ corr_a @ np.diag(std)

    if isinstance(corr, pd.DataFrame):
        cov = pd.DataFrame(data=cov, index=corr.index, columns=corr.columns)

    return cov


# =====================
# ===== SHRINKAGE =====
# =====================
def cov_shrinkage(cov, alpha=0.5):
    """
    Shrinks the covariance matrix `cov` towards homoskadasticity.
        scov = (1-alpha) * cov + alpha * (Tr(cov)/p) * eye(p)

    Parameters
    ----------
    cov: pandas.DataFrame, numpy.array
        Covariance matrix.

    alpha: float
        A scalar between 0 and 1. If alpha=0, there is no shrinkage and the
        output covariance will be the same as the input. If alpha=1, total
        shrinkage and the output will be a diagonal homoskedastic covariance.

    Returns
    -------
    scov: pandas.DataFrame, numpy.array
        Shrunk covariance matrix.
    """
    assert 0 <= alpha <= 1, "`alpha` must be a number between 0 and 1"

    p = cov.shape[0]
    anchor_cov = np.eye(p) * (np.trace(cov) / p)
    scov = (1 - alpha) * cov + alpha * anchor_cov
    return scov


def corr_shrinkage(cov, alpha=0.5):
    """
    Shrinks the correlation component of `cov` towards the identity.
        scorr = (1-alpha) * corr + alpha * eye(p)
    This method does not alter the main diagonal of the `cov` input. It
    preserves the variances.

    Parameters
    ----------
    cov: pandas.DataFrame, numpy.array
        Covariance matrix.

    alpha: float
        A scalar between 0 and 1. If alpha=0, there is no shrinkage and the
        output covariance will be the same as the input. If alpha=1, total
        shrinkage and the output will be just the main diagonal of the input
        covariance.
    """
    p = cov.shape[0]
    corr, vol = cov2corr(cov)
    scorr = (1 - alpha) * corr + alpha * np.eye(p)
    scov = np.diag(vol) @ scorr @ np.diag(vol)
    if isinstance(cov, pd.DataFrame):
        scov = pd.DataFrame(
            data=scov.values,
            index=cov.index,
            columns=cov.columns,
        )
    return scov