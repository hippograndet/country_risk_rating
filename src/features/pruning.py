"""
Additional feature pruning utilities for country risk rating prediction.

These filters complement the main selection pipeline and are available for
experimental use (e.g. in the experiments notebooks).
"""

import pandas as pd


def drop_unstable_features(
    X: pd.DataFrame,
    group_col: str,
    max_cv: float = 1.5
) -> pd.DataFrame:
    """
    Drop features with a high coefficient of variation across groups.

    Useful for removing indicators whose group-level means are erratic
    (e.g. a feature that swings wildly year-over-year).

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix including the grouping column.
    group_col : str
        Column name to group by (e.g. ``'YEAR'``).
    max_cv : float, optional
        Maximum allowed coefficient of variation. Columns exceeding this
        value are dropped (default 1.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with unstable columns removed.
    """
    keep_cols = []

    for col in X.columns:
        grouped = X.groupby(group_col)[col].mean()
        cv = grouped.std() / (grouped.mean() + 1e-6)
        if cv < max_cv:
            keep_cols.append(col)

    return X[keep_cols]


def drop_low_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    min_abs_corr: float = 0.05
) -> pd.DataFrame:
    """
    Drop features with a low absolute Pearson correlation with the target.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    y : pd.Series
        Target variable.
    min_abs_corr : float, optional
        Minimum absolute correlation required to keep a feature (default 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame containing only features meeting the correlation threshold.
    """
    corr = X.corrwith(y).abs()
    keep = corr[corr >= min_abs_corr].index
    return X[keep]
