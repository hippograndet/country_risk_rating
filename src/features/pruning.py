import pandas as pd
import numpy as np

def drop_unstable_features(
    X: pd.DataFrame,
    group_col: str,
    max_cv: float = 1.5
) -> pd.DataFrame:
    """
    Drops features with high coefficient of variation across groups (e.g. years)
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
    corr = X.corrwith(y).abs()
    keep = corr[corr >= min_abs_corr].index
    return X[keep]