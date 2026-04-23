"""
Feature selection utilities for country risk rating prediction.

Provides column-level filters (missingness, variance, correlation, mutual
information) and a row-level sparsity filter applied before the modelling
pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression


def filter_missingness(
    X: pd.DataFrame,
    max_missing_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Drop columns where the fraction of missing values exceeds a threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    max_missing_ratio : float, optional
        Maximum fraction of NaN values allowed per column (default 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with high-missingness columns removed.
    """
    missing_ratio = X.isna().mean()
    keep_cols = missing_ratio[missing_ratio <= max_missing_ratio].index

    return X[keep_cols]


def filter_low_variance(
    X: pd.DataFrame,
    threshold: float = 1e-4
) -> pd.DataFrame:
    """
    Drop numeric columns whose variance falls below a threshold.

    Non-numeric columns are always retained.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    threshold : float, optional
        Minimum variance required to keep a column (default 1e-4).

    Returns
    -------
    pd.DataFrame
        DataFrame with near-constant numeric columns removed.
    """
    float_features = X.columns[X.dtypes == float]
    non_float_features = X.columns[X.dtypes != float]

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X[float_features].fillna(0))

    return X.loc[:, list(non_float_features) + list(selector.get_feature_names_out())]


def filter_correlated(
    X: pd.DataFrame,
    max_corr: float = 0.9
) -> pd.DataFrame:
    """
    Drop numeric columns that are highly correlated with another column.

    For each pair exceeding ``max_corr``, the later column (upper-triangle
    traversal) is removed.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    max_corr : float, optional
        Pearson correlation threshold above which a column is dropped
        (default 0.9).

    Returns
    -------
    pd.DataFrame
        DataFrame with redundant correlated columns removed.
    """
    float_features = X.columns[X.dtypes == float]

    corr = X[float_features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > max_corr)]

    return X.drop(columns=drop_cols)


def select_by_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 100
) -> pd.DataFrame:
    """
    Retain the top-k numeric features by mutual information with the target.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix (numeric columns only are evaluated).
    y : pd.Series
        Target variable.
    top_k : int, optional
        Number of top features to keep (default 100).

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the top-k features by mutual information.
    """
    float_features = X.columns[X.dtypes == float]

    mi = mutual_info_regression(X[float_features].fillna(0), y)
    scores = pd.Series(mi, index=float_features)
    keep = scores.sort_values(ascending=False).head(top_k).index

    return X[keep]


def filter_sparse_rows(
    dataset: pd.DataFrame,
    max_missing_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Drop rows where the fraction of NaN columns exceeds max_missing_ratio.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset (features + target).
    max_missing_ratio : float, optional
        Maximum fraction of NaN values allowed per row (default 0.5).

    Returns
    -------
    pd.DataFrame
        Dataset with sparse rows removed.
    """
    row_missing = dataset.isna().mean(axis=1)
    return dataset[row_missing <= max_missing_ratio]


def get_dataset_feature_selected(
    dataset: pd.DataFrame,
    params: dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply the full column-level feature selection pipeline to a dataset.

    Removes rows with an invalid OECD rating, then sequentially applies
    missingness, low-variance, and correlation filters.

    Parameters
    ----------
    dataset : pd.DataFrame
        Merged dataset containing features and the ``OECD_RATING`` column.
    params : dict
        Selection thresholds with keys: ``max_missing_ratio``,
        ``low_var_threshold``, ``max_corr``.
    verbose : bool, optional
        Print shape after each filter step (default True).

    Returns
    -------
    pd.DataFrame
        Feature-selected dataset.
    """
    # non_float_features_t = list(dataset.columns[dataset.dtypes!=float])

    if verbose:
        print('Initial Dataset Shape:', dataset.shape)
    dataset = dataset[dataset['OECD_RATING'] != '-']
    if verbose:
        print('Drop Null Target Shape:', dataset.shape)
    dataset = filter_missingness(
        dataset,
        max_missing_ratio=params['max_missing_ratio']
    )
    if verbose:
        print('Filter Missingness Shape:', dataset.shape)
    dataset = filter_low_variance(
        dataset,
        threshold=params['low_var_threshold']
    )
    if verbose:
        print('Filter Low Variance Shape:', dataset.shape)
    dataset = filter_correlated(
        dataset,
        max_corr=params['max_corr']
    )
    # if verbose:
    #     print('Filter Correlated Shape:', dataset.shape)
    # dataset = selection.select_by_mutual_information(
    #     dataset.drop(columns=['OECD_RATING']),
    #     dataset['OECD_RATING'],
    #     top_k=params['top_k_mi']
    # )
    # print('Filter Mutual Information Shape:', dataset.shape)
    if verbose:
        print('Final Dataset Shape:', dataset.shape)

    return dataset

def get_dataset_feature_selected_counts(
    dataset: pd.DataFrame,
    params: dict,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the full column-level feature selection pipeline to a dataset.

    Removes rows with an invalid OECD rating, then sequentially applies
    missingness, low-variance, and correlation filters.

    Parameters
    ----------
    dataset : pd.DataFrame
        Merged dataset containing features and the ``OECD_RATING`` column.
    params : dict
        Selection thresholds with keys: ``max_missing_ratio``,
        ``low_var_threshold``, ``max_corr``.
    verbose : bool, optional
        Print shape after each filter step (default True).

    Returns
    -------
    pd.DataFrame
        Feature-selected dataset.
    """
    # non_float_features_t = list(dataset.columns[dataset.dtypes!=float])
    counts = {}
    counts['Initial'] = dataset.shape
    if verbose:
        print('Initial Dataset Shape:', dataset.shape)
    dataset = dataset[dataset['OECD_RATING'] != '-']
    counts['Drop Null Target'] = dataset.shape
    if verbose:
        print('Drop Null Target Shape:', dataset.shape)
    dataset = filter_missingness(
        dataset,
        max_missing_ratio=params['max_missing_ratio']
    )
    counts['Filter Missingness'] = dataset.shape
    if verbose:
        print('Filter Missingness Shape:', dataset.shape)
    dataset = filter_low_variance(
        dataset,
        threshold=params['low_var_threshold']
    )
    counts['Filter Low Variance'] = dataset.shape
    if verbose:
        print('Filter Low Variance Shape:', dataset.shape)
    dataset = filter_correlated(
        dataset,
        max_corr=params['max_corr']
    )
    counts['Filter Correlated'] = dataset.shape
    # if verbose:
    #     print('Filter Correlated Shape:', dataset.shape)
    # dataset = selection.select_by_mutual_information(
    #     dataset.drop(columns=['OECD_RATING']),
    #     dataset['OECD_RATING'],
    #     top_k=params['top_k_mi']
    # )
    # print('Filter Mutual Information Shape:', dataset.shape)
    counts['Final'] = dataset.shape
    if verbose:
        print('Final Dataset Shape:', dataset.shape)

    return dataset, counts
