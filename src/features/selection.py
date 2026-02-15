import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

def filter_missingness(
    X: pd.DataFrame,
    max_missing_ratio: float = 0.5
) -> pd.DataFrame:
    missing_ratio = X.isna().mean()
    keep_cols = missing_ratio[missing_ratio <= max_missing_ratio].index

    return X[keep_cols]


def filter_low_variance(
    X: pd.DataFrame,
    threshold: float = 1e-4
) -> pd.DataFrame:
    
    float_features = X.columns[X.dtypes==float]
    non_float_features = X.columns[X.dtypes!=float]

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X[float_features].fillna(0))

    return X.loc[:, list(non_float_features) + list(selector.get_feature_names_out())]


def filter_correlated(
    X: pd.DataFrame,
    max_corr: float = 0.9
) -> pd.DataFrame:
    
    float_features = X.columns[X.dtypes==float]

    corr = X[float_features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > max_corr)]

    return X.drop(columns=drop_cols)


def select_by_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 100
) -> pd.DataFrame:
    
    float_features = X.columns[X.dtypes==float]

    mi = mutual_info_regression(X[float_features].fillna(0), y)
    scores = pd.Series(mi, index=float_features)
    keep = scores.sort_values(ascending=False).head(top_k).index
    
    return X[keep]


def get_dataset_feature_selected(dataset, params, verbose=True):
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
        print(params)
        print('Final Dataset Shape:', dataset.shape)
        
    return dataset
