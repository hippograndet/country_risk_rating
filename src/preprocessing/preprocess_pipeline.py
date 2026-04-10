"""
Sklearn preprocessing pipeline factory for country risk rating models.

Builds a ``ColumnTransformer`` that applies imputation and scaling to numeric
features and, optionally, imputation and one-hot encoding to categorical
features.  The transformer is configured via a params dictionary so that
different configurations can be tracked in MLflow experiments.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer


def build_preprocessor(
    X: pd.DataFrame,
    params: dict
) -> ColumnTransformer:
    """
    Build a ``ColumnTransformer`` preprocessing pipeline from a params dict.

    Numeric pipeline
    ~~~~~~~~~~~~~~~~
    * Imputer: either KNN (``params['num_imputer'] == 'knn'``) or
      ``SimpleImputer`` for any other value.
    * Scaler: ``StandardScaler``.

    Categorical pipeline (only added when ``params['cat_imputer'] == 'uni'``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Imputer: ``SimpleImputer`` with constant fill.
    * Encoder: ``OneHotEncoder(handle_unknown='ignore')``.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix, used only to derive column dtypes via
        ``make_column_selector``.
    params : dict
        Configuration keys:

        * ``num_imputer`` (str): ``'knn'`` or any string for mean/mode.
        * ``num_imputer_knn_weights`` (str, optional): KNN weights
          (default ``'uniform'``).
        * ``num_imputer_knn_n_neighbors`` (int, optional): KNN neighbours
          (default ``5``).
        * ``num_imputer_uni_strategy`` (str, optional): SimpleImputer
          strategy (default ``'mean'``).
        * ``cat_imputer`` (str): ``'uni'`` to include a categorical pipeline.
        * ``cat_imputer_uni_strategy`` (str, optional): fill strategy
          (default ``'constant'``).
        * ``cat_imputer_uni_fill_value`` (str, optional): fill value
          (default ``'NaN'``).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessing transformer.
    """
    preprocessor_l = []

    if params['num_imputer'] == 'knn':
        num_imputer = KNNImputer(
            weights=params.get('num_imputer_knn_weights', 'uniform'),
            n_neighbors=params.get('num_imputer_knn_n_neighbors', 5)
        )
    else:
        num_imputer = SimpleImputer(
            strategy=params.get('num_imputer_uni_strategy', 'mean'),
            fill_value=params.get('num_imputer_uni_fill_value', np.nan)
        )

    num_pipeline_l = [
        ('imputer', num_imputer),
        ('scaler', StandardScaler())
    ]

    preprocessor_l.append(
        ('num', Pipeline(num_pipeline_l), make_column_selector(dtype_exclude='object'))
    )

    if params['cat_imputer'] == 'uni':
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(
                strategy=params.get('cat_imputer_uni_strategy', 'constant'),
                fill_value=params.get('cat_imputer_uni_fill_value', 'NaN')
            )),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor_l.append(
            ('cat', categorical_pipeline, make_column_selector(dtype_include='object'))
        )
    else:
        pass

    preprocessor = ColumnTransformer(preprocessor_l)

    return preprocessor
