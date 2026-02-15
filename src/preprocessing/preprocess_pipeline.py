import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer

def build_preprocessor(X, params):

    categorical_cols = X.columns[X.dtypes == 'object']
    numeric_cols = X.columns[X.dtypes != 'object']
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