from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.impute import KNNImputer, SimpleImputer

def build_preprocessor(X, params):

    categorical_cols = X.columns[X.dtypes == 'object']
    numeric_cols = X.columns[X.dtypes != 'object']
    preprocessor_l = []

    num_pipeline_l = []
    if params['num_imputer'] == 'knn':
        num_imputer = KNNImputer(
            weights=params['num_imputer_knn_weights'],
            n_neighbors=params['num_imputer_knn_n_neighbors']
        )
    elif params['num_imputer'] == 'knn':
        num_imputer = SimpleImputer(
            strategy=params['num_imputer_uni_strategy'],
            fill_value=params['num_imputer_uni_fill_value']
        )
    else:
        num_imputer = SimpleImputer(strategy='mean')

    num_pipeline_l.append(('imputer', num_imputer))

    if params['num_scaler']:
        num_pipeline_l.append(('scaler', StandardScaler()))

    preprocessor_l.append(
        ('num', Pipeline(num_pipeline_l), make_column_selector(dtype_exclude='object'))
    )

    if params['cat_imputer'] == 'uni':
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(
                strategy=params['cat_imputer_uni_strategy'],
                fill_value=params['cat_imputer_uni_fill_value']
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