from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.impute import KNNImputer, SimpleImputer

def build_preprocessor(X, params):

    categorical_cols = X.columns[X.dtypes == 'object']
    numeric_cols = X.columns[X.dtypes != 'object']
    preprocessor_l = []

    num_pipeline_l = []
    if 'knn' in params['num_imputer'].keys():
        num_imputer = KNNImputer(**params['num_imputer']['knn'])
    elif 'uni' in params['num_imputer'].keys():
        num_imputer = SimpleImputer(**params['num_imputer']['uni'])
    else:
        num_imputer = SimpleImputer(strategy='mean')

    num_pipeline_l.append(('imputer', num_imputer))

    if params['num_scaler']:
        num_pipeline_l.append(('scaler', StandardScaler()))

    preprocessor_l.append(
        ('num', Pipeline(num_pipeline_l), make_column_selector(dtype_exclude='object'))
    )

    if 'cat_imputer' in params['cat'].keys():
        cat_imputer_params = params['cat']['cat_imputer']
        if cat_imputer_params == {}:
            cat_imputer_params = {'strategy': 'most_frequent'}

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(**cat_imputer_params)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor_l.append(
            ('cat', categorical_pipeline, make_column_selector(dtype_include='object'))
        )
    else:
        pass
    
    preprocessor = ColumnTransformer(preprocessor_l)

    return preprocessor