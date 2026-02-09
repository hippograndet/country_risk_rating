from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.impute import KNNImputer, SimpleImputer

def build_preprocessor(X, numeric_imputer='knn', categorical_imputer='most_frequent', scaler=True):

    categorical_cols = X.columns[X.dtypes == 'object']
    numeric_cols = X.columns[X.dtypes != 'object']

    num_pipeline_l = []
    if numeric_imputer == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif numeric_imputer == 'uni_mean':
        imputer = SimpleImputer(strategy='mean')
    elif numeric_imputer == 'uni_median':
        imputer = SimpleImputer(strategy='median')
    num_pipeline_l.append(('imputer', imputer))

    if scaler:
        num_pipeline_l.append(('scaler', StandardScaler()))

    num_pipeline_l.append(('norm', Normalizer()))
    numeric_pipeline = Pipeline(num_pipeline_l)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=categorical_imputer)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, make_column_selector(dtype_exclude='object')), # numeric_cols
        ('cat', categorical_pipeline, make_column_selector(dtype_include='object')) # categorical_cols
    ])

    return preprocessor