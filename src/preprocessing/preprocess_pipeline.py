from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer

def build_preprocessor(X, numeric_imputer='knn', categorical_imputer='most_frequent', scaler=True):

    categorical_cols = X.columns[X.dtypes == 'object']
    numeric_cols = X.columns[X.dtypes != 'object']

    num_pipeline_l = []
    if numeric_imputer == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer()
    num_pipeline_l.append(('imputer', imputer))

    if scaler:
        num_pipeline_l.append(('scaler', StandardScaler()))

    numeric_pipeline = Pipeline(num_pipeline_l)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=categorical_imputer)),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor