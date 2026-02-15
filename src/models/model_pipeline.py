from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
# import pytorch


def get_model(model_name, params={}):

    if model_name == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_name == 'xgboost_classifier':
        model = xgb.XGBClassifier(**params)
    elif model_name == 'xgboost_regressor':
        model = xgb.XGBRegressor(**params)
    else:
        print('No model architecture with name', model_name)

    return model

def get_model_pipeline(model_name, preprocessor, params={}):

    model = get_model(model_name, params)

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return model_pipeline