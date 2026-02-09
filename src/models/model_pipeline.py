from sklearn.pipeline import Pipeline
# import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
# import pytorch


def get_model(model_name, params={}):

    if model_name == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_name == 'xgboost':
        model = xgb.XGBClassifier(objective='multi:softprob', **params)
    elif model_name == 'torch':
        # model = xgb.XGBClassifier(objective='multi:softprob', **params)
        pass
    else:
        model = LogisticRegression(**params)

    return model

def get_model_pipeline(model_name, preprocessor, params={}):

    model = get_model(model_name, params)

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return model_pipeline