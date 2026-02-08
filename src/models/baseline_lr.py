from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

def get_model(preprocessor, params={}):
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('model', LogisticRegression(**params))
    ])

    return model