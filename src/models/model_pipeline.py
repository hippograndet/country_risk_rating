"""
Model factory and sklearn Pipeline construction for country risk rating.

Supports three architectures: logistic regression (baseline), XGBoost
classifier, and XGBoost regressor.  Each is wrapped in a standard
``sklearn.pipeline.Pipeline`` that applies the preprocessor before the model.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def get_model(
    model_name: str,
    params: dict = {}
) -> object:
    """
    Instantiate a model by name with the given hyperparameters.

    Parameters
    ----------
    model_name : str
        One of ``'logistic_regression'``, ``'xgboost_classifier'``, or
        ``'xgboost_regressor'``.
    params : dict, optional
        Keyword arguments forwarded to the model constructor (default ``{}``).

    Returns
    -------
    object
        An unfitted sklearn-compatible estimator.
    """
    if model_name == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_name == 'xgboost_classifier':
        model = xgb.XGBClassifier(**params)
    elif model_name == 'xgboost_regressor':
        model = xgb.XGBRegressor(**params)
    else:
        print('No model architecture with name', model_name)
        print('Getting Default Model, XGBoost Classifier')
        model = xgb.XGBClassifier(**params)

    return model


def get_model_pipeline(
    model_name: str,
    preprocessor: ColumnTransformer,
    params: dict = {}
) -> Pipeline:
    """
    Build a two-step sklearn Pipeline: preprocessor → model.

    Parameters
    ----------
    model_name : str
        Model architecture identifier (see :func:`get_model`).
    preprocessor : ColumnTransformer
        Fitted or unfitted preprocessing transformer (e.g. from
        :func:`src.preprocessing.preprocess_pipeline.build_preprocessor`).
    params : dict, optional
        Hyperparameters forwarded to the model constructor (default ``{}``).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline with steps ``('preprocessor', ...)`` and
        ``('model', ...)``.
    """
    model = get_model(model_name, params)

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return model_pipeline
