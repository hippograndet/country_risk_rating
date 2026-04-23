"""
Evaluation utilities for country risk rating models.

Provides classification metrics (accuracy, precision, recall, F1), two
domain-specific metrics (blurred accuracy and distance accuracy ratio), and
high-level evaluation functions for classifiers, regressors, and ensemble
models.
"""

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np
import pandas as pd


def get_mean_rating_error(
    y_test: pd.DataFrame,
    y_pred: np.ndarray
) -> float:
    """
    Mean absolute error between predicted and true rating (in rating steps).

    Lower is better; 0 means every prediction is exact.

    Parameters
    ----------
    y_test : pd.DataFrame
        True ratings. Accepts either a DataFrame with an ``OECD_RATING``
        column or a plain array/Series.
    y_pred : array-like
        Predicted ratings (1–7 scale).

    Returns
    -------
    float
        Mean absolute error in rating steps.
    """
    cols = getattr(y_test, 'columns', [])
    y_true = y_test['OECD_RATING'] if 'OECD_RATING' in cols else np.asarray(y_test)
    return np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)).mean()


def get_blurred_accuracy(
    y_test: pd.DataFrame,
    y_pred: np.ndarray
) -> float:
    """
    Fraction of predictions within ±1 rating step of the true value.

    Useful here because adjacent OECD ratings often reflect similar risk
    levels; a one-step error is materially less severe than a five-step error.

    Parameters
    ----------
    y_test : array-like
        True ratings.
    y_pred : array-like
        Predicted ratings (1–7 scale).

    Returns
    -------
    float
        Value in [0, 1]; higher is better.
    """
    y_true = np.asarray(y_test, dtype=float).ravel()
    y_hat = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_hat - y_true) <= 1))


def evaluate_classification(
    y_test: pd.DataFrame,
    y_pred: np.ndarray,
    prefix: str
) -> dict:
    """
    Compute standard classification metrics plus domain-specific metrics.

    Parameters
    ----------
    y_test : array-like
        True ratings.
    y_pred : array-like
        Predicted ratings (1–7 scale).
    prefix : str
        String prepended to every metric key (e.g. ``'test_'``).

    Returns
    -------
    dict
        Keys: ``{prefix}accuracy``, ``{prefix}precision``, ``{prefix}recall``,
        ``{prefix}f1``, ``{prefix}blurred_accuracy``,
        ``{prefix}dist_accuracy_ratio``.
    """
    results = {
        prefix + 'accuracy': accuracy_score(y_test, y_pred),
        prefix + 'precision': precision_score(
            y_test, y_pred, average='macro', zero_division=np.nan),
        prefix + 'recall': recall_score(
            y_test, y_pred, average='macro', zero_division=np.nan),
        prefix + 'f1': f1_score(y_test, y_pred, average='macro'),
        prefix + 'blurred_accuracy': get_blurred_accuracy(y_test, y_pred),
        prefix + 'mean_rating_error': get_mean_rating_error(y_test, y_pred)
        # 'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
    }

    return results


def evaluate_model_classifier(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    prefix: str = '',
    verbose: bool = True
) -> dict:
    """
    Generate predictions from a classifier and evaluate them.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted classification pipeline.
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        True ratings.
    prefix : str, optional
        Prepended to every metric key (default ``''``).
    verbose : bool, optional
        Print metrics, classification report, and confusion matrix
        (default True).

    Returns
    -------
    dict
        Evaluation metrics (see :func:`evaluate_classification`).
    """
    y_pred = model.predict(X)

    if len(y_pred) == 0:
        print('No preds')
        return {}

    results = evaluate_classification(y, y_pred, prefix)

    if verbose:
        print('Classifier Results')
        for k, v in results.items():
            print(f'{k}: {v:.4f}')

        print('\nClassification Report')
        print(classification_report(y, y_pred))

        print('\nConfusion Matrix')
        print(confusion_matrix(y, y_pred))

    return results


def evaluate_model_regressor(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    prefix: str = '',
    verbose: bool = True
) -> dict:
    """
    Generate predictions from a regressor, round to integer ratings, and
    evaluate them.

    Continuous predictions are clipped to [1, 7] and rounded before
    classification metrics are computed.  Raw regression metrics (MAE, MSE)
    are also included.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted regression pipeline.
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        True ratings.
    prefix : str, optional
        Prepended to every metric key (default ``''``).
    verbose : bool, optional
        Print metrics, classification report, and confusion matrix
        (default True).

    Returns
    -------
    dict
        Classification metrics plus ``{prefix}mae`` and ``{prefix}mse``.
    """
    y_pred = model.predict(X)
    y_pred_class = np.round(np.clip(y_pred, 1, 7))

    if len(y_pred) == 0:
        print('No preds')
        return {}

    results = evaluate_classification(y, y_pred_class, prefix)

    results[prefix + 'mae'] = mean_absolute_error(y, y_pred)
    results[prefix + 'mse'] = mean_squared_error(y, y_pred)

    if verbose:
        print('Regressor Results')
        for k, v in results.items():
            print(f'{k}: {v:.4f}')

        print('\nClassification Report')
        print(classification_report(y, y_pred_class))

        print('\nConfusion Matrix')
        print(confusion_matrix(y, y_pred_class))

    return results


def evaluate_ensemble_model(
    models: dict,
    X: pd.DataFrame,
    y: pd.DataFrame,
    prefix: str = '',
    verbose: bool = True
) -> dict:
    """
    Evaluate a simple ensemble that averages classifier and regressor outputs.

    The ensemble prediction is the mean of the classifier's class prediction
    (shifted back to 1-based) and the regressor's continuous prediction,
    then rounded to the nearest integer rating.

    Parameters
    ----------
    models : dict
        Dictionary with keys ``'clas'`` (fitted classifier pipeline) and
        ``'reg'`` (fitted regressor pipeline).
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        True ratings.
    prefix : str, optional
        Prepended to every metric key (default ``''``).
    verbose : bool, optional
        Print the results dict (default True).

    Returns
    -------
    dict
        Classification metrics plus MAE and MSE on the averaged prediction.
    """
    y_pred_clas = models['clas'].predict(X) + 1

    y_pred_reg = models['reg'].predict(X)

    y_pred_mean = (y_pred_clas + y_pred_reg) / 2
    y_pred_mean_clas = np.round(np.clip(y_pred_mean, 1, 7))

    results = evaluate_classification(y, y_pred_mean_clas, prefix + 'meanPred_')
    results[prefix + 'meanPred_' + 'mae'] = mean_absolute_error(y, y_pred_mean)
    results[prefix + 'meanPred_' + 'mse'] = mean_squared_error(y, y_pred_mean)

    if verbose:
        print(results)

    return results
