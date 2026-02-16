from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error, 
    mean_squared_error
)
import numpy as np

def get_distance_accuracy_ratio(y_test, y_pred):
    y_diff = y_test['OECD_RATING'] - y_pred
    
    return y_diff.abs().sum() / len(y_test)

def get_blurred_accuracy(y_test, y_pred):

    acc_1 = accuracy_score(y_test, y_pred)
    acc_2 = accuracy_score(y_test + 1, y_pred)
    acc_3 = accuracy_score(y_test - 1, y_pred)

    return acc_1 + acc_2 + acc_3

def evaluate_classification(y_test, y_pred, prefix):

    results = {
        prefix + 'accuracy': accuracy_score(y_test, y_pred),
        prefix + 'precision': precision_score(y_test, y_pred, average='macro', zero_division=np.nan),
        prefix + 'recall': recall_score(y_test, y_pred, average='macro', zero_division=np.nan),
        prefix + 'f1': f1_score(y_test, y_pred, average='macro'),
        prefix + 'blurred_accuracy': get_blurred_accuracy(y_test, y_pred),
        prefix + 'dist_accuracy_ratio': get_distance_accuracy_ratio(y_test, y_pred)
        # 'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
    }
    
    return results

def evaluate_model_classifier(model, X, y, prefix = '', verbose=True):
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    if len(y_pred) == 0:
        print('No preds')
        return {}
    
    results = evaluate_classification(y, y_pred, prefix)

    if verbose:
        print('Baseline Logistic Regression Results')
        for k, v in results.items():
            print(f'{k}: {v:.4f}')

        print('\nClassification Report')
        print(classification_report(y, y_pred))

        print('\nConfusion Matrix')
        print(confusion_matrix(y, y_pred))

    return results


def evaluate_model_regressor(model, X, y, prefix = '', verbose=True):
    # Predictions
    y_pred = model.predict(X)
    y_pred_class = np.round(np.clip(y_pred, 1, 7))

    # print(y_pred)
    # Evaluation metrics

    if len(y_pred) == 0:
        print('No preds')
        return {}
    
    results = evaluate_classification(y, y_pred_class, prefix)

    # Regression Evaluation Metrics
    results[prefix + 'mae'] = mean_absolute_error(y, y_pred)
    results[prefix + 'mse'] = mean_squared_error(y, y_pred)

    if verbose:
        print('Baseline Logistic Regression Results')
        for k, v in results.items():
            print(f'{k}: {v:.4f}')

        print('\nClassification Report')
        print(classification_report(y, y_pred_class))

        print('\nConfusion Matrix')
        print(confusion_matrix(y, y_pred_class))

    return results

def evaluate_ensemble_model(models, X, y, prefix = '', verbose=True):

    y_pred_clas = models['clas'].predict(X) + 1

    y_pred_reg = models['reg'].predict(X)
    y_pred_reg_class = np.round(np.clip(y_pred_reg, 1, 7))

    y_pred_mean = (y_pred_clas + y_pred_reg) / 2
    y_pred_mean_clas = np.round(np.clip(y_pred_mean, 1, 7))


    results = evaluate_classification(y, y_pred_mean_clas, prefix + 'meanPred_')
    results[prefix + 'meanPred_' + 'mae'] = mean_absolute_error(y, y_pred_mean)
    results[prefix + 'meanPred_' + 'mse'] = mean_squared_error(y, y_pred_mean)

    if verbose:
        print(results)

    return results



