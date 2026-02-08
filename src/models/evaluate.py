from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X, y, prefix = '', verbose=True):
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Evaluation metrics
    results = {
        prefix + 'accuracy': accuracy_score(y, y_pred),
        prefix + 'precision': precision_score(y, y_pred, average='macro'),
        prefix + 'recall': recall_score(y, y_pred, average='macro'),
        prefix + 'f1': f1_score(y, y_pred, average='macro'),
        # 'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr')
    }

    if verbose:
        print('Baseline Logistic Regression Results')
        for k, v in results.items():
            print(f'{k}: {v:.4f}')

        print('\nClassification Report')
        print(classification_report(y, y_pred))

        print('\nConfusion Matrix')
        print(confusion_matrix(y, y_pred))

    return results