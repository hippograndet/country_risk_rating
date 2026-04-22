"""
Training entry point for country risk rating models.

Loads the processed feature matrix (X.csv / y.csv), applies the sklearn
preprocessing pipeline, trains the chosen model, evaluates it on the
temporal test split, and logs everything to MLflow.

Usage
-----
    python -m src.train                               # default: xgboost_classifier
    python -m src.train --model logistic_regression
    python -m src.train --model xgboost_regressor
    python -m src.train --model xgboost_classifier --register
"""

import argparse

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.models import evaluate, model_pipeline
from src.preprocessing import preprocess_pipeline
from src.utils import config, io

# ── Default configurations ────────────────────────────────────────────────────

PREPROCESSOR_PARAMS = {
    "num_imputer": "uni",
    "num_imputer_uni_strategy": "mean",
    "cat_imputer": "uni",
    "cat_imputer_uni_strategy": "most_frequent",
}

MODEL_PARAMS = {
    "logistic_regression": {
        "penalty": "l2",
        "C": 10,
        "solver": "newton-cg",
        "max_iter": 1000,
    },
    "xgboost_classifier": {
        "objective": "multi:softprob",
        "num_class": 7,
        "gamma": 0.0,
        "max_depth": 10,
        "min_child_weight": 1.0,
        "colsample_bytree": 0.8,
        "subsample": 1.0,
    },
    "xgboost_regressor": {
        "objective": "reg:squarederror",
        "gamma": 0.0,
        "max_depth": 10,
        "min_child_weight": 1.0,
        "colsample_bytree": 0.6,
        "subsample": 1.0,
    },
}

# ── Data helpers ──────────────────────────────────────────────────────────────


def load_data():
    X = io.load_csv(config.PROCESSED_DATA_DIR / "X.csv", index_col=0)
    y = io.load_csv(config.PROCESSED_DATA_DIR / "y.csv", index_col=0)
    return X, y


def get_splits(X, y, split_cfg):
    def _mask(data, bounds):
        return (data["YEAR"] >= bounds[0]) & (data["YEAR"] <= bounds[1])

    X_train = X[_mask(X, split_cfg["train_years"])]
    X_test = X[_mask(X, split_cfg["test_years"])]
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]
    return X_train, X_test, y_train, y_test


# ── Training & evaluation ─────────────────────────────────────────────────────


def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    """
    Train a model and return (fitted_pipeline, metrics_dict).

    y_test is kept as a DataFrame so that evaluate.evaluate_classification
    can compute dist_accuracy_ratio via y_test['OECD_RATING'].
    All predictions are decoded back to the original 1-7 rating scale before
    evaluation so metrics are comparable across model types.
    """
    preprocessor = preprocess_pipeline.build_preprocessor(X_train, PREPROCESSOR_PARAMS)
    model = model_pipeline.get_model_pipeline(
        model_name, preprocessor, MODEL_PARAMS[model_name]
    )

    if model_name == "xgboost_classifier":
        # XGBoost requires 0-indexed labels; encode for training, decode for eval
        encoder = LabelEncoder()
        encoder.fit(y_train.values.ravel())
        model.fit(X_train, encoder.transform(y_train.values.ravel()))
        y_pred = encoder.inverse_transform(model.predict(X_test))
        metrics = evaluate.evaluate_classification(y_test, y_pred, prefix="test_")

    elif model_name == "xgboost_regressor":
        model.fit(X_train, y_train.values.ravel())
        y_pred_raw = model.predict(X_test)
        y_pred = np.round(np.clip(y_pred_raw, 1, 7))
        metrics = evaluate.evaluate_classification(y_test, y_pred, prefix="test_")
        metrics["test_mae"] = float(np.mean(np.abs(y_test.values.ravel() - y_pred_raw)))
        metrics["test_mse"] = float(np.mean((y_test.values.ravel() - y_pred_raw) ** 2))

    else:  # logistic_regression (handles multi-class natively)
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        metrics = evaluate.evaluate_classification(y_test, y_pred, prefix="test_")

    return model, metrics


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    # Load data
    X, y = load_data()
    split_cfg = io.load_json(
        config.PROCESSED_DATA_DIR / "splits" / f"{args.split}.json"
    )
    X_train, X_test, y_train, y_test = get_splits(X, y, split_cfg)

    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")

    # MLflow setup
    mlflow.set_tracking_uri(config.PROJECT_ROOT / "models" / "mlruns")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.model}__{args.split}"):

        mlflow.log_params(
            {
                "model": args.model,
                "split": args.split,
                "train_years": str(split_cfg["train_years"]),
                "test_years": str(split_cfg["test_years"]),
                **MODEL_PARAMS[args.model],
                **PREPROCESSOR_PARAMS,
            }
        )

        model, metrics = train_and_evaluate(
            args.model, X_train, X_test, y_train, y_test
        )

        # MLflow accepts numeric scalars; cast numpy types to be safe
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=args.model if args.register else None,
        )

        run_id = mlflow.active_run().info.run_id

    print(f"\nModel     : {args.model}")
    print(f"Test F1   : {metrics.get('test_f1', float('nan')):.4f}")
    print(f"Test Acc  : {metrics.get('test_accuracy', float('nan')):.4f}")
    print(f"Run ID    : {run_id}")
    if args.register:
        print(f"Registered: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a country risk rating model and log results to MLflow."
    )
    parser.add_argument(
        "--model",
        default="xgboost_classifier",
        choices=list(MODEL_PARAMS.keys()),
        help="Model architecture to train (default: xgboost_classifier)",
    )
    parser.add_argument(
        "--split",
        default="temporal_v1",
        help="Split config name under data/3-processed/splits/ (default: temporal_v1)",
    )
    parser.add_argument(
        "--experiment",
        default="Country Risk Prediction",
        help="MLflow experiment name (default: 'Country Risk Prediction')",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the trained model in the MLflow Model Registry",
    )

    main(parser.parse_args())
