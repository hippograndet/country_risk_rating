"""
DEPRECATED — model persistence is now handled via MLflow.

Use ``mlflow.sklearn.log_model(..., registered_model_name=...)`` in
``src/train.py`` or ``mlflow.register_model(run_uri, name)`` in notebooks.
This file is kept for reference only and is no longer called.
"""

from joblib import dump
import json
from datetime import datetime
from pathlib import Path

from src.utils import config, io


def update_registry_index(
    model_name: str,
    framework: str,
    version: str,
    f1_score: float,
    path: Path
) -> None:
    """
    Append a new entry to the local model registry index CSV.

    Parameters
    ----------
    model_name : str
        Identifier for the model architecture.
    framework : str
        ML framework used (e.g. ``'sklearn'``).
    version : str
        Version string (e.g. ``'v1'``).
    f1_score : float
        Macro F1 score on the test set.
    path : Path
        Path to the saved model artefacts directory.
    """
    registry_index = io.load_csv(config.PROJECT_ROOT / 'models' / 'model_index.csv')

    registry_index.loc[len(registry_index)] = [
        model_name,
        framework,
        version,
        f1_score,
        path
    ]

    io.save_csv(registry_index, config.PROJECT_ROOT / 'models' / 'model_index.csv')


def save_to_registry(
    model: object,
    model_name: str = 'baseline_lr',
    framework: str = 'sklearn',
    test_metrics: dict = {},
    run_id: int = 0,
    version: int = 1
) -> None:
    """
    Persist a model artefact, metrics, and metadata to the local registry.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model pipeline to serialise.
    model_name : str, optional
        Registry identifier (default ``'baseline_lr'``).
    framework : str, optional
        ML framework label (default ``'sklearn'``).
    test_metrics : dict, optional
        Evaluation metrics to record alongside the model (default ``{}``).
    run_id : int, optional
        MLflow run ID associated with this model (default ``0``).
    version : int, optional
        Version number; determines the sub-directory name (default ``1``).
    """
    version = 'v' + str(version)
    registry_path = config.PROJECT_ROOT / 'models/registry' / model_name / version
    registry_path.mkdir(parents=True, exist_ok=True)

    dump(model, registry_path / 'model' / 'model.joblib')

    with open(registry_path / 'metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    with open(registry_path / 'metadata.json', 'w') as f:
        json.dump({
            'model': 'baseline_lr',
            'version': 'v1',
            'mlflow_run_id': run_id,
            'date': str(datetime.today()),
            'split': 'temporal_v1'
        }, f, indent=2)

    update_registry_index(
        model_name,
        framework,
        version,
        test_metrics['test_f1'],
        registry_path
    )
