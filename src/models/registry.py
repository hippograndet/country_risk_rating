from joblib import dump
import json
from datetime import datetime

from src.utils import config

def save_to_registry(model, model_type='baseline_lr', test_metrics={}, run_id=0):

    registry_path = config.PROJECT_ROOT / 'models/registry' / model_type / 'v1'
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
