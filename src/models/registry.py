from joblib import dump
import json
from datetime import datetime

from src.utils import config, io

def update_registry_index(model_name, framework, version, f1_score, path):
    registry_index = io.load_csv(config.PROJECT_ROOT / 'models' / 'model_index.csv')

    registry_index.loc[len(registry_index)] = [
        model_name,
        framework,
        version, 
        f1_score,
        path
    ]

    io.save_csv(registry_index, config.PROJECT_ROOT / 'models' / 'model_index.csv')

def save_to_registry(model, model_name='baseline_lr', framework='sklearn', test_metrics={}, run_id=0, version=1):

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
