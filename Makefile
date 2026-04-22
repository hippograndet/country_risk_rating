.PHONY: help setup test lint train train-lr train-xgb-reg train-xgb-cls mlflow-ui notebooks

help:
	@echo "Available commands:"
	@echo "  make setup        - Create virtualenv and install dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run flake8 on src and tests"
	@echo "  make train        - Train default model (xgboost_classifier)"
	@echo "  make train-lr     - Train logistic regression baseline"
	@echo "  make train-xgb-reg- Train xgboost regressor"
	@echo "  make train-xgb-cls- Train and register xgboost classifier"
	@echo "  make mlflow-ui    - Launch MLflow UI"
	@echo "  make notebooks    - Launch Jupyter notebooks"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -q

lint:
	flake8 src tests

train:
	python -m src.models.train

train-lr:
	python -m src.models.train --model logistic_regression

train-xgb-reg:
	python -m src.models.train --model xgboost_regressor

train-xgb-cls:
	python -m src.models.train --model xgboost_classifier --register

mlflow-ui:
	mlflow ui --backend-store-uri models/mlruns

notebooks:
	jupyter notebook notebooks/
