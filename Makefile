.PHONY: help setup setup-openssl-macos verify-ssl test lint train report explore train-classifier train-regressor mlflow-ui notebooks

PYTHON ?= python3

help:
	@echo "✅ After setup, choose one of these 3 paths:"
	@echo "  make report      📄 Read the full project report (start here)"
	@echo "  make explore     🔬 Explore as Data Scientist (Jupyter Notebooks)"
	@echo "  make train       🤖 Train model locally (default: XGBoost Classifier)"
	@echo ""
	@echo "Other commands:"
	@echo "  make setup        - Create virtualenv and install dependencies"
	@echo "  make setup-openssl-macos - macOS: OpenSSL-backed Python setup"
	@echo "  make verify-ssl   - Verify SSL backend"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make train-classifier - Train XGBoost Classifier model"
	@echo "  make train-regressor  - Train XGBoost Regressor model"
	@echo "  make mlflow-ui    - Launch MLflow experiment tracking UI"

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

setup-openssl-macos:
	@command -v brew >/dev/null 2>&1 || (echo "Homebrew is required on macOS for this target: https://brew.sh" && exit 1)
	brew install python@3.11 openssl@3
	"$$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel
	. .venv/bin/activate && python -m pip install -r requirements.txt
	$(MAKE) verify-ssl

verify-ssl:
	. .venv/bin/activate && python -c "import ssl,sys; v=ssl.OPENSSL_VERSION; print('SSL backend:', v); sys.exit(0 if v.startswith(('OpenSSL', 'LibreSSL')) else 1)"

test:
	PYTHONPATH=. pytest -q

lint:
	flake8 src tests

report:
	@echo "📄 Opening project report..."
	@open reports/Model_Report.md

explore:
	@echo "🔬 Launching Jupyter Notebooks..."
	. .venv/bin/activate && jupyter notebook notebooks/

train:
	. .venv/bin/activate && python -m src.models.train --model xgboost_classifier --register

train-classifier:
	. .venv/bin/activate && python -m src.models.train --model xgboost_classifier --register

train-regressor:
	. .venv/bin/activate && python -m src.models.train --model xgboost_regressor --register

# Legacy aliases for backwards compatibility
train-lr:
	python -m src.models.train --model logistic_regression

train-xgb-reg: train-regressor

train-xgb-cls: train-classifier

mlflow-ui:
	. .venv/bin/activate && mlflow ui --backend-store-uri models/mlruns

notebooks:
	. .venv/bin/activate && jupyter notebook notebooks/
