# OECD Country Risk Rating Prediction

> End-to-end machine learning pipeline predicting OECD country risk ratings (1–7) from World Bank macroeconomic indicators.

**Hippolyte Grandet** · MSc Artificial Intelligence, University of Edinburgh

---

## Overview

OECD country risk ratings reflect sovereign risk as assessed by expert committees. This project asks: *how well can publicly available macroeconomic data reproduce those assessments?*

The pipeline covers the full data science lifecycle — API ingestion, feature engineering, model training with MLflow tracking, and temporal evaluation — built to demonstrate both data science and MLOps practices.

---

## What this demonstrates

| Area | Details |
|---|---|
| **Data engineering** | API-driven ingestion (World Bank), PDF extraction (OECD), multi-source alignment |
| **Feature design** | 94 World Bank indicators across 6 economic dimensions + domain-derived engineered features |
| **ML pipeline** | Modular sklearn pipelines, temporal train/test splits, multi-class classification & regression |
| **MLOps** | MLflow experiment tracking, model registry, CLI training entrypoint (`src/train.py`) |
| **Code quality** | Reproducible structure (raw → interim → processed), separation of notebooks and src/ |

---

## Pipeline

```
┌─────────────────────────────────────────┐
│  DATA SOURCES                           │
│  World Bank API  ──►  94 macro          │
│  OECD PDF        ──►  risk ratings      │
│  Country metadata──►  ISO codes         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  INTEGRATION  (data/2-interim)          │
│  Merge by country × year                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  FEATURE SELECTION  (03-notebook / src) │
│  Missingness · Variance · Correlation   │
│  + Feature engineering (ENG_* columns) │
│  → 63 features, 4 081 observations      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  TRAINING  (src/train.py)               │
│  Temporal split: train 1999–2020        │
│                  test  2021–2024        │
│  sklearn preprocessing pipeline        │
│  MLflow: params · metrics · artifacts  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  EVALUATION  (04-notebook / MLflow UI)  │
│  Macro F1 · Accuracy · Blurred Acc.    │
└─────────────────────────────────────────┘
```

---

## Results

Test period: **2021–2024** (held-out temporal split, no leakage)

| Model | Macro F1 | Accuracy | Blurred Acc. (±1) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.604 | 0.711 | — |
| XGBoost Regressor | 0.487 | 0.560 | 0.896 |
| **XGBoost Classifier** | **0.747** | **0.811** | **0.955** |

**Blurred accuracy** measures the fraction of predictions within one rating step of the true value — relevant here because adjacent OECD ratings often reflect similar risk levels.

Key findings:
- Tree-based models capture non-linear interactions that linear models miss
- Temporal splits are essential — random splits artificially inflate performance
- Macroeconomic indicators explain ~75% of rating variation; residual error reflects expert judgment not captured by public data

Experiments are tracked in MLflow. Launch the UI with:
```bash
mlflow ui --backend-store-uri models/mlruns
```

---

## Quick Start

```bash
git clone https://github.com/hippograndet/country_risk_rating.git
cd country_risk_rating

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Train a model** (uses pre-built `data/3-processed/X.csv`):
```bash
python -m src.train                              # XGBoost classifier (default)
python -m src.train --model logistic_regression
python -m src.train --model xgboost_regressor
python -m src.train --model xgboost_classifier --register  # register in MLflow
```

**Explore interactively:**
```bash
jupyter notebook notebooks/
```

> **Note on data:** `data/3-processed/X.csv` is included so the pipeline runs immediately. To rebuild it from scratch, download the latest OECD country risk PDF from [country-risk.oecd.org](https://country-risk.oecd.org) and place it in `data/1-raw/`, then run notebooks 01 → 03 in sequence.

---

## Repository Structure

```
├── data/
│   ├── 0-metadata/        # Country codes and static reference data
│   ├── 1-raw/             # Unmodified source files (OECD PDFs, WB extracts)
│   ├── 2-interim/         # Intermediate outputs (merged, feature-selected)
│   └── 3-processed/       # Final X.csv / y.csv + split configs
│
├── notebooks/
│   ├── 01-Data_Extraction.ipynb
│   ├── 02-Data_Integration.ipynb
│   ├── 03-Feature_Selection.ipynb   # feature engineering + selection
│   ├── 04-Modeling_and_Evaluation.ipynb
│   ├── 05-Model_Comparing.ipynb
│   └── 06-Results_and_Interpretation.ipynb
│
├── src/
│   ├── extraction/        # World Bank API + OECD PDF parsing
│   ├── preprocessing/     # sklearn preprocessing pipeline
│   ├── features/          # Feature selection, pruning, engineering
│   ├── models/            # Model factory, evaluation, registry
│   ├── utils/             # Config, I/O, country utilities
│   └── train.py           # CLI training entrypoint
│
├── models/
│   ├── mlruns/            # MLflow experiment tracking
│   └── registry/          # Versioned model artifacts
│
└── reports/               # Analysis and model reports
```

`notebooks/` explain the reasoning behind decisions.  
`src/` contains the reproducible pipeline code used for training.

---

## Data Sources

| Source | Content | Access |
|---|---|---|
| [World Bank WDI](https://datatopics.worldbank.org/world-development-indicators/) | 94 macroeconomic indicators (1999–2024) | Free API (`wbgapi`) |
| [OECD Country Risk](https://country-risk.oecd.org) | Risk ratings 1–7 (target variable) | Free PDF download |

OECD ratings reflect expert committee judgment on sovereign default risk. Rating 1 = highest risk, 7 = lowest risk. Not all countries are rated (OECD members and select others only).

---

## Notes

- Models capture correlation, not causation — this is a forecasting exercise, not a causal model
- OECD ratings include subjective expert judgment that macroeconomic data alone cannot fully replicate
- Data availability varies across countries and years; missingness filtering is applied throughout
- Results are exploratory and not intended as financial advice
