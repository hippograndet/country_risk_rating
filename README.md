# OECD Country Risk Rating Prediction

End-to-end machine learning pipeline predicting OECD country risk ratings (1–7) from World Bank macroeconomic indicators.

**Hippolyte Grandet** · MSc Artificial Intelligence, University of Edinburgh

---

## Results

Test period: **2021–2024** (held-out temporal split, no data leakage)

| Model | Macro F1 | Accuracy | Blurred Acc. (±1) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.604 | 0.711 | — |
| XGBoost Regressor | 0.487 | 0.560 | 0.896 |
| **XGBoost Classifier** | **0.747** | **0.811** | **0.955** |

**Blurred accuracy** measures the fraction of predictions within one rating step of the true value — adjacent OECD ratings often reflect similar risk levels.

![Model Comparison](reports/plots/model_comparison.png)

The confusion matrix shows most misclassifications occur between adjacent rating categories:

![Confusion Matrix](reports/plots/confusion_matrix_xgb.png)

SHAP analysis reveals which economic dimensions drive predictions across the rating spectrum:

![SHAP Summary](reports/plots/shap_summary_rating_by_features.png)

### Key findings

- Tree-based models capture non-linear interactions that linear models miss
- Temporal splits are essential — random splits artificially inflate performance
- Macroeconomic indicators explain ~75% of rating variation; residual error reflects expert judgment not captured by public data

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
│  FEATURE SELECTION  (notebooks / src)   │
│  Missingness · Variance · Correlation   │
│  + Feature engineering (ENG_* columns)  │
│  → 63 features, 4 081 observations      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  TRAINING  (src/models/train.py)        │
│  Temporal split: train 1999–2020        │
│                  test  2021–2024        │
│  sklearn preprocessing pipeline         │
│  MLflow: params · metrics · artifacts   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  EVALUATION  (notebooks / MLflow UI)    │
│  Macro F1 · Accuracy · Blurred Acc.     │
└─────────────────────────────────────────┘
```

---

## Documentation

| Resource | Description |
|---|---|
| [Executive Summary](reports/Executive_Summary.md) | High-level overview of objectives, results, and implications |
| [Model Report](reports/Model_Report.md) | Full technical report: baseline → XGBoost, error analysis, feature importance, future directions |
| [Data Diagnostics](reports/Data_Diagnostics.md) | Data quality, missingness patterns, and feature selection methodology |
| [Model Validation](reports/Model_Validation.md) | Temporal evaluation, per-class breakdown, and SHAP interpretation |

### Notebooks

The notebooks walk through the full research process step-by-step:

| Notebook | Topic |
|---|---|
| `01-Data_Extraction` | Retrieving data from World Bank API and OECD PDFs |
| `02-Data_Integration` | Merging and aligning multi-source datasets |
| `03-Feature_Selection` | Feature engineering, filtering, and selection |
| `04-Modeling_and_Evaluation` | Preprocessing pipeline, training, and evaluation |
| `05-Model_Analysis` | Per-class performance, subgroup analysis, and SHAP interpretability |

---

## Data Sources

| Source | Content | Access |
|---|---|---|
| [World Bank WDI](https://datatopics.worldbank.org/world-development-indicators/) | 94 macroeconomic indicators (1999–2024) | Free API (`wbgapi`) |
| [OECD Country Risk](https://country-risk.oecd.org) | Risk ratings 1–7 (target variable) | Free PDF download |

---

## Repository Structure

```
├── data/
│   ├── 0-metadata/        # Country codes and static reference data
│   ├── 1-raw/             # Unmodified source files (OECD PDFs, WB extracts)
│   ├── 2-interim/         # Intermediate outputs (merged, feature-selected)
│   └── 3-processed/       # Final X.csv / y.csv + split configs
│
├── notebooks/             # Step-by-step analysis (01 → 05)
│
├── src/
│   ├── extraction/        # World Bank API + OECD PDF parsing
│   ├── preprocessing/     # sklearn preprocessing pipeline
│   ├── features/          # Feature selection, pruning, engineering
│   ├── models/            # Model factory, evaluation, registry
│   └── utils/             # Config, I/O, country utilities
│
├── models/
│   ├── mlruns/            # MLflow experiment tracking
│   └── registry/          # Versioned model artifacts
│
└── reports/               # Analysis reports and plots
```

---

## Local Setup

```bash
git clone https://github.com/hippograndet/country_risk_rating.git
cd country_risk_rating
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### macOS SSL fix

If you see `NotOpenSSLWarning` warnings:

```bash
make setup-openssl-macos
make verify-ssl
```

### Commands

```bash
make train              # Train XGBoost Classifier (best performing)
make train-regressor    # Train XGBoost Regressor
make test               # Run test suite
make lint               # Run code quality checks
make mlflow-ui          # Launch MLflow experiment tracking UI
make explore            # Open Jupyter notebooks
```

> **Note:** `data/3-processed/X.csv` is included so the pipeline runs immediately. To rebuild from scratch, download the latest OECD country risk PDF from [country-risk.oecd.org](https://country-risk.oecd.org), place it in `data/1-raw/`, and run notebooks 01 → 03 in sequence.

---

## Notes

- Models capture correlation, not causation — this is a forecasting exercise, not a causal model
- OECD ratings include subjective expert judgment that macroeconomic data alone cannot fully replicate
- Data availability varies across countries and years; missingness filtering is applied throughout
- Results are exploratory and not intended as financial advice
