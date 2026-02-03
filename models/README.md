# Models Folder Overview

This folder contains **all trained models and their associated artifacts**, organised as a lightweight **model registry**.

The purpose of this registry is to make every model:

* Easy to identify
* Easy to compare
* Easy to reproduce
* Easy to load, regardless of framework (XGBoost, PyTorch, etc.)

If you are looking for *experiments*, see the `notebooks/` folder.
If you are looking for *re-runnable training code*, see `src/training/`.

---

## 1. What lives in this folder

```
models/
├── registry/        # All trained models, versioned and structured
├── model_index.csv  # Flat index of all registered models
└── README.md        # This document
```

This folder does **not** contain:

* Training code
* Feature engineering logic
* Exploratory notebooks

It contains only **results of training**.

---

## 2. Registry structure

Models are organised by **model family**, then **version**:

```
models/registry/
├── baseline_logreg/
│   ├── v1/
│   └── v2/
├── xgboost/
│   ├── v1/
│   └── v2/
└── torch_mlp/
    ├── v1/
    └── v2/
```

Each version is a *fully self-contained model snapshot*.

---

## 3. Inside a model version

Each version folder follows the same structure, independent of framework:

```
v2/
├── model/           # Framework-specific model files
├── metrics.json     # Evaluation metrics
├── config.yaml      # Training configuration
├── metadata.json    # High-level context and decisions
└── training.log     # Optional training output
```

### 3.1 `model/`

Contains the serialized model artifact(s).

Examples:

* XGBoost: `model.bin`
* PyTorch: `model.pt`, `state_dict.pt`, optional preprocessors

Loading logic should rely on **metadata**, not file names.

---

### 3.2 `metrics.json`

Pure evaluation results.

Example:

```json
{
  "f1": 0.73,
  "precision": 0.70,
  "recall": 0.76,
  "auc": 0.81
}
```

No interpretation belongs here.

---

### 3.3 `config.yaml`

Describes *how the model was trained*:

* Hyperparameters
* Feature set
* Random seeds
* Training procedure

This file enables exact re-training.

---

### 3.4 `metadata.json`

Describes *why this model exists*.

Typical contents:

* Which baseline it improves on
* Performance deltas
* Qualitative observations
* Known trade-offs

This file is referenced directly in reports.

---

## 4. `model_index.csv`

This is a flat, human-readable index of all registered models.

It enables:

* Global comparison across frameworks
* Automatic selection of best models
* Report generation

Example columns:

* model_name
* framework
* version
* primary_metric
* score
* baseline
* path

All comparisons in reports should be derived from this file.

---

## 5. How models are used

Models are:

* Trained via scripts in `src/training/`
* Registered automatically into this folder
* Loaded via a unified model loader

Evaluation and inference code should **never** hard-code framework logic.

---

## 6. Relationship to reports

The `reports/` folder references this registry directly:

* Baseline performance
* Improvement deltas
* Hyperparameter justification
* Final model selection

This ensures reports are grounded in stored artifacts, not notebooks.

---

## 7. Design philosophy

* Notebooks explain *thinking*
* Scripts perform *work*
* This folder records *decisions*