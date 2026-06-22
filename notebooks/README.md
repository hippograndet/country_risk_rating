# Notebooks

Sequential notebooks covering each stage of the project pipeline, with visualizations.

## Overview

These notebooks explore the steps of this project, from data extraction and integration, to feature selection, model training, evaluation, and analysis.

Experiment notebooks in the `experiments/` folder are for parameter exploration and are not part of the main pipeline.

---

### 01-Data_Extraction.ipynb

Retrieving data from public datasets (OECD, World Bank) and formatting for later use. Includes initial missing data observations.

### 02-Data_Integration.ipynb

Merging and formatting datasets from all sources into a single country-year indexed dataset.

### 03-Feature_Selection.ipynb

Feature selection pipeline (missingness, variance, correlation filters) and domain feature engineering.

### 04-Modeling_and_Evaluation.ipynb

Applying the preprocessing pipeline, creating the temporal train/test split, then training and evaluating Logistic Regression, XGBoost Classifier, XGBoost Regressor, and Ensemble models.

### 05-Model_Analysis.ipynb

In-depth model comparison and interpretation. Per-class F1 breakdown, error distribution analysis, subgroup performance (by income group and region), and SHAP-based interpretability of the best model.
