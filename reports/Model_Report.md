# Country Risk Rating Prediction — Baseline to Model Improvement Report

## Executive Summary

This project addresses the problem of predicting **future country risk ratings** using historical economic, financial, and institutional indicators. Country risk ratings are widely used in investment decision-making, sovereign risk assessment, and policy analysis, making predictive accuracy, temporal robustness, and interpretability essential.

Using a temporally consistent dataset indexed by country and year, I developed and evaluated multiple machine learning models under strict anti-leakage constraints. A simple logistic regression baseline was established to provide an interpretable and transparent reference point. From this baseline, systematic improvements were explored through feature preprocessing, model family selection, and hyperparameter tuning.

Key findings include:

* Temporal train/validation/test splits are critical; random splits significantly overestimate performance.
* Feature preprocessing choices contribute meaningful gains but plateau quickly.
* Tree-based models (XGBoost) capture non-linear interactions missed by linear models and deliver the strongest performance gains.
* Neural network models offer marginal improvements at the cost of increased complexity and reduced interpretability.

The final recommended model balances predictive performance, stability over time, and operational simplicity. While further gains may be possible with additional data or sequence-based modeling, the current solution demonstrates a robust and business-aligned approach to country risk prediction.

---

## 1. Problem Overview

### Task definition

The task is a **supervised multi-class classification problem**: predicting a country’s future risk rating for a given year using historical economic and institutional indicators. Each observation corresponds to a specific *(country, year)* pair, and the target represents the officially assigned risk rating in a subsequent period.

### Context & motivation

Country risk ratings inform decisions in sovereign lending, foreign direct investment, export credit guarantees, and geopolitical risk assessment. Errors in prediction can have material financial consequences, particularly when risk is systematically underestimated.

Key constraints include:

* Strong temporal dependencies in both features and labels
* Limited sample size per country
* Class imbalance across rating categories
* The need for interpretability and stability, not just raw accuracy

### Evaluation goal

The primary evaluation metric is **macro-averaged F1 score**, reflecting balanced performance across all rating classes. Secondary metrics include accuracy and per-class recall to assess systematic bias.

---

## 2. Dataset & Experimental Setup

### 2.1 Dataset description

The dataset aggregates multiple public sources, including World Bank indicators and OECD country risk ratings. Features include macroeconomic indicators, trade measures, financial stability proxies, and institutional variables.

* Unit of observation: country-year
* Temporal coverage: multiple decades (varies by feature)
* Feature types: numerical and categorical
* Target variable: ordinal country risk rating

### 2.2 Temporal train / validation / test split

To reflect the real-world forecasting task, all experiments use **strict temporal splits**. Training data includes only years prior to validation and test periods. Validation data is used exclusively for model selection and hyperparameter tuning, while the test set is held out until final evaluation.

This approach prevents information leakage and ensures that reported performance reflects true forward-looking predictive ability.

### 2.3 Preprocessing pipeline

All models share a common preprocessing framework implemented via scikit-learn pipelines:

* Missing value imputation (mean, median, or KNN, depending on experiment)
* Feature scaling for numerical variables
* One-hot encoding for categorical variables

Preprocessing steps are fit **only on training data** and applied consistently to validation and test sets.

---

## 3. Baseline Model

### 3.1 Baseline choice

Logistic regression was selected as the baseline model due to its simplicity, interpretability, and widespread use in risk modeling contexts. It provides a transparent benchmark against which more complex models can be evaluated.

### 3.2 Baseline configuration

* Model: Multinomial logistic regression
* Regularization: L2
* Class weighting: balanced
* Features: full processed feature set

### 3.3 Baseline performance

The baseline achieves reasonable predictive performance but exhibits systematic underfitting, particularly in capturing non-linear relationships and feature interactions.

### 3.4 Baseline error analysis

Error analysis reveals:

* Confusion between adjacent risk categories
* Reduced performance during periods of economic instability
* Sensitivity to missing or lagging indicators

---

## 4. Improvement Strategy

### 4.1 Observed limitations

The baseline model struggles with:

* Non-linear feature interactions
* Temporal shifts in feature importance
* Minority class recall

### 4.2 Hypotheses for improvement

* Tree-based models can better capture non-linearities
* Alternative preprocessing strategies may reduce noise
* More expressive models may improve class separation

---

## 5. Model Experiments

### 5.1 Feature and preprocessing experiments

Multiple preprocessing variants were tested, including alternative imputation strategies and scaling configurations. Improvements were incremental but consistent.

### 5.2 Algorithmic experiments

The following model families were evaluated:

* Logistic regression (baseline and tuned variants)
* Gradient-boosted decision trees (XGBoost)
* Feedforward neural networks (PyTorch)

### 5.3 Training protocol

All models were trained using identical temporal splits and evaluated using the same metrics. Experiments and results were tracked using MLflow to ensure reproducibility and comparability.

---

## 6. Model Comparison

Quantitative results show that XGBoost provides the largest performance improvement over the baseline, particularly in macro-F1 score and minority class recall. Neural networks offer limited additional gains while increasing training complexity and reducing interpretability.

Qualitatively, tree-based models demonstrate greater robustness to missing data and temporal drift.

---

## 7. Error Analysis & Insights

Detailed error analysis highlights remaining challenges in predicting abrupt rating changes following exogenous shocks. Performance is strongest for stable countries with consistent economic trajectories.

Key insight: much of the residual error appears driven by factors not captured in the available indicators, suggesting diminishing returns from model complexity alone.

---

## 8. Trade-offs & Practical Considerations

Key trade-offs considered include:

* Predictive performance vs interpretability
* Model complexity vs retraining cost
* Stability over time vs sensitivity to new data

These considerations favor a moderately complex, well-regularized model.

---

## 9. Final Model Selection

### Chosen model

Gradient-boosted decision tree model (XGBoost)

### Justification

This model offers the best balance between performance, robustness, and operational feasibility, making it suitable for real-world risk assessment workflows.

---

## 10. Limitations & Future Work

* Incorporation of sequence-based models to better capture temporal dynamics
* Integration of exogenous shock indicators
* Extension to probabilistic forecasting of rating transitions

---

## 11. Reproducibility Notes

* All experiments tracked via MLflow
* Fixed temporal splits and random seeds
* Modular training and evaluation pipelines

---

## Appendix

Additional plots, hyperparameter grids, and ablation studies are available in the accompanying notebooks and experiment logs.