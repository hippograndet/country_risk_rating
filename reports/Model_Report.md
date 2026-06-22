# Model Report: Baseline to Best Model

## 1. Problem Overview

The task is **supervised multi-class classification**: predicting a country's OECD risk rating (1–7) for a given year from macroeconomic indicators. Each observation is a *(country, year)* pair. Ratings reflect sovereign default risk as assessed by OECD expert committees — rating 1 indicates the lowest risk, rating 7 the highest.

The primary evaluation metric is **macro-averaged F1 score**, which ensures balanced performance across all rating categories regardless of class frequency. Secondary metrics include accuracy and **blurred accuracy** (±1 step), which captures predictions within one rating of the true value — relevant because adjacent OECD ratings often reflect similar underlying risk.

## 2. Dataset & Experimental Setup

### Data sources

The dataset combines two public sources:

| Source | Content | Access |
|---|---|---|
| World Bank WDI | 94 macroeconomic indicators (1999–2024) | API (`wbgapi`) |
| OECD Country Risk | Risk ratings 1–7 (target variable) | PDF extraction (`camelot-py`) |

After merging by country and year, applying feature selection (missingness, variance, correlation, mutual information filters), and engineering domain features (growth rates, ratios, moving averages, z-scores), the final dataset contains **4,081 observations** across **63 features**.

See [Data_Diagnostics.md](Data_Diagnostics.md) for the full data quality assessment and feature selection methodology.

### Temporal split

All experiments use a strict temporal split to prevent data leakage:

- **Training**: 1999–2020 (3,368 observations)
- **Test**: 2021–2024 (713 observations)

No overlap between splits. Preprocessing is fit on training data only.

### Preprocessing pipeline

All models share a common sklearn pipeline:
- Median imputation for missing values
- Standard scaling for numerical features
- One-hot encoding for categorical features

## 3. Baseline: Logistic Regression

Multinomial logistic regression with L2 regularization and balanced class weights was chosen as the baseline for its simplicity and widespread use in risk modeling.

| Metric | Value |
|---|---|
| Macro F1 | 0.604 |
| Accuracy | 0.711 |

The baseline achieves reasonable performance but underfits non-linear relationships and feature interactions, particularly for mid-range ratings (3–5) where economic signals are more ambiguous.

## 4. Improvement Strategy

Three limitations guided the search for better models:

1. **Non-linear feature interactions** — economic indicators interact in ways linear models cannot capture (e.g., debt burden matters differently depending on GDP growth)
2. **Temporal feature importance shifts** — the relative importance of indicators changes across economic cycles
3. **Minority class recall** — extreme ratings (1 and 7) are underrepresented and poorly predicted by the baseline

Hypothesis: tree-based models can address all three through their ability to learn feature interactions, handle heterogeneous feature types, and partition the feature space adaptively.

## 5. Model Experiments

### XGBoost Regressor

Gradient-boosted regression treating the rating as a continuous target, with predictions rounded to the nearest integer class. This approach captures ordinal structure but introduces rounding errors and struggles with class boundaries.

### XGBoost Classifier

Gradient-boosted classification with multi-class softmax objective. Directly optimizes for class separation and produces calibrated class probabilities.

All experiments were tracked using MLflow with identical temporal splits and evaluation metrics.

## 6. Results

| Model | Macro F1 | Accuracy | Blurred Acc. (±1) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.604 | 0.711 | — |
| XGBoost Regressor | 0.487 | 0.560 | 0.896 |
| **XGBoost Classifier** | **0.747** | **0.811** | **0.955** |

![Model Comparison](plots/model_comparison.png)

The XGBoost Classifier improves over the baseline by **+0.143 macro F1** and **+10 percentage points accuracy**. The regressor underperforms due to rounding artifacts and loss of class boundary information.

### Error analysis

The confusion matrix shows most misclassifications occur between adjacent rating categories:

![Confusion Matrix](plots/confusion_matrix_xgb.png)

Error rates vary systematically across ratings — high-rated countries (6–7) are predicted most reliably, while mid-range ratings (3–5) exhibit higher uncertainty, reflecting their transitional economic nature.

![Error by Class](plots/error_by_class.png)

### Feature importance

The most influential predictors align with economic theory of sovereign risk:

![Feature Importance](plots/feature_importance_xgb.png)

SHAP analysis reveals how features influence predictions differently across the rating spectrum:

![SHAP Summary](plots/shap_summary_rating_by_features.png)

## 7. Final Model

**Selected model**: XGBoost Classifier

**Justification**: Best performance across all metrics, robust to missing data, interpretable through feature importance and SHAP analysis, and operationally simple to retrain and deploy.

**Key takeaways**:
- Tree-based models capture non-linear interactions that linear models miss
- Temporal splits are essential — random splits artificially inflate performance by ~15 percentage points
- Macroeconomic indicators explain ~75% of rating variation; residual error reflects expert judgment not captured by public data

## 8. Future Directions

- **Alternative data sources**: news sentiment, political stability indices, or market-based indicators (CDS spreads, bond yields) to capture qualitative risk dimensions
- **Ensemble methods**: combining classifier and regressor outputs for improved robustness at class boundaries
- **Probabilistic forecasting**: quantifying prediction uncertainty to enable risk-adjusted decision making
- **Sequence modeling**: leveraging temporal trajectories of indicators rather than point-in-time snapshots
- **Extended validation**: rolling temporal splits to assess model stability across different economic regimes
