# Model Validation: Temporal Evaluation and Performance Analysis

## Introduction

This report presents the temporal validation framework and performance analysis for the OECD country risk rating prediction models. The evaluation strategy employs a temporal split (train: 1999–2020, test: 2021–2024) to prevent data leakage and assess real-world forecasting capability. Three model types were evaluated: Logistic Regression (baseline), XGBoost Regressor, and XGBoost Classifier.

## Temporal Evaluation

### Split Strategy

The temporal split ensures that models are trained on historical data and evaluated on future observations, simulating real-world deployment conditions:

- **Training period**: 1999–2020 (3,368 observations)
- **Test period**: 2021–2024 (713 observations)
- **No overlap**: Prevents temporal leakage that would artificially inflate performance

The OECD rating distribution remains consistent across splits, validating the temporal separation:

![OECD Rating Distribution](plots/oecd_rating_distribution.png)

### Performance Metrics

Models were evaluated using multiple metrics to capture different aspects of prediction quality:

| Model | Macro F1 | Accuracy | Blurred Accuracy (±1) |
|---|---|---|---|
| Logistic Regression | 0.604 | 0.711 | — |
| XGBoost Regressor | 0.487 | 0.560 | 0.896 |
| XGBoost Classifier | **0.747** | **0.811** | **0.955** |

The XGBoost Classifier achieves the highest performance across all metrics, demonstrating superior ability to capture non-linear relationships in the data.

## Model Comparison

### Overall Performance

The model comparison plot highlights the performance hierarchy across all metrics:

![Model Comparison](plots/model_comparison.png)

The XGBoost Classifier shows particularly strong performance in macro F1, indicating balanced performance across all rating categories.

### Prediction Behavior

#### Confusion Matrix Analysis

The normalized confusion matrix for the XGBoost Classifier reveals strong diagonal performance with most errors occurring in adjacent rating categories:

![Confusion Matrix](plots/confusion_matrix_xgb.png)

This pattern is expected for ordinal classification where adjacent ratings represent similar risk levels.

#### Error by Class

Prediction errors vary systematically across rating categories:

![Error by Class](plots/error_by_class.png)

Higher-rated countries (ratings 6–7) show lower error rates, while mid-range ratings (3–5) exhibit higher uncertainty. This reflects the greater economic stability of high-rated countries and the transitional nature of mid-range ratings.

#### Per-Class Performance

F1 scores by rating category demonstrate the model's reliability across the risk spectrum:

![F1 Score per Rating](plots/f1_score_per_rating.png)

The XGBoost Classifier maintains consistent performance across all ratings, with particularly strong results for the most common categories.

## Feature Importance and Interpretation

### Global Feature Importance

The XGBoost Classifier identifies the most influential features for prediction:

![Feature Importance](plots/feature_importance_xgb.png)

Economic growth, fiscal balance, and external debt indicators emerge as the strongest predictors, aligning with economic theory of sovereign risk.

### SHAP Analysis

SHAP values provide model-agnostic interpretation of feature effects across predictions:

![SHAP Summary](plots/shap_summary_rating_by_features.png)

The analysis reveals how features influence predictions differently across the rating spectrum, with some features having opposing effects on high versus low ratings.

### Accuracy by Feature Category

Performance varies by economic dimension, with some feature categories providing more predictive signal than others:

![XGB Classifier Accuracy by Category](plots/xgb_classifier_accuracy_by_cat.png)

This insight can guide future feature engineering efforts and data collection priorities.

## Conclusion

The XGBoost Classifier demonstrates robust temporal forecasting capability, achieving 81% exact accuracy and 95.5% near-miss accuracy on held-out future data. The temporal validation framework provides confidence in the model's real-world applicability, while the error analysis reveals systematic patterns that align with economic intuition.

## Further Directions

- **Implement probabilistic predictions** to quantify uncertainty and enable risk-adjusted decision making.
- **Explore ensemble methods** that combine multiple model types for improved robustness.
- **Conduct stress testing** with extreme economic scenarios to assess model behavior under crisis conditions.
- **Implement continuous learning** to adapt to evolving economic relationships over time.
- **Validate on alternative temporal splits** to assess model stability across different time periods.