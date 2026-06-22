# Executive Summary: OECD Country Risk Rating Prediction

## Introduction

This report summarizes the end-to-end machine learning project that predicts OECD country risk ratings (1–7) from publicly available macroeconomic indicators sourced from the World Bank. The central question is: *how well can macroeconomic data reproduce expert-assessed sovereign risk ratings?*

The pipeline covers data extraction, feature engineering, model training with temporal validation, and experiment tracking. The best-performing model is an XGBoost Classifier, evaluated on a held-out temporal test set (2022–2026) to prevent data leakage.

## Key Results

The XGBoost Classifier achieved the strongest performance across all metrics on the temporal test split:

| Model | Macro F1 | Accuracy | Blurred Accuracy (±1) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.592 | 0.683 | 0.876 |
| XGBoost Regressor | 0.487 | 0.560 | 0.896 |
| **XGBoost Classifier** | **0.761** | **0.812** | **0.917** |

Blurred accuracy measures the fraction of predictions within one rating step of the true value — a relevant metric because adjacent OECD ratings often reflect similar risk levels.

![Model Comparison](plots/model_comparison.png)

The confusion matrix for the XGBoost Classifier shows strong diagonal performance with most misclassifications occurring in adjacent classes:

![Confusion Matrix](plots/confusion_matrix_xgb.png)

## Practical Implications

Three key findings emerge from this analysis:

1. **Tree-based models capture non-linear interactions** that linear models miss, explaining the performance gap between XGBoost and logistic regression.

2. **Temporal splits are essential** — random train/test splits artificially inflate performance by allowing future data to leak into training.

3. **Macroeconomic indicators explain approximately 80% of rating variation**; the residual error reflects expert judgment and qualitative factors not captured by public data alone.

The feature importance analysis reveals which economic dimensions drive predictions:

![Feature Importance](plots/feature_importance_xgb.png)

SHAP analysis provides model-agnostic insight into how features influence predictions across the rating spectrum:

![SHAP Summary](plots/shap_summary_rating_by_features.png)

## Conclusion

The XGBoost Classifier reliably reproduces OECD country risk ratings from macroeconomic data, achieving 81% exact accuracy and 92% near-miss accuracy on held-out temporal data. This demonstrates that public economic indicators contain substantial signal about sovereign risk, though expert judgment adds nuance that models cannot fully capture.

## Further Directions

- **Incorporate alternative data sources** such as news sentiment, political stability indices, or market-based indicators (CDS spreads, bond yields) to capture qualitative dimensions of risk.
- **Explore ensemble methods** that combine classifier and regressor outputs for improved robustness.
- **Implement probabilistic forecasting** to quantify uncertainty around point predictions, enabling risk-aware decision making.
- **Extend temporal coverage** as new annual data becomes available to validate model stability over longer horizons.
