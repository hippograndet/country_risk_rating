# Executive Summary: OECD Country Risk Rating Prediction

## Objective

This project predicts OECD country risk ratings (1–7) from publicly available macroeconomic indicators. The central question: *how well can macroeconomic data reproduce expert-assessed sovereign risk ratings?*

## Approach

### Data

The dataset combines two public sources:

| Source | Content |
|---|---|
| World Bank WDI | 74 macroeconomic indicators (1999–2026), accessed via API |
| OECD Country Risk | Sovereign risk ratings 1–7 (target variable), extracted from published PDFs |

After merging by country and year, applying feature selection filters (missingness, variance, correlation), and engineering domain features (growth rates, ratios, spreads), the final dataset contains **4,238 observations** across **64 features**. See [Data Diagnostics](Data_Diagnostics.md) for the full methodology.

### Evaluation Strategy

All models are evaluated on a strict **temporal split** — trained on 1999–2021 (3,755 observations) and tested on 2022–2026 (483 observations). This prevents data leakage and simulates real-world forecasting, where the model must predict future ratings from historical patterns.

### Models

Three models were compared:

- **Logistic Regression** — multinomial baseline with L2 regularization and balanced class weights. Simple, interpretable, but limited to linear feature relationships.
- **XGBoost Regressor** — gradient-boosted regression treating ratings as continuous, with predictions rounded to integers. Captures ordinal structure but suffers from median collapse (predictions cluster toward mid-range ratings).
- **XGBoost Classifier** — gradient-boosted classification with multi-class softmax. Directly optimises for class separation and produces calibrated class probabilities.

The primary evaluation metric is **macro-averaged F1 score**, ensuring balanced performance across all rating categories. Secondary metrics include accuracy and **blurred accuracy** (±1 step), which captures predictions within one rating of the true value.

## Key Results

| Model | Macro F1 | Accuracy | Blurred Accuracy (±1) |
|---|---|---|---|
| Logistic Regression | 0.592 | 0.683 | 0.876 |
| XGBoost Regressor | 0.487 | 0.560 | 0.896 |
| **XGBoost Classifier** | **0.761** | **0.812** | **0.917** |

![Model Comparison](plots/model_comparison.png)

The XGBoost Classifier achieves **81% exact accuracy** and **92% near-miss accuracy** on held-out future data, with no per-class F1 below 0.66. Detailed results, per-class breakdowns, and interpretation are in the [Model Results](Model_Results.md) report.

## Key Takeaways

1. **Macroeconomic indicators explain ~80% of rating variation.** The residual error reflects expert judgment and qualitative factors not captured by public data alone.

2. **Non-linear feature interactions matter.** The XGBoost Classifier outperforms logistic regression by +13 percentage points accuracy, with the largest gains on mid-range ratings (3–5) where economic signals are most ambiguous.

3. **Temporal validation is essential.** Random train/test splits would artificially inflate performance by allowing future data to leak into training. The temporal split provides a realistic estimate of forecasting capability.

## Further Reading

| Report | Focus |
|---|---|
| [Model Results](Model_Results.md) | Detailed performance, error analysis, subgroup breakdowns, SHAP interpretation |
| [Data Diagnostics](Data_Diagnostics.md) | Data quality, missingness patterns, feature selection pipeline |
