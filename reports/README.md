# Reports

Structured analysis reports and visualizations from the OECD country risk rating prediction project.

## Reports

### [Executive_Summary.md](Executive_Summary.md)
Stakeholder-facing overview of objectives, key results, and practical implications. Start here for a high-level understanding of the project.

### [Model_Report.md](Model_Report.md)
Technical report covering the full modeling journey — from logistic regression baseline to XGBoost classifier — with performance metrics, error analysis, feature importance, and future directions.

### [Data_Diagnostics.md](Data_Diagnostics.md)
Data quality assessment, missingness analysis, and feature selection methodology. Covers the pipeline from 94 raw indicators to 63 engineered features.

### [Model_Validation.md](Model_Validation.md)
Temporal evaluation framework, per-class performance breakdown, and SHAP-based model interpretation.

## Plots

All plots in `plots/` are referenced by the reports above.

| Plot | Description |
|---|---|
| `model_comparison.png` | Performance comparison across models |
| `confusion_matrix_xgb.png` | Normalized confusion matrix (XGBoost Classifier) |
| `feature_importance_xgb.png` | Top feature importances |
| `shap_summary_rating_by_features.png` | SHAP feature impact analysis |
| `oecd_rating_distribution.png` | Rating distribution across temporal splits |
| `missing_values_histogram.png` | Missingness distribution across features |
| `missingness_matrix.png` | Missing data patterns |
| `missingness_correlation_heatmap.png` | Missingness clustering |
| `country_feature_coverage_distribution.png` | Feature coverage by country |
| `feature_selection_waterfall.png` | Feature reduction pipeline |
| `dataset_overview.png` | Final dataset characteristics |
| `engineered_feature_correlations.png` | Engineered feature-target correlations |
| `error_by_class.png` | Prediction errors by rating class |
| `f1_score_per_rating.png` | Per-class F1 scores |
| `xgb_classifier_accuracy_by_cat.png` | Accuracy by feature category |
| `top_features_mean_shap_importance.png` | Top features by mean SHAP importance |
| `aggregate_model_metrics_comparison.png` | Aggregate metrics comparison |
