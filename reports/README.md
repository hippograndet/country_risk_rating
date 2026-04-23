# Reports Folder

This folder contains all generated visualizations and analysis reports from the pipeline notebooks.

## Generated Plots

| Plot File | Source Notebook | Description |
|---|---|---|
| `oecd_rating_distribution.png` | 01-Data_Extraction | Distribution of OECD country risk ratings 1-7 in extracted raw data |
| `missing_values_histogram.png` | 02-Data_Integration | Histogram showing missing value ratios across all features |
| `missingness_matrix.png` | 02-Data_Integration | Matrix visualization of missing values across observations |
| `missingness_correlation_heatmap.png` | 02-Data_Integration | Correlation heatmap of missingness patterns between features |
| `country_feature_coverage_distribution.png` | 02-Data_Integration | Distribution of average feature coverage per country |
| `feature_selection_waterfall.png` | 03-Feature_Selection | Feature count reduction through each selection filter step |
| `dataset_overview.png` | 00-Summary | Final dataset overview after preprocessing |
| `engineered_feature_correlations.png` | 00-Summary | Correlation of engineered features with target variable |
| `model_comparison.png` | 00-Summary | Performance comparison across all trained models |
| `confusion_matrix_xgb.png` | 00-Summary | Normalized confusion matrix for XGBoost classifier |
| `feature_importance_xgb.png` | 00-Summary | Top feature importance values from XGBoost model |
| `error_by_class.png` | 00-Summary | Prediction error distribution per risk rating class |

## Reports

- `Data_Analysis.md` - Summary data quality and coverage analysis
- `Model_Report.md` - Complete model performance report
- `Model_Improvement.md` - Proposed improvements and future work

All plots are automatically generated when executing the notebooks in sequential order.