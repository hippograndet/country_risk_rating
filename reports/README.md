# Reports Folder

This folder contains structured analysis reports and visualizations from the OECD country risk rating prediction project.

## Structured Reports

### Executive_Summary.md
A stakeholder-facing overview of the project's objectives, key results, and practical implications. Designed for data scientists and stakeholders who need high-level insights without deep technical detail.

**Key sections:**
- Introduction: Project purpose and methodology
- Key Results: Performance metrics and model comparison
- Practical Implications: Three main findings and their significance
- Conclusion: Summary of predictive capability
- Further Directions: Suggested improvements and extensions

**Key visualizations:**
- `model_comparison.png` - Performance comparison across models
- `confusion_matrix_xgb.png` - Prediction accuracy by rating
- `feature_importance_xgb.png` - Most influential predictors
- `shap_summary_rating_by_features.png` - Feature impact analysis

### Data_Diagnostics.md
A technical report on data quality, missingness patterns, and feature selection methodology. Written for data scientists who need to understand dataset reliability and preprocessing choices.

**Key sections:**
- Introduction: Data sources and preprocessing goals
- Data Quality: Missingness patterns and feature coverage
- Feature Selection: Pipeline methodology and engineered features
- Final Dataset: Characteristics and temporal distribution
- Conclusion: Quality assessment and limitations
- Further Directions: Data improvement opportunities

**Key visualizations:**
- `missing_values_histogram.png` - Missingness distribution
- `missingness_matrix.png` - Missing data patterns
- `missingness_correlation_heatmap.png` - Missingness clustering
- `country_feature_coverage_distribution.png` - Coverage by country
- `feature_selection_waterfall.png` - Feature reduction process
- `engineered_feature_correlations.png` - Feature-target relationships

### Model_Validation.md
A comprehensive technical report on temporal evaluation, model performance, and prediction behavior. Written for technical reviewers assessing model robustness and validation methodology.

**Key sections:**
- Introduction: Temporal validation framework
- Temporal Evaluation: Split strategy and performance metrics
- Model Comparison: Overall performance and prediction behavior
- Feature Importance: Global and local interpretation
- Conclusion: Model reliability and forecasting capability
- Further Directions: Validation improvements and extensions

**Key visualizations:**
- `oecd_rating_distribution.png` - Rating distribution across splits
- `model_comparison.png` - Comprehensive model comparison
- `confusion_matrix_xgb.png` - Classification accuracy
- `error_by_class.png` - Error patterns by rating
- `f1_score_per_rating.png` - Per-class performance
- `feature_importance_xgb.png` - Predictor importance
- `shap_summary_rating_by_features.png` - Feature interpretation
- `xgb_classifier_accuracy_by_cat.png` - Performance by feature category

## Generated Plots

All plots in the `plots/` subdirectory are referenced in the structured reports above.

| Plot File | Description | Report Usage |
|---|---|---|
| `oecd_rating_distribution.png` | OECD rating distribution across temporal splits | Model_Validation.md |
| `missing_values_histogram.png` | Missing value distribution across features | Data_Diagnostics.md |
| `missingness_matrix.png` | Missing data patterns visualization | Data_Diagnostics.md |
| `missingness_correlation_heatmap.png` | Missingness pattern correlations | Data_Diagnostics.md |
| `country_feature_coverage_distribution.png` | Feature coverage by country | Data_Diagnostics.md |
| `feature_selection_waterfall.png` | Feature reduction through selection pipeline | Data_Diagnostics.md |
| `dataset_overview.png` | Final dataset characteristics | Data_Diagnostics.md |
| `engineered_feature_correlations.png` | Feature-target correlations | Data_Diagnostics.md |
| `model_comparison.png` | Performance comparison across models | Executive_Summary.md, Model_Validation.md |
| `confusion_matrix_xgb.png` | Normalized confusion matrix | Executive_Summary.md, Model_Validation.md |
| `feature_importance_xgb.png` | Feature importance from XGBoost | Executive_Summary.md, Model_Validation.md |
| `error_by_class.png` | Prediction errors by rating class | Model_Validation.md |
| `f1_score_per_rating.png` | Per-class F1 scores | Model_Validation.md |
| `shap_summary_rating_by_features.png` | SHAP feature impact analysis | Executive_Summary.md, Model_Validation.md |
| `xgb_classifier_accuracy_by_cat.png` | Accuracy by feature category | Model_Validation.md |

## Legacy Reports

These legacy reports are maintained for reference but have been superseded by the structured reports above:

- `Data_Analysis.md` - Original data quality analysis (superseded by Data_Diagnostics.md)
- `Model_Report.md` - Original model performance report (superseded by Model_Validation.md)
- `Model_Improvement.md` - Proposed improvements (incorporated into Further Directions sections)

## Report Generation

The structured reports are designed to be:
- **Self-contained**: Each report provides complete context for its focus area
- **Audience-specific**: Tailored to different stakeholder needs
- **Professional**: Structured format with real metrics and visualizations
- **Actionable**: Clear recommendations for further work

All plots are automatically generated when executing the notebooks in sequential order, and the reports reference these visualizations directly.
