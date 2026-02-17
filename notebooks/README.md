# Jupyter Notebooks, that go through the steps of the project pipeline, with visualizations


## 🔍 Overview

    The notebooks that explore the sequential steps of this project, from data extraction, integration, and analysis, to model training, evalution and final comparisons. There are also experiment notebooks (expirements folder) for personal use, to explore optimal parameters.

---

## Notebooks

### 01-Data_Extraction.ipynb

In this notebook, we go through the steps retrieving and data from public datasets (OECD, WB) for later formatting for model prediction. Also has an initial mssing data observation.

---

### 02-Data_Integration.ipynb

Merging and formatting dataset for preprocessing, from all sources

---

### 03-Feature_Selection.ipynb

Select from imported features, features that satisfy different selection criterias (missing values, correlation, ...).

---

### 04-Modeling_and_Evaluation.ipynb

Applying preprocessing pipeline to dataset and creating Train, test split. Then training and evaluating different ML models on data.

---

### 05-Model_Comparing.ipynb

Comparing the results of the different models with a more thorough evaluation, to understand predictions better.

---

### 06-Results_and_Interpretation.ipynb

Analyzing and interpretting results from trained models.