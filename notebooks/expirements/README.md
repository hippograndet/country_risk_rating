
# Jupyter Notebooks for Expirementing with Data and Parameters

## 🔍 Overview

    The notebooks in this folder are not for external use. They are expiremental, uncleaned, used to run cross validations for models with varying parameters, to find the most optimal configurations for, feature selection, data preprocessing pipeline (imputation, sclarization), and individual model hyperparameters.

    - Feature_Engineering.ipynb
    Explores Features to engineer from original data, and feature selection thresholds for different criterias (correlation, missingness, low variance). Also explores different imputation methods for missing values.

    - LR_Hyper_Parameters.ipynb
    - XGB_Classifier_Hyper_Parameters.ipynb
    - XGB_Regressor_Hyper_Parameters.ipynb

    Runs cross validation over many hyperparameters configuration for each specific ML model architectures, followed by an analysis of results to select optimal hyperparameter for each models.