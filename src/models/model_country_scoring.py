import pandas as pd
import numpy as np
import datetime
import xgboost as xgb

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append('~/Desktop/TINUBU/country/country_scoring')
sys.path.append('~/Desktop/TINUBU/country')

from src import addresses
from src import helper_objects

def preprocess_dataset(final_df):
    final_df = final_df.drop(
        columns=[
            'YEAR', 'ISO3_COUNTRY_CODE', 'Info-Country_Name', 'Languages-Official_language', 'Languages-Regional_language', 
            'Languages-Minority_language', 'Languages-National_language', 'Languages-Widely_spoken', 'Geography-x_coord', 
            'Geography-y_coord', 'Geography-Region', 'Geography-Sub_Region', 'Geography-Intermediate_Region', 'Geography-Region_Code', 
            'Geography-Sub_Region_Code', 'Geography-Intermediate_Region_Code', 'Economy-Income_Group'
        ]
    )

    row_na_count = final_df.isna().sum(axis=1)
    not_empty_rows = row_na_count[row_na_count < (len(final_df.columns) - 3)]
    final_df = final_df.loc[not_empty_rows.index, :]
    
    final_df.loc[:, 'OECD_RATING'] = final_df.loc[:, 'OECD_RATING'].astype(str)
    final_df.loc[:, 'Legal_Systems-Civil_Law'] = final_df.loc[:, 'Legal_Systems-Civil_Law'].fillna(-1).astype(int)
    final_df.loc[:, 'Legal_Systems-Common_Law'] = final_df.loc[:, 'Legal_Systems-Common_Law'].fillna(-1).astype(int)
    final_df.loc[:, 'Legal_Systems-Customary'] = final_df.loc[:, 'Legal_Systems-Customary'].fillna(-1).astype(int)
    final_df.loc[:, 'Legal_Systems-Muslim'] = final_df.loc[:, 'Legal_Systems-Muslim'].fillna(-1).astype(int)
    final_df.loc[:, 'Legal_Systems-Mixed'] = final_df.loc[:, 'Legal_Systems-Mixed'].fillna(-1).astype(int)
    
    final_df = final_df.dropna(how='all', axis=1)
    
    categorical = []
    numerical = list(final_df.columns[1:])
    
    numeric_processors = Pipeline(
        steps=[('KNNimputer', KNNImputer(n_neighbors=10, weights='distance', keep_empty_features=False))
               ]
    )
    categorical_processors = Pipeline(
        steps=[('onehotencoder', OneHotEncoder())]
    )

    X_transformer = ColumnTransformer(
        transformers=[
            ('numeric_processing', numeric_processors, numerical), 
            ('categorical_processing', categorical_processors, categorical)
        ]
    )
    X_transformer.set_output(transform='pandas')

    y_transformer = LabelEncoder()
    
    X = final_df.drop(columns=['OECD_RATING'])
    y = final_df[['OECD_RATING']]
    
    X_transformer.fit(X)
    X_transformed = X_transformer.transform(X)
    X_transformed.columns = [c.replace('numeric_processing__', '') for c in X.columns]
    
    #X = pd.DataFrame(X_transformed, index=X.index, columns=X.columns)

    y = y_transformer.fit_transform(y)
    
    return X_transformed, y, X_transformer, y_transformer

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    clf = xgb.XGBClassifier(
        reg_alpha= 0.01,
        colsample_bytree=0.60,
        eta=0.3,
        eval_metric=['mlogloss'],
        gamma=0.00001,
        reg_lambda=1.04,
        max_depth=6,
        min_child_weight=0.2,
        num_class=7,
        objective='multi:softprob',
        subsample=0.73
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    
    return clf
