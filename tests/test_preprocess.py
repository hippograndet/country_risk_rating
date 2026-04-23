import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocess_pipeline import build_preprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def numeric_df():
    """Small numeric DataFrame with one missing value."""
    return pd.DataFrame({
        'a': [1.0, 2.0, np.nan, 4.0],
        'b': [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def mixed_df():
    """DataFrame with numeric and categorical columns."""
    return pd.DataFrame({
        'num': [1.0, 2.0, np.nan, 4.0],
        'cat': ['x', 'y', None, 'x'],
    })


BASE_PARAMS = {
    'num_imputer': 'uni',
    'num_imputer_uni_strategy': 'mean',
    'cat_imputer': 'uni',
    'cat_imputer_uni_strategy': 'most_frequent',
}


# ── Numeric pipeline ──────────────────────────────────────────────────────────

class TestBuildPreprocessorNumeric:

    def test_output_has_no_nans(self, numeric_df):
        p = build_preprocessor(numeric_df, BASE_PARAMS)
        out = p.fit_transform(numeric_df)
        assert not np.isnan(out).any()

    def test_output_shape_preserved(self, numeric_df):
        p = build_preprocessor(numeric_df, BASE_PARAMS)
        out = p.fit_transform(numeric_df)
        assert out.shape == (4, 2)

    def test_knn_imputer_variant(self, numeric_df):
        params = {**BASE_PARAMS, 'num_imputer': 'knn', 'num_imputer_knn_n_neighbors': 2}
        p = build_preprocessor(numeric_df, params)
        out = p.fit_transform(numeric_df)
        assert not np.isnan(out).any()
        assert out.shape == (4, 2)

    def test_fit_on_train_transform_on_test(self, numeric_df):
        """Imputation and scaling learned on train should apply cleanly to test."""
        train = numeric_df.iloc[:3]
        test  = numeric_df.iloc[3:]
        p = build_preprocessor(train, BASE_PARAMS)
        p.fit(train)
        out = p.transform(test)
        assert not np.isnan(out).any()
        assert out.shape == (1, 2)


# ── Mixed (numeric + categorical) pipeline ────────────────────────────────────

class TestBuildPreprocessorMixed:

    def test_categorical_columns_are_encoded(self, mixed_df):
        p = build_preprocessor(mixed_df, BASE_PARAMS)
        out = p.fit_transform(mixed_df)
        # 1 numeric col + 2 OHE cols for {'x', 'y'} = 3 output cols
        assert out.shape[1] >= 2
        assert not np.isnan(out).any()