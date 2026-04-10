import numpy as np
import pandas as pd
import pytest

from src.features import engineering, selection


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small synthetic dataset mirroring the real feature matrix structure."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "YEAR": [2010] * 10 + [2015] * 10,
        "GEO_REGION": ["EAS"] * 20,
        "INCOME_GROUP": ["HIC"] * 20,
        # Features used by engineering
        "FR.INR.LEND": np.random.uniform(5, 15, n),
        "FR.INR.DPST": np.random.uniform(1, 5, n),
        "NY.GDP.MKTP.KD.ZG": np.random.uniform(-2, 8, n),
        "SP.POP.GROW": np.random.uniform(0, 3, n),
        "BX.KLT.DINV.WD.GD.ZS": np.random.uniform(0, 5, n),
        "BM.KLT.DINV.WD.GD.ZS": np.random.uniform(0, 3, n),
        "DT.DOD.DECT.GN.ZS": np.random.uniform(10, 80, n),
        "DT.TDS.DECT.GN.ZS": np.random.uniform(1, 10, n),
        "GE.EST": np.random.uniform(-2, 2, n),
        "RQ.EST": np.random.uniform(-2, 2, n),
        "NY.GNS.ICTR.ZS": np.random.uniform(10, 35, n),
        "NE.GDI.TOTL.ZS": np.random.uniform(15, 40, n),
    })


# ── engineering.create_features ───────────────────────────────────────────────

class TestCreateFeatures:

    def test_adds_expected_columns(self, sample_df):
        result = engineering.create_features(sample_df)
        expected = [
            "ENG_interest_rate_spread",
            "ENG_gdp_per_capita_growth",
            "ENG_net_fdi_pct_gdp",
            "ENG_external_debt_burden",
            "ENG_governance_composite",
            "ENG_savings_investment_gap",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self, sample_df):
        original_cols = list(sample_df.columns)
        engineering.create_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_interest_rate_spread_values(self, sample_df):
        result = engineering.create_features(sample_df)
        expected = sample_df["FR.INR.LEND"] - sample_df["FR.INR.DPST"]
        pd.testing.assert_series_equal(
            result["ENG_interest_rate_spread"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_governance_composite_is_mean(self, sample_df):
        result = engineering.create_features(sample_df)
        expected = (sample_df["GE.EST"] + sample_df["RQ.EST"]) / 2
        pd.testing.assert_series_equal(
            result["ENG_governance_composite"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_nan_safe(self):
        """NaN in an input column propagates to the engineered feature, not to others."""
        df = pd.DataFrame({
            "FR.INR.LEND": [10.0, np.nan],
            "FR.INR.DPST": [3.0, 2.0],
            "NY.GDP.MKTP.KD.ZG": [2.0, 3.0],
            "SP.POP.GROW": [1.0, 1.5],
        })
        result = engineering.create_features(df)
        assert np.isnan(result["ENG_interest_rate_spread"].iloc[1])
        assert not np.isnan(result["ENG_gdp_per_capita_growth"].iloc[1])

    def test_graceful_with_missing_source_columns(self):
        """If source columns are absent, the engineered feature is simply not added."""
        df = pd.DataFrame({"YEAR": [2010, 2011], "GE.EST": [0.5, 0.6]})
        result = engineering.create_features(df)
        assert "ENG_interest_rate_spread" not in result.columns
        assert "ENG_governance_composite" not in result.columns


# ── selection.filter_missingness ──────────────────────────────────────────────

class TestFilterMissingness:

    def test_drops_high_missing_columns(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [np.nan, np.nan, 1.0],   # 67% missing → drop at 0.5 threshold
            "c": [1.0, np.nan, 3.0],       # 33% missing → keep
        })
        result = selection.filter_missingness(df, max_missing_ratio=0.5)
        assert "b" not in result.columns
        assert "a" in result.columns
        assert "c" in result.columns

    def test_keeps_all_when_no_missing(self, sample_df):
        result = selection.filter_missingness(sample_df.fillna(0), max_missing_ratio=0.5)
        assert result.shape[1] == sample_df.shape[1]


# ── selection.filter_sparse_rows ──────────────────────────────────────────────

class TestFilterSparseRows:

    def test_drops_rows_with_too_many_nans(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [1.0, np.nan, 4.0],
            "c": [1.0, np.nan, 5.0],   # row 1: 100% NaN → drop
        })
        result = selection.filter_sparse_rows(df, max_missing_ratio=0.5)
        assert len(result) == 2

    def test_keeps_all_complete_rows(self, sample_df):
        result = selection.filter_sparse_rows(sample_df.fillna(0), max_missing_ratio=0.5)
        assert len(result) == len(sample_df)


# ── selection.filter_low_variance ─────────────────────────────────────────────

class TestFilterLowVariance:

    def test_drops_constant_column(self):
        df = pd.DataFrame({
            "a": [1.0, 1.0, 1.0, 1.0],  # zero variance → drop
            "b": [1.0, 2.0, 3.0, 4.0],
        })
        result = selection.filter_low_variance(df, threshold=1e-4)
        assert "a" not in result.columns
        assert "b" in result.columns


# ── selection.filter_correlated ───────────────────────────────────────────────

class TestFilterCorrelated:

    def test_drops_one_of_perfectly_correlated_pair(self):
        x = np.linspace(0, 1, 20)
        df = pd.DataFrame({"a": x, "b": x * 2 + 1})  # perfectly correlated
        result = selection.filter_correlated(df, max_corr=0.9)
        assert result.shape[1] == 1
