"""
Feature engineering for country risk rating prediction.

Derives interpretable composite indicators from the raw World Bank features.
All operations are pointwise and NaN-safe: if either input is missing the
result is NaN, which the downstream imputer handles.

Engineered columns are prefixed with "ENG_" for easy identification.
"""

import pandas as pd


def create_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (e.g. from data/3-processed/X.csv).

    Returns
    -------
    pd.DataFrame
        Original columns plus new ENG_* columns.
    """
    X = X.copy()

    # -- Interest rate spread -----------------------------------------------
    # Lending minus deposit rate. A wide spread signals financial stress or
    # limited banking competition; a classic early-warning indicator.
    if {"FR.INR.LEND", "FR.INR.DPST"}.issubset(X.columns):
        X["ENG_interest_rate_spread"] = X["FR.INR.LEND"] - X["FR.INR.DPST"]

    # -- GDP per capita growth proxy ----------------------------------------
    # GDP growth minus population growth ≈ change in output per person.
    # Captures whether growth is outpacing demographic pressure.
    if {"NY.GDP.MKTP.KD.ZG", "SP.POP.GROW"}.issubset(X.columns):
        X["ENG_gdp_per_capita_growth"] = (
            X["NY.GDP.MKTP.KD.ZG"] - X["SP.POP.GROW"]
        )

    # -- Net FDI position ---------------------------------------------------
    # FDI inflows minus outflows (% of GDP).
    # Positive = net receiver of foreign investment → confidence signal.
    if {"BX.KLT.DINV.WD.GD.ZS", "BM.KLT.DINV.WD.GD.ZS"}.issubset(X.columns):
        X["ENG_net_fdi_pct_gdp"] = (
            X["BX.KLT.DINV.WD.GD.ZS"] - X["BM.KLT.DINV.WD.GD.ZS"]
        )

    # -- External debt burden -----------------------------------------------
    # Product of external debt stock (% GNI) and debt service (% GNI).
    # Combines the size of the debt pile with the cost of servicing it.
    if {"DT.DOD.DECT.GN.ZS", "DT.TDS.DECT.GN.ZS"}.issubset(X.columns):
        X["ENG_external_debt_burden"] = (
            X["DT.DOD.DECT.GN.ZS"] * X["DT.TDS.DECT.GN.ZS"]
        )

    # -- Governance composite -----------------------------------------------
    # Average of Government Effectiveness and Regulatory Quality (World Bank
    # Governance Indicators). Single institutional quality signal.
    if {"GE.EST", "RQ.EST"}.issubset(X.columns):
        X["ENG_governance_composite"] = (X["GE.EST"] + X["RQ.EST"]) / 2

    # -- Savings-investment gap ---------------------------------------------
    # Gross savings minus gross capital formation (both % of GDP).
    # Negative = country invests more than it saves -> depends on external
    # financing, raising vulnerability to capital flow reversals.
    if {"NY.GNS.ICTR.ZS", "NE.GDI.TOTL.ZS"}.issubset(X.columns):
        X["ENG_savings_investment_gap"] = (
            X["NY.GNS.ICTR.ZS"] - X["NE.GDI.TOTL.ZS"]
        )

    return X
