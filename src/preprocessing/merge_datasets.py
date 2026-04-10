"""
Dataset merging utilities for country risk rating prediction.

Handles alignment of the OECD rating matrix, World Bank features, and country
metadata into a single modelling-ready DataFrame.
"""

import pandas as pd

from src.utils import config, io


def format_oecd_df(
    oecd_df: pd.DataFrame,
    template_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align OECD ratings onto the template (country × year) index.

    Aggregates any sub-annual OECD updates to an annual granularity by taking
    the maximum rating per year, then maps each row of the template to its
    corresponding annual rating.

    Parameters
    ----------
    oecd_df : pd.DataFrame
        Raw OECD rating matrix with countries as rows and date columns.
    template_df : pd.DataFrame
        Template DataFrame indexed by ``COUNTRY_PERIOD_INDEX`` with columns
        ``ISO3_COUNTRY_CODE`` and ``YEAR``.

    Returns
    -------
    pd.DataFrame
        A copy of ``template_df`` with an ``OECD_RATING`` column filled in;
        missing entries are replaced with ``'-'``.
    """
    dft = oecd_df.T
    oecd_yearly = dft.groupby(pd.to_datetime(dft.index).year).max().T

    formated_oecd_df = template_df.copy()
    formated_oecd_df.loc[:, 'OECD_RATING'] = formated_oecd_df.apply(
        lambda r: oecd_yearly[r['YEAR']][r['ISO3_COUNTRY_CODE']] if r['YEAR'] in oecd_yearly.columns else '-',
        axis=1
    )
    formated_oecd_df = formated_oecd_df.fillna('-')

    return formated_oecd_df


def add_info_datasets(
    df: pd.DataFrame,
    countries_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join country-level metadata columns onto a dataset by ISO3 code.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with an ``ISO3_COUNTRY_CODE`` column.
    countries_info_df : pd.DataFrame
        Metadata DataFrame indexed by ISO3 code, with any number of info
        columns (e.g. region, income group).

    Returns
    -------
    pd.DataFrame
        ``df`` with each metadata column from ``countries_info_df`` appended.
    """
    for info_column in countries_info_df.columns:
        df.loc[:, info_column] = df['ISO3_COUNTRY_CODE'].map(countries_info_df[info_column])

    return df


def merge_info_features_ratings_datasets(
    formated_oecd_df: pd.DataFrame,
    formated_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join World Bank features onto the OECD-rated template DataFrame.

    The ``config.info_columns`` columns (``YEAR``, ``ISO3_COUNTRY_CODE``) are
    dropped from ``formated_features_df`` before joining to avoid duplication.

    Parameters
    ----------
    formated_oecd_df : pd.DataFrame
        Template DataFrame with OECD ratings already filled in (output of
        :func:`format_oecd_df`).
    formated_features_df : pd.DataFrame
        World Bank feature DataFrame sharing the same
        ``COUNTRY_PERIOD_INDEX`` index.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ratings, metadata, and World Bank features.
    """
    df_final = formated_oecd_df.copy()
    df_final = df_final.join(formated_features_df.drop(columns=config.info_columns))

    return df_final
