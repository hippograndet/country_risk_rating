import pandas as pd

from src.utils import config, countries, templates

def format_indicator_df(df_indicator_standard: pd.DataFrame, provider: str, dataset: str, indicator: str) -> pd.DataFrame:
    """given standardized indicator df, format to df with YEAR, QUARTER, ISO3 and Indicator columns.

    Args:
        df_indicator_standard (DataFrame): standardized indicator df, with each row being a data instance with year, quarter, country and value.
        provider (string): name of the provider of the indicator.
        dataset (string): name of the dataset of the indicator.
        indicator (string): name of the indicator

    Returns:
        DataFrame: formatted df with YEAR, QUARTER, ISO3 and Indicator columns.
    """
    df_indicator = df_indicator_standard.reset_index()

    # Keep only non-na quarterly or annualy values from countries with OECD grade.
    df_indicator_clean = df_indicator[df_indicator['FREQ'] == 'A']
    df_indicator_clean = df_indicator_clean[~df_indicator_clean['value'].isna()]

    if not df_indicator_clean.empty:
        df_indicator_clean.loc[:, 'value'] = df_indicator_clean['value'].astype(float)

        # Setting YEAR and QUARTER columns.
        if df_indicator_clean['PERIOD'].str.split('-', expand=True).shape[1] == 1:
            df_indicator_clean['YEAR'] = df_indicator_clean['PERIOD'].str.split('-', expand=True)
        else:
            df_indicator_clean[['YEAR', 'QUARTER']] = df_indicator_clean['PERIOD'].str.split('-', expand=True)
            df_indicator_clean = df_indicator_clean.drop(columns=['QUARTER'])
        df_indicator_clean.loc[:, 'YEAR'] = df_indicator_clean['YEAR'].astype(int)

        # Create iso3-year-quarter index, to have formated index
        df_indicator_clean['COUNTRY_PERIOD_INDEX'] = df_indicator_clean['ISO3_COUNTRY_CODE'] + '-' + df_indicator_clean['YEAR'].astype(str)
        df_indicator_clean = df_indicator_clean.set_index('COUNTRY_PERIOD_INDEX')

        df_indicator_clean = df_indicator_clean.drop_duplicates(subset=config.info_columns)
        df_indicator_clean = df_indicator_clean[config.info_columns + ['value']]

        # Indicator value column name with have format: provider-dataset-indicator.
        indicator_new_name = provider + '-' + dataset + '-' + indicator
        df_indicator_final = df_indicator_clean.rename(columns={'value': indicator_new_name})

        return df_indicator_final[indicator_new_name]
    else:
        df_indicator_final = df_indicator_clean

    return df_indicator_final

def format_indicator_standard(df_indicator: pd.DataFrame, provider: str) -> pd.DataFrame:
    """given indicator df read from api, standardizes it, so there isn't a difference between the indicators from different providers.

    Args:
        df_indicator (DataFrame): indicator df read from api, with a different format depending provider.
        provider (string): name of the provider of the indicator.

    Returns:
        DataFrame: standardized indicator df, with standardized column names.
    """

    if provider == 'WB':
        df_indicator_clean = df_indicator.rename(columns={
            'country': 'ISO3_COUNTRY_CODE', 
            'frequency': 'FREQ', 
            'original_period': 'PERIOD',
            'indicator': 'INDICATOR',
            'indicator (label)': 'INDICATOR DESCRIPTION'
        })
    elif provider == 'IMF':
        df_indicator_clean = df_indicator.rename(columns={
            'REF_AREA': 'ISO2_COUNTRY_CODE', 
            'original_period': 'PERIOD',
            'Indicator': 'INDICATOR DESCRIPTION'
        })
        df_indicator_clean = df_indicator_clean[df_indicator_clean['ISO2_COUNTRY_CODE'].str.len() == 2]
        df_indicator_clean['ISO3_COUNTRY_CODE'] = df_indicator_clean['ISO2_COUNTRY_CODE'].map(countries.country_registry.get_ISO3_from_ISO2)

    return df_indicator_clean
