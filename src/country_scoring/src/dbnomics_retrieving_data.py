import pandas as pd

import sys
sys.path.append('~/Desktop/TINUBU/country/country_scoring')
sys.path.append('~/Desktop/TINUBU/country')

from src import addresses

import src.dbnomics_scrapping_data as scrapper
import src.dbnomics_formatting_data as formatter
from src import helper_objects

def get_data_indicator(provider: str, dataset: str, indicator: str) -> pd.DataFrame:
    """get from dbnomics api specific indicator, then format it.

    Args:
        provider (string): name of the provider of the indicator.
        dataset (string): name of the dataset of the indicator.
        indicator (string): name of the indicator.

    Returns:
        DataFrame: df of indicator, formated.
    """
    df_indicator = scrapper.get_indicator_df_from_api(provider, dataset, indicator)
    if not df_indicator.empty:
        df_indicator_standard = formatter.format_indicator_standard(df_indicator, provider)

        df_indicator_clean = formatter.format_indicator_df(df_indicator_standard, provider, dataset, indicator)
    else:
        df_indicator_clean = df_indicator
        
    return df_indicator_clean

def get_data_dbnomics(indicators_file: str = 'dbnomics_indicators'):
    dbnomics_indicators = pd.read_csv(addresses.country_BE_address + 'raw_data/' + indicators_file + '.csv', on_bad_lines='skip', sep=';')
    template_df = helper_objects.template_yearly_df

    indicators_dfs = []
    for i, row in dbnomics_indicators.iterrows():
        print(round(i*100/len(dbnomics_indicators), 2), end='\r')
        provider = row['PROVIDER']
        dataset = row['DATASET']
        indicator_code = row['INDICATOR_CODE']
        indicator_name = row['INDICATOR_NAME']
        
        indicator_df = get_data_indicator(provider, dataset, indicator_code)
        if indicator_df.empty:
            print('No values for indicator', indicator)
            continue

        indicator_column = indicator_df[indicator_name]
        indicators_dfs.append(indicator_column)
    
    if len(indicators_dfs) > 0:
        print('Got a total of', len(indicators_dfs), 'indicators')
        dbnomics_df = pd.concat([template_df] + indicators_dfs, axis=1)
    else:
        dbnomics_df = template_df
        
    return dbnomics_df

def update_data_dbnomics(dbnomics_data_file: str = 'dbnomics_data'):
    
    dbnomics_df = pd.read_csv(helper_objects.intermed_data_address + dbnomics_data_file + '.csv', index_col=0)
    
    dbnomics_indicators = list(dbnomics_df.drop(columns=helper_objects.info_columns).columns)
    
    max_year = dbnomics_df.dropna(subset=dbnomics_indicators, how='all')['YEAR'].max()
    
    if max_year < helper_objects.current_year:
        dbnomics_df = get_data_dbnomics(indicators_file='dbnomics_indicators')
        return dbnomics_df
    else:
        print('No new values available')