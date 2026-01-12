import pandas as pd

from src.utils import config, io, templates
from src.extraction import dbnomics_client, dbnomics_formatting_raw_data

def get_data_indicator(provider: str, dataset: str, indicator: str) -> pd.DataFrame:
    """get from dbnomics api specific indicator, then format it.

    Args:
        provider (string): name of the provider of the indicator.
        dataset (string): name of the dataset of the indicator.
        indicator (string): name of the indicator.

    Returns:
        DataFrame: df of indicator, formated.
    """
    df_indicator = dbnomics_client.fetch_indicator(provider, dataset, indicator)
    if not df_indicator.empty:
        df_indicator_standard = dbnomics_formatting_raw_data.format_indicator_standard(df_indicator, provider)
        df_indicator_clean = dbnomics_formatting_raw_data.format_indicator_df(df_indicator_standard, provider, dataset, indicator)
    else:
        df_indicator_clean = df_indicator
        
    return df_indicator_clean

def get_data_dbnomics(indicators_file: str = 'indicators_mapping', date_freq='Y'):
    dbnomics_indicators = io.load_csv(
        config.RAW_DATA_DIR / ('dbnomics_indicators/' + indicators_file + '.csv'), 
        on_bad_lines='skip', 
        sep=';'
    )
    # if date_freq == 'Q':
    #     template_df = templates.get_template_quarterly_df
    # else:
    #     template_df = templates.get_template_yearly_df

    indicators_dfs = []
    for i, row in dbnomics_indicators.iterrows():
        print(str(round(i*100/len(dbnomics_indicators), 2)) + '%', end='\r')
        provider = row['PROVIDER']
        dataset = row['DATASET']
        indicator_code = row['INDICATOR_CODE']
        indicator_name = row['INDICATOR_NAME']
        
        indicator_df = get_data_indicator(provider, dataset, indicator_code)
        if indicator_df.empty:
            print('No values for indicator', indicator_code)
            continue

        # indicator_column = indicator_df[indicator_name]
        indicators_dfs.append(indicator_df)
    
    # return pd.concat(indicators_dfs, axis=1)
    if len(indicators_dfs) > 0:
        dbnomics_df = pd.concat(indicators_dfs, axis=1)
        print('Got a total of', len(indicators_dfs), 'indicators')
        # dbnomics_df = pd.concat([template_df] + indicators_dfs, axis=1)
    else:
        # dbnomics_df = template_df
        dbnomics_df = pd.DataFrame()
        
    return dbnomics_df