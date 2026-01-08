print('-----Installing Package Requirements for Getting Data from DBnomics API-----')
import os
os.system('pip install dbnomics')

import pandas as pd
import dbnomics

def get_api_link(provider: str, dataset: str, indicator: str) -> str:
    """get api link for the indicator.

    Args:
        provider (string): name of the provider of the indicator.
        dataset (string): name of the dataset of the indicator.
        indicator (string): name of the indicator.

    Returns:
        string: formated api link, to get indicator df from dbnomics api.
    """

    link_p1 = 'https://api.db.nomics.world/v22/series/'
    link_p3 = '%22%5D%7D&observations=1'

    if provider == 'WB':
        link_p2 = '?dimensions=%7B%22indicator%22%3A%5B%22'
    elif provider == 'IMF':
        link_p2 = '?dimensions=%7B%22INDICATOR%22%3A%5B%22'
        # link_p3 = link_p3 + '&q=imf'
    else:
        return ''

    api_link = link_p1 + provider + '/' + dataset + link_p2 + indicator + link_p3
    return api_link

def get_indicator_df_from_api(provider: str, dataset: str, indicator: str) -> pd.DataFrame:
    """get unformated indicator df from dbnomics api.

    Args:
        provider (string): name of the provider of the indicator.
        dataset (string): name of the dataset of the indicator.
        indicator (string): name of the indicator.
    Returns:
        DataFrame: indicator df from dbnomics api.
    """

    api_link = get_api_link(provider, dataset, indicator)
    try:
        df_indicator = dbnomics.fetch_series_by_api_link(
            api_link,
            max_nb_series=600
        )
    except Exception:
        print('Error Fetching Series for indicator', indicator)
        df_indicator = pd.DataFrame()

    return df_indicator