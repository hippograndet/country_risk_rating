import wbgapi as wb
import pandas as pd
from typing import Dict, List

def format_wb_query(df_query: pd.DataFrame, indicators) -> pd.DataFrame:
    df_formatted = df_query.rename(columns={s: int(s[2:]) for s in df_query.columns})

    if len(indicators) == 1:
        df_formatted = pd.DataFrame(df_formatted.unstack())
        df_formatted = df_formatted.reset_index()
        df_formatted = df_formatted.rename(columns={'economy': 'ISO3_COUNTRY_CODE', 'level_0': 'YEAR', 0: indicators[0]})
    else:
        df_formatted = df_query.stack().unstack(level='series')
        df_formatted = df_formatted.reset_index().rename(columns={'economy': 'ISO3_COUNTRY_CODE', 'level_1': 'YEAR'})

    df_formatted.loc[:, 'COUNTRY_PERIOD_INDEX'] = df_formatted['ISO3_COUNTRY_CODE'] + '-' + df_formatted['YEAR'].astype(str)
    df_formatted = df_formatted.set_index('COUNTRY_PERIOD_INDEX')

    return df_formatted

def fetch_wb_indicator(indicator: str,
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    
    df = wb.data.DataFrame(
        indicator,
        economy=countries,
        time=range(start_year,end_year+1)
    )

    return format_wb_query(df, [indicator])

def fetch_wb_subset_indicators(
    indicators: List[str],
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
        
    df = wb.data.DataFrame(
        indicators,
        economy=countries,
        time=range(start_year,end_year+1)
    )

    return format_wb_query(df, indicators)

def get_world_bank_indicators(
    indicators: Dict[str, str],
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Fetches annual World Bank indicators and returns a tidy DataFrame
    indexed by country and year.

    Parameters
    ----------
    indicators : dict
        Mapping from World Bank indicator codes to readable names
    countries : list
        List of ISO country codes (e.g. ['USA', 'FRA', 'DEU'])
    start_year : int
    end_year : int

    Returns
    -------
    pd.DataFrame
        Columns: country, year, <indicator_1>, <indicator_2>, ...
    """

    ind_series = []
    for ind_code, ind_name in indicators.items():
        print(ind_name, ind_code)

        df_ind = fetch_wb_indicator(ind_code, countries, start_year, end_year)
        ind_series.append(df_ind[ind_code])
    
    final_df = pd.concat(ind_series, axis=1)

    return final_df