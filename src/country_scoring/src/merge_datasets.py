import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append('~/Desktop/TINUBU/country/country_scoring')
sys.path.append('~/Desktop/TINUBU/country')

from src import addresses
from src import helper_objects

def get_format_oecd_df(oecd_matrix_file_name: str = 'oecd_rating_matrix') -> pd.DataFrame:
    """given OECD pdf file name, return df with each rows corresponds to a rating for a country at a specific quarter.

    Args:
        oecd_matrix_file_name (str, optional): name of the matrix file name, if empty, create it. Defaults to ''.

    Returns:
        DataFrame: formated df where each rows corresponds to a rating for a country at a specific quarter.
    """
    
    try:
        yearly_oecd_df = pd.read_csv(addresses.country_BE_address + 'intermed_data/oecd/' + oecd_matrix_file_name + '_yearly.csv', index_col=0)
    except FileNotFoundError:
        os.system("scripts/set_oecd_matrix.py")
        yearly_oecd_df = pd.read_csv(addresses.country_BE_address + 'intermed_data/oecd/' + oecd_matrix_file_name + '_yearly.csv', index_col=0)

    formated_dates = []
    for y in range(1999, helper_objects.current_year+1):
        s = str(y)
        formated_dates.append(s)
        
    formated_oecd_df = helper_objects.template_yearly_df
    formated_oecd_df.loc[:, 'OECD_RATING'] = '-'
        
    for iso in yearly_oecd_df.index:
        for col, format_date in zip(yearly_oecd_df.columns, formated_dates):
            index = iso + '-' + format_date
        
            rating = yearly_oecd_df.at[iso, col]
            formated_oecd_df.at[index, 'OECD_RATING'] = rating
            
    formated_oecd_df.to_csv(addresses.country_BE_address + 'intermed_data/oecd/formated_oecd_ratings_yearly.csv')

    return formated_oecd_df

def get_info_df():
    country_info = pd.read_csv(addresses.country_BE_address + 'raw_data/countries_info.csv', index_col=0, header=[0, 1])
    country_info = country_info.set_index(('Info', 'ISO code'))

    return country_info

def get_final_df():
    df_oecd = get_format_oecd_df()
    country_info = get_info_df()

    df_final = df_oecd.copy()
    for info_column in country_info.columns:
        formated_column = info_column[0].replace(' ', '_') + '-' + info_column[1].replace(' ', '_')
        
        df_final.loc[:, formated_column] = df_final['ISO3_COUNTRY_CODE'].map(country_info[info_column])

    dbnomics_df = pd.read_csv(addresses.country_BE_address + 'intermed_data/dbnomics_yearly_data.csv', index_col=0)
    df_final = df_final.join(dbnomics_df.drop(columns=helper_objects.info_columns))
    
    return df_final
