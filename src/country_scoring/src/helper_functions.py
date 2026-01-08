import pandas as pd
import os

import sys
sys.path.append('~/Desktop/TINUBU/country')

from src import addresses

def get_oecd_countries(data_address):
    try:
        oecd_df = pd.read_csv(addresses.country_BE_address + 'intermed_data/oecd/oecd_rating_matrix_raw.csv')
    except FileNotFoundError:
        raise FileNotFoundError('No OECD Matrix, so we can\'t get oecd_countries. Run set_oecd script') 
        
    oecd_countries = list(oecd_df['ISO3_COUNTRY_CODE'].unique())
    oecd_countries_df = pd.DataFrame(oecd_countries, columns=['ISO3_COUNTRY_CODE']).sort_values(by=['ISO3_COUNTRY_CODE'], ascending=True).reset_index(drop=True)
    
    return oecd_countries_df
    
def get_template_quarterly_df(isos, last_year=2023):
    rows = []
    for iso in isos:
        for year in range(1999, last_year+1):
            for q in range(0, 5):
                quarter = 'Q' + str(q)
                index = iso + '-' + str(year) + '-' + quarter

                rows.append(
                    {
                        'COUNTRY_PERIOD_INDEX': index,
                        'YEAR': year,
                        'QUARTER': quarter,
                        'ISO3_COUNTRY_CODE': iso
                    }
                )

    template_quarterly_df = pd.DataFrame(rows).set_index('COUNTRY_PERIOD_INDEX')

    return template_quarterly_df

def get_template_yearly_df(isos, last_year=2023):
    rows = []
    for iso in isos:
        for year in range(1999, last_year+1):

            index = iso + '-' + str(year)

            rows.append(
                {
                    'COUNTRY_PERIOD_INDEX': index,
                    'YEAR': year,
                    'ISO3_COUNTRY_CODE': iso
                }
            )

    template_yearly_df = pd.DataFrame(rows).set_index('COUNTRY_PERIOD_INDEX')

    return template_yearly_df

