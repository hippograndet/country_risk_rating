import pandas as pd
import os

from src.utils import config

def get_template_quarterly_df(isos, last_year=2023):
    rows = []
    for iso in isos:
        for year in range(config.start_year, last_year+1):
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
        for year in range(config.start_year, last_year+1):

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

