import pandas as pd
import numpy as np

from src.utils import config, io

def format_oecd_df(oecd_df, template_df):

    dft = oecd_df.T
    oecd_yearly = dft.groupby(pd.to_datetime(dft.index).year).max().T

    formated_oecd_df = template_df.copy()
    formated_oecd_df.loc[:, 'OECD_RATING'] = formated_oecd_df.apply(
        lambda r: oecd_yearly[r['YEAR']][r['ISO3_COUNTRY_CODE']] if r['YEAR'] in oecd_yearly.columns else '-',
        axis=1
    )
    formated_oecd_df = formated_oecd_df.fillna('-')

    return formated_oecd_df        

def add_info_datasets(df, countries_info_df):

    for info_column in countries_info_df.columns:
        formated_column = info_column[0].replace(' ', '_') + '-' + info_column[1].replace(' ', '_')
        df.loc[:, formated_column] = df['ISO3_COUNTRY_CODE'].map(countries_info_df[info_column])

    return df


def merge_info_features_ratings_datasets(formated_oecd_df, formated_features_df):

    df_final = formated_oecd_df.copy()
    df_final = df_final.join(formated_features_df.drop(columns=config.info_columns))
    
    return df_final