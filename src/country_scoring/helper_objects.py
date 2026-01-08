import pandas as pd
from datetime import date

import sys
sys.path.append('~/Desktop/TINUBU/country')

from src import countries, addresses
from country_scoring.src import helper_functions

today = date.today()
current_year = today.year

info_columns = ['YEAR', 'ISO3_COUNTRY_CODE']

country_info_columns = [
    'Info-Country_Name', 'Legal_Systems-Civil_Law', 'Legal_Systems-Common_Law', 'Legal_Systems-Customary', 'Legal_Systems-Muslim', 'Legal_Systems-Mixed', 
    'Languages-Official_language', 'Languages-Regional_language', 'Languages-Minority_language', 'Languages-National_language', 'Languages-Widely_spoken', 'Geography-x_coord',
    'Geography-y_coord', 'Geography-Region', 'Geography-Sub_Region', 'Geography-Intermediate_Region', 'Geography-Region_Code', 'Geography-Sub_Region_Code', 
    'Geography-Intermediate_Region_Code', 'Economy-Income_Group'
]

try:
    oecd_countries = list(pd.read_csv(addresses.country_BE_address + 'raw_data/oecd_countries.csv', index_col=0)['ISO3_COUNTRY_CODE'])
except FileNotFoundError:
    oecd_countries = helper_functions.get_oecd_countries(intermed_data_address)
    oecd_countries.to_csv(addresses.country_BE_address + 'raw_data/oecd_countries.csv')

try:
    template_quarterly_df = pd.read_csv(addresses.country_BE_address + 'raw_data/template_quarterly_df.csv', index_col=0)
except FileNotFoundError:
    template_quarterly_df = helper_functions.get_template_quarterly_df(oecd_countries, last_year=current_year)
    template_quarterly_df.to_csv(addresses.country_BE_address + 'raw_data/template_quarterly_df.csv')
    
try:
    template_yearly_df = pd.read_csv(addresses.country_BE_address + 'raw_data/template_yearly_df.csv', index_col=0)
except FileNotFoundError:
    template_yearly_df = helper_functions.get_template_yearly_df(oecd_countries, last_year=current_year)
    template_yearly_df.to_csv(addresses.country_BE_address + 'raw_data/template_yearly_df.csv')
    
Countries = countries.CountriesInfo(addresses.country_BE_address + 'raw_data/country_mapping.csv')