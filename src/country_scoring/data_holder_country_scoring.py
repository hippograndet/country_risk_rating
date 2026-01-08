import pandas as pd
import numpy as np
import datetime

import sys
sys.path.append('~/Desktop/TINUBU/country/country_scoring')
sys.path.append('~/Desktop/TINUBU/country')

from src import addresses
from src import helper_objects, merge_datasets
from scripts import set_dbnomics_data, set_oecd_matrix

class DataHolderCountryScoring():
    
    def __init__(self):        
        self.data = pd.read_csv(addresses.country_BE_address + 'intermed_data/country_scoring_data.csv', index_col=0)
        
    def update(update_oecd: bool = True, update_dbnomics: bool = True):
        set_dbnomics_data.main()
        set_oecd_matrix.main()
        
        df_final = merge_datasets.get_final_df()
        
        if df_final.empty:
            print('Update got empty DF')
        else:
            self.data = df_final
            df_final.to_csv(addresses.country_BE_address + 'intermed_data/country_scoring_data.csv') 