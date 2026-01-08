import pandas as pd

import sys
sys.path.append('/root/country')

from src import addresses

class CountriesInfo:
    def __init__(self, address):
        self.address = address

        self.__set_country_mapping()
        self.__set_ISO2_list()
        self.__set_ISO3_list()
        self.__set_ISO2_to_ISO3()
        self.__set_ISO3_to_ISO2()
        self.__set_name_to_ISO3()
        self.__set_ISO3_to_name()
        self.__set_ISO3_df()

    # Set Functions
    def __set_country_mapping(self):
        try:
            self.country_mapping = pd.read_csv(self.address, keep_default_na=False, na_values=[''])
        except FileNotFoundError:
            # try:
            #     self.country_mapping = pd.read_csv('./affinity_score/' + self.address, keep_default_na=False, na_values=[''])
            # except FileNotFoundError:
            raise Exception('Country Mapping file not found at:\n\t' + self.address)

    def __set_ISO2_list(self):
        self.iso2_list = list(self.country_mapping['ISO 2-Alpha'])

    def __set_ISO3_list(self):
        self.iso3_list = list(self.country_mapping['ISO'])

    def __set_ISO2_to_ISO3(self):
        self.iso2_to_iso3 = dict(self.country_mapping.set_index('ISO 2-Alpha')['ISO'])

    def __set_ISO3_to_ISO2(self):
        self.iso3_to_iso2 = dict(self.country_mapping.set_index('ISO')['ISO 2-Alpha'])
        
    def __set_name_to_ISO3(self):
        name_to_iso3 = dict(self.country_mapping.set_index('Country Name 1')['ISO'])
        i = list(self.country_mapping.columns).index('Country Name 1') + 1

        for col in self.country_mapping.columns[i:]:
            name_to_iso3_temp = dict(self.country_mapping.dropna(subset=[col]).set_index(col)['ISO'])
            name_to_iso3.update(name_to_iso3_temp)
            
        self.name_to_iso3 = name_to_iso3
        
    def __set_ISO3_to_name(self):
        self.iso3_to_name = dict(self.country_mapping.set_index('ISO')['Country Name 1'])
        
    def __set_ISO3_df(self):
        self.iso3_df = pd.DataFrame(self.iso3_list).set_index(0)

    # Get Functions
    def get_ISO2_list(self):
        """
        Returns:
            list of strings: list of all the ISO Alpha-2 codes of the countries we have data on
        """
        return self.iso2_list

    def get_ISO3_list(self):
        """
        Returns:
            list of strings: list of all the ISO Alpha-3 codes of the countries we have data on
        """
        return self.iso3_list
            
    def get_ISO3_df(self):
        """
        Returns:
            DataFrame: as index, all ISO Alpha-3 codes of the countries we have data on, no columns
        """
        return self.iso3_df

    def get_ISO3_from_ISO2(self, iso2: str):
        """
        Args:
            iso2 (string): 2 letter string, representing a ISO Alpha-2 code of a country

        Returns:
            string: returns the corresponding ISO Alpha-3 code if country covered, otherwise, returns empty string ''
        """
        try:
            return self.iso2_to_iso3[iso2]
        except KeyError:
            return ''

    def get_ISO2_from_ISO3(self, iso3: str):
        """
        Args:
            iso3 (string): 3 letter string, representing a ISO Alpha-3 code of a country

        Returns:
            string: returns the corresponding ISO Alpha-2 code if country covered, otherwise, returns empty string ''
        """
        try:
            return self.iso3_to_iso2[iso3]
        except KeyError:
            return ''
    
    def get_ISO3_from_name(self, name: str):
        """
        Args:
            name (string): representing a country name

        Returns:
            string: returns the corresponding ISO Alpha-3 code if country covered and name has been added,
            otherwise, returns empty string ''
        """
        try:
            return self.name_to_iso3[name]
        except KeyError:
            return ''

    def get_name_from_ISO3(self, iso3: str):
        """
        Args:
            iso3 (string): 3 letter string, representing a ISO Alpha-3 code of a country

        Returns:
            string: returns the first corresponding country name if covered, otherwise, returns empty string ''
        """
        try:
            return self.iso3_to_name[iso3]
        except KeyError:
            return ''

    def check_ISO2_in_countries(self, iso2: str):
        return iso2 in self.iso2_list

    def check_ISO3_in_countries(self, iso3: str):
        return iso3 in self.iso3_list    