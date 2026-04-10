"""
Country registry and metadata for country risk rating prediction.

Provides :class:`CountriesRegistry`, a lookup object built from the static
``data/0-metadata/country_mapping.csv`` file, and a module-level singleton
``country_registry`` used throughout the pipeline.
"""

import pandas as pd

from src.utils import config, io


class CountriesRegistry:
    """
    Lookup registry for country ISO codes and OECD membership.

    Built from ``data/0-metadata/country_mapping.csv`` at instantiation time.
    Exposes bidirectional mappings between ISO Alpha-2 codes, ISO Alpha-3
    codes, and country names, as well as OECD-subset lists.

    Parameters
    ----------
    address : str or Path
        Path to the country mapping CSV file.
    """

    def __init__(self, address: str) -> None:
        self.address = address
        self.country_mapping = pd.DataFrame()

        self.__set_country_mapping()
        self.__set_ISO2_list()
        self.__set_ISO3_list()
        self.__set_OECD_ISO2_list()
        self.__set_OECD_ISO3_list()
        self.__set_ISO2_to_ISO3()
        self.__set_ISO3_to_ISO2()
        self.__set_name_to_ISO3()
        self.__set_ISO3_to_name()
        self.__set_ISO3_df()

    # ── Private setters ───────────────────────────────────────────────────────

    def __set_country_mapping(self) -> None:
        try:
            self.country_mapping = io.load_csv(self.address, keep_default_na=False, na_values=[''])
        except FileNotFoundError:
            raise Exception('Country Mapping file not found at:\n\t' + self.address)

    def __set_ISO2_list(self) -> None:
        self.iso2_list = list(self.country_mapping['ISO 2-Alpha'])

    def __set_ISO3_list(self) -> None:
        self.iso3_list = list(self.country_mapping['ISO'])

    def __set_OECD_ISO2_list(self) -> None:
        self.oecd_iso2_list = list(self.country_mapping[self.country_mapping['OECD Country'] == True]['ISO 2-Alpha'])

    def __set_OECD_ISO3_list(self) -> None:
        self.oecd_iso3_list = list(self.country_mapping[self.country_mapping['OECD Country'] == True]['ISO'])

    def __set_ISO2_to_ISO3(self) -> None:
        self.iso2_to_iso3 = dict(self.country_mapping.set_index('ISO 2-Alpha')['ISO'])

    def __set_ISO3_to_ISO2(self) -> None:
        self.iso3_to_iso2 = dict(self.country_mapping.set_index('ISO')['ISO 2-Alpha'])

    def __set_name_to_ISO3(self) -> None:
        name_to_iso3 = dict(self.country_mapping.set_index('Country Name 1')['ISO'])
        i = list(self.country_mapping.columns).index('Country Name 1') + 1

        for col in self.country_mapping.columns[i:]:
            name_to_iso3_temp = dict(self.country_mapping.dropna(subset=[col]).set_index(col)['ISO'])
            name_to_iso3.update(name_to_iso3_temp)

        self.name_to_iso3 = name_to_iso3

    def __set_ISO3_to_name(self) -> None:
        self.iso3_to_name = dict(self.country_mapping.set_index('ISO')['Country Name 1'])

    def __set_ISO3_df(self) -> None:
        self.iso3_df = pd.DataFrame(self.iso3_list).set_index(0)

    # ── Public getters ────────────────────────────────────────────────────────

    def get_ISO2_list(self) -> list:
        """
        Return all ISO Alpha-2 codes in the registry.

        Returns
        -------
        list of str
            ISO Alpha-2 codes for every country in the mapping file.
        """
        return self.iso2_list

    def get_ISO3_list(self) -> list:
        """
        Return all ISO Alpha-3 codes in the registry.

        Returns
        -------
        list of str
            ISO Alpha-3 codes for every country in the mapping file.
        """
        return self.iso3_list

    def get_OECD_ISO2_list(self) -> list:
        """
        Return ISO Alpha-2 codes for countries in the OECD dataset.

        Returns
        -------
        list of str
            ISO Alpha-2 codes for OECD-rated countries.
        """
        return self.oecd_iso2_list

    def get_OECD_ISO3_list(self) -> list:
        """
        Return ISO Alpha-3 codes for countries in the OECD dataset.

        Returns
        -------
        list of str
            ISO Alpha-3 codes for OECD-rated countries.
        """
        return self.oecd_iso3_list

    def get_ISO3_df(self) -> pd.DataFrame:
        """
        Return a DataFrame indexed by ISO Alpha-3 codes with no columns.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame whose index contains all ISO Alpha-3 codes.
        """
        return self.iso3_df

    def get_ISO3_from_ISO2(self, iso2: str) -> str:
        """
        Convert an ISO Alpha-2 code to the corresponding ISO Alpha-3 code.

        Parameters
        ----------
        iso2 : str
            Two-letter ISO Alpha-2 country code.

        Returns
        -------
        str
            Corresponding ISO Alpha-3 code, or ``''`` if not found.
        """
        try:
            return self.iso2_to_iso3[iso2]
        except KeyError:
            return ''

    def get_ISO2_from_ISO3(self, iso3: str) -> str:
        """
        Convert an ISO Alpha-3 code to the corresponding ISO Alpha-2 code.

        Parameters
        ----------
        iso3 : str
            Three-letter ISO Alpha-3 country code.

        Returns
        -------
        str
            Corresponding ISO Alpha-2 code, or ``''`` if not found.
        """
        try:
            return self.iso3_to_iso2[iso3]
        except KeyError:
            return ''

    def get_ISO3_from_name(self, name: str) -> str:
        """
        Convert a country name to the corresponding ISO Alpha-3 code.

        Searches across all name columns in the mapping file.

        Parameters
        ----------
        name : str
            Country name (any alias registered in the mapping file).

        Returns
        -------
        str
            Corresponding ISO Alpha-3 code, or ``''`` if not found.
        """
        try:
            return self.name_to_iso3[name]
        except KeyError:
            return ''

    def get_name_from_ISO3(self, iso3: str) -> str:
        """
        Return the primary name for an ISO Alpha-3 code.

        Parameters
        ----------
        iso3 : str
            Three-letter ISO Alpha-3 country code.

        Returns
        -------
        str
            Primary country name, or ``''`` if not found.
        """
        try:
            return self.iso3_to_name[iso3]
        except KeyError:
            return ''

    def check_ISO2_in_countries(self, iso2: str) -> bool:
        """
        Check whether an ISO Alpha-2 code is present in the registry.

        Parameters
        ----------
        iso2 : str
            Two-letter ISO Alpha-2 country code.

        Returns
        -------
        bool
        """
        return iso2 in self.iso2_list

    def check_ISO3_in_countries(self, iso3: str) -> bool:
        """
        Check whether an ISO Alpha-3 code is present in the registry.

        Parameters
        ----------
        iso3 : str
            Three-letter ISO Alpha-3 country code.

        Returns
        -------
        bool
        """
        return iso3 in self.iso3_list

    def check_ISO3_in_oecd(self, iso3: str) -> bool:
        """
        Check whether an ISO Alpha-3 code belongs to the OECD-rated subset.

        Parameters
        ----------
        iso3 : str
            Three-letter ISO Alpha-3 country code.

        Returns
        -------
        bool
        """
        return iso3 in self.oecd_iso3_list


country_info_columns: list = [
    'Info-Country_Name', 'Legal_Systems-Civil_Law', 'Legal_Systems-Common_Law', 'Legal_Systems-Customary', 'Legal_Systems-Muslim', 'Legal_Systems-Mixed',
    'Languages-Official_language', 'Languages-Regional_language', 'Languages-Minority_language', 'Languages-National_language', 'Languages-Widely_spoken', 'Geography-x_coord',
    'Geography-y_coord', 'Geography-Region', 'Geography-Sub_Region', 'Geography-Intermediate_Region', 'Geography-Region_Code', 'Geography-Sub_Region_Code',
    'Geography-Intermediate_Region_Code', 'Economy-Income_Group'
]

country_registry: CountriesRegistry = CountriesRegistry(config.METADATA_DIR / 'country_mapping.csv')
