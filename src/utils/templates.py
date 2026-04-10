"""
Template DataFrame factories for country risk rating prediction.

Generates skeleton DataFrames indexed by ``COUNTRY_PERIOD_INDEX`` that serve
as the alignment target when merging OECD ratings and World Bank features.
"""

import pandas as pd

from src.utils import config


def get_template_quarterly_df(
    isos: list,
    last_year: int = 2023
) -> pd.DataFrame:
    """
    Build an empty quarterly template DataFrame for a set of countries.

    Covers every (country, year, quarter) combination from ``config.start_year``
    through ``last_year`` inclusive.

    Parameters
    ----------
    isos : list of str
        ISO Alpha-3 country codes to include.
    last_year : int, optional
        Last year to include (default 2023).

    Returns
    -------
    pd.DataFrame
        Columns: ``YEAR``, ``QUARTER``, ``ISO3_COUNTRY_CODE``.
        Index: ``COUNTRY_PERIOD_INDEX`` (e.g. ``'FRA-2020-Q1'``).
    """
    rows = []
    for iso in isos:
        for year in range(config.start_year, last_year + 1):
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


def get_template_yearly_df(
    isos: list,
    last_year: int = 2023
) -> pd.DataFrame:
    """
    Build an empty yearly template DataFrame for a set of countries.

    Covers every (country, year) combination from ``config.start_year``
    through ``last_year`` inclusive.

    Parameters
    ----------
    isos : list of str
        ISO Alpha-3 country codes to include.
    last_year : int, optional
        Last year to include (default 2023).

    Returns
    -------
    pd.DataFrame
        Columns: ``YEAR``, ``ISO3_COUNTRY_CODE``.
        Index: ``COUNTRY_PERIOD_INDEX`` (e.g. ``'FRA-2020'``).
    """
    rows = []
    for iso in isos:
        for year in range(config.start_year, last_year + 1):

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
