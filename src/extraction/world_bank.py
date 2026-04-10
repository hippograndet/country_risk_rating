"""
World Bank data extraction for country risk rating prediction.

Provides functions to fetch annual macroeconomic indicators from the World
Bank API (via ``wbgapi``) and reshape them into a tidy DataFrame indexed by
``COUNTRY_PERIOD_INDEX`` (e.g. ``'FRA-2020'``).
"""

import time
import wbgapi as wb
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def format_wb_query(
    df_query: pd.DataFrame,
    indicators: List[str]
) -> pd.DataFrame:
    """
    Reshape and reindex a raw World Bank API response.

    Renames year columns from ``'YR<year>'`` strings to integers, unstacks
    the multi-index into a tidy format, and creates the composite
    ``COUNTRY_PERIOD_INDEX``.

    Parameters
    ----------
    df_query : pd.DataFrame
        DataFrame as returned by ``wbgapi.data.DataFrame``.
    indicators : list of str
        Indicator codes that were queried; governs whether single- or
        multi-indicator reshaping logic is applied.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame indexed by ``COUNTRY_PERIOD_INDEX``
        (e.g. ``'FRA-2020'``) with columns ``ISO3_COUNTRY_CODE``, ``YEAR``,
        and one column per indicator.
    """
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


def fetch_wb_indicator(
    indicator: str,
    countries: List[str],
    start_year: int,
    end_year: int,
    retries: int = 5,
    backoff: float = 10.0,
) -> pd.DataFrame:
    """
    Fetch a single World Bank indicator for a set of countries and years.

    Retries automatically on transient API errors (503, connection errors)
    with exponential backoff.

    Parameters
    ----------
    indicator : str
        World Bank indicator code (e.g. ``'NY.GDP.MKTP.KD.ZG'``).
    countries : list of str
        ISO Alpha-3 country codes (e.g. ``['USA', 'FRA', 'DEU']``).
    start_year : int
        First year to retrieve data for (inclusive).
    end_year : int
        Last year to retrieve data for (inclusive).
    retries : int, optional
        Number of retry attempts on failure (default 5).
    backoff : float, optional
        Initial wait in seconds between retries; doubles each attempt (default 10).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame indexed by ``COUNTRY_PERIOD_INDEX`` with one column
        for the requested indicator.
    """
    wait = backoff
    for attempt in range(retries + 1):
        try:
            df = wb.data.DataFrame(
                indicator,
                economy=countries,
                time=range(start_year, end_year + 1)
            )
            return format_wb_query(df, [indicator])
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  API error ({e}), retrying in {wait:.0f}s… (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            wait *= 2


def fetch_wb_subset_indicators(
    indicators: List[str],
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Fetch multiple World Bank indicators in a single API call.

    Parameters
    ----------
    indicators : list of str
        World Bank indicator codes.
    countries : list of str
        ISO Alpha-3 country codes.
    start_year : int
        First year to retrieve data for (inclusive).
    end_year : int
        Last year to retrieve data for (inclusive).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame indexed by ``COUNTRY_PERIOD_INDEX`` with one column
        per indicator.
    """
    df = wb.data.DataFrame(
        indicators,
        economy=countries,
        time=range(start_year, end_year + 1)
    )

    return format_wb_query(df, indicators)


def get_world_bank_indicators(
    indicators: Dict[str, str],
    countries: List[str],
    start_year: int,
    end_year: int,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch all indicators one by one and concatenate into a single DataFrame.

    Fetches each indicator separately (to handle differing availability).
    If ``cache_dir`` is provided, each successfully fetched indicator is
    saved as a CSV there immediately, and any already-cached indicator is
    loaded from disk instead of re-fetched — so a mid-run crash can be
    resumed without re-downloading completed indicators.

    Parameters
    ----------
    indicators : dict
        Mapping from World Bank indicator codes to human-readable names
        (names are printed as progress output; codes become column names).
    countries : list of str
        ISO Alpha-3 country codes.
    start_year : int
        First year to retrieve data for (inclusive).
    end_year : int
        Last year to retrieve data for (inclusive).
    cache_dir : Path, optional
        Directory to store per-indicator CSV caches.  If ``None``, caching
        is disabled.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``COUNTRY_PERIOD_INDEX`` with one column per
        indicator code.
    """
    # WB data typically lags by 1 year; years beyond that are not yet indexed
    # and cause a 502 because wbgapi omits the 'YR' prefix for unknown years.
    import datetime
    safe_end = min(end_year, datetime.date.today().year - 1)
    if safe_end < end_year:
        print(f"Capping end year to {safe_end} (WB data not yet available beyond that)")
    end_year = safe_end

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    ind_series = []
    total = len(indicators)
    for i, (ind_code, ind_name) in enumerate(indicators.items(), 1):
        print(f"[{i}/{total}] {ind_name} ({ind_code})", end=' ')

        cache_path = cache_dir / f"{ind_code.replace('.', '_')}.csv" if cache_dir else None

        if cache_path and cache_path.exists():
            print("(from cache)")
            series = pd.read_csv(cache_path, index_col=0).squeeze()
            series.name = ind_code
        else:
            print("(fetching…)")
            try:
                df_ind = fetch_wb_indicator(ind_code, countries, start_year, end_year)
                series = df_ind[ind_code]
                if cache_path:
                    series.to_csv(cache_path)
                ind_series.append(series)
            except Exception as e:
                print(f"  Skipping {ind_code} after all retries failed: {e}")
                continue

    return pd.concat(ind_series, axis=1)