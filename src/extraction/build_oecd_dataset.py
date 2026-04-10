"""
OECD country risk rating extraction for country risk rating prediction.

Parses the OECD country risk PDF (via camelot), cleans raw rating values,
and reshapes the result into a tidy annual matrix indexed by ISO3 country
code.
"""

import re
import os

import camelot
import pandas as pd
from datetime import datetime

DATE_PATTERN = re.compile(r'\d{2}-\w{3}-\d{2,4}')

from src.utils import config, io


def get_date_dict(
    oecd_df_clean: pd.DataFrame,
    selected_date: str = 'first_date'
) -> dict:
    """
    Map each date-range column header to a single representative datetime.

    OECD PDF column headers encode a date interval (e.g.
    ``'01-Jan-2020\\n31-Dec-2020'``); this function picks either the first or
    last date of each interval.

    Parameters
    ----------
    oecd_df_clean : pd.DataFrame
        DataFrame whose columns are date-interval strings in the format
        ``'<start>\\n<end>'``.
    selected_date : str, optional
        Which endpoint to use: ``'first_date'`` (default) or ``'last_date'``.

    Returns
    -------
    dict
        Mapping from original column string to the chosen ``datetime``.
    """
    def _parse_date(s: str) -> datetime:
        for fmt in ('%d-%b-%Y', '%d-%b-%y'):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date string: {s!r}")

    date_dict_total = {}
    for date_string in oecd_df_clean.columns:
        parts = str(date_string).split('\n')
        first_date_str = parts[0].strip()
        last_date_str = parts[1].strip() if len(parts) > 1 else ''

        # Skip columns that have no parseable date at all
        if not first_date_str:
            continue

        first_date = _parse_date(first_date_str)
        last_date = _parse_date(last_date_str) if last_date_str else first_date

        # mid_date = first_date + (last_date - first_date)/2

        if selected_date == 'first_date':
            date_dict_total[date_string] = first_date
        elif selected_date == 'last_date':
            date_dict_total[date_string] = last_date
        # elif selected_date == 'mid_date':
        #     date_dict_total[date_string] = mid_date
        else:
            date_dict_total[date_string] = first_date

    return date_dict_total


def clean_oecd_value(value: str) -> str:
    """
    Normalise a raw rating string read from the OECD PDF table.

    Handles multi-character strings that encode special values (e.g.
    asterisk-annotated ratings), and maps rating ``'0'`` to ``'1'`` so
    that the scale is balanced from 1 to 7.

    Parameters
    ----------
    value : str
        Raw cell value as extracted by camelot.

    Returns
    -------
    str
        Cleaned rating string (``'1'``–``'7'``) or ``'-'`` for unrated
        countries/periods.
    """
    # Coerce to string; treat NaN / empty cells as unrated
    if pd.isna(value):
        return '-'
    s = str(value).strip()
    if not s:
        return '-'

    if len(s) == 1:
        v = s
    else:
        if s[1] == '5':
            v = '-'
        elif s[1] == '6':
            v = '0'
        elif s[1] == '7':
            v = '0'
        elif s[1] == '8':
            v = s[-1]
        elif s[1] == '9':
            v = '0'
        else:
            v = s[-1]

    # Map grade 0 to 1 to balance the dataset scale
    if v == '0':
        v = '1'

    return v


def get_yearly_dates(
    start_year: int = 1999,
    end_year: int = 2024
) -> list:
    """
    Return a list of year-start datetimes from ``start_year`` to ``end_year``.

    Parameters
    ----------
    start_year : int, optional
        First year (inclusive, default 1999).
    end_year : int, optional
        Last year (inclusive, default 2024).

    Returns
    -------
    list of datetime
        One ``datetime`` per year, each set to 1 January of that year.
    """
    start_date = '01/01/' + str(start_year)
    end_date = '01/01/' + str(end_year)

    years = list(pd.date_range(
        datetime.strptime(start_date, '%d/%m/%Y'),
        datetime.strptime(end_date, '%d/%m/%Y'),
        freq='YS',
        # inclusive='both'
    ))

    return years


def get_last_grade_idx(
    dates_columns: pd.Series,
    date_selected: datetime
) -> int:
    """
    Return the index of the latest rating date that does not exceed a target date.

    Parameters
    ----------
    dates_columns : pd.Index or pd.Series of datetime
        All available rating dates.
    date_selected : datetime
        Reference date; the function finds the last date on or before this.

    Returns
    -------
    int
        Index position of the most recent date ≤ ``date_selected``, or
        ``-1`` if no such date exists.
    """
    prev_dates = dates_columns[dates_columns <= date_selected]
    if len(prev_dates) == 0:
        # No previous dates
        idx_last_date = -1
    else:
        idx_last_date = prev_dates.argmax()

    return idx_last_date

def oecd_df_to_format(oecd_df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a raw OECD rating matrix to an annual (calendar-year) grid.

    For each year from ``config.start_year`` to the year of the latest rating,
    the function looks up the most recent rating available on or before
    1 January of that year.

    Parameters
    ----------
    oecd_df_clean : pd.DataFrame
        Raw OECD rating matrix with datetime columns (output of
        :func:`create_raw_oecd_rating_dataset`).

    Returns
    -------
    pd.DataFrame
        Annual rating matrix with ISO3 country codes as rows and
        ``'YYYY-MM-DD'`` strings as columns; empty columns are dropped.
    """
    latest_date = oecd_df_clean.columns.max()

    dates_interval_l = get_yearly_dates(start_year=config.start_year, end_year=latest_date.year + 1)

    formated_dates = []
    for d in dates_interval_l:
        formated_dates.append(d.strftime('%Y-%m-%d'))

    formated_df = pd.DataFrame(columns=formated_dates, index=oecd_df_clean.index)
    for i in range(len(formated_dates)):
        idx_last_date = get_last_grade_idx(oecd_df_clean.columns, dates_interval_l[i])
        if idx_last_date == -1:
            formated_df.loc[:, formated_dates[i]] = '-'
        else:
            formated_df.loc[:, formated_dates[i]] = oecd_df_clean[oecd_df_clean.columns[idx_last_date]]

    formated_df = formated_df.dropna(how='all', axis=1)

    return formated_df

def pages_to_df(tables: camelot.core.TableList) -> pd.DataFrame:
    """
    Concatenate camelot pages into a single wide rating DataFrame.

    Each page covers a subset of countries for one or more rating periods.
    Pages sharing the same leading date column are concatenated vertically;
    pages with a new date are added as additional columns.

    Parameters
    ----------
    tables : camelot.core.TableList
        List of tables as returned by ``camelot.read_pdf``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with ISO3 country codes as the index and rating-period
        strings as columns.
    """
    current_date = ''
    total_list = []
    specific_date_list = []
    for i in range(tables.n - 1):
        print('On page:', str(i + 1), end='\r')

        page_df = tables[i].df.copy()

        print(page_df)

        page_df = page_df.iloc[1:, 1:]
        page_df = page_df.reset_index()
        page_df = page_df.iloc[:, 1:]

        page_df.at[0, page_df.columns[0]] = 'ISO3_COUNTRY_CODE'
        page_df.at[0, page_df.columns[1]] = 'COUNTRY_NAME'

        page_df.columns = page_df.loc[0]
        page_df = page_df.iloc[2:, :]
        page_df = page_df.set_index('ISO3_COUNTRY_CODE')
        df_page_clean = page_df.drop(columns=['COUNTRY_NAME'])

        # If page has same date as previous page, then append to specific date list
        if df_page_clean.columns[0] == current_date:
            specific_date_list.append(df_page_clean)
        # If page has a new date, concat the previous pages and start a new list
        else:
            if len(specific_date_list) > 0:
                total_list.append(pd.concat(specific_date_list))

            current_date = df_page_clean.columns[0]
            specific_date_list = [df_page_clean]

    if len(specific_date_list) > 0:
        total_list.append(pd.concat(specific_date_list))

    df_oecd = pd.concat(total_list, axis=1)
    return df_oecd

def extract_page(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one raw camelot DataFrame into a clean (ISO3 index, date columns) slice.

    Handles two OECD PDF layout variants:

    **Old format** (pre-2026 PDFs): Row 1 already contains complete
    ``'start\\nend'`` date-range strings for each rating column.  Row 2 is a
    language label row (``'English'``) with no date content and is skipped.

    **New format** (2026+ PDFs): The date range is split across two rows —
    Row 1 holds start dates (with the first one merged into the
    ``'Country Name'`` cell) and Row 2 holds end dates.  Both rows are
    inspected and their date strings are reconstructed into
    ``'start\\nend'`` pairs.

    In both formats the ISO3 codes for all countries on the page are merged
    by the PDF renderer into a single ``'\\n'``-joined cell in the first data
    row (col 1).  This function splits that cell and assigns one ISO3 code
    to each country row.

    Parameters
    ----------
    raw_df : pd.DataFrame
        A single table as returned by camelot (integer-indexed rows and
        columns).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with ``ISO3_COUNTRY_CODE`` as the index and
        ``'start\\nend'`` date-range strings as columns, or ``None`` if the
        page does not contain usable rating data.
    """
    # Need at least: row-numbers row, header row, English/end-dates row, one data row
    if raw_df.shape[0] < 4 or raw_df.shape[1] < 4:
        return pd.DataFrame()

    row1 = raw_df.iloc[1]   # start-dates, or full "start\nend" date strings (old)
    row2 = raw_df.iloc[2]   # "English" label (old) or end-dates (new)

    # ── Format detection ─────────────────────────────────────────────────────
    # New format: rating columns in row2 (cols 3+) contain date strings.
    new_format = any(
        bool(DATE_PATTERN.search(str(v)))
        for v in row2.iloc[3:]
    )

    # ── Reconstruct date-range column headers ─────────────────────────────────
    rating_cols = list(raw_df.columns[3:])  # positional indices for rating cols

    if new_format:
        # In the new format the first date pair is merged into col 2
        # (e.g. "13-Oct-23\nCountry Name (1)" / "09-Feb-24\nEnglish").
        date_headers = []
        for col in rating_cols:
            start = DATE_PATTERN.search(str(row1.iloc[col]))
            end   = DATE_PATTERN.search(str(row2.iloc[col]))
            if start and end:
                date_headers.append(f"{start.group()}\n{end.group()}")
            elif start:
                date_headers.append(start.group())
            else:
                date_headers.append(str(row1.iloc[col]).strip())

        # Overwrite the first header slot with the date extracted from col 2
        first_start = DATE_PATTERN.search(str(row1.iloc[2]))
        first_end   = DATE_PATTERN.search(str(row2.iloc[2]))
        if first_start and first_end:
            date_headers = [f"{first_start.group()}\n{first_end.group()}"] + date_headers[1:]
    else:
        # Old format: each rating column already holds "start\nend"
        date_headers = [str(row1.iloc[col]).strip() for col in rating_cols]

    # ── Data rows ─────────────────────────────────────────────────────────────
    data = raw_df.iloc[3:].reset_index(drop=True)

    # Col 1 of the first data row holds all ISO3 codes merged with '\n'
    iso_cell = str(data.iloc[0, 1])
    iso_list = [
        x.strip() for x in iso_cell.split('\n')
        if x.strip() and len(x.strip()) == 3
    ]

    # Country name is in col 2; keep only rows that have a non-empty name
    country_rows = data[data.iloc[:, 2].str.strip() != ''].reset_index(drop=True)

    n = min(len(iso_list), len(country_rows))
    if n == 0:
        return pd.DataFrame()

    result = country_rows.iloc[:n, 3:3 + len(date_headers)].copy()
    result.columns = date_headers
    result.index = iso_list[:n]
    result.index.name = 'ISO3_COUNTRY_CODE'

    return result


def pages_to_df_v2(tables: camelot.core.TableList) -> pd.DataFrame:
    """
    Concatenate camelot pages into a single wide rating DataFrame.

    Each page is parsed by :func:`extract_page`, which handles both the old
    (pre-2026) and new (2026+) OECD PDF layout variants.  Pages that yield
    no usable data are silently skipped.  Pages that share the same leading
    date column (i.e. cover different alphabetical slices of the same rating
    period) are concatenated vertically before being added as a column group.

    Parameters
    ----------
    tables : camelot.core.TableList
        List of tables as returned by ``camelot.read_pdf``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with ``ISO3_COUNTRY_CODE`` as the index and
        ``'start\\nend'`` date-range strings as columns.

    Raises
    ------
    ValueError
        If no usable rating tables were found across all pages.
    """
    current_date = ''
    total_list = []
    date_group = []

    for i in range(tables.n):
        print('On page:', str(i + 1), end='\r')

        page_result = extract_page(tables[i].df)

        if page_result.empty:
            continue

        first_col = str(page_result.columns[0]).strip()
        if not first_col:
            continue

        if first_col == current_date:
            date_group.append(page_result)
        else:
            if date_group:
                total_list.append(pd.concat(date_group))
            current_date = first_col
            date_group = [page_result]

    if date_group:
        total_list.append(pd.concat(date_group))

    if not total_list:
        raise ValueError("No usable rating tables found in the PDF.")

    df_oecd = pd.concat(total_list, axis=1)
    df_oecd = df_oecd[~df_oecd.index.duplicated(keep='last')]
    df_oecd = df_oecd.loc[:, ~df_oecd.columns.duplicated(keep='last')]

    return df_oecd

def create_raw_oecd_rating_dataset(oecd_fname: str) -> tuple:
    """
    Parse an OECD country risk PDF and return a cleaned rating DataFrame.

    Robust version of :func:`create_raw_oecd_rating_dataset` that uses
    :func:`pages_to_df_v2` to handle newer PDF layouts (e.g. single-column
    final sections, footnote-only pages).

    If the named PDF is not found, falls back to the most recent PDF in the
    directory.

    Parameters
    ----------
    oecd_fname : str
        Base name of the OECD PDF file (without extension), e.g.
        ``'03-07-2026'``.

    Returns
    -------
    tuple of (str, pd.DataFrame)
        ``(oecd_fname, dataset)`` where ``oecd_fname`` is the name actually
        used (may differ from input if the fallback path was taken) and
        ``dataset`` is a wide DataFrame with ISO3 codes as rows and
        ``datetime`` columns.
    """
    pdf_address = config.RAW_DATA_DIR / 'oecd_country_ratings_pdfs'

    try:
        pdf_file_address = pdf_address / (oecd_fname + '.pdf')
        tables = io.load_pdf(pdf_file_address, pages='all')
    except FileNotFoundError:
        pdf_file_names = [f for f in os.listdir(pdf_address) if f.endswith('.pdf')]
        new_oecd_fname = get_latest_pdf_name(pdf_file_names)
        pdf_file_address = pdf_address / new_oecd_fname
        tables = io.load_pdf(pdf_file_address, pages='all')
        oecd_fname = new_oecd_fname.split('.')[0]

    oecd_df = pages_to_df_v2(tables)

    # Clean OECD values
    oecd_rating_dataset = oecd_df.apply(
        lambda column: column.apply(clean_oecd_value)
    )

    # Convert column headers (date-range strings) to datetime objects;
    # drop any columns that could not be parsed (no valid date string).
    oecd_rating_dataset = oecd_rating_dataset.rename(
        columns=get_date_dict(oecd_rating_dataset)
    )
    oecd_rating_dataset = oecd_rating_dataset.loc[
        :, [c for c in oecd_rating_dataset.columns if isinstance(c, datetime)]
    ]

    return oecd_fname, oecd_rating_dataset


def get_clean_oecd_rating_df(oecd_fname: str = '03-07-2026') -> pd.DataFrame:
    """
    Load or build the OECD rating matrix for a given PDF snapshot (robust version).

    Drop-in replacement for :func:`get_clean_oecd_rating_df` that uses the
    v2 parsing pipeline.  Checks first for a pre-built CSV cache in
    ``data/1-raw/oecd_country_ratings_datasets/``; if not found, parses the
    corresponding PDF and saves the result.

    Parameters
    ----------
    oecd_fname : str, optional
        Base name of the OECD PDF/CSV file (without extension), e.g.
        ``'03-07-2026'``. Defaults to ``'03-07-2026'``.

    Returns
    -------
    pd.DataFrame
        Annual rating matrix with ISO3 country codes as rows and
        ``datetime`` columns.
    """

    df_address = config.RAW_DATA_DIR / 'oecd_country_ratings_datasets'
    raw_oecd_rating_dataset_name = df_address / (oecd_fname + '.csv')
    raw_oecd_rating_dataset = pd.DataFrame()
    try:
        raw_oecd_rating_dataset = io.load_csv(raw_oecd_rating_dataset_name, index_col=0)
        # Drop unnamed columns and strip pandas duplicate suffixes (e.g. "2023-10-13.1")
        cols = raw_oecd_rating_dataset.columns
        cols = cols[~cols.str.startswith('Unnamed')]
        cols = cols[~cols.str.contains(r'\.\d+$', regex=True)]
        raw_oecd_rating_dataset = raw_oecd_rating_dataset.loc[:, cols]
        raw_oecd_rating_dataset.columns = pd.to_datetime(raw_oecd_rating_dataset.columns)
        print('Reading CSV')
    except FileNotFoundError:
        oecd_fname, raw_oecd_rating_dataset = create_raw_oecd_rating_dataset(oecd_fname)
        raw_oecd_rating_dataset_name = df_address / (oecd_fname + '.csv')
        io.save_csv(raw_oecd_rating_dataset, raw_oecd_rating_dataset_name, index=True)

    # Resample to annual calendar-year grid
    oecd_rating_dataset = oecd_df_to_format(oecd_df_clean=raw_oecd_rating_dataset)

    return oecd_rating_dataset


def get_latest_pdf_name(fnames: list) -> str:
    """
    Return the filename with the most recent date encoded in its name.

    File names are expected to follow the ``'DD-MM-YYYY.pdf'`` convention.

    Parameters
    ----------
    fnames : list of str
        Candidate file names.

    Returns
    -------
    str
        The filename whose date component is the latest.
    """
    return max(
        fnames,
        key=lambda f: datetime.strptime(f.split('.')[0], "%d-%m-%Y")
    )

