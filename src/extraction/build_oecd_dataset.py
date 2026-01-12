import camelot # camelot.io as
import pandas as pd
# import datetime
from datetime import datetime

import os
# sys.path.append('/root/country/country_scoring')
# sys.path.append('/root/country')

from src.utils import config, io



def get_date_dict(oecd_df_clean: pd.DataFrame, selected_date: str = 'first_date') -> dict:
    """get dictionary that maps name of columns of the clean oecd df, to their corresponding date chosen by selected_date.

    Args:
        oecd_df_clean (DataFrame): df of oecd ratings with as index country iso3 code, and as columns a date interval.
        selected_date (str, optional): date to choose, choice between first, middle or last date for every column, which is a date interval, . Defaults to 'first_date'.

    Returns:
        dict: mapping from column name with format first_date\\nlast_date.
    """
    date_dict_total = {}
    for date_string in oecd_df_clean.columns:
        first_date = date_string.split('\n')[0]
        last_date = date_string.split('\n')[1]
        try:
            first_date = datetime.strptime(first_date, '%d-%b-%Y')
            last_date = datetime.strptime(last_date, '%d-%b-%Y')
        except ValueError:
            first_date = datetime.strptime(first_date, '%d-%b-%y')
            last_date = datetime.strptime(last_date, '%d-%b-%y')

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
    """converts string value read from df created from OECD pdf, and returns its corresponding rating in a string.

    Args:
        value (string): input value, if '0', convert to '1', and if value has a asterix, change score to corresponding value.

    Returns:
        string: final value of the rating from input string.
    """
    v = ''
    if len(value) == 1:
        v = value
    else:
        if value[1] == '5':
            v = '-'
        elif value[1] == '6':
            v = '0'
        elif value[1] == '7':
            v = '0'
        elif value[1] == '8':
            v = value[-1]
        elif value[1] == '9':
            v = '0'
        else:
            v = value[-1]

    # Make all grades of 0 to 1, to balance dataset
    if v == '0':
        v = '1'

    return v    

def get_quarterly_dates(start_year: int = 1999, end_year: int = 2024) -> list:
    """Given start and end year, return a list of the date of each quarter between the years, last year excluded.

    Args:
        start_year (int, optional): year to start getting quarters from. Defaults to 1999.
        end_year (int, optional): year to stop getting quarters from. Defaults to 2024.

    Returns:
        list: datetime objects.
    """
    start_date = '01/01/' + str(start_year)
    end_date = '01/01/' + str(end_year)

    quarters = list(pd.date_range(
        datetime.strptime(start_date, '%d/%m/%Y'), 
        datetime.strptime(end_date, '%d/%m/%Y'), 
        freq='QS', 
            #inclusive='both'
    ))

    return quarters

def get_yearly_dates(start_year: int = 1999, end_year: int = 2024) -> list:
    """Given start and end year, return a list of the date of each years start date between the years, last year excluded.

    Args:
        start_year (int, optional): year to start getting quarters from. Defaults to 1999.
        end_year (int, optional): year to stop getting quarters from. Defaults to 2024.

    Returns:
        list: datetime objects.
    """
    start_date = '01/01/' + str(start_year)
    end_date = '01/01/' + str(end_year)

    years = list(pd.date_range(
        datetime.strptime(start_date, '%d/%m/%Y'), 
        datetime.strptime(end_date, '%d/%m/%Y'), 
        freq='YS', 
            #inclusive='both'
    ))

    return years

def get_last_grade_idx(dates_columns: pd.Series, date_selected: datetime) -> int:
    """get latest grade date before selected date

    Args:
        dates_columns (pd Series (Index) of dates): all date columns.
        date_selected (datetime): selected date, to select last previous from.
    Returns:
        int: index of last date column before selected date.
    """
    prev_dates = dates_columns[dates_columns <= date_selected]
    if len(prev_dates) == 0:
        # No previous dates
        idx_last_date = -1
    else:
        idx_last_date = prev_dates.argmax()

    return idx_last_date

def pages_to_df(tables) -> pd.DataFrame:
    """converts camelot pages to one whole df.

    Args:
        tables (camelot.core.TableList): list of tables read from OECD rating pdf.

    Returns:
        DataFrame: columns corresponds to date and rows correspond to country.
    """

    # read pdf with one dataframe per page
    current_date = ''
    total_list = []
    specific_date_list = []
    for i in range(tables.n-1):
        print('On page:', str(i+1), end='\r')
            
        page_df = tables[i].df.copy()

        page_df = page_df.iloc[1: , 1:]
        page_df = page_df.reset_index()
        page_df = page_df.iloc[: , 1:]

        page_df.at[0, page_df.columns[0]] = 'ISO3_COUNTRY_CODE'
        page_df.at[0, page_df.columns[1]] = 'COUNTRY_NAME'

        page_df.columns = page_df.loc[0]
        page_df = page_df.iloc[2: , :]
        page_df = page_df.set_index('ISO3_COUNTRY_CODE')
        df_page_clean = page_df.drop(columns=['COUNTRY_NAME'])

        # If page has same date as previous page, then append to specific date list
        if df_page_clean.columns[0] == current_date:
            specific_date_list.append(df_page_clean)
        # If page has a new date, then concat the previous pages for previous dates, and start a new specific date pages list
        else:
            if len(specific_date_list) > 0:
                total_list.append(pd.concat(specific_date_list))

            current_date = df_page_clean.columns[0]
            specific_date_list = [df_page_clean]

    if len(specific_date_list) > 0:
        total_list.append(pd.concat(specific_date_list))

    df_oecd = pd.concat(total_list, axis=1)
    return df_oecd

def oecd_df_to_format(oecd_df_clean: pd.DataFrame, date_freq: str = 'Q') -> pd.DataFrame:
    """convert input df into formated df, specified by format.

    Args:
        oecd_df_clean (DataFrame): formated raw matrix of OECD ratings.
        date_freq (str, optional): Frequency at which ratings are recorded, 'M' for monthly and 'Y' for yearly.
        
    Returns:
        DataFrame: df with format, rows are countries, and columns are quarters if format Q, years if format Y, and original raw format if else.
    """

    earliest_date = oecd_df_clean.columns.min()
    latest_date = oecd_df_clean.columns.max()

    if date_freq == 'Q':
        dates_interval_l = get_quarterly_dates(start_year=config.start_year, end_year=latest_date.year + 1)
    elif date_freq == 'Y':    
        dates_interval_l = get_yearly_dates(start_year=config.start_year, end_year=latest_date.year + 1)
    else:
        print('Date Frequency', date_freq, 'not implemented.')
        return oecd_df_clean

    formated_dates = []
    for d in dates_interval_l:
        formated_dates.append(d.strftime('%Y-%m-%d'))

    formated_df = pd.DataFrame(columns=formated_dates, index=oecd_df_clean.index)
    for i in range(len(formated_dates)-1):
        idx_last_date = get_last_grade_idx(oecd_df_clean.columns, dates_interval_l[i])
        if idx_last_date == -1:
            formated_df.loc[:, formated_dates[i]] = '-'
        else:
            formated_df.loc[:, formated_dates[i]] = oecd_df_clean[oecd_df_clean.columns[idx_last_date]]

    formated_df = formated_df.dropna(how='all', axis=1)        
    
    return formated_df

def get_latest_pdf_name(fnames: list) -> str:
    """get latest file name, based off date in name.

    Args:
        fnames (list of string): List of file names to find latest in.
    Returns:
        string: name of file with latest date.
    """

    return max(
        fnames,
        key=lambda f: datetime.strptime(f.split('.')[0], "%d-%m-%Y")
    )

def create_raw_oecd_rating_dataset(oecd_fname):
    """create raw dataset from pdf.

    Args:
        oecd_fname (str, optional): name of the OECD pdf file to create dataset from, latest if none specified/found.

    Returns:
        DataFrame: formated df with rows as countries and columns as date.
    """

    pdf_address = config.RAW_DATA_DIR / 'oecd_country_ratings_pdfs'

    try:
        pdf_file_address = pdf_address / (oecd_fname + '.pdf')
        tables = io.load_pdf(pdf_file_address, pages='all')
    except FileNotFoundError:
        pdf_file_names = os.listdir(pdf_address)
        new_oecd_fname = get_latest_pdf_name(pdf_file_names)
        pdf_file_address = pdf_address / oecd_fname
        tables = io.load_pdf(pdf_file_address, pages='all')

        oecd_fname = new_oecd_fname.split('.')[0]

    oecd_df = pages_to_df(tables)

    # Clean OECD values
    oecd_rating_dataset = oecd_df.apply(
        lambda column: column.apply(clean_oecd_value)
    )

    # Clean column names
    oecd_rating_dataset = oecd_rating_dataset.rename(
        columns=get_date_dict(oecd_rating_dataset)
    )

    return oecd_fname, oecd_rating_dataset

def get_clean_oecd_rating_df(oecd_fname: str = '30-06-2023', date_freq: str = 'Q') -> pd.DataFrame:
    """get input pdf file as df, with rows as countries and columns as date.

    Args:
        oecd_fname (str, optional): name of the OECD file, first check if dataset already exists (in csv format) otherwise create from pdf, latest if none specified.
        date_freq (str, optional): Frequency at which ratings are recorded, 'M' for monthly and 'Y' for yearly.

    Returns:
        DataFrame: formated df with rows as countries and columns as date.
    """

    df_address = config.RAW_DATA_DIR / 'oecd_country_ratings_datasets'
    raw_oecd_rating_dataset_name = df_address / (oecd_fname + '.csv')
    raw_oecd_rating_dataset = pd.DataFrame()
    try:
        raw_oecd_rating_dataset = io.load_csv(raw_oecd_rating_dataset_name, index_col=0)
        raw_oecd_rating_dataset.columns = pd.to_datetime(raw_oecd_rating_dataset.columns)
        print('Reading CSV')
    except FileNotFoundError:
        # No dataset form raw data, need to convert pdf to dataset (csv) form
        oecd_fname, raw_oecd_rating_dataset = create_raw_oecd_rating_dataset(oecd_fname)
        raw_oecd_rating_dataset_name = df_address / (oecd_fname + '.csv')
        io.save_csv(raw_oecd_rating_dataset, raw_oecd_rating_dataset_name, index=True)

    # Format Dataset to the sepecified time interval needed, format can be raw, yearly (Y), and quarterly (Q)
    oecd_rating_dataset = oecd_df_to_format(
        oecd_df_clean=raw_oecd_rating_dataset, 
        date_freq=date_freq
    )

    return oecd_rating_dataset