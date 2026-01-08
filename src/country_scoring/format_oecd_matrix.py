
print('-----Installing Package Requirements for Formatting OECD Matrix-----')
import os
os.system('apt-get update && apt-get install -y python3-opencv')
os.system('pip install opencv-python')
os.system('pip install ghostscript==9.20')
os.system('pip install "camelot-py[base]"')

import camelot # camelot.io as
import pandas as pd
import datetime

import sys, os
sys.path.append('/root/country/country_scoring')
sys.path.append('/root/country')

from src import addresses
from src import helper_objects

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
            first_date = datetime.datetime.strptime(first_date, '%d-%b-%Y').date()
            last_date = datetime.datetime.strptime(last_date, '%d-%b-%Y').date()
        except ValueError:
            first_date = datetime.datetime.strptime(first_date, '%d-%b-%y').date()
            last_date = datetime.datetime.strptime(last_date, '%d-%b-%y').date()

        mid_date = first_date + (last_date - first_date)/2

        if selected_date == 'first_date':
            date_dict_total[date_string] = first_date
        elif selected_date == 'last_date':
            date_dict_total[date_string] = last_date
        elif selected_date == 'mid_date':
            date_dict_total[date_string] = mid_date
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

def get_quarters(start_year: int = 1999, end_year: int = 2024) -> list:
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
        pd.to_datetime(start_date), 
        pd.to_datetime(end_date), 
        freq='QS', 
        #inclusive='both'
    ))

    return quarters

def get_years(start_year: int = 1999, end_year: int = 2024) -> list:
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
        pd.to_datetime(start_date), 
        pd.to_datetime(end_date), 
        freq='YS',  
        #inclusive='both'
    ))

    return years

def get_grade(country_ratings: pd.Series, date_mid: datetime) -> str:
    """from the series that map, date to rating for specific country, returns the best rating for that quarter.

    Args:
        country_ratings (pd.Series): row, mapping datr to rating.
        date_mid (datetime): the middle date of quarter, if multiple grades, get rating before the middle date.

    Returns:
        str: rating given for specific quarter, gotten from the country_ratings.
    """
    grades = list(country_ratings.unique())
    #print(country_ratings)
    #print(grades)

    grade = ''
    if len(grades) != 1:
        for date, grade in country_ratings.items():
            if date <= date_mid:
                grade = grade
            else:
                break
    else:
        grade = grades[0]

    return grade

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

def get_dates_in_interval(dates_columns: list, date1: datetime, date2: datetime) -> list:
    """from dates columns list, return the list of dates in between date1 and date2, return the last date before date1, if no date in interval.

    Args:
        dates_columns (list): datetime objects.
        date1 (datetime): lower bound of date interval.
        date2 (datetime): upper bound of date interval.

    Returns:
        list: datetime objects in interval.
    """
    dates_in_interval = []
    last_date_before = ''
    for date in dates_columns:

        if date1 <= date and date <= date2:
            dates_in_interval.append(date)
        elif date > date2:
            break
        else:
            last_date_before = date

    if len(dates_in_interval) == 0:
        dates_in_interval = [last_date_before]

    return dates_in_interval

def oecd_df_to_format(oecd_df_clean: pd.DataFrame, format: str = 'Q') -> pd.DataFrame:
    """convert input df into formated df, specified by format.

    Args:
        oecd_df_clean (DataFrame): formated raw matrix of OECD ratings.
        format (string): Time interval to format the matrix in.
        
    Returns:
        DataFrame: df with format, rows are countries, and columns are quarters if format Q, years if format Y, and original raw format if else.
    """

    if format == 'Q':
        dates = get_quarters(start_year=1999, end_year=helper_objects.current_year+1)
    elif format == 'Y':    
        dates = get_years(start_year=1999, end_year=helper_objects.current_year+1)
    else:
        print('format', format, 'not coded')
        return oecd_df_clean

    formated_dates = []
    for d in dates:
        formated_dates.append(d.strftime('%Y-%m-%d'))

    iso_codes = oecd_df_clean.index
    dates_columns = oecd_df_clean.columns

    formated_df = pd.DataFrame(columns=formated_dates, index=iso_codes)
    for i in range(len(dates)-1):
        date_i = dates[i]
        date_j = dates[i+1]
        date_mid = (date_i + (date_j - date_i)/2).date()

        dates_in_interval = get_dates_in_interval(dates_columns, date_i, date_j)
        #print(dates_in_interval)
        dft = oecd_df_clean[dates_in_interval]
        for iso in iso_codes:
            country_grades = dft.loc[iso]
            grade = get_grade(country_grades, date_mid)
            formated_df.at[iso, formated_dates[i]] = grade

    formated_df = formated_df.dropna(how='all', axis=1)        
    
    return formated_df

def get_oecd_rating_matrix(oecd_raw_file_name: str = 'oecd_country_ratings', format: str = 'Q') -> pd.DataFrame:
    """get input pdf file as df, with rows as countries and columns as date.

    Args:
        oecd_raw_file_name (str, optional): name of the OECD pdf file, if oecd_rating_matrix, then get raw pre-cleaned matrix. Defaults to 'oecd_country_ratings'.
        quarterly (bool, optional): if true, convert df to quarterly, that is get 1 grade per country per year. Defaults to True.

    Returns:
        DataFrame: formated df with rows as countries and columns as date.
    """
    
    if oecd_raw_file_name == 'oecd_rating_matrix':
        oecd_rating_matrix = pd.read_csv(addresses.country_BE_address + 'intermed_data/oecd/' + oecd_raw_file_name + '_raw.csv', index_col=0)
        oecd_rating_matrix.columns = pd.to_datetime(oecd_rating_matrix.columns)
    else:
        try:
            file =  'data/' + oecd_raw_file_name + '.pdf' # helper_objects.oecd_raw_data_address +
        
            tables = camelot.read_pdf(
                filepath=file,
                pages='all'
            )  

        except FileNotFound:
            file =  addresses.country_BE_address + 'raw_data/' + oecd_raw_file_name + '.pdf' # helper_objects.oecd_raw_data_address +
        
            tables = camelot.read_pdf(
                filepath=file,
                pages='all'
            )  

        oecd_df = pages_to_df(tables)

        # Clean OECD values
        oecd_rating_matrix = oecd_df.apply(
            lambda column: column.apply(clean_oecd_value)
        )

        # Clean column names
        oecd_rating_matrix = oecd_rating_matrix.rename(
            columns=get_date_dict(oecd_rating_matrix)
        )

    # Format Matrix to the time interval needed, format can be raw, yearly (Y), and quarterly (Q)
    oecd_rating_matrix = oecd_df_to_format(
        oecd_df_clean=oecd_rating_matrix, 
        format=format
    )

    return oecd_rating_matrix