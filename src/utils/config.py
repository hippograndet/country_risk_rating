from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

today = date.today()
current_year = today.year

info_columns = ['YEAR', 'ISO3_COUNTRY_CODE']

country_info_columns = [
    'Info-Country_Name', 'Legal_Systems-Civil_Law', 'Legal_Systems-Common_Law', 'Legal_Systems-Customary', 'Legal_Systems-Muslim', 'Legal_Systems-Mixed', 
    'Languages-Official_language', 'Languages-Regional_language', 'Languages-Minority_language', 'Languages-National_language', 'Languages-Widely_spoken', 'Geography-x_coord',
    'Geography-y_coord', 'Geography-Region', 'Geography-Sub_Region', 'Geography-Intermediate_Region', 'Geography-Region_Code', 'Geography-Sub_Region_Code', 
    'Geography-Intermediate_Region_Code', 'Economy-Income_Group'
]
