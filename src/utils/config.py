from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "1-raw"
INTERIM_DATA_DIR = DATA_DIR / "2-interim"
PROCESSED_DATA_DIR = DATA_DIR / "3-processed"
METADATA_DIR = DATA_DIR / "0-metadata"

start_year = 1999
today = date.today()
current_year = today.year

info_columns = ['YEAR', 'ISO3_COUNTRY_CODE']
