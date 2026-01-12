from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

start_year = 1999
today = date.today()
current_year = today.year

info_columns = ['YEAR', 'ISO3_COUNTRY_CODE']
