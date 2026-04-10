"""
Project-wide path and date configuration for country risk rating prediction.

All paths are resolved relative to the project root so the module works
regardless of the working directory from which scripts are invoked.
"""

from pathlib import Path
from datetime import date

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "1-raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "2-interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "3-processed"
METADATA_DIR: Path = DATA_DIR / "0-metadata"

start_year: int = 1999
today: date = date.today()
current_year: int = today.year

info_columns: list = ['YEAR', 'ISO3_COUNTRY_CODE']
