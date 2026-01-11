
from src.extraction import get_clean_oecd_rating_df
from src.utils.io import save_csv
from src.utils.config import INTERIM_DATA_DIR

df = get_clean_oecd_rating_df(...)
save_csv(df, INTERIM_DATA_DIR / "oecd_ratings_clean.csv")
