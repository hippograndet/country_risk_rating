from pathlib import Path
import pandas as pd
import camelot


def ensure_dir(path: Path) -> None:
    """
    Ensure that a directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_pdf(path: Path, **kwargs) -> camelot.core.TableList:
    """
    Load a pdf file into a camelot TableList.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return camelot.read_pdf(
        filepath=path,
        pages='all'
    )  


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, ensuring the directory exists.
    """
    ensure_dir(path.parent)
    df.to_csv(path, index=index)