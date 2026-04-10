"""
I/O utilities for country risk rating prediction.

Thin wrappers around pandas, camelot, and json that enforce consistent error
handling and directory creation across the pipeline.
"""

from pathlib import Path
import json
import logging

import pandas as pd
import camelot

# pdfminer logs spurious warnings about unrecognised spot/Pantone color values
# (e.g. "'P19' is an invalid float value") that don't affect table extraction.
logging.getLogger('pdfminer').setLevel(logging.ERROR)


def ensure_dir(path: Path) -> None:
    """
    Create a directory (and any missing parents) if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_pdf(path: Path, **kwargs) -> camelot.core.TableList:
    """
    Load a PDF file into a camelot ``TableList``.

    Parameters
    ----------
    path : Path
        Path to the PDF file.
    **kwargs
        Additional keyword arguments are accepted but not forwarded (camelot
        options are fixed internally).

    Returns
    -------
    camelot.core.TableList
        List of tables extracted from all pages of the PDF.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return camelot.read_pdf(
        filepath=path,
        pages='all', #'all'
    )


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    **kwargs
        Forwarded to ``pandas.read_csv`` (e.g. ``index_col``, ``dtype``).

    Returns
    -------
    pd.DataFrame
        Contents of the CSV file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_json(path: Path, **kwargs) -> dict:
    """
    Load a JSON file into a Python object.

    Parameters
    ----------
    path : Path
        Path to the JSON file.
    **kwargs
        Accepted but not forwarded (reserved for future use).

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.load(open(path))


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to serialise.
    path : Path
        Destination file path (parent directories are created if missing).
    index : bool, optional
        Whether to write the DataFrame index to the CSV (default False).
    """
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
