"""
Burn-date bookkeeping for the data release.

Reads the burn log once and exposes the burn_id <-> date mapping plus a helper
that tags each row of an instrument record with the burn it falls on. A row is
tagged only if its calendar date matches a burn day; all other rows carry an
empty burn_id so the released file keeps the full continuous record.

Author: Nathan Lima
Created: 2026-08-06
"""

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.data_paths import get_common_file  # noqa: E402


@lru_cache(maxsize=1)
def burn_dates() -> dict:
    """
    Return the burn_id -> calendar date map from the burn log.

    Returns
    -------
    dict
        Keys are burn ids ('burn1'..'burn10'); values are ``datetime.date``.
    """
    bl = pd.read_excel(get_common_file("burn_log"), sheet_name="Sheet2")
    bl = bl.dropna(subset=["Burn ID", "Date"])
    out = {}
    for _, row in bl.iterrows():
        out[str(row["Burn ID"]).strip()] = pd.to_datetime(row["Date"]).date()
    return out


def tag_burn_id(df: pd.DataFrame, datetime_col: str) -> pd.Series:
    """
    Return a burn_id label for each row based on its calendar date.

    Parameters
    ----------
    df : pd.DataFrame
        Instrument record with a datetime column.
    datetime_col : str
        Name of the datetime column to read the calendar date from.

    Returns
    -------
    pd.Series
        String series aligned to ``df.index``; each element is the burn id when
        the row's date matches a burn day, otherwise an empty string.
    """
    date_to_burn = {d: b for b, d in burn_dates().items()}
    dates = pd.to_datetime(df[datetime_col], errors="coerce").dt.date
    return dates.map(date_to_burn).fillna("").astype(str)
