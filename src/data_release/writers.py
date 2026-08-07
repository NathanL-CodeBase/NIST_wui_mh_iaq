"""
CSV writer for the data release.

Writes each prepared DataFrame to the release directory with a leading standard
column order (datetime, burn_id, location, instrument, serial_number) and
UTF-8-with-BOM encoding so Excel reads the micro sign correctly. Existing output
is never overwritten silently: an ISO-dated suffix is appended when a target
file already exists, matching the project's safety defaults.

Author: Nathan Lima
Created: 2026-08-06
"""

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.data_paths import resolver  # noqa: E402

# Standard leading columns shared by every released file, in order.
LEADING_COLS = ["datetime", "burn_id", "location", "instrument", "serial_number"]


def release_dir() -> Path:
    """Resolve and create the MIDAS release output directory."""
    out = Path(resolver.get_common_file("midas_release"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put the standard leading columns first, keeping the rest in their order.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared instrument frame; may or may not contain every leading column.

    Returns
    -------
    pd.DataFrame
        Reordered copy with present leading columns first.
    """
    leading = [c for c in LEADING_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in leading]
    return df[leading + rest]


def write_release_csv(df: pd.DataFrame, filename: str, today: str) -> Path:
    """
    Write a prepared frame to the release directory without silent overwrite.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared instrument frame.
    filename : str
        Target filename (e.g. 'SMPS_bedroom2_number.csv').
    today : str
        ISO date string (YYYY-MM-DD) used to version a name collision. Passed in
        rather than read from the clock so a full release run stamps one date.

    Returns
    -------
    Path
        The path actually written.
    """
    out_dir = release_dir()
    target = out_dir / filename
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        target = out_dir / f"{stem}_{today}{suffix}"

    df = order_columns(df)
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return target
