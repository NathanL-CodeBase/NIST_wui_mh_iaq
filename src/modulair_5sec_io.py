"""
MODULAIR-PM 5 s and portal data I/O for the peak-window analysis.

Loads the raw on-instrument 5 s SD-card record (no QA/QC) and the
QA/QC-filtered 1-minute portal product for the two indoor QuantAQ MODULAIR-PM
units deployed in the WUI manufactured-home smoke experiments. All paths
resolve through data_config.json; no raw data are written or committed.

The on-instrument record logs one row every 5 seconds (0.2 Hz), not 5 Hz; the
"5 s" naming throughout reflects that actual cadence.

Raw 5 s channels exposed:
    - OPC-N3 bins bin0..bin23 (0.35 to 40 um), raw counts per 5 s window
    - PMS5003 nephelometer bins neph_bin0..neph_bin5 (size thresholds
      > 0.3, > 0.5, > 1.0, > 2.5, > 5.0, > 10 um), raw signal values
    - flag (bitmask, no QA/QC removal applied)

Timestamps in both products are stored as UTC (ISO 8601 with 'Z'). The
experiments were run in EDT (UTC-4); a fixed -4 h offset converts to local
time, then a per-unit clock-correction shift is applied (bedroom -2.97 min,
kitchen 0 min), matching testing/quantaq_5sec_timeseries.py.

Author: Nathan Lima
Created: 2026-06-24
Updated: 2026-06-25 (renamed from modulair_5hz_io; data is 5 s cadence)
"""

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.data_paths import resolver  # noqa: E402

# ==============================================================================
# CONSTANTS
# ==============================================================================

UTC_OFFSET_HRS = -4  # sensor timestamps are UTC; experiments were EDT (UTC-4)

# OPC-N3 raw count bins and PMS5003 nephelometer raw-signal bins.
OPC_BINS = [f"bin{i}" for i in range(24)]          # bin0..bin23
NEPH_BINS = [f"neph_bin{i}" for i in range(6)]     # neph_bin0..neph_bin5

# OPC-N3 bin lower/upper edges (um). QuantAQ/Alphasense OPC-N3 24-bin layout.
# Used only for the bin-response grid x-axis (lower edge) and for the
# "three smallest bins" size labels in the manuscript sentences.
OPC_BIN_EDGES_UM = [
    (0.35, 0.46), (0.46, 0.66), (0.66, 1.0), (1.0, 1.3), (1.3, 1.7),
    (1.7, 2.3), (2.3, 3.0), (3.0, 4.0), (4.0, 5.2), (5.2, 6.5),
    (6.5, 8.0), (8.0, 10.0), (10.0, 12.0), (12.0, 14.0), (14.0, 16.0),
    (16.0, 18.0), (18.0, 20.0), (20.0, 22.0), (22.0, 25.0), (25.0, 28.0),
    (28.0, 31.0), (31.0, 34.0), (34.0, 37.0), (37.0, 40.0),
]

# PMS5003 nephelometer size thresholds (lower edge, um) per the prompt.
NEPH_THRESHOLDS_UM = [0.3, 0.5, 1.0, 2.5, 5.0, 10.0]

# Per-unit configuration. "unit" is the analysis label used in outputs.
UNIT_CONFIG = {
    "MODULAIR-PM1": {
        "config_key": "quantaq_bedroom",
        "device_sn": "MOD-PM-00194",
        "location": "bedroom2",
        "location_label": "Bedroom 2",
        "time_shift_min": -2.97,
    },
    "MODULAIR-PM2": {
        "config_key": "quantaq_kitchen",
        "device_sn": "MOD-PM-00197",
        "location": "morning_room",
        "location_label": "Morning Room",
        "time_shift_min": 0.0,
    },
}

# Burns with 5 s QuantAQ deployment and their dates (YYYYMMDD for filenames).
BURN_DATES = {
    "burn4": "2024-05-09",
    "burn5": "2024-05-13",
    "burn6": "2024-05-17",
    "burn7": "2024-05-20",
    "burn8": "2024-05-23",
    "burn9": "2024-05-28",
    "burn10": "2024-05-31",
}

# Burns where Bedroom 2 was sealed: exclude from between-location comparisons,
# but keep for per-burn instrument-behavior analysis.
BEDROOM_SEALED_BURNS = {"burn5", "burn6"}


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

def _instr_entry(config_key: str) -> dict:
    """Return the data_config.json instruments[config_key] dict."""
    instr = resolver.config.get("instruments", {}).get(config_key)
    if instr is None:
        raise KeyError(f"Instrument '{config_key}' not in data_config.json.")
    return instr


def get_unit_5sec_path(unit: str) -> Path:
    """Resolve the 5 s SD-card directory for a MODULAIR-PM analysis unit."""
    cfg = UNIT_CONFIG[unit]
    raw = _instr_entry(cfg["config_key"]).get("path_5sec")
    if raw is None:
        raise KeyError(
            f"No 'path_5sec' key for '{cfg['config_key']}' in data_config.json."
        )
    return Path(raw)


def get_unit_portal_file(unit: str) -> Path | None:
    """Resolve the 1-minute portal CSV for a unit (matches device SN glob)."""
    cfg = UNIT_CONFIG[unit]
    portal_dir = Path(_instr_entry(cfg["config_key"])["path"])
    if not portal_dir.exists():
        return None
    matches = sorted(portal_dir.glob(f"{cfg['device_sn']}*.csv"))
    return matches[0] if matches else None


# ==============================================================================
# 5 SECOND LOADER
# ==============================================================================

def _to_local(ts_iso: pd.Series, time_shift_min: float) -> pd.Series:
    """Parse a UTC ISO timestamp series and convert to local EDT + clock shift."""
    ts = pd.to_datetime(
        ts_iso.astype(str).str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    ).dt.tz_localize(None) + pd.Timedelta(hours=UTC_OFFSET_HRS)
    if time_shift_min:
        ts = ts + pd.Timedelta(minutes=time_shift_min)
    return ts


def load_5sec_burn(unit: str, burn_id: str) -> pd.DataFrame | None:
    """
    Load the raw 5 s record for one unit and burn day, in local time.

    The CSV has a 3-row device-header block (deviceModel, deviceID, deviceSN)
    before the column-header row, so skiprows=3 is required. No QA/QC filtering
    is applied: all rows (including flagged rows) are retained.

    Parameters
    ----------
    unit : str
        'MODULAIR-PM1' (Bedroom 2) or 'MODULAIR-PM2' (Morning Room).
    burn_id : str
        e.g. 'burn6'.

    Returns
    -------
    pd.DataFrame or None
        Columns include 'timestamp' (local), all OPC_BINS, all NEPH_BINS,
        and 'flag'. Returns None if the file is missing or unreadable.
    """
    cfg = UNIT_CONFIG[unit]
    date_str = pd.Timestamp(BURN_DATES[burn_id]).strftime("%Y%m%d")
    fpath = get_unit_5sec_path(unit) / f"DATA_{date_str}.csv"
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath, skiprows=3, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"    [5sec] cannot read {fpath.name}: {exc}")
        return None

    df["timestamp"] = _to_local(df["timestamp_iso"], cfg["time_shift_min"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Coerce all channel columns to numeric (raw files are clean floats, but
    # guard against stray strings without dropping any rows).
    for col in OPC_BINS + NEPH_BINS + ["flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ==============================================================================
# PORTAL (1-MINUTE QA/QC) LOADER
# ==============================================================================

def load_portal_burn(unit: str, burn_id: str) -> pd.DataFrame | None:
    """
    Load the QA/QC-filtered 1-minute portal product for one unit and burn day.

    Parameters
    ----------
    unit : str
    burn_id : str

    Returns
    -------
    pd.DataFrame or None
        Columns include 'timestamp' (local), 'pm1', 'pm25', 'pm10',
        'neph_bin0', 'bin0', and 'flag'. Filtered to the burn day. Returns
        None if no portal file is found.
    """
    cfg = UNIT_CONFIG[unit]
    fpath = get_unit_portal_file(unit)
    if fpath is None:
        return None
    df = pd.read_csv(fpath, low_memory=False)

    # Portal stores both UTC ('timestamp') and local ('timestamp_local').
    ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(
        df[ts_col].astype(str).str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    ).dt.tz_localize(None)
    # timestamp_local is already EDT wall-clock; apply only the clock shift so
    # it aligns with the 5 s local axis.
    if cfg["time_shift_min"] and ts_col == "timestamp_local":
        df["timestamp"] = df["timestamp"] + pd.Timedelta(minutes=cfg["time_shift_min"])
    elif ts_col == "timestamp":  # UTC fallback
        df["timestamp"] = (
            df["timestamp"]
            + pd.Timedelta(hours=UTC_OFFSET_HRS)
            + pd.Timedelta(minutes=cfg["time_shift_min"])
        )

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    burn_date = pd.Timestamp(BURN_DATES[burn_id]).date()
    df = df[df["timestamp"].dt.date == burn_date].reset_index(drop=True)
    return df if not df.empty else None


# ==============================================================================
# BURN-LOG EVENT TIMES
# ==============================================================================

def load_event_times() -> pd.DataFrame:
    """
    Load burn-log event timestamps as full local pd.Timestamp columns.

    Returns
    -------
    pd.DataFrame
        Indexed by 'Burn ID'; columns 'Date', 'ignition', 'garage_closed',
        'pac_on' (CR Box / portable air cleaner activation). Times are local
        EDT wall-clock.
    """
    bl = pd.read_excel(resolver.get_common_file("burn_log"), sheet_name="Sheet2")
    bl["Date"] = pd.to_datetime(bl["Date"])

    col_map = {
        "ignition": "Ignition",
        "garage_closed": "garage closed",
        "pac_on": "CR Box on",
    }
    out = pd.DataFrame({"Burn ID": bl["Burn ID"], "Date": bl["Date"]})
    for new, src in col_map.items():
        if src not in bl.columns:
            out[new] = pd.NaT
            continue
        out[new] = bl.apply(
            lambda r, c=src: (
                pd.Timestamp(f"{r['Date'].strftime('%Y-%m-%d')} {bl.loc[r.name, c]}")
                if pd.notna(bl.loc[r.name, c])
                and str(bl.loc[r.name, c]).strip().lower() not in ("no", "nan")
                else pd.NaT
            ),
            axis=1,
        )
    return out.set_index("Burn ID")
