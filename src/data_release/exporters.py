"""
Per-instrument exporters for the data release.

Each ``prepare_<instrument>`` function reads one instrument's raw record through
the project data resolver, reshapes it to the shared tidy schema (datetime,
burn_id, location, instrument, serial_number, then measurement columns with
units in the name), and returns a DataFrame. No artifact corrections are applied
and timestamps keep the instrument's native clock.

Exporters reuse existing repository parsers where they exist:
    - SMPS: scripts.export_smps_total_concentration.read_transposed_smps_file
    - AeroTrak Bedroom 2 foreign-row removal: src.aerotrak_coincidence logic
    - MODULAIR-PM 5 s loader: src.modulair_5sec_io.load_5sec_burn

Author: Nathan Lima
Created: 2026-08-06
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_paths import resolver, get_instrument_path  # noqa: E402
from src.data_release import release_config as rc  # noqa: E402
from src.data_release.burns import tag_burn_id  # noqa: E402


def _attach_common(df: pd.DataFrame, product: dict, datetime_col: str) -> pd.DataFrame:
    """
    Add the shared leading columns to a prepared frame and rename the datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Frame that already holds a datetime column named ``datetime_col`` and
        the instrument's measurement columns.
    product : dict
        Registry entry for this product.
    datetime_col : str
        Name of the existing datetime column; renamed to 'datetime'.

    Returns
    -------
    pd.DataFrame
        Frame with datetime, burn_id, location, instrument, serial_number added.
    """
    df = df.copy()
    if datetime_col != "datetime":
        df = df.rename(columns={datetime_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df["burn_id"] = tag_burn_id(df, "datetime")
    df["location"] = product["location"]
    df["instrument"] = product["instrument"]
    df["serial_number"] = product["serial_number"]
    return df


# ==============================================================================
# SMPS
# ==============================================================================

def prepare_smps(product: dict) -> pd.DataFrame:
    """
    Build the SMPS number-concentration release frame (Bedroom 2, all burns).

    Reuses the transposed-file parser from the SMPS export utility. Each burn day
    has its own file; they are concatenated, de-duplicated on datetime, and the
    diameter-midpoint bins are ordered by size. Units are number concentration
    (#/cm3).

    Parameters
    ----------
    product : dict
        Registry entry (kind 'smps').

    Returns
    -------
    pd.DataFrame
        datetime, leading columns, Lower/Upper Size(nm), sorted midpoint bins
        (#/cm3), summary statistics, and Total Concentration (#/cm3).
    """
    from scripts.export_smps_total_concentration import read_transposed_smps_file

    conc_type = "NumConc"
    size_units = "nm (#/cm³)"
    smps_dir = get_instrument_path(product["config_key"])
    files = sorted(smps_dir.glob(f"MH_apollo_bed_*_{conc_type}.xlsx"))
    if not files:
        raise FileNotFoundError(f"No SMPS {conc_type} files in {smps_dir}")

    frames = []
    units = None
    for fp in files:
        part = read_transposed_smps_file(fp, conc_type)
        if len(part):
            frames.append(part)
            if units is None:
                units = part["_units"].iloc[0]

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("datetime").drop_duplicates(
        subset="datetime", keep="first"
    )
    combined = combined.drop(columns=["_units"])

    # Sort diameter-midpoint columns by size. A stray non-numeric diameter
    # label in some files parses to NaN and yields a spurious "nan nm" column;
    # drop it so only real size bins are released.
    nan_bin = f"nan {size_units}"
    if nan_bin in combined.columns:
        combined = combined.drop(columns=[nan_bin])
    midpoints = []
    for col in combined.columns:
        if col.endswith(" " + size_units):
            try:
                val = float(col.replace(" " + size_units, ""))
            except ValueError:
                continue
            if not np.isnan(val):
                midpoints.append((val, col))
    midpoint_cols = [c for _, c in sorted(midpoints)]

    meta_cols = [c for c in ["Lower Size(nm)", "Upper Size(nm)"] if c in combined.columns]
    stat_cols = [
        c for c in [
            "D50(nm)", "Median(nm)", "Mean(nm)", "Geo. Mean(nm)",
            "Mode(nm)", "Geo. Std. Dev.",
        ] if c in combined.columns
    ]
    if units:
        combined = combined.rename(
            columns={"Total Concentration": f"Total Concentration ({units})"}
        )
        total_col = f"Total Concentration ({units})"
    else:
        total_col = "Total Concentration"

    ordered = ["datetime"] + meta_cols + midpoint_cols + stat_cols + [total_col]
    combined = combined[ordered]
    return _attach_common(combined, product, "datetime")


# ==============================================================================
# DUSTTRAK
# ==============================================================================

def prepare_dusttrak(product: dict) -> pd.DataFrame:
    """
    Build the DustTrak release frame (all five channels, all burns).

    Reads the combined all_data.xlsx, converts each channel from mg/m3 to ug/m3,
    and keeps the native 1 min clock. All five channels are released; the README
    documents that only TOTAL is usable because of a size-fraction configuration
    error and that the photometer reads high against carbonaceous smoke.

    Parameters
    ----------
    product : dict
        Registry entry (kind 'dusttrak').

    Returns
    -------
    pd.DataFrame
        datetime, leading columns, and PM1/PM2.5/PM4/PM10/TOTAL (ug/m3).
    """
    dt_dir = get_instrument_path(product["config_key"])
    fp = dt_dir / "all_data.xlsx"
    if not fp.exists():
        raise FileNotFoundError(f"DustTrak combined file not found: {fp}")

    raw = pd.read_excel(fp)
    raw.columns = raw.columns.str.strip()

    channel_map = {
        "PM1 [mg/m3]": "PM1 (µg/m³)",
        "PM2.5 [mg/m3]": "PM2.5 (µg/m³)",
        "PM4 [mg/m3]": "PM4 (µg/m³)",
        "PM10 [mg/m3]": "PM10 (µg/m³)",
        "TOTAL [mg/m3]": "TOTAL (µg/m³)",
    }
    out = pd.DataFrame({"datetime": pd.to_datetime(raw["datetime"], errors="coerce")})
    for src, dst in channel_map.items():
        if src in raw.columns:
            out[dst] = pd.to_numeric(raw[src], errors="coerce") * 1000.0  # mg->ug

    return _attach_common(out, product, "datetime")


# ==============================================================================
# AEROTRAK
# ==============================================================================

def prepare_aerotrak(product: dict) -> pd.DataFrame:
    """
    Build an AeroTrak count-concentration release frame for one location.

    Reads all_data.xlsx, computes per-channel count concentration (#/cm3) from
    the differential counts and sample volume, and keeps the native 1 min clock.
    For Bedroom 2 (AeroTrak1) the Morning Room rows that were merged into the
    export are removed first, reusing the exact fingerprint rule from
    src.aerotrak_coincidence (timestamp plus all six differential counts must
    match the Morning Room file). Size cut-points are taken from the file and
    written into the column names.

    Parameters
    ----------
    product : dict
        Registry entry (kind 'aerotrak'); 'aerotrak_label' is 'AeroTrak1'
        (Bedroom 2) or 'AeroTrak2' (Morning Room).

    Returns
    -------
    pd.DataFrame
        datetime, leading columns, and per-channel count concentration columns
        named by size range, e.g. 'N 0.3-0.5um (#/cm³)'.
    """
    from src.aerotrak_coincidence import (
        CHANNELS,
        _drop_foreign_rows,
    )

    label = product["aerotrak_label"]
    path = get_instrument_path(product["config_key"]) / "all_data.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"AeroTrak file not found: {path}")

    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df["Date and Time"] = pd.to_datetime(df["Date and Time"], errors="coerce")

    # Remove Morning Room rows merged into the Bedroom 2 export (AeroTrak1 only),
    # before any time shift, while raw timestamps still line up between files.
    if label == "AeroTrak1":
        df = _drop_foreign_rows(df, "AeroTrak2")

    # Status filter to the instrument's own valid samples.
    if "Flow Status" in df.columns and "Laser Status" in df.columns:
        df = df[(df["Flow Status"] == "OK") & (df["Laser Status"] == "OK")].copy()
    df = df.reset_index(drop=True)

    vol_cm3 = pd.to_numeric(df["Volume (L)"], errors="coerce") * 1000.0

    # Channel cut-point sizes from the first valid row.
    size_val = {}
    for ch, _lo, _hi in CHANNELS:
        col = f"{ch} Size (µm)"
        if col in df.columns:
            v = df[col].dropna()
            if not v.empty:
                size_val[ch] = float(v.iloc[0])

    out = pd.DataFrame({"datetime": df["Date and Time"]})
    for i, (ch, _lo, _hi) in enumerate(CHANNELS):
        if ch not in size_val:
            continue
        lo_um = size_val[ch]
        next_ch = CHANNELS[i + 1][0] if i < len(CHANNELS) - 1 else None
        hi_um = float(size_val[next_ch]) if next_ch and next_ch in size_val else 25.0
        diff_col = f"{ch} Diff (#)"
        if diff_col not in df.columns:
            continue
        counts = pd.to_numeric(df[diff_col], errors="coerce")
        name = f"N {lo_um:g}-{hi_um:g}um (#/cm³)"
        out[name] = counts / vol_cm3

    return _attach_common(out, product, "datetime")


# ==============================================================================
# MODULAIR-PM PORTAL (1-MINUTE QA/QC PRODUCT)
# ==============================================================================

# Portal columns released, in order: mass, OPC-N3 bins, nephelometer bins, flag.
_PORTAL_KEEP = (
    ["pm1", "pm25", "pm10", "opcn3_pm1", "opcn3_pm25", "opcn3_pm10"]
    + [f"bin{i}" for i in range(24)]
    + [f"neph_bin{i}" for i in range(6)]
    + ["sample_rh", "sample_temp", "sample_pres", "flag"]
)

_PORTAL_RENAME = {
    "pm1": "PM1 (µg/m³)",
    "pm25": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "opcn3_pm1": "OPC-N3 PM1 (µg/m³)",
    "opcn3_pm25": "OPC-N3 PM2.5 (µg/m³)",
    "opcn3_pm10": "OPC-N3 PM10 (µg/m³)",
    "sample_rh": "sample_rh (%)",
    "sample_temp": "sample_temp (°C)",
    "sample_pres": "sample_pres (hPa)",
}


def prepare_modulair_portal(product: dict) -> pd.DataFrame:
    """
    Build a MODULAIR-PM portal (1 min QA/QC) release frame for one unit.

    Reads the cloud-portal CSV, keeps the native local timestamp
    (timestamp_local), and releases mass, OPC-N3 bin counts, PMS5003
    nephelometer bins, environment channels, and the QA/QC flag as recorded. No
    30 s centering or clock shift is applied: released timestamps are the portal
    label. The README documents the trailing-minute label convention and the
    QA/QC peak-removal artifact.

    Parameters
    ----------
    product : dict
        Registry entry (kind 'modulair_portal').

    Returns
    -------
    pd.DataFrame
        datetime, leading columns, and the kept portal channels.
    """
    portal_dir = get_instrument_path(product["config_key"])
    matches = sorted(portal_dir.glob(f"{product['serial_number']}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No portal CSV for {product['serial_number']} in {portal_dir}"
        )
    df = pd.read_csv(matches[0], low_memory=False)

    ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp"
    ts = pd.to_datetime(
        df[ts_col].astype(str).str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    )
    out = pd.DataFrame({"datetime": ts})
    for col in _PORTAL_KEEP:
        if col in df.columns:
            out[col] = df[col]
    out = out.rename(columns=_PORTAL_RENAME)

    # The MODULAIR-PM units were installed in the test house only after burn3;
    # earlier portal rows are from a different location and are excluded.
    cutoff = pd.Timestamp(rc.MODULAIR_DEPLOY_START)
    out = out[pd.to_datetime(out["datetime"], errors="coerce") >= cutoff]

    return _attach_common(out, product, "datetime")


# ==============================================================================
# MODULAIR-PM RAW 5 SECOND RECORD
# ==============================================================================

def prepare_modulair_5s(product: dict) -> pd.DataFrame:
    """
    Build a MODULAIR-PM raw 5 s (no QA/QC) release frame for one unit.

    Concatenates the raw SD-card record across the burn days for which the unit
    was deployed, using the loader from src.modulair_5sec_io (3-row device
    header, timestamp_iso in UTC). Released timestamps keep the raw record's UTC
    clock converted to local EDT (fixed -4 h), with no per-unit clock-correction
    shift, so the release carries the instrument's own time base. OPC-N3 bins,
    nephelometer bins, and the flag are released as recorded.

    Parameters
    ----------
    product : dict
        Registry entry (kind 'modulair_5s'); 'unit' selects the MODULAIR unit.

    Returns
    -------
    pd.DataFrame
        datetime, leading columns, OPC-N3 bin0-23, neph_bin0-5, flag.
    """
    from src.modulair_5sec_io import (
        BURN_DATES,
        NEPH_BINS,
        OPC_BINS,
        UNIT_CONFIG,
        get_unit_5sec_path,
    )

    unit = product["unit"]
    # Reload each burn day's raw file directly so we can strip the per-unit
    # clock-correction shift that load_5sec_burn applies (release keeps native).
    frames = []
    src_dir = get_unit_5sec_path(unit)
    for burn_id in BURN_DATES:
        date_str = pd.Timestamp(BURN_DATES[burn_id]).strftime("%Y%m%d")
        fpath = src_dir / f"DATA_{date_str}.csv"
        if not fpath.exists():
            continue
        try:
            raw = pd.read_csv(fpath, skiprows=3, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"    [5s] cannot read {fpath.name}: {exc}")
            continue
        ts = pd.to_datetime(
            raw["timestamp_iso"].astype(str).str.replace("T", " ").str.replace("Z", ""),
            errors="coerce",
        ) + pd.Timedelta(hours=-4)  # UTC -> local EDT, native (no clock shift)
        part = pd.DataFrame({"datetime": ts})
        for col in OPC_BINS + NEPH_BINS + ["flag"]:
            if col in raw.columns:
                part[col] = pd.to_numeric(raw[col], errors="coerce")
        frames.append(part)

    if not frames:
        raise FileNotFoundError(f"No raw 5 s files found for {unit}")

    out = pd.concat(frames, ignore_index=True)
    return _attach_common(out, product, "datetime")


# ==============================================================================
# PURPLEAIR
# ==============================================================================

def prepare_purpleair(product: dict) -> pd.DataFrame:
    """
    Build the PurpleAir PM2.5 release frame (Morning Room, burns 6-10).

    Parses the per-day raw-pm25-gm CSVs for the Morning Room sensor (PA2). Each
    file holds a DateTime column, an 'Average' summary, and the two channel
    columns 'PA2-30::3D A' and 'PA2-30::3D B' (PM2.5, ug/m3). The 'Average'
    column is only populated on the first row of each file, so the released
    average is recomputed as the mean of the two channels. Native 2 min clock.

    Parameters
    ----------
    product : dict
        Registry entry (kind 'purpleair').

    Returns
    -------
    pd.DataFrame
        datetime, leading columns, and PM2.5 A/B/Average (ug/m3).
    """
    pa_dir = get_instrument_path(product["config_key"])
    day_dirs = sorted(d for d in pa_dir.iterdir() if d.is_dir())

    frames = []
    for day in day_dirs:
        matches = sorted(day.glob("*_PA2_raw-pm25-gm.csv"))
        if not matches:
            continue
        raw = pd.read_csv(matches[0])
        a_col = next((c for c in raw.columns if c.endswith(" A")), None)
        b_col = next((c for c in raw.columns if c.endswith(" B")), None)
        if a_col is None or b_col is None:
            continue
        part = pd.DataFrame({
            "datetime": pd.to_datetime(raw["DateTime"], errors="coerce"),
        })
        part["PM2.5 A (µg/m³)"] = pd.to_numeric(raw[a_col], errors="coerce")
        part["PM2.5 B (µg/m³)"] = pd.to_numeric(raw[b_col], errors="coerce")
        part["PM2.5 Average (µg/m³)"] = part[
            ["PM2.5 A (µg/m³)", "PM2.5 B (µg/m³)"]
        ].mean(axis=1)
        frames.append(part)

    if not frames:
        raise FileNotFoundError(f"No PurpleAir PA2 raw-pm25 files under {pa_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="datetime", keep="first")
    return _attach_common(out, product, "datetime")
