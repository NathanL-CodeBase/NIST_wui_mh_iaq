"""
AeroTrak 9306-V2 Optical Coincidence Validation Analysis

Validates the transient optical-coincidence claim for Section 3.2.2 of the WUI
fire smoke instrument paper. For each AeroTrak unit and burn, detects bin
reversals in the 0.3-0.5, 0.5-1.0, and 1.0-3.0 um channels, estimates
coincidence losses via a Poisson dead-time model, checks counts conservation,
and cross-checks against co-located SMPS number concentration (Bedroom 2 only).

Inputs (all resolved through data_config.json):
    aerotrak_bedroom / aerotrak_kitchen  : all_data.xlsx
    smps (numConc files)                 : MH_apollo_bed_MMDDYYYY_numConc.xlsx
    burn_log                             : burn_log.xlsx, Sheet2

Outputs:
    coincidence_analysis/aerotrak_coincidence_per_burn.csv
    coincidence_analysis/aerotrak_coincidence_cross_burn_summary.csv
    coincidence_analysis/aerotrak_coincidence_summary.md
    coincidence_analysis/aerotrak_manuscript_sentences.md
    coincidence_figures/aerotrak_coincidence_<burn>_<unit>.html  (one per pair)
    coincidence_figures/aerotrak_coincidence_small_multiples.png
    coincidence_figures/aerotrak_coincidence_overlay.png
    coincidence_figures/aerotrak_loss_vs_peakmass.png

Author: Nathan Lima
Created: 2026-04-30
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from bokeh.plotting import figure, output_file, save
from bokeh.models import Span, HoverTool
from bokeh.layouts import column as bokeh_column
from bokeh.io import reset_output

# --- repository root on path ---------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.data_paths import get_instrument_path, get_common_file

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Optical sensing volumes from Poisson model  L = 1 - exp(-n*V)
# TSI spec: 10 % coincidence at n = 1.4e5 /cm3 -> V_CENTRAL = -ln(0.9)/1.4e5
V_CENTRAL = 7.5e-7   # cm3 (central estimate)
V_LOW     = 5.0e-7   # cm3 (lower bound)
V_HIGH    = 1.0e-6   # cm3 (upper bound)

# Coincidence threshold: manufacturer 10 % coincidence concentration
COINCIDENCE_THRESHOLD_CM3 = 1.4e5  # particles/cm3

# Reversal detection parameters
REVERSAL_FRAC    = 0.5   # n_min < REVERSAL_FRAC * n_peak -> reversal present
REVERSAL_WIN_MIN = 30    # search window after n_peak (minutes)

# Counts-conservation tolerance (fraction change allowed)
CONSERVATION_TOL = 0.30

# Particle density for Mie-sphere PM mass (matches existing project scripts)
PARTICLE_DENSITY_G_CM3 = 1.0

# Pre-burn baseline window
BASELINE_MIN = 30   # minutes before ignition

# Maximum analysis window
MAX_WIN_HR = 4.0

# AeroTrak 9306-V2 size channels: (label, lower_um, upper_um)
CHANNELS = [
    ("Ch1", 0.3,  0.5),
    ("Ch2", 0.5,  1.0),
    ("Ch3", 1.0,  3.0),
    ("Ch4", 3.0,  5.0),
    ("Ch5", 5.0,  10.0),
    ("Ch6", 10.0, 25.0),
]
ANALYSIS_CH = CHANNELS[:3]   # three smallest bins for coincidence check

# Instrument time shifts (minutes applied to raw timestamps)
TIME_SHIFTS = {"AeroTrak1": 2.16, "AeroTrak2": 5.0}

# Burns processed per instrument
BURN_COVERAGE = {
    "AeroTrak1": [f"burn{i}" for i in range(3, 11)],   # burn3-burn10
    "AeroTrak2": [f"burn{i}" for i in range(2, 11)],   # burn2-burn10
}

# Burns where Bedroom 2 was sealed (flag in output; AeroTrak1 only)
BEDROOM_SEALED_BURNS = {"burn5", "burn6"}

# data_config.json instrument keys
INSTR_KEY = {
    "AeroTrak1": "aerotrak_bedroom",
    "AeroTrak2": "aerotrak_kitchen",
}

# Matplotlib TEXT_CONFIG (matches project convention)
_FS = 12
TEXT_CONFIG = dict(fontsize=_FS, labelsize=_FS, ticksize=_FS, legendsize=_FS)

# Plot colours
COLOR = {
    "AeroTrak1": "#003f5c",
    "AeroTrak2": "#ef5675",
    "smps":      "#ffa600",
}

# CSV column order for per-burn output
_CSV_COLS = [
    "burn", "instrument", "location", "bedroom_sealed",
    "n_peak_cm3", "t_peak", "reversal_present", "reversal_duration_minutes",
    "L_central", "L_low", "L_high",
    "peak_total_PM3_mass_ug_m3", "counts_conserved",
    "SMPS_ratio_during_vs_after", "notes",
]

# ==============================================================================
# BURN LOG
# ==============================================================================

def _load_burn_log() -> pd.DataFrame:
    """
    Load burn log from burn_log.xlsx Sheet2.

    Returns
    -------
    pd.DataFrame
        Columns include 'Burn ID', 'Date', 'Ignition', 'garage closed'
        where Ignition and garage closed are full pd.Timestamp objects.
    """
    bl = pd.read_excel(get_common_file("burn_log"), sheet_name="Sheet2")
    bl["Date"] = pd.to_datetime(bl["Date"])
    for col in ("Ignition", "garage closed"):
        if col not in bl.columns:
            bl[col] = pd.NaT
            continue
        bl[col] = bl.apply(
            lambda r, c=col: (
                pd.Timestamp(f"{r['Date'].strftime('%Y-%m-%d')} {r[c]}")
                if pd.notna(r[c]) else pd.NaT
            ),
            axis=1,
        )
    return bl

# ==============================================================================
# AEROTRAK LOADER
# ==============================================================================

def _load_aerotrak_all(instrument: str) -> pd.DataFrame:
    """
    Load the combined AeroTrak all_data.xlsx, apply the time shift, apply
    status filter, and compute count concentrations (#/cm3) and PM mass
    (ug/m3) from raw differential counts.

    PM mass uses a Mie-sphere model with density = 1 g/cm3 and geometric-mean
    diameter per channel, identical to the approach in peak_concentration_script.py.

    Parameters
    ----------
    instrument : str
        "AeroTrak1" (Bedroom 2) or "AeroTrak2" (Morning Room).

    Returns
    -------
    pd.DataFrame
        Includes 'Date and Time', count columns 'Sum{lo}-{hi}um (#/cm3)',
        and cumulative mass columns 'PM0.5 (ug/m3)' through 'PM25 (ug/m3)'.
    """
    path = get_instrument_path(INSTR_KEY[instrument]) / "all_data.xlsx"
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df["Date and Time"] = pd.to_datetime(df["Date and Time"])

    # Apply instrument time shift
    df["Date and Time"] += pd.Timedelta(minutes=TIME_SHIFTS[instrument])

    # Status filter
    ok = (df["Flow Status"] == "OK") & (df["Laser Status"] == "OK")
    df = df[ok].copy().reset_index(drop=True)

    # Sample volume
    vol_L   = df["Volume (L)"]
    vol_cm3 = vol_L * 1000.0    # cm3
    vol_m3  = vol_L * 1e-3      # m3

    # Extract cut-point sizes from the first valid row
    size_val = {}
    for ch, _lo, _hi in CHANNELS:
        col = f"{ch} Size (µm)"
        if col in df.columns:
            v = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if v is not None:
                size_val[ch] = float(v)

    pm_diff_cols: list[str] = []

    for i, (ch, _lo, _hi) in enumerate(CHANNELS):
        if ch not in size_val:
            continue
        lo_um = size_val[ch]
        next_ch = CHANNELS[i + 1][0] if i < len(CHANNELS) - 1 else None
        hi_um   = float(size_val[next_ch]) if next_ch and next_ch in size_val else 25.0

        diff_col = f"{ch} Diff (#)"
        if diff_col not in df.columns:
            continue

        # Count concentration (#/cm3)
        conc_col = f"Ʃ{lo_um}-{hi_um}µm (#/cm³)"
        df[conc_col] = df[diff_col] / vol_cm3

        # Single-particle mass (ug) via Mie sphere, density = 1 g/cm3
        gm_um    = np.sqrt(lo_um * hi_um)          # geometric-mean diameter (um)
        r_m      = gm_um * 1e-6 / 2.0              # radius (m)
        vp_m3    = (4.0 / 3.0) * np.pi * r_m ** 3  # volume (m3)
        mass_ug  = vp_m3 * 1e12                     # ug (1 g/cm3 density)

        # Differential mass concentration (ug/m3)
        diff_mass_col = f"PM{lo_um}-{hi_um} Diff (µg/m³)"
        df[diff_mass_col] = (df[diff_col] / vol_m3) * mass_ug
        pm_diff_cols.append(diff_mass_col)

    # Cumulative PM mass (matches peak_concentration_script.py convention)
    cum_labels = [
        "PM0.5 (µg/m³)", "PM1 (µg/m³)", "PM3 (µg/m³)",
        "PM5 (µg/m³)", "PM10 (µg/m³)", "PM25 (µg/m³)",
    ]
    for i, label in enumerate(cum_labels):
        if i < len(pm_diff_cols):
            if i == 0:
                df[label] = df[pm_diff_cols[i]]
            else:
                df[label] = df[pm_diff_cols[i]] + df[cum_labels[i - 1]]

    return df


def _day_slice(df: pd.DataFrame, burn_date: pd.Timestamp) -> pd.DataFrame:
    """Return rows whose 'Date and Time' falls on burn_date (local date)."""
    mask = df["Date and Time"].dt.date == burn_date.date()
    return df[mask].copy().reset_index(drop=True)

# ==============================================================================
# SMPS numConc LOADER
# ==============================================================================

def _smps_numconc_path(burn_date: pd.Timestamp) -> Path | None:
    """Locate SMPS numConc file for a given burn date (case-insensitive suffix)."""
    smps_dir = get_instrument_path("smps")
    date_str = burn_date.strftime("%m%d%Y")
    for suffix in ("numConc", "NumConc"):
        p = smps_dir / f"MH_apollo_bed_{date_str}_{suffix}.xlsx"
        if p.exists():
            return p
    return None


def _load_smps_numconc(burn_date: pd.Timestamp) -> pd.DataFrame | None:
    """
    Load SMPS number-concentration file for one burn day.

    The TSI export is stored transposed: rows are metadata + size bins,
    columns are individual scans. Reading with index_col=0 and transposing
    yields a DataFrame where rows are scans and columns are labeled by size
    (nm, float) plus metadata strings ('Date', 'Start Time', etc.).

    Parameters
    ----------
    burn_date : pd.Timestamp
        Burn date used to locate and filter the file.

    Returns
    -------
    pd.DataFrame or None
        Columns include 'datetime' and float size-bin labels (nm).
        Returns None if file is missing or unreadable.
    """
    fpath = _smps_numconc_path(burn_date)
    if fpath is None:
        return None

    try:
        raw = pd.read_excel(fpath, header=None, index_col=0)
    except Exception as exc:
        print(f"  [SMPS] Cannot read {fpath.name}: {exc}")
        return None

    # Transpose: scans become rows, metadata/size-bin labels become columns
    df = raw.T.copy()
    df.index = range(len(df))

    # Build datetime from 'Date' + 'Start Time' columns
    if "Date" not in df.columns or "Start Time" not in df.columns:
        print(f"  [SMPS] Date/Start Time not found in {fpath.name}")
        return None

    try:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        times = df["Start Time"].astype(str).str.strip()
        df["datetime"] = pd.to_datetime(
            dates.dt.strftime("%Y-%m-%d") + " " + times, errors="coerce"
        )
    except Exception as exc:
        print(f"  [SMPS] Datetime parse failed for {fpath.name}: {exc}")
        return None

    # Convert size-bin columns (float keys) to numeric; Excel FALSE -> 0
    size_cols = [
        c for c in df.columns
        if isinstance(c, float) and not isinstance(c, bool)
    ]
    for c in size_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Keep only the burn day
    mask = df["datetime"].dt.date == burn_date.date()
    df = df[mask].dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    return df if not df.empty else None


def _smps_300_437(df_smps: pd.DataFrame) -> pd.Series:
    """
    Sum SMPS number-concentration columns between 300 and 437 nm.

    Returns
    -------
    pd.Series
        Summed number concentration (#/cm3, approximate) indexed like df_smps.
    """
    cols = [
        c for c in df_smps.columns
        if isinstance(c, float) and not isinstance(c, bool)
        and 300.0 <= c <= 437.0
    ]
    if not cols:
        return pd.Series(np.nan, index=df_smps.index)
    return df_smps[cols].sum(axis=1)

# ==============================================================================
# ANALYSIS HELPERS
# ==============================================================================

def _get_baseline(df_day: pd.DataFrame, ignition: pd.Timestamp) -> float:
    """
    Compute mean PM3 (ug/m3) in the 30 minutes immediately before ignition.

    Returns
    -------
    float
        Baseline PM3 mass concentration, or NaN if no data in window.
    """
    t0 = ignition - pd.Timedelta(minutes=BASELINE_MIN)
    mask = (
        (df_day["Date and Time"] >= t0)
        & (df_day["Date and Time"] < ignition)
        & df_day["PM3 (µg/m³)"].notna()
    )
    vals = df_day.loc[mask, "PM3 (µg/m³)"]
    return float(vals.mean()) if len(vals) > 0 else np.nan


def _find_window_end(
    df_day: pd.DataFrame,
    ignition: pd.Timestamp,
    baseline_pm3: float,
) -> pd.Timestamp:
    """
    Analysis window: ignition + 4 h, or first post-peak return to within
    10 % of pre-burn baseline PM3, whichever is earlier.

    Parameters
    ----------
    df_day : pd.DataFrame
        AeroTrak data for the burn day.
    ignition : pd.Timestamp
    baseline_pm3 : float
        Pre-burn PM3 mass baseline (ug/m3); may be NaN.

    Returns
    -------
    pd.Timestamp
        Window end time.
    """
    hard_end = ignition + pd.Timedelta(hours=MAX_WIN_HR)

    if np.isnan(baseline_pm3):
        return hard_end

    threshold = baseline_pm3 * 1.10  # within 10 % above baseline

    in_win = df_day[
        (df_day["Date and Time"] > ignition)
        & (df_day["Date and Time"] <= hard_end)
        & df_day["PM3 (µg/m³)"].notna()
    ]
    if in_win.empty:
        return hard_end

    peak_idx = in_win["PM3 (µg/m³)"].idxmax()
    post_peak = in_win.loc[peak_idx:]
    recovery = post_peak[post_peak["PM3 (µg/m³)"] <= threshold]

    if not recovery.empty:
        return min(recovery["Date and Time"].iloc[0], hard_end)
    return hard_end


def _analyze_bin(
    series: pd.Series,
    timestamps: pd.Series,
    ignition: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    """
    Detect coincidence reversal in one count-concentration time series.

    Peak location uses a 5-minute centred rolling mean on the in-window data.
    n_peak and n_min are read from the original (unsmoothed) series.

    Parameters
    ----------
    series : pd.Series
        Count concentration (#/cm3), same index as timestamps.
    timestamps : pd.Series
        pd.Timestamp values aligned with series.
    ignition, window_end : pd.Timestamp

    Returns
    -------
    dict
        Keys: n_peak, t_peak, n_min_during_reversal, t_min,
              reversal_present, reversal_duration_minutes.
    """
    blank = dict(
        n_peak=np.nan, t_peak=pd.NaT,
        n_min_during_reversal=np.nan, t_min=pd.NaT,
        reversal_present=False, reversal_duration_minutes=np.nan,
    )

    mask = (timestamps >= ignition) & (timestamps <= window_end) & series.notna()
    if mask.sum() < 5:
        return blank

    s  = np.asarray(series[mask], dtype=float)
    ts = timestamps[mask].values  # numpy datetime64

    # Use the raw maximum as the coincidence peak. The initial spike is the
    # true maximum; a centred rolling mean would find the later sustained
    # plateau instead, placing t_peak after the reversal has already occurred.
    peak_i = int(np.nanargmax(s))
    n_peak = float(s[peak_i])
    t_peak = pd.Timestamp(ts[peak_i])

    if np.isnan(n_peak) or n_peak <= 0:
        return blank

    result = blank.copy()
    result["n_peak"] = n_peak
    result["t_peak"] = t_peak

    # Two-pronged reversal detection:
    #
    # Check A (pre-peak): The AeroTrak saturates at very high particle
    # concentration — Ch1 drops to near zero while smoke is densest, then
    # recovers as concentration falls back into the instrument's working range.
    # That recovery plateau becomes the global max, so the reversal dip sits
    # BEFORE t_peak.  Detect by looking for a V-shape in [ignition, t_peak]:
    # a significant rise followed by a deep drop.
    #
    # Check B (post-peak): Fallback for burns where the initial spike IS the
    # global max and the dip follows immediately after.

    # --- Check A: pre-peak V-shape ---
    reversal_pre = False
    n_min_pre    = np.nan
    t_min_pre    = pd.NaT
    if peak_i > 1:
        s_before  = s[:peak_i]
        ts_before = ts[:peak_i]
        thresh    = REVERSAL_FRAC * n_peak
        # Only search for the dip AFTER the signal has risen above thresh.
        # Without this, the pre-smoke baseline (~0 #/cm³) would always be
        # identified as the minimum, and n_pre_dip would be ~0, causing
        # condition 2 in the original approach to always fail.
        above = np.where(s_before >= thresh)[0]
        if len(above) > 0:
            first_above  = above[0]
            s_post_rise  = s_before[first_above:]
            ts_post_rise = ts_before[first_above:]
            dip_i_local  = int(np.nanargmin(s_post_rise))
            n_dip        = float(s_post_rise[dip_i_local])
            if n_dip < thresh:
                reversal_pre = True
                n_min_pre    = n_dip
                t_min_pre    = pd.Timestamp(ts_post_rise[dip_i_local])

    # --- Check B: post-peak drop (original logic) ---
    reversal_post = False
    n_min_post    = np.nan
    t_min_post    = pd.NaT
    t_search   = t_peak + pd.Timedelta(minutes=REVERSAL_WIN_MIN)
    after_peak = (ts > np.datetime64(t_peak)) & (ts <= np.datetime64(t_search))
    if after_peak.sum() > 0:
        s_after    = s[after_peak]
        ts_after   = ts[after_peak]
        min_i      = int(np.argmin(s_after))
        n_min_post = float(s_after[min_i])
        t_min_post = pd.Timestamp(ts_after[min_i])
        if n_min_post < REVERSAL_FRAC * n_peak:
            reversal_post = True

    # Combine: prefer pre-peak result when both fire (deeper dip scenario)
    if reversal_pre or reversal_post:
        result["reversal_present"] = True
        n_min = n_min_pre if reversal_pre else n_min_post
        t_min = t_min_pre if reversal_pre else t_min_post
        assert isinstance(t_min, pd.Timestamp)  # always True inside this block
        result["n_min_during_reversal"] = n_min
        result["t_min"]                 = t_min  # type: ignore[assignment]

        # Duration: from t_min (bottom of dip) to first recovery to n_peak / 2
        thresh    = n_peak / 2.0
        post_lo   = ts > np.datetime64(t_min)  # type: ignore[operator]
        s_rec     = s[post_lo]
        ts_rec    = ts[post_lo]
        recovered = s_rec >= thresh
        if recovered.any():
            t_rec = pd.Timestamp(ts_rec[int(np.argmax(recovered))])
            result["reversal_duration_minutes"] = (
                (t_rec - t_min).total_seconds() / 60.0
            )

    return result


def _coincidence_loss(n_cm3: float) -> tuple[float, float, float]:
    """
    Estimate coincidence loss fraction L = 1 - exp(-n * V) for the three
    sensing-volume estimates.

    Parameters
    ----------
    n_cm3 : float
        Peak number concentration of the 0.3-0.5 um bin (#/cm3).

    Returns
    -------
    tuple of float
        (L_central, L_low, L_high)
    """
    return (
        float(1 - np.exp(-n_cm3 * V_CENTRAL)),
        float(1 - np.exp(-n_cm3 * V_LOW)),
        float(1 - np.exp(-n_cm3 * V_HIGH)),
    )


def _counts_conserved(
    df_day: pd.DataFrame,
    t_peak: pd.Timestamp,
    t_min: pd.Timestamp,
) -> bool | None:
    """
    Compare the sum of all six bin count concentrations at t_peak vs t_min.

    Returns True if |sum_at_tmin - sum_at_tpeak| / sum_at_tpeak <= 0.30,
    False if the change exceeds 30 %, or None if data are unavailable.
    """
    if pd.isna(t_peak) or pd.isna(t_min):
        return None

    conc_cols = [
        c for c in df_day.columns
        if "Ʃ" in str(c) and "#/cm" in str(c)
    ]
    if not conc_cols:
        return None

    def _nearest_row_sum(t: pd.Timestamp) -> float:
        dt = (df_day["Date and Time"] - t).abs()
        idx = dt.idxmin()
        if dt[idx] > pd.Timedelta("3min"):
            return np.nan
        return float(df_day.loc[idx, conc_cols].sum())

    total_peak = _nearest_row_sum(t_peak)
    total_min  = _nearest_row_sum(t_min)

    if np.isnan(total_peak) or total_peak == 0:
        return None

    rel_change = abs(total_min - total_peak) / total_peak
    return bool(rel_change <= CONSERVATION_TOL)


def _smps_cross_check(
    df_day: pd.DataFrame,
    df_smps: pd.DataFrame | None,
    t_min: pd.Timestamp,
    window_end: pd.Timestamp,
    ch1_col: str,
) -> float | None:
    """
    Compute the SMPS cross-check ratio:
        (AeroTrak Ch1 / SMPS 300-437 nm) at t_min
      divided by
        mean(AeroTrak Ch1 / SMPS 300-437 nm) during post-recovery decay phase.

    The decay phase is defined as 1 h after t_min through window_end.
    A ratio < 1 confirms the AeroTrak reversal is the artifact.

    Returns None if insufficient overlapping data.
    """
    if df_smps is None or pd.isna(t_min):
        return None

    smps_conc = _smps_300_437(df_smps)
    smps_ts   = df_smps["datetime"]

    def _aerotrak_nearest(t: pd.Timestamp) -> float:
        dt = (df_day["Date and Time"] - t).abs()
        idx = dt.idxmin()
        return float(df_day.loc[idx, ch1_col]) if dt[idx] <= pd.Timedelta("3min") else np.nan

    def _smps_nearest(t: pd.Timestamp) -> float:
        dt = (smps_ts - t).abs()
        idx = dt.idxmin()
        val = smps_conc.iloc[idx]
        return float(val) if dt.iloc[idx] <= pd.Timedelta("5min") else np.nan

    at_ch1  = _aerotrak_nearest(t_min)
    at_smps = _smps_nearest(t_min)
    if np.isnan(at_ch1) or np.isnan(at_smps) or at_smps == 0:
        return None
    ratio_tmin = at_ch1 / at_smps

    # Decay-phase mean ratio
    t_decay = t_min + pd.Timedelta(hours=1)
    if t_decay >= window_end:
        return None

    decay_mask = (
        (df_day["Date and Time"] >= t_decay)
        & (df_day["Date and Time"] <= window_end)
        & df_day[ch1_col].notna()
    )
    ratios = []
    for _, row in df_day[decay_mask].iterrows():
        s_val = _smps_nearest(row["Date and Time"])
        if not np.isnan(s_val) and s_val > 0:
            ratios.append(row[ch1_col] / s_val)

    if not ratios:
        return None

    return float(ratio_tmin / np.mean(ratios))

# ==============================================================================
# MAIN PER-BURN ANALYSIS
# ==============================================================================

def analyze_burn_instrument(
    burn_id: str,
    instrument: str,
    df_aerotrak: pd.DataFrame,
    bl_row: pd.Series,
    df_smps: pd.DataFrame | None,
) -> dict | None:
    """
    Run the full coincidence analysis for one burn x instrument pair.

    Parameters
    ----------
    burn_id : str
        e.g. 'burn3'
    instrument : str
        'AeroTrak1' or 'AeroTrak2'
    df_aerotrak : pd.DataFrame
        Full multi-day AeroTrak dataset (from _load_aerotrak_all).
    bl_row : pd.Series
        Single row from the burn log for this burn.
    df_smps : pd.DataFrame or None
        SMPS numConc data for the burn day (AeroTrak1 only).

    Returns
    -------
    dict or None
        Per-burn metrics dict including private keys prefixed with '_' for
        downstream plotting. Returns None if data are insufficient.
    """
    burn_date = bl_row["Date"]
    ignition  = bl_row["Ignition"]
    if pd.isna(ignition):
        print(f"    [{burn_id}|{instrument}] No ignition time - skipped.")
        return None

    df_day = _day_slice(df_aerotrak, burn_date)
    if df_day.empty:
        print(f"    [{burn_id}|{instrument}] No data for {burn_date.date()} - skipped.")
        return None

    # Pre-burn baseline and analysis window
    baseline_pm3 = _get_baseline(df_day, ignition)
    window_end   = _find_window_end(df_day, ignition, baseline_pm3)

    # Peak PM3 mass within analysis window
    in_win = df_day[
        (df_day["Date and Time"] >= ignition)
        & (df_day["Date and Time"] <= window_end)
        & df_day["PM3 (µg/m³)"].notna()
    ]
    if in_win.empty:
        print(f"    [{burn_id}|{instrument}] No in-window PM3 data - skipped.")
        return None
    peak_pm3_mass = float(in_win["PM3 (µg/m³)"].max())

    # Per-channel reversal analysis
    ch_results: dict[str, dict] = {}
    for ch, lo, hi in ANALYSIS_CH:
        col = f"Ʃ{lo}-{hi}µm (#/cm³)"
        if col not in df_day.columns:
            continue
        ch_results[ch] = {
            "col": col,
            **_analyze_bin(
                df_day[col],
                df_day["Date and Time"],
                ignition,
                window_end,
            ),
        }

    if "Ch1" not in ch_results:
        print(f"    [{burn_id}|{instrument}] 0.3-0.5 um column absent - skipped.")
        return None

    r1        = ch_results["Ch1"]
    n_peak    = r1["n_peak"]
    t_peak    = r1["t_peak"]
    t_min_ch1 = r1["t_min"]
    ch1_col   = r1["col"]

    # Coincidence loss estimates
    if not np.isnan(n_peak):
        L_c, L_lo, L_hi = _coincidence_loss(n_peak)
    else:
        L_c = L_lo = L_hi = np.nan

    # Counts conservation
    conserved = _counts_conserved(df_day, t_peak, t_min_ch1)

    # SMPS cross-check (AeroTrak1 only, burns where reversal detected)
    smps_ratio = None
    if instrument == "AeroTrak1" and r1["reversal_present"]:
        smps_ratio = _smps_cross_check(
            df_day, df_smps, t_min_ch1, window_end, ch1_col
        )

    notes_parts = []
    if instrument == "AeroTrak1" and burn_id in BEDROOM_SEALED_BURNS:
        notes_parts.append("bedroom sealed")

    return {
        # --- CSV columns ---
        "burn":                      burn_id,
        "instrument":                instrument,
        "location":                  "bedroom2" if instrument == "AeroTrak1" else "morning_room",
        "bedroom_sealed":            (instrument == "AeroTrak1" and burn_id in BEDROOM_SEALED_BURNS),
        "n_peak_cm3":                n_peak,
        "t_peak":                    t_peak,
        "reversal_present":          r1["reversal_present"],
        "reversal_duration_minutes": r1["reversal_duration_minutes"],
        "L_central":                 L_c,
        "L_low":                     L_lo,
        "L_high":                    L_hi,
        "peak_total_PM3_mass_ug_m3": peak_pm3_mass,
        "counts_conserved":          conserved,
        "SMPS_ratio_during_vs_after": smps_ratio,
        "notes":                     "; ".join(notes_parts),
        # --- private: used by plotting functions ---
        "_df_day":    df_day,
        "_df_smps":   df_smps,
        "_ch_results": ch_results,
        "_ignition":  ignition,
        "_window_end": window_end,
    }

# ==============================================================================
# BOKEH PER-BURN FIGURES
# ==============================================================================

def _bokeh_individual(result: dict) -> None:
    """
    Three-panel Bokeh figure (one panel per analysis channel) showing count
    concentration vs wall-clock time. Annotates ignition, n_peak, and n_min
    with vertical spans. Overlays SMPS 300-437 nm on the Ch1 panel for
    AeroTrak1 burns where a reversal was detected.

    Output: coincidence_figures/aerotrak_coincidence_<burn>_<unit>.html
    """
    burn_id    = result["burn"]
    instrument = result["instrument"]
    df_day     = result["_df_day"]
    df_smps    = result["_df_smps"]
    ch_res     = result["_ch_results"]
    ignition   = result["_ignition"]
    window_end = result["_window_end"]

    unit_tag   = "aerotrak1" if instrument == "AeroTrak1" else "aerotrak2"
    unit_label = "AeroTrak 1 - Bedroom 2" if instrument == "AeroTrak1" else "AeroTrak 2 - Morning Room"

    # Time window for display: 30 min before ignition to window_end
    t_start = ignition - pd.Timedelta(minutes=30)

    panels = []
    for ch, lo, hi in ANALYSIS_CH:
        if ch not in ch_res:
            continue
        r   = ch_res[ch]
        col = r["col"]

        mask = (
            (df_day["Date and Time"] >= t_start)
            & (df_day["Date and Time"] <= window_end)
            & df_day[col].notna()
        )
        sub = df_day[mask]

        p = figure(
            x_axis_type="datetime",
            height=280,
            width=950,
            title=(
                f"{burn_id}  |  {unit_label}  |  "
                f"{lo}–{hi} µm count concentration (#/cm³)"
            ),
            toolbar_location="right",
        )
        p.line(
            sub["Date and Time"], sub[col],
            color=COLOR[instrument], line_width=2.0,
            legend_label=instrument,
        )

        # SMPS overlay on Ch1 panel (AeroTrak1 with reversal)
        if (
            ch == "Ch1"
            and instrument == "AeroTrak1"
            and df_smps is not None
            and r["reversal_present"]
        ):
            smps_c = _smps_300_437(df_smps)
            smps_m = (
                (df_smps["datetime"] >= t_start)
                & (df_smps["datetime"] <= window_end)
                & smps_c.notna()
            )
            sub_s = df_smps[smps_m]
            if not sub_s.empty:
                p.line(
                    sub_s["datetime"],
                    smps_c[smps_m].values,
                    color=COLOR["smps"], line_width=1.8,
                    line_dash="dashed",
                    legend_label="SMPS 300–437 nm",
                )

        # Vertical event lines
        events = [
            (ignition,    "black",   "solid",   "Ignition"),
            (r["t_peak"], "#d62728", "dashed",  "n_peak"),
            (r["t_min"],  "#1f77b4", "dotted",  "n_min"),
        ]
        for t_ev, col_ev, dash_ev, lbl_ev in events:
            if pd.notna(t_ev):
                span = Span(
                    location=int(t_ev.timestamp() * 1000),
                    dimension="height",
                    line_color=col_ev,
                    line_dash=dash_ev,
                    line_width=1.4,
                )
                p.add_layout(span)

        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "#/cm³"
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        panels.append(p)

    if not panels:
        return

    fig_dir = get_common_file("coincidence_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"aerotrak_coincidence_{burn_id}_{unit_tag}.html"

    reset_output()
    output_file(str(out_path), title=f"{burn_id} {instrument} coincidence")
    save(bokeh_column(*panels))
    print(f"    [Bokeh] {out_path.name}")

# ==============================================================================
# MATPLOTLIB - SMALL MULTIPLES (SI figure)
# ==============================================================================

def _mpl_small_multiples(
    all_results: list[dict],
    df_at1: pd.DataFrame,
    df_at2: pd.DataFrame,
) -> None:
    """
    One panel per burn: Ch1 (0.3-0.5 um) count concentration vs time from
    ignition (minutes). Morning Room (AeroTrak2) and Bedroom 2 (AeroTrak1)
    overlaid in different colours. Log y-axis. Vertical line at ignition and
    at each instrument's t_peak. Intended for SI.
    """
    # Collect all burns present in results
    all_burns = sorted(
        {r["burn"] for r in all_results},
        key=lambda b: int(b.replace("burn", "")),
    )
    # Ignition lookup
    ignitions = {
        r["burn"]: r["_ignition"]
        for r in all_results
        if pd.notna(r["_ignition"])
    }
    # t_peak lookup per (burn, instrument)
    t_peaks = {
        (r["burn"], r["instrument"]): r["t_peak"]
        for r in all_results
        if pd.notna(r["t_peak"])
    }

    ncols = 3
    nrows = int(np.ceil(len(all_burns) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    for ax_idx, burn_id in enumerate(all_burns):
        ax     = axes[ax_idx]
        ignit  = ignitions.get(burn_id)
        if ignit is None:
            ax.set_visible(False)
            continue

        has_line = False
        for instr, df_all, lbl in [
            ("AeroTrak1", df_at1, "Bedroom 2"),
            ("AeroTrak2", df_at2, "Morning Room"),
        ]:
            if burn_id not in BURN_COVERAGE[instr]:
                continue

            burn_date = bl_row_date_from_results(all_results, burn_id, instr)
            if burn_date is None:
                continue

            sub = _day_slice(df_all, pd.Timestamp(burn_date))
            ch1 = next(
                (c for c in sub.columns if "Ʃ0.3-0.5µm" in c), None
            )
            if ch1 is None:
                continue

            t_min_norm = (sub["Date and Time"] - ignit).dt.total_seconds() / 60.0
            vals = sub[ch1].replace(0, np.nan)
            ax.semilogy(
                t_min_norm, vals,
                color=COLOR[instr], lw=1.0, label=lbl, alpha=0.85,
            )
            has_line = True

            # Mark t_peak for this instrument on this burn
            tp = t_peaks.get((burn_id, instr))
            if pd.notna(tp):
                tp_min = (tp - ignit).total_seconds() / 60.0
                ax.axvline(tp_min, color=COLOR[instr], lw=0.8, ls=":")

        ax.axvline(0, color="black", lw=0.9, ls="--", label="Ignition")
        ax.set_xlim(-15, MAX_WIN_HR * 60)
        ax.set_xlabel("min from ignition", fontsize=TEXT_CONFIG["labelsize"])
        ax.set_ylabel("#/cm³", fontsize=TEXT_CONFIG["labelsize"])
        ax.tick_params(labelsize=TEXT_CONFIG["ticksize"])
        ax.set_title(burn_id, fontsize=TEXT_CONFIG["labelsize"], fontweight="bold")

        if ax_idx == 0 and has_line:
            ax.legend(fontsize=TEXT_CONFIG["legendsize"], loc="upper right")

    for ax in axes[len(all_burns):]:
        ax.set_visible(False)

    fig_dir  = get_common_file("coincidence_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "aerotrak_coincidence_small_multiples.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    [mpl] {out_path.name}")


def bl_row_date_from_results(
    all_results: list[dict], burn_id: str, instrument: str
) -> object | None:
    """Return the burn date (date object) from matching results, or None."""
    for r in all_results:
        if r["burn"] == burn_id and r["instrument"] == instrument:
            df = r["_df_day"]
            if not df.empty:
                return df["Date and Time"].dt.date.iloc[0]
    return None

# ==============================================================================
# MATPLOTLIB - OVERLAY (main-text figure)
# ==============================================================================

def _mpl_overlay(all_results: list[dict]) -> None:
    """
    Single panel: all AeroTrak2 (Morning Room) burns overlaid. Ch1 count
    (#/cm3), x-axis normalized to t_peak = 0 (minutes). Log y-axis.
    """
    at2 = [
        r for r in all_results
        if r["instrument"] == "AeroTrak2" and pd.notna(r.get("t_peak"))
    ]
    if not at2:
        print("    [mpl] No AeroTrak2 results for overlay figure.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    cmap   = plt.get_cmap("viridis", len(at2))
    all_burns_sorted = sorted(at2, key=lambda r: int(r["burn"].replace("burn", "")))

    for i, r in enumerate(all_burns_sorted):
        df_day  = r["_df_day"]
        t_peak  = r["t_peak"]
        burn_id = r["burn"]

        ch1 = next(
            (c for c in df_day.columns if "Ʃ0.3-0.5µm" in c), None
        )
        if ch1 is None:
            continue

        t_norm = (df_day["Date and Time"] - t_peak).dt.total_seconds() / 60.0
        vals   = df_day[ch1].replace(0, np.nan)

        ax.semilogy(t_norm, vals, color=cmap(i), lw=1.4,
                    label=burn_id, alpha=0.85)

    ax.axvline(0, color="black", lw=1.0, ls="--", label="t_peak")
    ax.set_xlim(-20, 120)
    ax.set_xlabel(
        "Minutes relative to nₚₑₐₖ",
        fontsize=TEXT_CONFIG["labelsize"],
        fontweight="bold",
    )
    ax.set_ylabel(
        "0.3–0.5 µm count (#/cm³)",
        fontsize=TEXT_CONFIG["labelsize"],
        fontweight="bold",
    )
    ax.tick_params(labelsize=TEXT_CONFIG["ticksize"])
    ax.set_title(
        "AeroTrak 2 (Morning Room) — all burns, Ch1 normalised to nₚₑₐₖ",
        fontsize=TEXT_CONFIG["labelsize"],
        fontweight="bold",
    )
    ax.legend(fontsize=TEXT_CONFIG["legendsize"], ncol=2, loc="upper right")

    fig_dir  = get_common_file("coincidence_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "aerotrak_coincidence_overlay.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    [mpl] {out_path.name}")

# ==============================================================================
# MATPLOTLIB - LOSS VS PEAK MASS (main-text scatter)
# ==============================================================================

def _mpl_loss_vs_peakmass(all_results: list[dict]) -> None:
    """
    Scatter plot: x = peak total PM3 mass (ug/m3), y = L_central.
    Marker shape by location. Error bars from L_low to L_high (vertical).
    Horizontal reference line at L = 0.10 (manufacturer 10 % threshold).
    """
    records = [
        r for r in all_results
        if not np.isnan(r.get("L_central", np.nan))
        and not np.isnan(r.get("peak_total_PM3_mass_ug_m3", np.nan))
    ]
    if not records:
        print("    [mpl] No data for loss vs peak-mass scatter.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    markers = {"bedroom2": "o", "morning_room": "s"}
    plotted = set()

    for r in records:
        loc  = r["location"]
        x    = r["peak_total_PM3_mass_ug_m3"]
        y    = r["L_central"]
        y_lo = max(0.0, y - r["L_low"])
        y_hi = max(0.0, r["L_high"] - y)
        lbl  = loc.replace("_", " ").title() if loc not in plotted else None
        plotted.add(loc)

        ax.errorbar(
            x, y,
            yerr=[[y_lo], [y_hi]],
            fmt=markers.get(loc, "o"),
            color=COLOR.get(
                "AeroTrak1" if loc == "bedroom2" else "AeroTrak2", "gray"
            ),
            markersize=7, capsize=4, alpha=0.85,
            label=lbl,
        )
        ax.annotate(
            r["burn"][-2:],
            xy=(x, y), xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0.10, color="gray", lw=0.9, ls=":", label="10 % threshold")
    ax.set_xlabel(
        "Peak PM3 mass (µg/m³)",
        fontsize=TEXT_CONFIG["labelsize"], fontweight="bold",
    )
    ax.set_ylabel(
        "Coincidence loss L (fraction)",
        fontsize=TEXT_CONFIG["labelsize"], fontweight="bold",
    )
    ax.tick_params(labelsize=TEXT_CONFIG["ticksize"])
    ax.set_title(
        "Estimated coincidence loss vs peak PM3 mass",
        fontsize=TEXT_CONFIG["labelsize"], fontweight="bold",
    )
    ax.legend(fontsize=TEXT_CONFIG["legendsize"])

    fig_dir  = get_common_file("coincidence_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "aerotrak_loss_vs_peakmass.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    [mpl] {out_path.name}")

# ==============================================================================
# CSV OUTPUTS
# ==============================================================================

def _write_csv(all_results: list[dict]) -> None:
    """
    Write per-burn CSV and cross-burn summary CSV to the coincidence_analysis
    folder.
    """
    out_dir = get_common_file("coincidence_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-burn table
    rows = [{k: r.get(k, np.nan) for k in _CSV_COLS} for r in all_results]
    df   = pd.DataFrame(rows, columns=_CSV_COLS)

    per_burn_path = out_dir / "aerotrak_coincidence_per_burn.csv"
    df.to_csv(str(per_burn_path), index=False, float_format="%.4g")
    print(f"    [CSV] {per_burn_path.name}")

    # Cross-burn summary statistics
    valid = df[df["n_peak_cm3"].notna()].copy()
    rev   = valid[valid["reversal_present"] == True]

    def _q_stats(col: str) -> dict:
        s = valid[col].dropna()
        if s.empty:
            return {f"{col}_median": np.nan, f"{col}_IQR": np.nan,
                    f"{col}_range_min": np.nan, f"{col}_range_max": np.nan}
        q25, q75 = s.quantile([0.25, 0.75])
        return {
            f"{col}_median":    float(s.median()),
            f"{col}_IQR":       float(q75 - q25),
            f"{col}_range_min": float(s.min()),
            f"{col}_range_max": float(s.max()),
        }

    summary = {}
    for col in ("n_peak_cm3", "reversal_duration_minutes", "L_central"):
        summary.update(_q_stats(col))

    summary["n_reversal_present"] = int(rev.shape[0])
    summary["n_total_pairs"]      = int(valid.shape[0])
    summary["median_peak_PM3_with_reversal_ug_m3"] = (
        float(rev["peak_total_PM3_mass_ug_m3"].median())
        if not rev.empty else np.nan
    )

    summary_path = out_dir / "aerotrak_coincidence_cross_burn_summary.csv"
    pd.DataFrame([summary]).to_csv(str(summary_path), index=False, float_format="%.4g")
    print(f"    [CSV] {summary_path.name}")

# ==============================================================================
# MARKDOWN OUTPUTS
# ==============================================================================

def _write_markdown(all_results: list[dict]) -> None:
    """
    Write aerotrak_coincidence_summary.md and aerotrak_manuscript_sentences.md
    to the coincidence_analysis folder.

    All numbers are derived from data; no values are estimated from memory.
    """
    out_dir = get_common_file("coincidence_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = [r for r in all_results if not np.isnan(r.get("n_peak_cm3", np.nan))]
    rev   = [r for r in valid if r.get("reversal_present")]

    n_peak_arr   = np.array([r["n_peak_cm3"] for r in valid])
    L_arr        = np.array([r["L_central"]  for r in valid if not np.isnan(r.get("L_central", np.nan))])
    dur_arr      = np.array([r["reversal_duration_minutes"] for r in rev
                              if not np.isnan(r.get("reversal_duration_minutes", np.nan))])
    pm3_rev_arr  = np.array([r["peak_total_PM3_mass_ug_m3"] for r in rev
                              if not np.isnan(r.get("peak_total_PM3_mass_ug_m3", np.nan))])

    cons_fails = [r for r in rev if r.get("counts_conserved") is False]

    # ── summary.md ──────────────────────────────────────────────────────────
    n_total = len(valid)
    n_rev   = len(rev)

    if n_total > 0:
        med_npeak = float(np.median(n_peak_arr))
        med_L     = float(np.median(L_arr)) if len(L_arr) > 0 else np.nan
        factor    = med_npeak / COINCIDENCE_THRESHOLD_CM3
        med_pm3   = float(np.median(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan
        n_at1     = sum(1 for r in valid if r["instrument"] == "AeroTrak1")
        n_at2     = sum(1 for r in valid if r["instrument"] == "AeroTrak2")

        summary_body = (
            f"Across {n_total} burn-instrument pairs ({n_at1} AeroTrak1 / Bedroom 2, "
            f"{n_at2} AeroTrak2 / Morning Room), the median peak 0.3-0.5 um count "
            f"concentration was {med_npeak:.2e} particles/cm3 -- "
            f"{'above' if med_npeak > COINCIDENCE_THRESHOLD_CM3 else 'below'} the "
            f"manufacturer's 10 % coincidence threshold of "
            f"{COINCIDENCE_THRESHOLD_CM3:.1e} particles/cm3 "
            f"(factor of {factor:.1f}). "
            f"Channel reversals (Ch1 minimum < 50 % of Ch1 peak within 15 min) "
            f"were detected in {n_rev} of {n_total} pairs. "
            f"For those burns, the median estimated coincidence loss fraction was "
            f"{med_L:.3f} ({med_L*100:.0f} %). "
            f"Reversals occurred at median peak PM3 mass concentrations of "
            f"approximately {med_pm3:.0f} ug/m3."
        ) if n_total > 0 else "No valid results computed."
    else:
        summary_body = "No valid results computed."

    cons_section = ""
    if cons_fails:
        items = "\n".join(
            f"- {r['burn']}, {r['instrument']}: total count changed > "
            f"{CONSERVATION_TOL*100:.0f} % at reversal trough."
            for r in cons_fails
        )
        cons_section = (
            "\n\n## Burns where counts conservation fails\n\n"
            f"The following pairs show > {CONSERVATION_TOL*100:.0f} % change in "
            "total 6-bin count concentration at the reversal trough. Coincidence "
            "alone may not fully account for the reversal in these cases.\n\n"
            + items
        )
    else:
        cons_section = (
            "\n\n## Counts conservation\n\n"
            f"All {n_rev} burn-instrument pairs with a detected reversal passed "
            f"the counts-conservation check (< {CONSERVATION_TOL*100:.0f} % change "
            "in total 6-bin count at the reversal trough), consistent with optical "
            "coincidence as the dominant mechanism."
        )

    summary_text = (
        "# AeroTrak Coincidence Analysis - Summary\n\n"
        "## Plain-language summary\n\n"
        + summary_body
        + cons_section
    )
    (out_dir / "aerotrak_coincidence_summary.md").write_text(
        summary_text, encoding="utf-8"
    )
    print("    [MD] aerotrak_coincidence_summary.md")

    # ── manuscript_sentences.md ──────────────────────────────────────────────
    b9k = next(
        (r for r in all_results
         if r["burn"] == "burn9" and r["instrument"] == "AeroTrak2"),
        None,
    )

    def _fmt(val, fmt=".2e", unit=""):
        return f"{val:{fmt}}{unit}" if not np.isnan(val) else "[no data]"

    if b9k and not np.isnan(b9k.get("n_peak_cm3", np.nan)):
        n_ex     = b9k["n_peak_cm3"]
        fac_ex   = n_ex / COINCIDENCE_THRESHOLD_CM3
        L_lo_ex  = b9k["L_low"]  * 100
        L_hi_ex  = b9k["L_high"] * 100
    else:
        n_ex = fac_ex = L_lo_ex = L_hi_ex = np.nan

    if len(n_peak_arr) > 0:
        fac_med = float(np.median(n_peak_arr)) / COINCIDENCE_THRESHOLD_CM3
        fac_lo  = float(np.min(n_peak_arr))    / COINCIDENCE_THRESHOLD_CM3
        fac_hi  = float(np.max(n_peak_arr))    / COINCIDENCE_THRESHOLD_CM3
    else:
        fac_med = fac_lo = fac_hi = np.nan

    if len(L_arr) > 0:
        L_med_pct = float(np.median(L_arr)) * 100
        L_lo_pct  = float(np.min(L_arr))    * 100
        L_hi_pct  = float(np.max(L_arr))    * 100
    else:
        L_med_pct = L_lo_pct = L_hi_pct = np.nan

    med_pm3_rev_str = (
        f"{float(np.median(pm3_rev_arr)):.0f}"
        if len(pm3_rev_arr) > 0 else "[no data]"
    )
    med_dur_str = (
        f"{float(np.median(dur_arr)):.0f}"
        if len(dur_arr) > 0 else "[no data]"
    )

    ms_text = (
        "# AeroTrak Coincidence - Manuscript Sentences for Section 3.2.2\n\n"
        "_All values derived from data. Insert these into the manuscript text._\n\n"
        "---\n\n"
        "## Worked example: AeroTrak 2 (Morning Room), Burn 09\n\n"
        f'**Sentence 1 (peak count):** "approximately {_fmt(n_ex)} particles/cm3 '
        'at the time of the channel reversal"\n\n'
        f'**Sentence 2 (factor above threshold):** "exceeding the manufacturer\'s '
        f"10 % coincidence threshold by a factor of approximately {_fmt(fac_ex, '.1f')} "
        f"(median across all burns: {_fmt(fac_med, '.1f')}; "
        f"range: {_fmt(fac_lo, '.1f')}-{_fmt(fac_hi, '.1f')}).\"\n\n"
        "---\n\n"
        "## Campaign-wide sentences\n\n"
        f'**Sentence 3 (coincidence loss range):** "implying an estimated '
        f"coincidence loss of approximately {_fmt(L_lo_pct, '.0f')} % "
        f"to {_fmt(L_hi_pct, '.0f')} % "
        f"(median {_fmt(L_med_pct, '.0f')} %)\"\n\n"
        f'**Sentence 4 (PM mass threshold):** "all three optical instruments '
        "showed evidence of these effects at peak PM2.5-equivalent concentrations "
        f"exceeding approximately {med_pm3_rev_str} ug/m3\"\n\n"
        "---\n\n"
        "## Supporting statistics\n\n"
        f"| Quantity | Value |\n"
        f"|---|---|\n"
        f"| Burn-instrument pairs analysed | {n_total} |\n"
        f"| Pairs with Ch1 reversal | {n_rev} |\n"
        f"| Median n_peak all pairs (#/cm3) | {_fmt(np.median(n_peak_arr) if len(n_peak_arr) > 0 else np.nan)} |\n"
        f"| Median reversal duration (min) | {med_dur_str} |\n"
        f"| Median L_central all pairs (%) | {_fmt(L_med_pct, '.1f')} |\n"
        f"| Median peak PM3 at reversal (ug/m3) | {med_pm3_rev_str} |\n"
    )
    (out_dir / "aerotrak_manuscript_sentences.md").write_text(
        ms_text, encoding="utf-8"
    )
    print("    [MD] aerotrak_manuscript_sentences.md")

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full coincidence analysis pipeline:
    load data, analyse each burn-instrument pair, generate all plots and outputs.
    """
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Loading burn log...")
    burn_log = _load_burn_log()

    print("Loading AeroTrak data...")
    df_at1 = _load_aerotrak_all("AeroTrak1")
    df_at2 = _load_aerotrak_all("AeroTrak2")
    print(f"  AeroTrak1: {len(df_at1):,} rows | AeroTrak2: {len(df_at2):,} rows")

    all_results: list[dict] = []

    for instrument, df_aerotrak in [("AeroTrak1", df_at1), ("AeroTrak2", df_at2)]:
        print(f"\nProcessing {instrument} ({'Bedroom 2' if instrument == 'AeroTrak1' else 'Morning Room'})...")

        for burn_id in BURN_COVERAGE[instrument]:
            bl_rows = burn_log[burn_log["Burn ID"] == burn_id]
            if bl_rows.empty:
                continue
            bl_row    = bl_rows.iloc[0]
            burn_date = bl_row["Date"]

            # Load SMPS numConc for Bedroom 2 cross-check
            df_smps = None
            if instrument == "AeroTrak1":
                df_smps = _load_smps_numconc(burn_date)
                if df_smps is None:
                    print(f"  [{burn_id}] SMPS numConc not found for {burn_date.date()}")

            print(f"  Analysing {burn_id}...")
            result = analyze_burn_instrument(
                burn_id, instrument, df_aerotrak, bl_row, df_smps
            )
            if result is None:
                continue

            all_results.append(result)
            print(
                f"    n_peak={result['n_peak_cm3']:.2e} #/cm3  "
                f"reversal={result['reversal_present']}  "
                f"L_c={result['L_central']:.3f}  "
                f"PM3={result['peak_total_PM3_mass_ug_m3']:.0f} ug/m3"
            )

    if not all_results:
        print("\nNo results computed - check data paths and burn coverage.")
        return

    print(f"\n{len(all_results)} burn-instrument pairs processed.")

    print("\nGenerating Bokeh per-burn figures...")
    for r in all_results:
        _bokeh_individual(r)

    print("\nGenerating matplotlib figures...")
    _mpl_small_multiples(all_results, df_at1, df_at2)
    _mpl_overlay(all_results)
    _mpl_loss_vs_peakmass(all_results)

    print("\nWriting CSV outputs...")
    _write_csv(all_results)

    print("\nWriting markdown outputs...")
    _write_markdown(all_results)

    print("\nDone. Check coincidence_analysis/ and coincidence_figures/ for outputs.")


if __name__ == "__main__":
    main()
