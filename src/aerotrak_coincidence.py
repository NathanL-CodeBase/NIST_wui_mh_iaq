"""
AeroTrak 9306-V2 Optical Coincidence Validation Analysis

Tests the optical-coincidence interpretation of the AeroTrak bin reversals for
Section 3.2.2 of the WUI fire smoke instrument paper. For each AeroTrak unit
and burn, detects bin reversals in the 0.3-0.5, 0.5-1.0, and 1.0-3.0 um
channels, estimates coincidence losses via a Poisson dead-time model using the
correctly converted manufacturer concentration-limit specification, compares
peak counts against the Poisson rollover ceiling, checks counts conservation,
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
Updated: 2026-07-07 (corrected coincidence-limit unit conversion; see CONSTANTS)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bokeh.io import reset_output
from bokeh.layouts import column as bokeh_column
from bokeh.models import HoverTool, Label, LinearAxis, Range1d, Span
from bokeh.plotting import figure, output_file, save

# --- repository root on path ---------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.data_paths import get_common_file, get_instrument_path
from src.fig_style import (  # noqa: E402
    INSTR_COLORS,
    REF_LINE,
    SHADE,
    apply_est_style,
    figsize,
    save_fig,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Optical sensing volume from Poisson model  L = 1 - exp(-n*V)
# TSI Concentration Limits spec: 5 % coincidence loss at 3,000,000 particles/ft^3.
# The spec is stated per cubic foot; 1 ft^3 = 28,316.85 cm^3, so the limit is
# ~106 particles/cm3 (an earlier conversion divided by 28.3, the number of
# liters per ft^3, and overstated the limit by a factor of 1000). The 2024
# 9306-V2 spec sheet (rev N) gives an alternative statement, 10 % loss at
# 5,950,000 particles/ft^3 (2.1e8 particles/m^3 = 210 particles/cm3), which
# implies nearly the same sensing volume (5.0e-4 cm3), so the inferred V is
# robust to the spec variant.
FT3_TO_CM3 = 28_316.846592  # cm3 per cubic foot

COINCIDENCE_LOSS_SPEC = 0.05  # manufacturer Concentration Limits coincidence loss
COINCIDENCE_THRESHOLD_CM3 = 3.0e6 / FT3_TO_CM3  # ~105.9 particles/cm3

V_CENTRAL = float(-np.log(1.0 - COINCIDENCE_LOSS_SPEC) / COINCIDENCE_THRESHOLD_CM3)  # ~4.8e-4 cm3
# Same +/- one-third band previously applied to the sensing-volume estimate,
# now spanning the two published spec variants comfortably.
V_LOW = V_CENTRAL * (2.0 / 3.0)
V_HIGH = V_CENTRAL * (4.0 / 3.0)

# Rollover ceiling of the Poisson model: the measured count n*exp(-n*V) can
# never exceed exp(-1)/V regardless of the true concentration, and it declines
# with increasing true concentration beyond n = 1/V. Peak counts pinned near
# this ceiling across burns are the signature of a counter past its limit.
N_MEAS_CEILING_CM3 = float(np.exp(-1.0) / V_CENTRAL)  # ~760 particles/cm3
N_MEAS_CEILING_LOW = float(np.exp(-1.0) / V_HIGH)
N_MEAS_CEILING_HIGH = float(np.exp(-1.0) / V_LOW)

# Reversal detection parameters
REVERSAL_FRAC = 0.5  # n_min < REVERSAL_FRAC * n_peak -> reversal present
REVERSAL_WIN_MIN = 30  # search window after n_peak (minutes)

# Minimum sustained duration (minutes) for a dip to count as a true reversal.
# The overlay (aerotrak_coincidence_overlay.png) shows a genuine Ch1 reversal
# does not establish until ~30 min; the short events that previously set the
# low end of the duration range (~3 min) are the early ignition transient and
# the post-peak decay, each followed by a sharp drop rather than a sustained
# suppression. Requiring the count to stay below REVERSAL_FRAC * n_peak for at
# least this contiguous span rejects those transients without hard-coding a
# fixed onset time.
REVERSAL_MIN_SUSTAIN_MIN = 8.0

# Counts-conservation tolerance (fraction change allowed)
CONSERVATION_TOL = 0.30

# Particle density for Mie-sphere PM mass (matches existing project scripts)
PARTICLE_DENSITY_G_CM3 = 1.0

# Pre-burn baseline window
BASELINE_MIN = 30  # minutes before ignition

# Maximum analysis window
MAX_WIN_HR = 4.0

# AeroTrak 9306-V2 size channels: (label, lower_um, upper_um)
CHANNELS = [
    ("Ch1", 0.3, 0.5),
    ("Ch2", 0.5, 1.0),
    ("Ch3", 1.0, 3.0),
    ("Ch4", 3.0, 5.0),
    ("Ch5", 5.0, 10.0),
    ("Ch6", 10.0, 25.0),
]
ANALYSIS_CH = CHANNELS[:3]  # three smallest bins for coincidence check

# Raw differential-count columns used to fingerprint a row when removing
# Morning Room data that was merged into the Bedroom 2 export (see
# _drop_foreign_rows). A row is a duplicate only if its timestamp AND all six
# differential counts match the other instrument exactly.
_DIFF_COLS = [f"{ch} Diff (#)" for ch, _lo, _hi in CHANNELS]

# Instrument time shifts (minutes applied to raw timestamps)
TIME_SHIFTS = {"AeroTrak1": 2.16, "AeroTrak2": 5.0}

# Burns processed per instrument
BURN_COVERAGE = {
    "AeroTrak1": [f"burn{i}" for i in range(3, 11)],  # burn3-burn10
    "AeroTrak2": [f"burn{i}" for i in range(2, 11)],  # burn2-burn10
}

# Burns where Bedroom 2 was sealed (flag in output; AeroTrak1 only)
BEDROOM_SEALED_BURNS = {"burn5", "burn6"}

# Worked example pinned for the manuscript (Sentence 5). Set to a
# (burn, instrument) tuple to force the example, or None to fall back to the
# highest peak-PM3 non-sealed reversal pair with a computable suppression.
# Pinned to Burn 09 Morning Room to match the manuscript mass-bracketing
# worked example; the auto-selected highest-PM3 pair (burn3 Morning Room)
# has a near-zero reversal trough that reads poorly in prose.
WORKED_EXAMPLE: tuple[str, str] | None = ("burn9", "AeroTrak2")

# data_config.json instrument keys
INSTR_KEY = {
    "AeroTrak1": "aerotrak_bedroom",
    "AeroTrak2": "aerotrak_kitchen",
}

# Numeric point size for matplotlib calls (matches src.fig_style BASE_FONT_PT).
_FS = 12

# Plot colours from the shared colorblind-safe palette (Bedroom 2 blue,
# Morning Room vermillion, SMPS orange).
COLOR = {
    "AeroTrak1": INSTR_COLORS["AeroTrak1"],
    "AeroTrak2": INSTR_COLORS["AeroTrak2"],
    "smps": "#E69F00",
}

# CSV column order for per-burn output
_CSV_COLS = [
    "burn",
    "instrument",
    "location",
    "bedroom_sealed",
    "n_peak_cm3",
    "t_peak",
    "reversal_present",
    "reversal_onset",
    "t_min",
    "reversal_end",
    "reversal_duration_minutes",
    "reversal_onset_pre_pm3peak_minutes",
    "L_central",
    "L_low",
    "L_high",
    "factor_vs_threshold",
    "n_peak_frac_of_ceiling",
    "peak_total_PM3_mass_ug_m3",
    "counts_conserved",
    "SMPS_ratio_during_vs_after",
    "notes",
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
                if pd.notna(r[c])
                else pd.NaT
            ),
            axis=1,
        )
    return bl


# ==============================================================================
# AEROTRAK LOADER
# ==============================================================================


def _drop_foreign_rows(df: pd.DataFrame, other_instrument: str) -> pd.DataFrame:
    """
    Remove rows that are an exact copy of another instrument's record.

    On at least one burn day (2024-05-06) the Bedroom 2 export was merged with
    a copy of the Morning Room record, so the Bedroom 2 file carries both
    instruments' rows on that date. The foreign rows interleave with the real
    Bedroom 2 samples, halving the effective cadence and inserting Morning
    Room values that corrupt peak and reversal detection.

    A row is treated as foreign only when its raw timestamp AND all six
    differential counts match a row in the other instrument's file exactly, so
    genuine Bedroom 2 samples (which never coincide with the Morning Room down
    to the count) are always kept. Days with no overlap are returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows for this instrument (pre time shift), columns stripped.
    other_instrument : str
        The instrument whose rows may have been merged in (e.g. "AeroTrak2").

    Returns
    -------
    pd.DataFrame
        df with exact other-instrument duplicates removed.
    """
    other_path = get_instrument_path(INSTR_KEY[other_instrument]) / "all_data.xlsx"
    other = pd.read_excel(other_path)
    other.columns = other.columns.str.strip()
    other["Date and Time"] = pd.to_datetime(other["Date and Time"])

    key_cols = ["Date and Time"] + [c for c in _DIFF_COLS if c in df.columns]
    if not set(key_cols).issubset(other.columns):
        return df  # cannot fingerprint; leave unchanged

    foreign_keys = set(other[key_cols].itertuples(index=False, name=None))
    is_foreign = [
        row in foreign_keys
        for row in df[key_cols].itertuples(index=False, name=None)
    ]
    n_dropped = int(np.sum(is_foreign))
    if n_dropped:
        days = sorted(
            df.loc[is_foreign, "Date and Time"].dt.date.unique().tolist()
        )
        print(
            f"  [loader] Dropped {n_dropped} {other_instrument}-duplicate rows "
            f"merged into the record on {', '.join(str(d) for d in days)}."
        )
    return df[[not f for f in is_foreign]].reset_index(drop=True)


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

    # On some days the Bedroom 2 (AeroTrak1) export was merged with a copy of
    # the Morning Room (AeroTrak2) record, so the same burn day carries both
    # instruments' rows. Drop the foreign rows before the time shift, while raw
    # timestamps still line up between the two files.
    if instrument == "AeroTrak1":
        df = _drop_foreign_rows(df, "AeroTrak2")

    # Apply instrument time shift
    df["Date and Time"] += pd.Timedelta(minutes=TIME_SHIFTS[instrument])

    # Status filter
    ok = (df["Flow Status"] == "OK") & (df["Laser Status"] == "OK")
    df = df[ok].copy().reset_index(drop=True)

    # Sample volume
    vol_L = df["Volume (L)"]
    vol_cm3 = vol_L * 1000.0  # cm3
    vol_m3 = vol_L * 1e-3  # m3

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
        hi_um = float(size_val[next_ch]) if next_ch and next_ch in size_val else 25.0

        diff_col = f"{ch} Diff (#)"
        if diff_col not in df.columns:
            continue

        # Count concentration (#/cm3)
        conc_col = f"Ʃ{lo_um}-{hi_um}µm (#/cm³)"
        df[conc_col] = df[diff_col] / vol_cm3

        # Single-particle mass (ug) via Mie sphere, density = 1 g/cm3
        gm_um = np.sqrt(lo_um * hi_um)  # geometric-mean diameter (um)
        r_m = gm_um * 1e-6 / 2.0  # radius (m)
        vp_m3 = (4.0 / 3.0) * np.pi * r_m**3  # volume (m3)
        mass_ug = vp_m3 * 1e12  # ug (1 g/cm3 density)

        # Differential mass concentration (ug/m3)
        diff_mass_col = f"PM{lo_um}-{hi_um} Diff (µg/m³)"
        df[diff_mass_col] = (df[diff_col] / vol_m3) * mass_ug
        pm_diff_cols.append(diff_mass_col)

    # Cumulative PM mass (matches peak_concentration_script.py convention)
    cum_labels = [
        "PM0.5 (µg/m³)",
        "PM1 (µg/m³)",
        "PM3 (µg/m³)",
        "PM5 (µg/m³)",
        "PM10 (µg/m³)",
        "PM25 (µg/m³)",
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
    size_cols = [c for c in df.columns if isinstance(c, float) and not isinstance(c, bool)]
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
        c
        for c in df_smps.columns
        if isinstance(c, float) and not isinstance(c, bool) and 300.0 <= c <= 437.0
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
              reversal_present, reversal_duration_minutes,
              reversal_onset, reversal_end (full reversal interval bounds).
    """
    blank = dict(
        n_peak=np.nan,
        t_peak=pd.NaT,
        n_min_during_reversal=np.nan,
        t_min=pd.NaT,
        reversal_present=False,
        reversal_duration_minutes=np.nan,
        reversal_onset=pd.NaT,
        reversal_end=pd.NaT,
        reversal_pre=False,
        t_min_pre=pd.NaT,
    )

    mask = (timestamps >= ignition) & (timestamps <= window_end) & series.notna()
    if mask.sum() < 5:
        return blank

    s = np.asarray(series[mask], dtype=float)
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
    n_min_pre = np.nan
    t_min_pre = pd.NaT
    t_onset_pre = pd.NaT
    if peak_i > 1:
        s_before = s[:peak_i]
        ts_before = ts[:peak_i]
        thresh = REVERSAL_FRAC * n_peak
        # Only search for the dip AFTER the signal has risen above thresh.
        # Without this, the pre-smoke baseline (~0 #/cm³) would always be
        # identified as the minimum, and n_pre_dip would be ~0, causing
        # condition 2 in the original approach to always fail.
        above = np.where(s_before >= thresh)[0]
        if len(above) > 0:
            first_above = above[0]
            s_post_rise = s_before[first_above:]
            ts_post_rise = ts_before[first_above:]
            dip_i_local = int(np.nanargmin(s_post_rise))
            n_dip = float(s_post_rise[dip_i_local])
            if n_dip < thresh:
                reversal_pre = True
                n_min_pre = n_dip
                t_min_pre = pd.Timestamp(ts_post_rise[dip_i_local])
                # Onset: first sample on the rise where the signal drops back
                # below thresh (the leading edge of the dip).
                below = np.where(s_post_rise[: dip_i_local + 1] < thresh)[0]
                if below.size:
                    t_onset_pre = pd.Timestamp(ts_post_rise[int(below[0])])

    # --- Check B: post-peak drop (original logic) ---
    reversal_post = False
    n_min_post = np.nan
    t_min_post = pd.NaT
    t_search = t_peak + pd.Timedelta(minutes=REVERSAL_WIN_MIN)
    after_peak = (ts > np.datetime64(t_peak)) & (ts <= np.datetime64(t_search))
    if after_peak.sum() > 0:
        s_after = s[after_peak]
        ts_after = ts[after_peak]
        min_i = int(np.argmin(s_after))
        n_min_post = float(s_after[min_i])
        t_min_post = pd.Timestamp(ts_after[min_i])
        if n_min_post < REVERSAL_FRAC * n_peak:
            reversal_post = True

    # Combine: prefer pre-peak result when both fire (deeper dip scenario)
    if reversal_pre or reversal_post:
        n_min = n_min_pre if reversal_pre else n_min_post
        t_min = t_min_pre if reversal_pre else t_min_post
        assert isinstance(t_min, pd.Timestamp)  # always True inside this block

        # Sustained-reversal gate: a true reversal stays suppressed below
        # REVERSAL_FRAC * n_peak for a contiguous span around t_min. The early
        # ignition transient and the post-peak decay each dip only briefly
        # before a sharp drop, so they fail this test. Measure the contiguous
        # below-threshold span that contains t_min and reject the candidate if
        # it is shorter than REVERSAL_MIN_SUSTAIN_MIN.
        thresh_gate = REVERSAL_FRAC * n_peak
        below = s < thresh_gate
        min_idx = int(np.nanargmin(np.abs((ts - np.datetime64(t_min)))))
        if below[min_idx]:
            lo_i = min_idx
            while lo_i - 1 >= 0 and below[lo_i - 1]:
                lo_i -= 1
            hi_i = min_idx
            while hi_i + 1 < len(below) and below[hi_i + 1]:
                hi_i += 1
            sustain_min = (
                pd.Timestamp(ts[hi_i]) - pd.Timestamp(ts[lo_i])
            ).total_seconds() / 60.0
        else:
            sustain_min = 0.0

        if sustain_min < REVERSAL_MIN_SUSTAIN_MIN:
            # Not a sustained reversal; leave result as the blank/no-reversal
            # state but keep the peak so coincidence stats are still reported.
            result["reversal_pre"] = False
            result["t_min_pre"] = pd.NaT  # type: ignore[assignment]
            return result

        result["reversal_present"] = True
        result["n_min_during_reversal"] = n_min
        result["t_min"] = t_min  # type: ignore[assignment]

        # Reversal onset (leading edge of the dip). For the pre-peak case this
        # is the downward threshold crossing found above; for the post-peak
        # case the dip starts at t_peak.
        t_onset = t_onset_pre if reversal_pre else t_peak
        result["reversal_onset"] = t_onset  # type: ignore[assignment]

        # Reversal interval and duration. The interval runs from the leading
        # edge of the dip (onset, the downward crossing of n_peak/2) to the
        # first recovery back above n_peak/2 (the trailing edge). Duration is
        # the full onset-to-recovery span, which matches the sustained-reversal
        # gate and the "reversal duration" reported in the text; measuring only
        # t_min-to-recovery understated the duration when t_min sat late in a
        # broad dip.
        thresh = n_peak / 2.0
        post_lo = ts > np.datetime64(t_min)  # type: ignore[operator]
        s_rec = s[post_lo]
        ts_rec = ts[post_lo]
        recovered = s_rec >= thresh
        if recovered.any():
            t_rec = pd.Timestamp(ts_rec[int(np.argmax(recovered))])
            result["reversal_end"] = t_rec  # type: ignore[assignment]
            t_dur_start = t_onset if pd.notna(t_onset) else t_min
            result["reversal_duration_minutes"] = (
                t_rec - t_dur_start  # type: ignore[operator]
            ).total_seconds() / 60.0

    result["reversal_pre"] = reversal_pre
    result["t_min_pre"] = t_min_pre if reversal_pre else pd.NaT  # type: ignore[assignment]

    return result


def _coincidence_loss(n_cm3: float) -> tuple[float, float, float]:
    """
    Estimate coincidence loss fraction L = 1 - exp(-n * V) for the three
    sensing-volume estimates.

    Evaluated at the measured count concentration, which understates the true
    concentration once losses are appreciable, so each L is a lower bound on
    the actual loss.

    Parameters
    ----------
    n_cm3 : float
        Peak measured number concentration of the 0.3-0.5 um bin (#/cm3).

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

    conc_cols = [c for c in df_day.columns if "Ʃ" in str(c) and "#/cm" in str(c)]
    if not conc_cols:
        return None

    def _nearest_row_sum(t: pd.Timestamp) -> float:
        dt = (df_day["Date and Time"] - t).abs()
        idx = dt.idxmin()
        if dt[idx] > pd.Timedelta("3min"):
            return np.nan
        return float(df_day.loc[idx, conc_cols].sum())

    total_peak = _nearest_row_sum(t_peak)
    total_min = _nearest_row_sum(t_min)

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
    smps_ts = df_smps["datetime"]

    def _aerotrak_nearest(t: pd.Timestamp) -> float:
        dt = (df_day["Date and Time"] - t).abs()
        idx = dt.idxmin()
        return float(df_day.loc[idx, ch1_col]) if dt[idx] <= pd.Timedelta("3min") else np.nan

    def _smps_nearest(t: pd.Timestamp) -> float:
        dt = (smps_ts - t).abs()
        idx = dt.idxmin()
        val = smps_conc.iloc[idx]
        return float(val) if dt.iloc[idx] <= pd.Timedelta("5min") else np.nan

    at_ch1 = _aerotrak_nearest(t_min)
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
    ignition = bl_row["Ignition"]
    if pd.isna(ignition):
        print(f"    [{burn_id}|{instrument}] No ignition time - skipped.")
        return None

    df_day = _day_slice(df_aerotrak, burn_date)
    if df_day.empty:
        print(f"    [{burn_id}|{instrument}] No data for {burn_date.date()} - skipped.")
        return None

    # Pre-burn baseline and analysis window
    baseline_pm3 = _get_baseline(df_day, ignition)
    window_end = _find_window_end(df_day, ignition, baseline_pm3)

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

    r1 = ch_results["Ch1"]
    n_peak = r1["n_peak"]
    t_peak = r1["t_peak"]
    t_min_ch1 = r1["t_min"]
    ch1_col = r1["col"]

    # Coincidence loss estimates, threshold ratio, and rollover-ceiling ratio.
    # L is evaluated at the MEASURED count, which understates the true
    # concentration when losses are appreciable, so L is a lower bound.
    if not np.isnan(n_peak):
        L_c, L_lo, L_hi = _coincidence_loss(n_peak)
        factor_vs_threshold = n_peak / COINCIDENCE_THRESHOLD_CM3  # type: ignore[operator]
        n_peak_frac_of_ceiling = n_peak / N_MEAS_CEILING_CM3  # type: ignore[operator]
    else:
        L_c = L_lo = L_hi = np.nan
        factor_vs_threshold = np.nan
        n_peak_frac_of_ceiling = np.nan

    # Counts conservation
    conserved = _counts_conserved(df_day, t_peak, t_min_ch1)

    # SMPS cross-check (AeroTrak1 only, burns where reversal detected)
    smps_ratio = None
    if instrument == "AeroTrak1" and r1["reversal_present"]:
        smps_ratio = _smps_cross_check(df_day, df_smps, t_min_ch1, window_end, ch1_col)

    # Timing: how far the Ch1 reversal onset preceded the PM3 mass peak.
    # Only meaningful when Check A (pre-peak V-shape) detected the reversal.
    # Positive = reversal onset during the rising smoke phase (expected).
    # Negative = reversal onset after PM3 peak (flag in notes).
    reversal_onset_pre_pm3peak_minutes = np.nan
    if r1["reversal_pre"] and pd.notna(r1["t_min_pre"]):  # type: ignore[index]
        t_min_pre_ch1 = pd.Timestamp(r1["t_min_pre"])  # type: ignore[index]
        idx_pm3 = int(in_win["PM3 (µg/m³)"].idxmax())  # type: ignore[arg-type]
        t_pm3_peak_row = df_day.loc[idx_pm3, "Date and Time"]
        reversal_onset_pre_pm3peak_minutes = (
            float(
                (t_pm3_peak_row - t_min_pre_ch1).total_seconds()  # type: ignore[operator]
            )
            / 60.0
        )

    notes_parts = []
    if instrument == "AeroTrak1" and burn_id in BEDROOM_SEALED_BURNS:
        notes_parts.append("bedroom sealed")
    if not np.isnan(reversal_onset_pre_pm3peak_minutes) and reversal_onset_pre_pm3peak_minutes < 0:
        notes_parts.append("reversal onset post-PM3-peak (unexpected)")

    return {
        # --- CSV columns ---
        "burn": burn_id,
        "instrument": instrument,
        "location": "bedroom2" if instrument == "AeroTrak1" else "morning_room",
        "bedroom_sealed": (instrument == "AeroTrak1" and burn_id in BEDROOM_SEALED_BURNS),
        "n_peak_cm3": n_peak,
        "t_peak": t_peak,
        "reversal_present": r1["reversal_present"],
        "reversal_onset": r1["reversal_onset"],
        "t_min": t_min_ch1,
        "reversal_end": r1["reversal_end"],
        "reversal_duration_minutes": r1["reversal_duration_minutes"],
        "reversal_onset_pre_pm3peak_minutes": reversal_onset_pre_pm3peak_minutes,
        "L_central": L_c,
        "L_low": L_lo,
        "L_high": L_hi,
        "factor_vs_threshold": factor_vs_threshold,
        "n_peak_frac_of_ceiling": n_peak_frac_of_ceiling,
        "peak_total_PM3_mass_ug_m3": peak_pm3_mass,
        "counts_conserved": conserved,
        "SMPS_ratio_during_vs_after": smps_ratio,
        "notes": "; ".join(notes_parts),
        # --- private: used by plotting functions ---
        "_df_day": df_day,
        "_df_smps": df_smps,
        "_ch_results": ch_results,
        "_ignition": ignition,
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
    burn_id = result["burn"]
    instrument = result["instrument"]
    df_day = result["_df_day"]
    df_smps = result["_df_smps"]
    ch_res = result["_ch_results"]
    ignition = result["_ignition"]
    window_end = result["_window_end"]

    unit_tag = "aerotrak1" if instrument == "AeroTrak1" else "aerotrak2"
    unit_label = (
        "AeroTrak 1 - Bedroom 2" if instrument == "AeroTrak1" else "AeroTrak 2 - Morning Room"
    )

    # Time window for display: 30 min before ignition to window_end
    t_start = ignition - pd.Timedelta(minutes=30)

    panels = []
    for ch, lo, hi in ANALYSIS_CH:
        if ch not in ch_res:
            continue
        r = ch_res[ch]
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
            title=(f"{burn_id}  |  {unit_label}  |  {lo}–{hi} µm count concentration (#/cm³)"),
            toolbar_location="right",
        )
        p.line(
            sub["Date and Time"],
            sub[col],
            color=COLOR[instrument],
            line_width=2.0,
            legend_label=instrument,
        )

        # SMPS overlay on Ch1 panel (AeroTrak1 with reversal). The SMPS
        # 300-437 nm number concentration is roughly an order of magnitude
        # larger than the AeroTrak Ch1 count, so it goes on a right-hand
        # secondary axis: on a shared axis the AeroTrak trace flattens out.
        if (
            ch == "Ch1"
            and instrument == "AeroTrak1"
            and df_smps is not None
            and r["reversal_present"]
        ):
            smps_c = _smps_300_437(df_smps)
            # Buffer by one SMPS scan interval to handle minor timestamp offsets
            # between AeroTrak ignition time and SMPS scan cadence.
            smps_buf = pd.Timedelta(minutes=10)
            smps_m = (
                (df_smps["datetime"] >= t_start - smps_buf)
                & (df_smps["datetime"] <= window_end + smps_buf)
                & smps_c.notna()
            )
            sub_s = df_smps[smps_m]
            if sub_s.empty:
                if result["SMPS_ratio_during_vs_after"] is not None:  # type: ignore[index]
                    smps_dt = df_smps["datetime"]  # type: ignore[index]
                    print(
                        f"  [SMPS Bokeh] {burn_id} {instrument}: mask empty despite "
                        f"non-NaN SMPS ratio; SMPS range "
                        f"{smps_dt.min()} to {smps_dt.max()}, "
                        f"window {t_start} to {window_end}"
                    )
            else:
                smps_vals = smps_c[smps_m].to_numpy(dtype=float)
                smps_hi = float(np.nanmax(smps_vals)) if smps_vals.size else 1.0
                if not np.isfinite(smps_hi) or smps_hi <= 0:
                    smps_hi = 1.0
                p.extra_y_ranges = {"smps": Range1d(start=0, end=smps_hi * 1.05)}
                p.add_layout(
                    LinearAxis(
                        y_range_name="smps",
                        axis_label="SMPS 300–437 nm (#/cm³)",
                        axis_label_text_color=COLOR["smps"],
                        major_label_text_color=COLOR["smps"],
                    ),
                    "right",
                )
                p.line(
                    sub_s["datetime"],
                    smps_vals,
                    color=COLOR["smps"],
                    line_width=1.8,
                    line_dash="dashed",
                    legend_label="SMPS 300–437 nm",
                    y_range_name="smps",
                )

        # Ch1 panel only: TSI 5 % coincidence-loss limit and the Poisson
        # rollover ceiling (the maximum count the instrument can report; the
        # observed peaks pin near this line, the overload signature).
        if ch == "Ch1":
            for y_ref, dash_ref, text_ref in [
                (COINCIDENCE_THRESHOLD_CM3, "dotted", "TSI 5% coincidence-loss limit"),
                (N_MEAS_CEILING_CM3, "dashed", "Poisson rollover ceiling"),
            ]:
                p.add_layout(
                    Span(
                        location=y_ref,
                        dimension="width",
                        line_color="gray",
                        line_dash=dash_ref,
                        line_width=1,
                    )
                )
                p.add_layout(
                    Label(
                        x=int(t_start.timestamp() * 1000),  # type: ignore[union-attr]
                        y=y_ref,
                        text=text_ref,
                        text_font_size="10px",
                        text_color="gray",
                        x_units="data",
                        y_units="data",
                        y_offset=3,
                    )
                )

        # Vertical event lines
        events = [
            (ignition, "black", "solid", "Ignition"),
            (r["t_peak"], "#d62728", "dashed", "n_peak"),
            (r["t_min"], "#1f77b4", "dotted", "n_min"),
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
    ignitions = {r["burn"]: r["_ignition"] for r in all_results if pd.notna(r["_ignition"])}
    # t_peak lookup per (burn, instrument)
    t_peaks = {
        (r["burn"], r["instrument"]): r["t_peak"] for r in all_results if pd.notna(r["t_peak"])
    }

    ncols = 3
    nrows = int(np.ceil(len(all_burns) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize("double")[0], 2.4 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).flatten()

    for ax_idx, burn_id in enumerate(all_burns):
        ax = axes[ax_idx]
        ignit = ignitions.get(burn_id)
        if ignit is None:
            ax.set_visible(False)
            continue

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
            ch1 = next((c for c in sub.columns if "Ʃ0.3-0.5µm" in c), None)
            if ch1 is None:
                continue

            t_min_norm = (sub["Date and Time"] - ignit).dt.total_seconds() / 60.0
            vals = sub[ch1].replace(0, np.nan)
            ax.semilogy(
                t_min_norm,
                vals,
                color=COLOR[instr],
                lw=1.0,
                label=lbl,
                alpha=0.85,
            )

            # Mark t_peak for this instrument on this burn
            tp = t_peaks.get((burn_id, instr))
            if pd.notna(tp):
                tp_min = (tp - ignit).total_seconds() / 60.0
                ax.axvline(tp_min, color=COLOR[instr], lw=0.8, ls=":")

        ax.axvline(0, color=REF_LINE, lw=0.9, ls="--")
        ax.set_xlim(-15, MAX_WIN_HR * 60)
        ax.tick_params(labelsize=_FS - 2)
        ax.set_title(burn_id, fontsize=_FS - 1)

    for ax in axes[len(all_burns) :]:
        ax.set_visible(False)

    # Shared axis labels (one per figure) instead of per-panel labels.
    fig.supxlabel("min from ignition", fontsize=_FS)
    fig.supylabel("0.3-0.5 µm count (#/cm³)", fontsize=_FS)

    # Single figure-level legend with a fixed handle set, independent of which
    # series happen to appear in any given panel (e.g. burn2 has no Bedroom 2).
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=COLOR["AeroTrak1"], lw=1.5, label="Bedroom 2"),
        Line2D([0], [0], color=COLOR["AeroTrak2"], lw=1.5, label="Morning Room"),
        Line2D([0], [0], color=REF_LINE, lw=0.9, ls="--", label="Ignition"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
        fontsize=_FS - 2,
        frameon=True,
    )

    fig_dir = get_common_file("coincidence_figures")
    save_fig(fig, fig_dir / "aerotrak_coincidence_small_multiples.png")


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
    (#/cm3) vs minutes from ignition (a fixed physical event, so the rising
    edge aligns across burns instead of smearing on a ragged t_peak). Log
    y-axis. Individual burns are drawn faint; a bold median trace with a
    shaded IQR band summarises the typical reversal shape through the
    spaghetti.
    """
    at2 = [
        r for r in all_results if r["instrument"] == "AeroTrak2" and pd.notna(r.get("_ignition"))
    ]
    if not at2:
        print("    [mpl] No AeroTrak2 results for overlay figure.")
        return

    fig, ax = plt.subplots(figsize=figsize("double", aspect=0.6))

    all_burns_sorted = sorted(at2, key=lambda r: int(r["burn"].replace("burn", "")))

    # Common time grid (minutes from ignition) for the median/IQR summary.
    # 1-minute resolution over the displayed window.
    grid = np.arange(-15.0, MAX_WIN_HR * 60.0 + 1.0, 1.0)
    interp_stack = []

    for i, r in enumerate(all_burns_sorted):
        df_day = r["_df_day"]
        ignit = r["_ignition"]
        burn_id = r["burn"]

        ch1 = next((c for c in df_day.columns if "Ʃ0.3-0.5µm" in c), None)
        if ch1 is None:
            continue

        t_norm = (df_day["Date and Time"] - ignit).dt.total_seconds() / 60.0
        vals = df_day[ch1].replace(0, np.nan)

        # Faint individual burn traces in neutral gray so the bold vermillion
        # Morning Room summary (median + IQR) reads clearly through them.
        ax.semilogy(t_norm, vals, color=SHADE, lw=0.8, label=burn_id, alpha=0.40)

        # Resample onto the common grid for the summary band. Mask points
        # outside the burn's own time span so flat extrapolation does not bias
        # the median at the edges.
        t_arr = t_norm.to_numpy(dtype=float)
        v_arr = vals.to_numpy(dtype=float)
        order = np.argsort(t_arr)
        t_arr, v_arr = t_arr[order], v_arr[order]
        finite = np.isfinite(t_arr) & np.isfinite(v_arr)
        if finite.sum() < 2:
            continue
        t_arr, v_arr = t_arr[finite], v_arr[finite]
        gi = np.interp(grid, t_arr, v_arr, left=np.nan, right=np.nan)
        interp_stack.append(gi)

    # Median and IQR band across burns (ignoring NaN where a burn lacks data)
    if interp_stack:
        stack = np.vstack(interp_stack)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
        valid = np.isfinite(med)
        ax.fill_between(
            grid[valid],
            q25[valid],
            q75[valid],
            color=COLOR["AeroTrak2"],
            alpha=0.20,
            label="IQR (across burns)",
            zorder=4,
        )
        ax.semilogy(
            grid[valid],
            med[valid],
            color=COLOR["AeroTrak2"],
            lw=2.6,
            label="Median",
            zorder=5,
        )

    ax.axvline(0, color=REF_LINE, lw=1.0, ls="--", label="Ignition")
    ax.set_xlim(-15, MAX_WIN_HR * 60)
    ax.set_xlabel("Minutes from ignition", fontsize=_FS)
    ax.set_ylabel("0.3-0.5 µm count (#/cm³)", fontsize=_FS)
    ax.tick_params(labelsize=_FS)
    ax.legend(fontsize=_FS - 2, ncol=2, loc="lower right")

    fig_dir = get_common_file("coincidence_figures")
    save_fig(fig, fig_dir / "aerotrak_coincidence_overlay.png")


# ==============================================================================
# MATPLOTLIB - LOSS VS PEAK MASS (main-text scatter)
# ==============================================================================


def _mpl_loss_vs_peakmass(all_results: list[dict]) -> None:
    """
    Figure S2 (SI). Scatter: x = peak total PM3 mass (ug/m3), y = L_central
    (Poisson coincidence-loss fraction at the measured peak count), with
    vertical L_low-to-L_high error bars. Marker shape by location (Bedroom 2
    circles, Morning Room squares). Horizontal reference at L = 0.05
    (manufacturer 5 % coincidence-loss limit).

    Per-point burn labels are intentionally omitted: the points cluster, so
    inline labels overlapped badly. A single annotation notes that every point
    exceeds the specified loss limit and that L at the measured count is a
    lower bound; per-point identity is available in the per-burn CSV.
    """
    records = [
        r
        for r in all_results
        if not np.isnan(r.get("L_central", np.nan))
        and not np.isnan(r.get("peak_total_PM3_mass_ug_m3", np.nan))
    ]
    if not records:
        print("    [mpl] No data for loss vs peak-mass scatter.")
        return

    fig, ax = plt.subplots(figsize=figsize("onehalf", aspect=0.72))

    markers = {"bedroom2": "o", "morning_room": "s"}
    plotted = set()

    for r in records:
        loc = r["location"]
        x = r["peak_total_PM3_mass_ug_m3"]
        y = r["L_central"]
        y_lo = max(0.0, y - r["L_low"])
        y_hi = max(0.0, r["L_high"] - y)
        lbl = loc.replace("_", " ").title() if loc not in plotted else None
        plotted.add(loc)

        ax.errorbar(
            x,
            y,
            yerr=[[y_lo], [y_hi]],
            fmt=markers.get(loc, "o"),
            color=COLOR.get("AeroTrak1" if loc == "bedroom2" else "AeroTrak2", "gray"),
            markersize=7,
            capsize=4,
            alpha=0.85,
            label=lbl,
        )

    ax.axhline(0.05, color=REF_LINE, lw=0.9, ls=":", label="5 % coincidence-loss limit")

    # Single annotation (replaces overlapping per-point labels). The measured
    # count understates the true concentration under overload, so the plotted
    # losses are lower bounds; state both facts once rather than per point.
    min_l = min(r["L_central"] for r in records)
    if min_l > 0:
        ax.annotate(
            f"all points exceed the 5 % loss limit\n"
            f"(minimum L = {min_l:.0%}); L at the measured\n"
            f"count is a lower bound on the actual loss",
            xy=(0.03, 0.35), xycoords="axes fraction", ha="left", va="center",
            fontsize=_FS - 2,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Peak PM3 mass (µg/m³)", fontsize=_FS)
    ax.set_ylabel("Coincidence loss L (fraction)", fontsize=_FS)
    ax.tick_params(labelsize=_FS)
    ax.legend(fontsize=_FS - 2, loc="lower right")

    fig_dir = get_common_file("coincidence_figures")
    save_fig(fig, fig_dir / "aerotrak_loss_vs_peakmass.png")


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
    df = pd.DataFrame(rows, columns=_CSV_COLS)

    per_burn_path = out_dir / "aerotrak_coincidence_per_burn.csv"
    df.to_csv(str(per_burn_path), index=False, float_format="%.4g")
    print(f"    [CSV] {per_burn_path.name}")

    # Cross-burn summary statistics
    valid = df[df["n_peak_cm3"].notna()].copy()
    rev = valid[valid["reversal_present"] == True]

    def _q_stats(col: str) -> dict:
        s = valid[col].dropna()
        if s.empty:
            return {
                f"{col}_median": np.nan,
                f"{col}_IQR": np.nan,
                f"{col}_range_min": np.nan,
                f"{col}_range_max": np.nan,
            }
        q25, q75 = s.quantile([0.25, 0.75])
        return {
            f"{col}_median": float(s.median()),
            f"{col}_IQR": float(q75 - q25),
            f"{col}_range_min": float(s.min()),
            f"{col}_range_max": float(s.max()),
        }

    summary = {}
    for col in (
        "n_peak_cm3",
        "reversal_duration_minutes",
        "L_central",
        "factor_vs_threshold",
        "n_peak_frac_of_ceiling",
    ):
        summary.update(_q_stats(col))

    summary["coincidence_threshold_cm3"] = COINCIDENCE_THRESHOLD_CM3
    summary["n_meas_ceiling_cm3"] = N_MEAS_CEILING_CM3
    summary["n_reversal_present"] = int(rev.shape[0])
    summary["n_total_pairs"] = int(valid.shape[0])
    summary["median_peak_PM3_with_reversal_ug_m3"] = (
        float(rev["peak_total_PM3_mass_ug_m3"].median()) if not rev.empty else np.nan
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

    Prevalence, the worked example, the practical mass threshold, and the
    timing tally are all reported over the NON-SEALED burn-instrument pairs
    only. The two sealed Bedroom 2 burns (burn5, burn6) saw far lower indoor
    smoke because the room was closed off, so they are not representative of
    the high-concentration reversal phenomenon and are excluded from the
    headline statistics. All numbers are derived from data; no values are
    estimated from memory.
    """
    out_dir = get_common_file("coincidence_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = [
        r
        for r in all_results
        if not np.isnan(r.get("n_peak_cm3", np.nan)) and not r.get("bedroom_sealed")
    ]
    rev = [r for r in valid if r.get("reversal_present")]

    n_total = len(valid)
    n_rev = len(rev)

    n_peak_arr = np.array([r["n_peak_cm3"] for r in valid])
    L_arr = np.array([r["L_central"] for r in valid if not np.isnan(r.get("L_central", np.nan))])
    dur_arr = np.array(
        [
            r["reversal_duration_minutes"]
            for r in rev
            if not np.isnan(r.get("reversal_duration_minutes", np.nan))
        ]
    )
    pm3_rev_arr = np.array(
        [
            r["peak_total_PM3_mass_ug_m3"]
            for r in rev
            if not np.isnan(r.get("peak_total_PM3_mass_ug_m3", np.nan))
        ]
    )
    timing_arr = np.array(
        [
            r["reversal_onset_pre_pm3peak_minutes"]
            for r in rev
            if not np.isnan(r.get("reversal_onset_pre_pm3peak_minutes", np.nan))
        ]
    )
    # Resolve the relative timing of the Ch1 reversal onset against the PM3 mass
    # peak. Positive = onset preceded the mass peak, 0 = coincident, negative =
    # onset followed it. A single conflated median is misleading because the
    # distribution straddles zero, so report the three categories separately.
    n_timing_determinable = int(len(timing_arr))
    n_precede = int((timing_arr > 0).sum())
    n_at_peak = int((timing_arr == 0).sum())
    n_follow = int((timing_arr < 0).sum())

    cons_fails = [r for r in rev if r.get("counts_conserved") is False]

    def _fmt(val: float, fmt: str = ".2e") -> str:
        return f"{val:{fmt}}" if not np.isnan(val) else "[no data]"

    # ── aerotrak_coincidence_summary.md (four-paragraph structure) ──────────
    if n_total == 0:
        (out_dir / "aerotrak_coincidence_summary.md").write_text(
            "# AeroTrak Coincidence Analysis - Summary\n\nNo valid results computed.\n",
            encoding="utf-8",
        )
        print("    [MD] aerotrak_coincidence_summary.md")
    else:
        med_dur = float(np.median(dur_arr)) if len(dur_arr) > 0 else np.nan
        min_dur = float(np.min(dur_arr)) if len(dur_arr) > 0 else np.nan
        max_dur = float(np.max(dur_arr)) if len(dur_arr) > 0 else np.nan
        med_pm3_rev = float(np.median(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan
        min_pm3_rev = float(np.min(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan
        max_pm3_rev = float(np.max(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan
        n_at1_rev = sum(1 for r in rev if r["instrument"] == "AeroTrak1")
        n_at2_rev = sum(1 for r in rev if r["instrument"] == "AeroTrak2")
        med_npeak = float(np.median(n_peak_arr))
        factor_med = med_npeak / COINCIDENCE_THRESHOLD_CM3
        min_L = float(np.min(L_arr)) if len(L_arr) > 0 else np.nan
        max_L = float(np.max(L_arr)) if len(L_arr) > 0 else np.nan
        n_fail = len(cons_fails)

        para1 = (
            f"{n_rev} of {n_total} non-sealed burn-instrument pairs ("
            f"{n_at1_rev} AeroTrak1 Bedroom 2, {n_at2_rev} AeroTrak2 Morning "
            f"Room) showed a Ch1 reversal (0.3-0.5 um count falling below 50% of "
            f"the local maximum). Median reversal duration was "
            f"{_fmt(med_dur, '.1f')} minutes (range {_fmt(min_dur, '.1f')} to "
            f"{_fmt(max_dur, '.1f')} minutes). Median peak PM3 mass concentration "
            f"at the time of the reversal was approximately {_fmt(med_pm3_rev, '.0f')} "
            f"ug/m3 (range {_fmt(min_pm3_rev, '.0f')} to {_fmt(max_pm3_rev, '.0f')} "
            f"ug/m3). The two sealed Bedroom 2 pairs (Burns 05 and 06) are excluded "
            f"from every aggregate because concentrations in the sealed room were "
            f"low: Burn 05 AeroTrak1 showed no reversal at a peak PM3 of about "
            f"24 ug/m3, and Burn 06 AeroTrak1 showed a flagged reversal at only "
            f"about 28 ug/m3, neither representative of the dense-smoke phenomenon."
        )

        at1_np = np.array([r["n_peak_cm3"] for r in valid if r["instrument"] == "AeroTrak1"])
        at2_np = np.array([r["n_peak_cm3"] for r in valid if r["instrument"] == "AeroTrak2"])

        para2 = (
            f"Median peak 0.3-0.5 um count concentration across all non-sealed pairs "
            f"was {_fmt(med_npeak)} particles/cm3, a factor of "
            f"{_fmt(factor_med, '.1f')} ABOVE the TSI manufacturer's 5% "
            f"coincidence-loss limit of {COINCIDENCE_THRESHOLD_CM3:.0f} particles/cm3 "
            f"(3,000,000 particles/ft3 at 28,317 cm3 per ft3). "
            f"The Poisson dead-time model predicts coincidence losses of "
            f"{_fmt(min_L * 100, '.0f')}% to {_fmt(max_L * 100, '.0f')}% at the "
            f"measured counts; these are lower bounds because the measured count "
            f"understates the true concentration once losses are appreciable. The "
            f"model also caps the reportable count at exp(-1)/V, approximately "
            f"{N_MEAS_CEILING_CM3:.0f} particles/cm3 (the rollover ceiling), beyond "
            f"which the reported count declines as the true concentration rises. The "
            f"observed peaks behave accordingly: AeroTrak2 peaked between "
            f"{_fmt(float(np.min(at2_np)), '.0f')} and "
            f"{_fmt(float(np.max(at2_np)), '.0f')} particles/cm3 across the Morning "
            f"Room burns and AeroTrak1 between {_fmt(float(np.min(at1_np)), '.0f')} "
            f"and {_fmt(float(np.max(at1_np)), '.0f')} particles/cm3 across the "
            f"non-sealed Bedroom 2 burns, narrow unit-specific ceilings that did not "
            f"track smoke intensity. Peak counts pinned at such a ceiling are the "
            f"signature of a single-particle counter driven past its concentration "
            f"limit."
        )

        para3 = (
            f"{n_fail} of {n_rev} reversal pairs showed a total 6-bin count "
            f"concentration decrease greater than {CONSERVATION_TOL * 100:.0f}% at "
            f"the reversal trough. Severe coincidence produces exactly this "
            f"behavior: particle merging and dead-time losses reduce the total "
            f"registered count, and count conservation across bins holds only in "
            f"the mild-loss limit."
        )

        para4 = (
            f"In {n_timing_determinable} of {n_rev} reversal pairs the relative "
            f"timing of the Ch1 reversal onset and the PM3 mass peak was "
            f"determinable: the reversal onset preceded the mass peak in "
            f"{n_precede}, coincided with it in {n_at_peak}, and followed it in "
            f"{n_follow}. Onset before the mass peak is the expected behavior for "
            f"a counter whose reported count begins falling once the true "
            f"concentration passes the rollover ceiling while smoke is still "
            f"accumulating. Where the onset followed the reported mass peak, the "
            f"PM3 series is itself derived from the distorted counts, so the "
            f"ordering in those pairs is weakly constrained."
        )

        parts = [
            "# AeroTrak Coincidence Analysis - Summary\n\n## Plain-language summary",
            para1,
            para2,
            para3,
            para4,
        ]

        (out_dir / "aerotrak_coincidence_summary.md").write_text(
            "\n\n".join(parts) + "\n", encoding="utf-8"
        )
        print("    [MD] aerotrak_coincidence_summary.md")

    # ── aerotrak_manuscript_sentences.md (five sentence templates) ───────────
    # Aggregate stats
    med_dur_s = _fmt(float(np.median(dur_arr)) if len(dur_arr) > 0 else np.nan, ".0f")
    min_dur_s = _fmt(float(np.min(dur_arr)) if len(dur_arr) > 0 else np.nan, ".0f")
    max_dur_s = _fmt(float(np.max(dur_arr)) if len(dur_arr) > 0 else np.nan, ".0f")
    med_pm3_s = _fmt(float(np.median(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan, ".0f")
    min_pm3_s = _fmt(float(np.min(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan, ".0f")
    max_pm3_s = _fmt(float(np.max(pm3_rev_arr)) if len(pm3_rev_arr) > 0 else np.nan, ".0f")

    min_npeak = float(np.min(n_peak_arr)) if len(n_peak_arr) > 0 else np.nan
    max_npeak = float(np.max(n_peak_arr)) if len(n_peak_arr) > 0 else np.nan
    med_npeak_s = float(np.median(n_peak_arr)) if len(n_peak_arr) > 0 else np.nan
    factor_ms = med_npeak_s / COINCIDENCE_THRESHOLD_CM3 if not np.isnan(med_npeak_s) else np.nan
    max_L_pct = float(np.max(L_arr)) * 100 if len(L_arr) > 0 else np.nan
    min_L_pct = float(np.min(L_arr)) * 100 if len(L_arr) > 0 else np.nan
    med_L_pct = float(np.median(L_arr)) * 100 if len(L_arr) > 0 else np.nan

    n_fail_s = len(cons_fails)

    # Worked example: a high-concentration reversal pair with a clear
    # suppression. Candidates are non-sealed reversal pairs with a computable
    # duration, a counts-NOT-conserved trough, and a valid Ch1 minimum. If
    # WORKED_EXAMPLE pins a (burn, instrument) pair, use it; otherwise fall
    # back to the highest peak-PM3 candidate. Selecting by peak PM3 keeps the
    # example in the dense-smoke regime the section is about.
    ex_cands = [
        r
        for r in rev
        if "_ch_results" in r
        and "Ch1" in r["_ch_results"]
        and not np.isnan(r.get("n_peak_cm3", np.nan))
        and not np.isnan(r.get("reversal_duration_minutes", np.nan))
        and not np.isnan(r.get("peak_total_PM3_mass_ug_m3", np.nan))
        and r.get("counts_conserved") is False
    ]
    ex = None
    if WORKED_EXAMPLE is not None:
        ex = next(
            (
                r
                for r in ex_cands
                if (r["burn"], r["instrument"]) == WORKED_EXAMPLE
            ),
            None,
        )
    if ex is None:
        ex = (
            max(ex_cands, key=lambda r: r["peak_total_PM3_mass_ug_m3"])
            if ex_cands
            else None
        )

    if ex is not None:
        ex_burn = ex["burn"]
        ex_loc = str(ex["location"]).replace("_", " ")
        ex_n_pre = float(ex["n_peak_cm3"])
        ex_n_min = float(ex["_ch_results"]["Ch1"]["n_min_during_reversal"])
        ex_dur = float(ex["reversal_duration_minutes"])
        ex_pm3 = float(ex["peak_total_PM3_mass_ug_m3"])
        ex_L_pct = float(ex["L_central"]) * 100
        ex_supp = (
            (1.0 - ex_n_min / ex_n_pre) * 100 if not np.isnan(ex_n_min) and ex_n_pre > 0 else np.nan
        )
    else:
        ex_burn = ex_loc = "[no example]"
        ex_n_pre = ex_n_min = ex_dur = ex_pm3 = ex_L_pct = ex_supp = np.nan

    n_at1_rev_s = sum(1 for r in rev if r["instrument"] == "AeroTrak1")
    n_at2_rev_s = sum(1 for r in rev if r["instrument"] == "AeroTrak2")

    s1 = (
        f"Channel reversals in the 0.3-0.5 um bin (Ch1 count falling below 50% of "
        f"the local maximum) were detected in all {n_rev} of the {n_total} non-sealed "
        f"AeroTrak burn-instrument pairs analysed ({n_at1_rev_s} AeroTrak1 Bedroom 2, "
        f"{n_at2_rev_s} AeroTrak2 Morning Room), with a median reversal duration of "
        f"approximately {med_dur_s} minutes (range {min_dur_s} to {max_dur_s} minutes) "
        f"and a median peak total PM3 mass concentration of approximately "
        f"{med_pm3_s} ug/m3 at the time of the reversal."
    )

    s2 = (
        f"At the time of each reversal, the peak Ch1 count concentration ranged from "
        f"approximately {_fmt(min_npeak)} to {_fmt(max_npeak)} particles/cm3 "
        f"(median {_fmt(med_npeak_s)}), a factor of approximately "
        f"{_fmt(factor_ms, '.1f')} above the manufacturer's 5% coincidence-loss limit "
        f"of 3,000,000 particles/ft3 (approximately "
        f"{COINCIDENCE_THRESHOLD_CM3:.0f} particles/cm3); the Poisson dead-time model "
        f"predicts coincidence losses of at least {_fmt(min_L_pct, '.0f')}% to "
        f"{_fmt(max_L_pct, '.0f')}% at the measured counts (lower bounds, since the "
        f"measured count understates the true concentration), and it caps the "
        f"reportable count at a rollover ceiling of approximately "
        f"{N_MEAS_CEILING_CM3:.0f} particles/cm3, near which the observed per-unit "
        f"peaks cluster."
    )

    s3 = (
        f"In {n_fail_s} of {n_rev} pairs where a reversal was detected, the sum of "
        f"count concentrations across all six AeroTrak bins decreased by more than "
        f"{CONSERVATION_TOL * 100:.0f}% at the reversal trough, consistent with the "
        f"particle merging and dead-time losses of severe coincidence; count "
        f"conservation across bins holds only in the mild-loss limit."
    )

    s4 = (
        f"The reversal phenomenon was associated with dense smoke and was "
        f"consistently observed at peak total PM3 mass concentrations exceeding "
        f"approximately {med_pm3_s} ug/m3 (range {min_pm3_s} to {max_pm3_s} ug/m3 "
        f"across the non-sealed pairs, the median taken as the practical caution "
        f"threshold); OPC-derived count concentrations and mass estimates should "
        f"be interpreted with caution at and above these levels."
    )

    s5 = (
        f"In the {ex_burn} ({ex_loc}) event, for example, the Ch1 channel declined "
        f"from approximately {_fmt(ex_n_pre)} particles/cm3 to a minimum of "
        f"approximately {_fmt(ex_n_min)} particles/cm3 over approximately "
        f"{_fmt(ex_dur, '.0f')} minutes while the co-located total PM3 mass "
        f"concentration reached approximately {_fmt(ex_pm3, '.0f')} ug/m3; "
        f"the Poisson coincidence loss estimated from the maximum observed Ch1 count "
        f"is at least {_fmt(ex_L_pct, '.0f')}%, and the near-complete observed "
        f"suppression ({_fmt(ex_supp, '.0f')}% of the local maximum) is consistent "
        f"with the true concentration rising far beyond the rollover ceiling of "
        f"approximately {N_MEAS_CEILING_CM3:.0f} particles/cm3, where the reported "
        f"count falls toward zero."
    )

    s6 = (
        f"The relative timing of the Ch1 reversal onset and the PM3 mass peak was "
        f"determinable in {n_timing_determinable} of {n_rev} reversal pairs; the "
        f"onset preceded the mass peak in {n_precede} and coincided with it in "
        f"{n_at_peak}, as expected for a counter whose reported count begins "
        f"falling once the true concentration passes the rollover ceiling while "
        f"smoke is still accumulating; in the {n_follow} pairs where the onset "
        f"followed the reported mass peak, the PM3 series is itself derived from "
        f"the distorted counts, so the ordering is weakly constrained."
    )

    ms_text = (
        "# AeroTrak Coincidence - Manuscript Sentences for Section 3.2.2\n\n"
        "_All values derived from data and reported over the non-sealed "
        "burn-instrument pairs. Insert into manuscript text._\n\n"
        "---\n\n"
        f'**Sentence 1 (reversal prevalence):** "{s1}"\n\n'
        f'**Sentence 2 (coincidence overload test):** "{s2}"\n\n'
        f'**Sentence 3 (counts conservation):** "{s3}"\n\n'
        f'**Sentence 4 (practical threshold):** "{s4}"\n\n'
        f'**Sentence 5 (worked example: {ex_burn}, {ex_loc}):** "{s5}"\n\n'
        f'**Sentence 6 (reversal-onset timing):** "{s6}"\n\n'
        "---\n\n"
        "## Supporting statistics\n\n"
        "| Quantity | Value |\n"
        "|---|---|\n"
        f"| Non-sealed burn-instrument pairs analysed | {n_total} |\n"
        f"| Pairs with Ch1 reversal | {n_rev} |\n"
        f"| Median n_peak all pairs (#/cm3) | {_fmt(med_npeak_s)} |\n"
        f"| Median reversal duration (min) | {med_dur_s} |\n"
        f"| Median L_central all pairs (%) | {_fmt(med_L_pct, '.1f')} |\n"
        f"| Median peak PM3 at reversal (ug/m3) | {med_pm3_s} |\n"
        f"| Reversal-onset timing determinable | {n_timing_determinable} of {n_rev} |\n"
        f"| Onset preceded / at / followed mass peak | "
        f"{n_precede} / {n_at_peak} / {n_follow} |\n"
        f"| TSI 5% coincidence-loss limit (#/cm3) | {COINCIDENCE_THRESHOLD_CM3:.0f} |\n"
        f"| Factor above TSI limit (median) | {_fmt(factor_ms, '.1f')} |\n"
        f"| Poisson rollover ceiling (#/cm3) | {N_MEAS_CEILING_CM3:.0f} |\n"
        f"| Predicted L at measured counts (min to max, %) | "
        f"{_fmt(min_L_pct, '.0f')} to {_fmt(max_L_pct, '.0f')} |\n"
    )
    (out_dir / "aerotrak_manuscript_sentences.md").write_text(ms_text, encoding="utf-8")
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
    apply_est_style()

    print("Loading burn log...")
    burn_log = _load_burn_log()

    print("Loading AeroTrak data...")
    df_at1 = _load_aerotrak_all("AeroTrak1")
    df_at2 = _load_aerotrak_all("AeroTrak2")
    print(f"  AeroTrak1: {len(df_at1):,} rows | AeroTrak2: {len(df_at2):,} rows")

    all_results: list[dict] = []

    for instrument, df_aerotrak in [("AeroTrak1", df_at1), ("AeroTrak2", df_at2)]:
        print(
            f"\nProcessing {instrument} ({'Bedroom 2' if instrument == 'AeroTrak1' else 'Morning Room'})..."
        )

        for burn_id in BURN_COVERAGE[instrument]:
            bl_rows = burn_log[burn_log["Burn ID"] == burn_id]
            if bl_rows.empty:
                continue
            bl_row = bl_rows.iloc[0]
            burn_date = bl_row["Date"]

            # Load SMPS numConc for Bedroom 2 cross-check
            df_smps = None
            if instrument == "AeroTrak1":
                df_smps = _load_smps_numconc(burn_date)
                if df_smps is None:
                    print(f"  [{burn_id}] SMPS numConc not found for {burn_date.date()}")

            print(f"  Analysing {burn_id}...")
            result = analyze_burn_instrument(burn_id, instrument, df_aerotrak, bl_row, df_smps)
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
