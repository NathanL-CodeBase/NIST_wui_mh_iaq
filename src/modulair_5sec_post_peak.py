"""
MODULAIR-PM 5 s post-peak OPC-N3 small-bin inversion analysis (SI for 3.2.3).

During the post-peak decay phase the smallest OPC-N3 bins of the MODULAIR-PM
units increase in count rather than decaying with the smoke. This is distinct
from the peak-window suppression characterized in modulair_5sec_peak_window.py
(Section 3.2.3). This script quantifies the inversion per burn-unit pair and
contrasts the OPC-N3 bin-0 decay trajectory against the co-located AeroTrak Ch1
count and (Bedroom 2 only) the SMPS 300-437 nm number concentration, both of
which decay monotonically.

The reference time t_ref is garage-door closure from the burn log, the same
x origin as the Burn 09 comparison figure. The decay window runs from t_ref to
ignition + 4 h or PM3 return to baseline, whichever is earlier (consistent with
the Prompt 1 / coincidence definition). The nephelometer saturation flag is
still read from modulair_5sec_peak_per_burn.csv.

Two post-peak checks feed the manuscript text:
    - Morning Room and Bedroom 2: ratio of portal PM2.5 to co-located AeroTrak
      PM3 at t_ref + 0.5 h and t_ref + 2 h, and its fractional change;
    - Bedroom 2: OPC-N3 bin 0 over the SMPS 300-437 nm number concentration at
      t_ref + 2 h (absolute ratio, same units).

Scope: every record with a garage-closure time is tested, so whether the
inversion occurs WITHOUT prior nephelometer saturation can be reported.
Headline counts are also reported split by saturation state.

The script stays descriptive: it does not propose a mechanism and does not cite
any OPC-N3 coincidence threshold (no published value exists for the OPC-N3).

All paths resolve through data_config.json via src/modulair_5sec_io.py. No raw
5 s data are written or committed. Missing data are flagged, not imputed.

Outputs (under common_folders quantaq_analysis / quantaq_figures):
    modulair_5sec_post_peak_per_burn.csv
    modulair_5sec_post_peak_cross_burn_summary.csv
    modulair_5sec_post_peak_<burn>_<unit>.html       (Bokeh, one per pair)
    YYYY-MM-DD_modulair_5sec_post_peak_morning_room.pdf/.png   (SI)
    YYYY-MM-DD_modulair_5sec_post_peak_bedroom_smps.pdf/.png   (SI)
    modulair_5sec_post_peak_summary.md

Usage (from the repository root, after modulair_5sec_peak_window.py):
    python src/modulair_5sec_post_peak.py
    python src/modulair_5sec_post_peak.py --overwrite

Author: Nathan Lima
Created: 2026-06-26
Update log:
    2026-10-28: Reviewer feedback on the v4 manuscript (peak window unclear).
        Reference time changed from t_peak_end to garage-door closure (t_ref)
        for all post-peak statistics and figures. Added the portal PM2.5 to
        AeroTrak PM3 ratio at t_ref + 0.5 h and + 2 h and the absolute OPC-N3
        bin 0 to SMPS ratio at t_ref + 2 h. The two SI figures drop the
        peak-window shading, add a PAC-on line, and save as ISO-dated PDF and
        600 dpi PNG. The legacy normalized small-multiples, overlay, and +2 h
        bias bar figures are no longer regenerated. Added --overwrite.
"""

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from bokeh.io import output_file, reset_output, save  # noqa: E402
from bokeh.layouts import column as bokeh_column  # noqa: E402
from bokeh.models import BoxAnnotation, Legend, LegendItem, Span  # noqa: E402
from bokeh.plotting import figure  # noqa: E402
from scipy import stats  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.aerotrak_coincidence import (  # noqa: E402
    _load_aerotrak_all,
    _load_smps_numconc,
    _smps_300_437,
)
from src.data_paths import get_common_file  # noqa: E402
from src.fig_style import (  # noqa: E402
    ROLE_COLORS,
    SHADE,
    UNIT_COLORS,
    apply_est_style,
    figsize,
)
from src.modulair_5sec_io import (  # noqa: E402
    BURN_DATES,
    UNIT_CONFIG,
    load_5sec_burn,
    load_event_times,
    load_portal_burn,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

BURNS = list(BURN_DATES.keys())  # burn4..burn10
UNITS = list(UNIT_CONFIG.keys())  # MODULAIR-PM1, MODULAIR-PM2
UNIT_LOCATION = {u: UNIT_CONFIG[u]["location"] for u in UNITS}

# Location -> AeroTrak instrument label used by the coincidence loader.
LOC_AEROTRAK = {"bedroom2": "AeroTrak1", "morning_room": "AeroTrak2"}

# Three smallest OPC-N3 bins (edges confirmed against the raw file / io module).
OPC_SMALL_BINS = ["bin0", "bin1", "bin2"]
BIN0_LABEL = "0.35-0.46 um"
BIN1_LABEL = "0.46-0.66 um"
BIN2_LABEL = "0.66-1.0 um"

# Reference time points after t_ref (garage closure, hours) for the trajectory comparison.
REF_HOURS = [0.5, 1.0, 2.0, 4.0]
# A subset used for the inversion-presence test (per the prompt: +0.5/1/2 h).
INVERSION_TEST_HOURS = [0.5, 1.0, 2.0]

# Inversion classification: any bin count at a test hour exceeds the t_ref
# value by more than this fraction.
INVERSION_FRAC = 0.50  # > 50% increase
# Return-to-baseline persistence for the inversion-end / duration test.
RETURN_PERSIST_MIN = 15.0  # must stay below t_ref value this long (min)

# Minimum t_ref count for a ratio (peak_inversion_ratio,
# opc_ratio_vs_aerotrak) to be reported. Some OPC-N3 small-bin counts at
# t_ref sit near zero (e.g. burn10 MODULAIR-PM2 bin0 ~0.01 counts), so a
# ratio against them divides by a tiny denominator and explodes to implausible
# values (thousands-fold). inversion_present stays driven by the absolute
# +0.5/1/2 h elevation; the ratios are returned as NaN ("no_data") below this
# floor. Matches the MIN_BASELINE_FOR_RATIO=0.5 convention in
# modulair_5sec_peak_window.py.
MIN_COUNT_FOR_RATIO = 0.5

# Analysis-window definition (matches Prompt 1 / coincidence definition).
MAX_WIN_HR = 4.0
BASELINE_MIN = 30  # pre-burn baseline window (minutes before ignition)

# Local sampling half-width: when reading a 5 s bin count "at" a reference time,
# average the samples within this many seconds of the target to suppress the
# single-sample 5 s noise without smearing the trajectory.
SAMPLE_HALF_WIDTH_S = 30.0

# Input CSV from the peak-window analysis (carries the saturated flag).
PEAK_CSV = "modulair_5sec_peak_per_burn.csv"

# Portal PM2.5 to AeroTrak PM3 ratio check: hours after t_ref and the largest
# fractional change counted as "stable" in the summary.
PM_RATIO_HOURS = (0.5, 2.0)
PM_RATIO_STABLE_FRAC = 0.25

# Event-line styles shared with the Burn 09 comparison figure. Drawn in grey
# here because the AeroTrak traces in these figures are black dashed.
GARAGE_LS = "-"
PAC_LS = "--"
EVENT_COLOR = SHADE
REF_LABEL = "garage closure"

# SMPS 300-437 nm is the OPC-N3 bin-0/-1 overlapping size range proxy for the
# PM2.5-bias cross-check (Bedroom 2 only).
SMPS_LO_NM, SMPS_HI_NM = 300.0, 437.0

# Numeric point size for matplotlib calls (matches src.fig_style BASE_FONT_PT).
_FS = 12

# Per-unit and reference-instrument colors from the shared colorblind-safe
# palette (bedroom blue, morning room vermillion, AeroTrak black, SMPS orange).
UNIT_COLOR = dict(UNIT_COLORS)
AEROTRAK_COLOR = ROLE_COLORS["AeroTrak"]
SMPS_COLOR = ROLE_COLORS["SMPS"]


# ==============================================================================
# INPUT LOADERS
# ==============================================================================


def _load_peak_windows() -> pd.DataFrame:
    """
    Load the peak-window per-burn CSV and return the t_peak_end / saturated /
    plateau fields keyed by (burn, unit).

    Returns
    -------
    pd.DataFrame
        Columns 'burn', 'unit', 'location', 't_peak_end' (pd.Timestamp/NaT),
        'saturated' (bool), 'peak_window_method' (str). Only rows with
        data_present.
    """
    path = get_common_file("quantaq_analysis") / PEAK_CSV
    df = pd.read_csv(path)
    df = df[df["data_present"] == True].copy()  # noqa: E712
    df["t_peak_end"] = pd.to_datetime(df["t_peak_end"], errors="coerce")
    df["saturated"] = df["saturated"].astype(bool)
    keep = ["burn", "unit", "location", "t_peak_end", "saturated",
            "peak_window_method"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _peak_window_lookup(pw: pd.DataFrame, burn_id: str, unit: str) -> dict:
    """Return the peak-window row for one pair as a dict (NaT/False if absent)."""
    row = pw[(pw["burn"] == burn_id) & (pw["unit"] == unit)]
    if row.empty:
        return dict(t_peak_end=pd.NaT, saturated=False,
                    peak_window_method="none")
    r = row.iloc[0]
    return dict(
        t_peak_end=r["t_peak_end"],
        saturated=bool(r["saturated"]),
        peak_window_method=str(r.get("peak_window_method", "none")),
    )


def _load_aerotrak_peak_mass() -> pd.DataFrame:
    """
    Load peak_total_PM3_mass_ug_m3 from the coincidence per-burn CSV keyed by
    (burn, location), for the Spearman peak-mass vs inversion-duration test.
    """
    path = get_common_file("coincidence_analysis") / "aerotrak_coincidence_per_burn.csv"
    df = pd.read_csv(path)
    keep = ["burn", "location", "peak_total_PM3_mass_ug_m3"]
    return df[[c for c in keep if c in df.columns]].copy()


# ==============================================================================
# WINDOW + SAMPLING HELPERS
# ==============================================================================


def _decay_window_end(
    df_at: pd.DataFrame | None,
    burn_date: pd.Timestamp,
    ignition: pd.Timestamp,
) -> pd.Timestamp:
    """
    End of the post-peak decay window: ignition + 4 h, or the AeroTrak PM3
    return to within 10% of the pre-burn baseline (whichever is earlier),
    matching the Prompt 1 analysis-window definition.

    The AeroTrak PM3 series is used for the baseline-return test because the
    MODULAIR-PM portal PM2.5 is itself derived from the OPC-N3 being
    characterized here and must not define the reference window. When the
    AeroTrak day record is unavailable, the hard ignition + 4 h cap is used.
    """
    if pd.isna(ignition):
        return pd.NaT
    hard_end = ignition + pd.Timedelta(hours=MAX_WIN_HR)
    if df_at is None or "PM3 (µg/m³)" not in df_at.columns:
        return hard_end

    day = df_at[df_at["Date and Time"].dt.date == burn_date.date()]
    pm3 = "PM3 (µg/m³)"
    pre = day[
        (day["Date and Time"] >= ignition - pd.Timedelta(minutes=BASELINE_MIN))
        & (day["Date and Time"] < ignition)
        & day[pm3].notna()
    ]
    baseline = float(pre[pm3].mean()) if not pre.empty else np.nan
    if not np.isfinite(baseline):
        return hard_end

    thresh = baseline * 1.10
    in_win = day[
        (day["Date and Time"] > ignition)
        & (day["Date and Time"] <= hard_end)
        & day[pm3].notna()
    ]
    if in_win.empty:
        return hard_end
    peak_idx = in_win[pm3].idxmax()
    post = in_win.loc[peak_idx:]
    rec = post[post[pm3] <= thresh]
    return min(rec["Date and Time"].iloc[0], hard_end) if not rec.empty else hard_end


def _value_at(ts: pd.Series, vals: pd.Series, target: pd.Timestamp) -> float:
    """
    Mean of the samples within SAMPLE_HALF_WIDTH_S of target (5 s noise
    suppression). Returns NaN if no sample lies within that half-width.
    """
    if pd.isna(target):
        return np.nan
    lo = target - pd.Timedelta(seconds=SAMPLE_HALF_WIDTH_S)
    hi = target + pd.Timedelta(seconds=SAMPLE_HALF_WIDTH_S)
    m = (ts >= lo) & (ts <= hi)
    if not m.any():
        return np.nan
    v = pd.to_numeric(vals[m], errors="coerce").dropna()
    return float(v.mean()) if not v.empty else np.nan


# ==============================================================================
# OPC-N3 INVERSION DETECTION
# ==============================================================================


def _bin_inversion(
    ts: pd.Series,
    vals: pd.Series,
    t0: pd.Timestamp,
    t_end: pd.Timestamp,
) -> dict:
    """
    Characterize the post-peak inversion for one OPC-N3 bin over [t0, t_end].

    t0 is t_ref (garage-door closure). The bin count at t0 is the reference;
    inversion is a sustained rise above it during decay.

    Returns
    -------
    dict
        count_t0, counts_at_<h>h for each REF_HOURS point, inversion_present
        (bool, any INVERSION_TEST_HOURS point > (1+INVERSION_FRAC)*count_t0),
        peak_inversion_ratio, t_inversion_peak (pd.Timestamp/NaT),
        inversion_duration_hours.
    """
    out = {
        "count_t0": np.nan,
        "inversion_present": False,
        "peak_inversion_ratio": np.nan,
        "t_inversion_peak": pd.NaT,
        "inversion_duration_hours": np.nan,
    }
    for h in REF_HOURS:
        out[f"count_{h}h"] = np.nan
    if pd.isna(t0) or pd.isna(t_end) or t_end <= t0:
        return out

    count_t0 = _value_at(ts, vals, t0)
    out["count_t0"] = count_t0

    for h in REF_HOURS:
        tref = t0 + pd.Timedelta(hours=h)
        if tref > t_end:
            continue
        out[f"count_{h}h"] = _value_at(ts, vals, tref)

    # Inversion presence: any +0.5/1/2 h point exceeds count_t0 by > 50%.
    if not np.isfinite(count_t0) or count_t0 <= 0:
        return out
    thresh = (1.0 + INVERSION_FRAC) * count_t0
    test_vals = [out[f"count_{h}h"] for h in INVERSION_TEST_HOURS]
    present = any(np.isfinite(v) and v > thresh for v in test_vals)
    out["inversion_present"] = bool(present)
    if not present:
        return out

    # Window samples for the peak-ratio and duration measurements.
    m = (ts >= t0) & (ts <= t_end)
    wts = ts[m].reset_index(drop=True)
    wv = pd.to_numeric(vals[m], errors="coerce").reset_index(drop=True)
    if wv.dropna().empty:
        return out

    # Peak inversion: max bin count over the decay window vs count_t0. The
    # ratio is only meaningful when count_t0 clears MIN_COUNT_FOR_RATIO; below
    # that floor the tiny denominator makes the ratio explode, so it is left NaN
    # (inversion_present and duration are unaffected, being based on absolute
    # elevation above count_t0).
    peak_i = int(wv.idxmax())
    peak_val = float(wv.iloc[peak_i])
    if count_t0 >= MIN_COUNT_FOR_RATIO:
        out["peak_inversion_ratio"] = peak_val / count_t0
    out["t_inversion_peak"] = pd.Timestamp(wts.iloc[peak_i])

    # Duration: from t0 to the first time the count returns below count_t0 and
    # stays below for at least RETURN_PERSIST_MIN consecutive minutes.
    arr = wv.to_numpy(dtype=float)
    tarr = wts.to_numpy()
    n = len(arr)
    persist = pd.Timedelta(minutes=RETURN_PERSIST_MIN)
    end_t = pd.NaT
    for i in range(peak_i, n):
        if not (np.isfinite(arr[i]) and arr[i] < count_t0):
            continue
        # Check the run stays below count_t0 for at least persist.
        run_start = pd.Timestamp(tarr[i])
        ok = True
        j = i
        while j < n and (pd.Timestamp(tarr[j]) - run_start) < persist:
            if np.isfinite(arr[j]) and arr[j] >= count_t0:
                ok = False
                break
            j += 1
        # Require the run to actually span the persistence window.
        spanned = j < n and (pd.Timestamp(tarr[j - 1]) - run_start) >= persist
        if ok and (spanned or j >= n):
            end_t = run_start
            break
    if pd.notna(end_t):
        out["inversion_duration_hours"] = (end_t - t0).total_seconds() / 3600.0
    else:
        # Never returned and held below count_t0 within the window: report the
        # full window length as a right-censored lower bound on the duration.
        out["inversion_duration_hours"] = (t_end - t0).total_seconds() / 3600.0
    return out


# ==============================================================================
# AEROTRAK Ch1 AND SMPS DECAY TRAJECTORIES
# ==============================================================================


def _aerotrak_ch1_series(df_at: pd.DataFrame, burn_date: pd.Timestamp) -> tuple:
    """
    Return (timestamps, Ch1 0.3-0.5 um count concentration) for the burn day.

    The count-concentration column is built by _load_aerotrak_all as
    'Ʃ0.3-0.5µm (#/cm³)'; match it by substring to be robust to the exact
    cut-point formatting.
    """
    day = df_at[df_at["Date and Time"].dt.date == burn_date.date()]
    ch1 = next((c for c in day.columns if "Ʃ0.3-0.5" in str(c)), None)
    if ch1 is None or day.empty:
        return None, None
    return day["Date and Time"], day[ch1]


def _nearest_value(ts: pd.Series, vals: pd.Series, target: pd.Timestamp,
                   half_s: float = 180.0) -> float:
    """
    Value of a 1-min series at the sample nearest target.

    Returns NaN when no sample lies within half_s seconds or the value is not
    numeric (portal QA/QC-removed minutes stay NaN; nothing is interpolated).
    """
    if ts is None or vals is None or pd.isna(target) or len(ts) == 0:
        return np.nan
    dt = (ts - target).abs()
    idx = dt.idxmin()
    if dt[idx] > pd.Timedelta(seconds=half_s):
        return np.nan
    v = pd.to_numeric(pd.Series([vals[idx]]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else np.nan


def _trajectory_at(ts, vals, t0: pd.Timestamp) -> dict:
    """
    Normalized trajectory of a reference series relative to its value at t0.

    Reads the series value at t0 and at each REF_HOURS point (3-min half-width
    for the coarser-cadence AeroTrak/SMPS), returning count and ratio-to-t0.
    """
    out = {"ref_t0": np.nan}
    for h in REF_HOURS:
        out[f"ref_{h}h"] = np.nan
        out[f"ref_ratio_{h}h"] = np.nan
    if ts is None or vals is None or pd.isna(t0):
        return out

    def _nearest(target: pd.Timestamp, half_s: float = 180.0) -> float:
        dt = (ts - target).abs()
        idx = dt.idxmin()
        if dt[idx] > pd.Timedelta(seconds=half_s):
            return np.nan
        return float(pd.to_numeric(pd.Series([vals[idx]]), errors="coerce").iloc[0])

    ref0 = _nearest(t0)
    out["ref_t0"] = ref0
    for h in REF_HOURS:
        v = _nearest(t0 + pd.Timedelta(hours=h))
        out[f"ref_{h}h"] = v
        out[f"ref_ratio_{h}h"] = (
            v / ref0 if (np.isfinite(v) and np.isfinite(ref0) and ref0 > 0) else np.nan
        )
    return out


def _smps_300_437_series(df_smps: pd.DataFrame | None) -> tuple:
    """Return (timestamps, summed 300-437 nm number concentration) or (None, None)."""
    if df_smps is None:
        return None, None
    conc = _smps_300_437(df_smps)
    return df_smps["datetime"], conc


# ==============================================================================
# PER-PAIR DRIVER
# ==============================================================================


def analyze_pair(
    burn_id: str,
    unit: str,
    pw: pd.DataFrame,
    events: pd.DataFrame,
    df_at_by_loc: dict,
    smps_cache: dict,
) -> dict:
    """
    Run the post-peak inversion analysis for one burn-unit pair.

    Returns a dict of CSV fields plus private '_' keys (loaded frames, event
    times, trajectories) used by the figure functions. A pair with no 5 s data
    or no garage-closure time is flagged and skipped, never imputed.
    """
    location = UNIT_LOCATION[unit]
    pwin = _peak_window_lookup(pw, burn_id, unit)
    saturated = pwin["saturated"]

    ev = events.loc[burn_id] if burn_id in events.index else None
    ignition = ev["ignition"] if ev is not None else pd.NaT
    garage = ev["garage_closed"] if ev is not None else pd.NaT
    pac_on = ev["pac_on"] if ev is not None else pd.NaT
    burn_date = pd.Timestamp(BURN_DATES[burn_id])
    # Reference time for every post-peak statistic: garage-door closure.
    t0 = garage

    base = dict(
        burn=burn_id, unit=unit, location=location, saturated=saturated,
        t_ref=t0,
        data_present=False, inversion_present=False, notes="",
        _df=None, _ignition=ignition, _garage=garage, _pac_on=pac_on,
        _t_end=pd.NaT, _at_ts=None, _at_vals=None, _smps_ts=None,
        _smps_vals=None,
    )

    df = load_5sec_burn(unit, burn_id)
    if df is None or df.empty:
        base["notes"] = "5 s data missing"
        print(f"    [{burn_id}|{unit}] no 5 s data - flagged, not imputed.")
        return base
    if pd.isna(t0):
        base["data_present"] = True
        base["_df"] = df
        base["notes"] = "no garage-closure time in burn log; inversion test skipped"
        print(f"    [{burn_id}|{unit}] no garage-closure time - skipped.")
        return base

    df_at = df_at_by_loc.get(location)
    t_end = _decay_window_end(df_at, burn_date, ignition)
    if pd.isna(t_end) or t_end <= t0:
        # Fall back to a fixed 4 h decay window anchored on t_ref.
        t_end = t0 + pd.Timedelta(hours=MAX_WIN_HR)

    ts = df["timestamp"]
    rec = dict(base)
    rec["data_present"] = True
    rec["_df"] = df
    rec["_t_end"] = t_end

    # Per-bin inversion characterization for the three smallest OPC-N3 bins.
    any_inversion = False
    for b in OPC_SMALL_BINS:
        res = _bin_inversion(ts, df[b], t0, t_end)
        any_inversion = any_inversion or res["inversion_present"]
        for k, v in res.items():
            rec[f"{b}_{k}"] = v
    rec["inversion_present"] = bool(any_inversion)

    # AeroTrak Ch1 trajectory over the same window.
    at_ts, at_vals = _aerotrak_ch1_series(df_at, burn_date) if df_at is not None else (None, None)
    rec["_at_ts"], rec["_at_vals"] = at_ts, at_vals
    at_traj = _trajectory_at(at_ts, at_vals, t0)
    for k, v in at_traj.items():
        rec[f"aerotrak_{k}"] = v

    # OPC bin-0 ratio vs AeroTrak Ch1 ratio at each reference hour. The bin-0
    # ratio is only reported when the bin-0 count at t_ref clears
    # MIN_COUNT_FOR_RATIO (a near-zero denominator otherwise explodes the ratio).
    bin0_t0 = rec.get("bin0_count_t0", np.nan)
    bin0_ok = np.isfinite(bin0_t0) and bin0_t0 >= MIN_COUNT_FOR_RATIO
    for h in REF_HOURS:
        opc_ratio = (
            rec.get(f"bin0_count_{h}h", np.nan) / bin0_t0
            if (bin0_ok and np.isfinite(rec.get(f"bin0_count_{h}h", np.nan)))
            else np.nan
        )
        at_ratio = rec.get(f"aerotrak_ref_ratio_{h}h", np.nan)
        rec[f"opc_bin0_ratio_{h}h"] = opc_ratio
        rec[f"aerotrak_ch1_ratio_{h}h"] = at_ratio
        rec[f"opc_ratio_vs_aerotrak_{h}h"] = (
            opc_ratio / at_ratio
            if (np.isfinite(opc_ratio) and np.isfinite(at_ratio) and at_ratio > 0)
            else np.nan
        )

    # SMPS 300-437 nm trajectory (Bedroom 2 only).
    smps_ts = smps_vals = None
    if location == "bedroom2":
        if burn_id not in smps_cache:
            smps_cache[burn_id] = _load_smps_numconc(burn_date)
        smps_ts, smps_vals = _smps_300_437_series(smps_cache[burn_id])
    rec["_smps_ts"], rec["_smps_vals"] = smps_ts, smps_vals
    smps_traj = _trajectory_at(smps_ts, smps_vals, t0)
    for k, v in smps_traj.items():
        rec[f"smps_{k}"] = v
    rec["smps_available"] = bool(smps_ts is not None)

    # Absolute OPC-N3 bin 0 over SMPS 300-437 nm (both particles/cm3), Bedroom 2.
    for h in REF_HOURS:
        b0 = rec.get(f"bin0_count_{h}h", np.nan)
        sm = rec.get(f"smps_ref_{h}h", np.nan)
        rec[f"bin0_over_smps_{h}h"] = (
            b0 / sm if (np.isfinite(b0) and np.isfinite(sm) and sm > 0) else np.nan
        )

    # Portal PM2.5 over co-located AeroTrak PM3 at t_ref + 0.5 h and + 2 h.
    portal = load_portal_burn(unit, burn_id)
    pm_ratios = {}
    for h in PM_RATIO_HOURS:
        tref = t0 + pd.Timedelta(hours=h)
        pm25 = _nearest_value(portal["timestamp"], portal["pm25"], tref) \
            if (portal is not None and "pm25" in portal.columns) else np.nan
        pm3 = np.nan
        if df_at is not None and "PM3 (µg/m³)" in df_at.columns:
            day = df_at[df_at["Date and Time"].dt.date == burn_date.date()]
            pm3 = _nearest_value(day["Date and Time"], day["PM3 (µg/m³)"], tref)
        ratio = pm25 / pm3 if (np.isfinite(pm25) and np.isfinite(pm3) and pm3 > 0) else np.nan
        rec[f"portal_pm25_{h}h"] = pm25
        rec[f"aerotrak_pm3_{h}h"] = pm3
        rec[f"pm25_over_pm3_{h}h"] = ratio
        pm_ratios[h] = ratio
    r_lo, r_hi = pm_ratios[PM_RATIO_HOURS[0]], pm_ratios[PM_RATIO_HOURS[1]]
    rec["pm25_over_pm3_change_frac"] = (
        (r_hi - r_lo) / r_lo if (np.isfinite(r_lo) and np.isfinite(r_hi) and r_lo > 0) else np.nan
    )

    # Notes.
    notes = []
    if not saturated:
        notes.append("no nephelometer saturation during peak window")
    if location == "morning_room":
        notes.append("no SMPS at this location")
    elif smps_ts is None:
        notes.append("SMPS data unavailable for this burn")
    rec["notes"] = "; ".join(notes)
    return rec


# ==============================================================================
# CSV OUTPUTS
# ==============================================================================


def _per_burn_columns() -> list:
    """Build the per-burn CSV column order (one row per burn-unit pair)."""
    cols = ["burn", "unit", "location", "data_present", "saturated",
            "t_ref", "inversion_present"]
    for b in OPC_SMALL_BINS:
        cols += [f"{b}_count_t0"]
        cols += [f"{b}_count_{h}h" for h in REF_HOURS]
        cols += [f"{b}_inversion_present", f"{b}_peak_inversion_ratio",
                 f"{b}_t_inversion_peak", f"{b}_inversion_duration_hours"]
    cols += ["aerotrak_ref_t0"]
    cols += [f"aerotrak_ref_{h}h" for h in REF_HOURS]
    for h in REF_HOURS:
        cols += [f"opc_bin0_ratio_{h}h", f"aerotrak_ch1_ratio_{h}h",
                 f"opc_ratio_vs_aerotrak_{h}h"]
    cols += ["smps_available", "smps_ref_t0"]
    cols += [f"smps_ref_{h}h" for h in REF_HOURS]
    cols += [f"smps_ref_ratio_{h}h" for h in REF_HOURS]
    cols += [f"bin0_over_smps_{h}h" for h in REF_HOURS]
    for h in PM_RATIO_HOURS:
        cols += [f"portal_pm25_{h}h", f"aerotrak_pm3_{h}h", f"pm25_over_pm3_{h}h"]
    cols += ["pm25_over_pm3_change_frac"]
    cols += ["notes"]
    return cols


def _write_per_burn_csv(results: list, out_dir: Path) -> pd.DataFrame:
    """Write modulair_5sec_post_peak_per_burn.csv; return the DataFrame."""
    cols = _per_burn_columns()
    rows = [{k: r.get(k, np.nan) for k in cols} for r in results]
    df = pd.DataFrame(rows, columns=cols)
    path = out_dir / "modulair_5sec_post_peak_per_burn.csv"
    df.to_csv(path, index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")
    return df


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple:
    """Spearman rho and p over finite paired values (NaN-safe)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[m], y[m])
    return float(rho), float(p)


def _bias_factor_rows(results: list) -> list:
    """
    Multi-burn post-peak PM2.5-bias factor, one row per qualifying Bedroom 2 pair.

    The SMPS 300-437 nm channel is the only same-size reference that is
    mechanistically independent of the OPC scattering artifact (the co-located
    AeroTrak Ch1 shares the same OPC reversal/inversion behavior and is therefore
    NOT a valid reference). The SMPS is deployed in Bedroom 2 only, so the bias
    factor is quantified there. The factor is opc_bin0_ratio_2.0h /
    smps_ref_ratio_2.0h: how much more elevated the OPC-N3 small bin is than the
    SMPS at +2 h, both relative to their t_ref value.

    The qualifying condition is "Bedroom 2 pair with a defined peak window and a
    co-located SMPS", NOT inversion presence: the Bedroom 2 OPC-N3 bin 0 decays
    rather than inverting, but it still reads high relative to the SMPS, which is
    the PM2.5-bias point.

    Returns
    -------
    list of dict
        Rows with keys burn, unit, location, opc_ratio, ref_ratio, ref_source
        ("SMPS"), factor. Only pairs with a positive finite SMPS ratio and a
        finite OPC ratio are included.
    """
    rows = []
    for r in results:
        if r.get("location") != "bedroom2" or not r.get("smps_available"):
            continue
        if pd.isna(r.get("t_ref")):
            continue
        opc = r.get("opc_bin0_ratio_2.0h", np.nan)
        smps = r.get("smps_ref_ratio_2.0h", np.nan)
        if not (np.isfinite(opc) and np.isfinite(smps) and smps > 0):
            continue
        rows.append(dict(
            burn=r["burn"], unit=r["unit"], location=r["location"],
            opc_ratio=float(opc), ref_ratio=float(smps), ref_source="SMPS",
            factor=float(opc) / float(smps),
        ))
    return rows


def _write_cross_burn_csv(results: list, peak_mass: pd.DataFrame, out_dir: Path) -> dict:
    """
    Write modulair_5sec_post_peak_cross_burn_summary.csv and return the summary
    dict for the markdown writers.

    Reports inversion prevalence (overall and split by saturation), duration and
    peak-ratio stats over the qualifying bin-0 inversions, the Spearman rho of
    bin-0 inversion duration vs peak AeroTrak PM3 mass, and the
    nephelometer-saturation vs inversion-presence cross-tabulation.
    """
    present = [r for r in results if r.get("data_present") and pd.notna(r.get("t_ref"))]
    n_tested = len(present)
    inv = [r for r in present if r.get("inversion_present")]
    n_inv = len(inv)

    sat = [r for r in present if r.get("saturated")]
    sat_inv = [r for r in sat if r.get("inversion_present")]
    unsat = [r for r in present if not r.get("saturated")]
    unsat_inv = [r for r in unsat if r.get("inversion_present")]

    # Bin-0 inversion duration and peak-ratio over pairs with a bin-0 inversion.
    bin0_inv = [r for r in present if r.get("bin0_inversion_present")]
    dur = np.array([r.get("bin0_inversion_duration_hours", np.nan) for r in bin0_inv], dtype=float)
    dur = dur[np.isfinite(dur)]
    ratio = np.array([r.get("bin0_peak_inversion_ratio", np.nan) for r in bin0_inv], dtype=float)
    ratio = ratio[np.isfinite(ratio)]

    # Spearman: bin-0 inversion duration vs co-located peak AeroTrak PM3 mass.
    pm = peak_mass.set_index(["burn", "location"])["peak_total_PM3_mass_ug_m3"]
    dvec, mvec = [], []
    for r in present:
        d = r.get("bin0_inversion_duration_hours", np.nan)
        key = (r["burn"], r["location"])
        mass = float(pm.get(key, np.nan)) if key in pm.index else np.nan
        dvec.append(d)
        mvec.append(mass)
    rho, pval = _spearman(np.array(mvec, dtype=float), np.array(dvec, dtype=float))

    # Multi-burn post-peak PM2.5-bias factor (OPC-N3 bin0 +2 h ratio relative to
    # the SMPS 300-437 nm +2 h ratio, Bedroom 2 only; the SMPS is the only
    # mechanistically independent same-size reference).
    bias_rows = _bias_factor_rows(present)
    factors = np.array([row["factor"] for row in bias_rows], dtype=float)
    factors = factors[np.isfinite(factors)]
    n_smps_ref = sum(1 for row in bias_rows if row["ref_source"] == "SMPS")

    # Manuscript checks: PM2.5/PM3 ratio stability (Morning Room) and the
    # absolute OPC-N3 bin 0 / SMPS ratio at t_ref + 2 h (Bedroom 2).
    mr = [r for r in present if r.get("location") == "morning_room"]
    chg = np.array([r.get("pm25_over_pm3_change_frac", np.nan) for r in mr], dtype=float)
    chg_f = chg[np.isfinite(chg)]
    n_stable = int(np.sum(np.abs(chg_f) < PM_RATIO_STABLE_FRAC))
    bd = [r for r in present if r.get("location") == "bedroom2"]
    b_s = np.array([r.get("bin0_over_smps_2.0h", np.nan) for r in bd], dtype=float)
    b_s = b_s[np.isfinite(b_s)]

    summary = dict(
        n_pairs_tested=n_tested,
        n_inversion_present=n_inv,
        n_saturated=len(sat),
        n_saturated_with_inversion=len(sat_inv),
        n_unsaturated=len(unsat),
        n_unsaturated_with_inversion=len(unsat_inv),
        bin0_n_inversions=len(bin0_inv),
        bin0_inversion_duration_median_h=float(np.median(dur)) if dur.size else np.nan,
        bin0_inversion_duration_min_h=float(np.min(dur)) if dur.size else np.nan,
        bin0_inversion_duration_max_h=float(np.max(dur)) if dur.size else np.nan,
        bin0_peak_ratio_median=float(np.median(ratio)) if ratio.size else np.nan,
        bin0_peak_ratio_min=float(np.min(ratio)) if ratio.size else np.nan,
        bin0_peak_ratio_max=float(np.max(ratio)) if ratio.size else np.nan,
        spearman_rho_dur_vs_peakmass=rho,
        spearman_p_dur_vs_peakmass=pval,
        # Cross-tab cells (saturation x inversion).
        sat_and_inv=len(sat_inv),
        sat_and_no_inv=len(sat) - len(sat_inv),
        unsat_and_inv=len(unsat_inv),
        unsat_and_no_inv=len(unsat) - len(unsat_inv),
        # Multi-burn PM2.5-bias factor at +2 h (OPC-N3 bin0 ratio over the SMPS
        # 300-437 nm ratio), Bedroom 2 pairs with a co-located SMPS.
        pm25_bias_n=int(factors.size),
        pm25_bias_factor_median=float(np.median(factors)) if factors.size else np.nan,
        pm25_bias_factor_min=float(np.min(factors)) if factors.size else np.nan,
        pm25_bias_factor_max=float(np.max(factors)) if factors.size else np.nan,
        pm25_bias_n_smps_ref=n_smps_ref,
        mr_pm25_pm3_n=int(chg_f.size),
        mr_pm25_pm3_n_stable=n_stable,
        mr_pm25_pm3_change_abs_max=float(np.max(np.abs(chg_f))) if chg_f.size else np.nan,
        bd_bin0_over_smps_2h_n=int(b_s.size),
        bd_bin0_over_smps_2h_min=float(np.min(b_s)) if b_s.size else np.nan,
        bd_bin0_over_smps_2h_max=float(np.max(b_s)) if b_s.size else np.nan,
    )
    path = out_dir / "modulair_5sec_post_peak_cross_burn_summary.csv"
    pd.DataFrame([summary]).to_csv(path, index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")
    return summary


# ==============================================================================
# BOKEH PER-PAIR FIGURE
# ==============================================================================

_BOKEH_TOOLS = "pan,box_zoom,wheel_zoom,crosshair,reset,save"
_OPC_FIG_COLORS = {"bin0": "#004D00", "bin1": "#2D8B2D", "bin2": "#95D595"}
_OPC_FIG_LABELS = {"bin0": BIN0_LABEL, "bin1": BIN1_LABEL, "bin2": BIN2_LABEL}


def _decay_band(p, t0, t_end) -> None:
    """Shade the post-peak decay window [t0, t_end]."""
    if pd.notna(t0) and pd.notna(t_end):
        p.add_layout(
            BoxAnnotation(
                left=int(pd.Timestamp(t0).timestamp() * 1000),
                right=int(pd.Timestamp(t_end).timestamp() * 1000),
                fill_color="#cccccc", fill_alpha=0.25,
            )
        )


def _event_spans(p, rec: dict) -> None:
    """Vertical lines: ignition (dotted), garage closed (t_ref, solid), PAC on (dashed)."""
    for t_ev, color, dash in [
        (rec["_ignition"], AEROTRAK_COLOR, "dotted"),
        (rec["_garage"], AEROTRAK_COLOR, "solid"),
        (rec["_pac_on"], AEROTRAK_COLOR, "dashed"),
    ]:
        if pd.notna(t_ev):
            p.add_layout(Span(
                location=int(pd.Timestamp(t_ev).timestamp() * 1000),
                dimension="height", line_color=color, line_dash=dash, line_width=1.4,
            ))


def _bokeh_pair(rec: dict, fig_dir: Path) -> None:
    """
    Three-panel Bokeh figure for one burn-unit pair (SI):
        top    = PMS5003 nephelometer bin 0 raw signal,
        middle = OPC-N3 bins 0-2 number concentration (#/cm3),
        bottom = co-located AeroTrak Ch1 (0.3-0.5 um) count concentration.
    All panels carry event lines and a shaded post-peak decay band.

    Output: quantaq_figures/modulair_5sec_post_peak_<burn>_<unit>.html
    """
    if not rec.get("data_present") or pd.isna(rec.get("t_ref")):
        return
    df = rec["_df"]
    t0 = rec["t_ref"]
    t_end = rec["_t_end"]

    # Display from 30 min before t_ref through the decay-window end + 15 min.
    t_lo = t0 - pd.Timedelta(minutes=30)
    t_hi = (t_end if pd.notna(t_end) else t0 + pd.Timedelta(hours=MAX_WIN_HR)) + pd.Timedelta(minutes=15)
    sub = df[(df["timestamp"] >= t_lo) & (df["timestamp"] <= t_hi)]
    if sub.empty:
        return

    unit_tag = "pm1" if rec["unit"] == "MODULAIR-PM1" else "pm2"
    loc_label = UNIT_CONFIG[rec["unit"]]["location_label"]
    title = f"{rec['burn']}  |  {rec['unit']} ({loc_label})  -  post-peak decay"

    # --- top: PMS5003 neph bin0 ---
    p_top = figure(x_axis_type="datetime", width=1300, height=300,
                   title=f"{title}  -  PMS5003 neph bin0 raw signal", tools=_BOKEH_TOOLS)
    p_top.line(sub["timestamp"], pd.to_numeric(sub["neph_bin0"], errors="coerce"),
               color="#67000D", line_width=1.3)
    p_top.yaxis.axis_label = "PMS5003 bin 0 (particles)"
    _decay_band(p_top, t0, t_end)
    _event_spans(p_top, rec)

    # --- middle: OPC-N3 bins 0-2 ---
    p_mid = figure(x_axis_type="datetime", width=1300, height=300, x_range=p_top.x_range,
                   title="OPC-N3 bins 0-2 number concentration", tools=_BOKEH_TOOLS)
    items = []
    for b in OPC_SMALL_BINS:
        r = p_mid.line(sub["timestamp"], pd.to_numeric(sub[b], errors="coerce"),
                       color=_OPC_FIG_COLORS[b], line_width=1.4)
        items.append(LegendItem(label=f"{b} ({_OPC_FIG_LABELS[b]})", renderers=[r]))
    p_mid.yaxis.axis_label = "OPC-N3 (#/cm³)"
    _decay_band(p_mid, t0, t_end)
    _event_spans(p_mid, rec)
    p_mid.add_layout(Legend(items=items, click_policy="hide", label_text_font_size="8pt"), "right")

    # --- bottom: AeroTrak Ch1 ---
    p_bot = figure(x_axis_type="datetime", width=1300, height=300, x_range=p_top.x_range,
                   title="Co-located AeroTrak Ch1 (0.3-0.5 um) count concentration",
                   tools=_BOKEH_TOOLS)
    at_ts, at_vals = rec["_at_ts"], rec["_at_vals"]
    if at_ts is not None:
        m = (at_ts >= t_lo) & (at_ts <= t_hi)
        p_bot.line(at_ts[m], pd.to_numeric(at_vals[m], errors="coerce"),
                   color=AEROTRAK_COLOR, line_width=1.6)
    p_bot.yaxis.axis_label = "AeroTrak Ch1 (#/cm3)"
    p_bot.xaxis.axis_label = "Local time (EDT)"
    _decay_band(p_bot, t0, t_end)
    _event_spans(p_bot, rec)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"modulair_5sec_post_peak_{rec['burn']}_{unit_tag}.html"
    reset_output()
    output_file(str(out_path), title=f"post-peak {rec['burn']} {rec['unit']}")
    save(bokeh_column(p_top, p_mid, p_bot))
    print(f"    [Bokeh] {out_path.name}")


# ==============================================================================
# MATPLOTLIB - SI FIGURES
# ==============================================================================


def _smooth_1min(vals: pd.Series) -> pd.Series:
    """Centered 12-sample (1 min) rolling mean of a 5 s series, for display only."""
    return pd.to_numeric(vals, errors="coerce").rolling(12, center=True, min_periods=6).mean()


def _hours_since(ts: pd.Series, t0: pd.Timestamp) -> pd.Series:
    """Hours between each timestamp and t0."""
    return (ts - t0).dt.total_seconds() / 3600.0


def _save_pdf_png_tight(fig, stem: Path, overwrite: bool = False) -> list:
    """
    Save a figure as vector PDF and 600 dpi PNG with a tight bounding box.

    The SI post-peak figures place their legend outside the axes grid, so the
    tight box is kept (unlike src.project_style.save_pdf_png, which preserves
    the creation size). An existing file raises unless overwrite is True.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    stem : Path
        Output path without suffix (ISO-dated by the caller).
    overwrite : bool
        Replace existing files instead of raising.

    Returns
    -------
    list of Path
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    for p in paths:
        if p.exists() and not overwrite:
            plt.close(fig)
            raise FileExistsError(f"{p} exists. Rerun with --overwrite or rename it.")
    for p in paths:
        fig.savefig(str(p), dpi=600, bbox_inches="tight")
        print(f"    [fig] {p.name}")
    plt.close(fig)
    return paths


def _event_lines(ax, rec: dict, t0: pd.Timestamp) -> None:
    """Garage closure (x = 0, solid) and PAC on (dashed), in hours since t0."""
    ax.axvline(0.0, color=EVENT_COLOR, lw=0.9, ls=GARAGE_LS)
    pac = rec.get("_pac_on", pd.NaT)
    if pd.notna(pac):
        ax.axvline((pac - t0).total_seconds() / 3600.0, color=EVENT_COLOR, lw=0.9, ls=PAC_LS)


def _mpl_post_peak_pm25_v2(results: list, df_at_by_loc: dict, fig_dir: Path,
                           overwrite: bool = False) -> list:
    """
    SI figure: Morning Room post-peak record, one row per burn, two columns on
    a shared time axis (hours since garage-door closure, t_ref):
        (left)  mass: portal MODULAIR-PM2 PM2.5 vs co-located AeroTrak2 PM3;
        (right) number: OPC-N3 bin 0 vs AeroTrak2 Ch1 (0.3-0.5 um).
    Both columns are absolute values on log axes. Vertical lines mark garage
    closure (x = 0, solid) and portable air cleaner activation (dashed). Mass
    and number are separate panels rather than two y-scales on one axis.

    Reading: if the portal PM2.5 decays more slowly than the AeroTrak PM3 while
    OPC-N3 bin 0 is elevated, the post-peak bin 0 rise is carried into the
    delivered PM2.5.
    """
    from matplotlib.lines import Line2D

    unit = "MODULAIR-PM2"
    pairs = [r for r in results
             if r["unit"] == unit and r.get("data_present") and pd.notna(r.get("t_ref"))]
    pairs = sorted(pairs, key=lambda r: int(r["burn"].replace("burn", "")))
    if not pairs:
        print("    [mpl] no Morning Room pairs for post-peak PM2.5 figure.")
        return []
    df_at = df_at_by_loc.get("morning_room")
    color = UNIT_COLOR[unit]
    x_lo = -0.5

    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 2, figsize=(figsize("double")[0], 1.6 * nrows + 0.8),
                             sharex=True, squeeze=False)
    for i, rec in enumerate(pairs):
        ax_m, ax_n = axes[i]
        t0, t_end = rec["t_ref"], rec["_t_end"]
        x_hi = _hours_since(pd.Series([t_end]), t0).iloc[0] if pd.notna(t_end) else MAX_WIN_HR
        burn_no = int(rec["burn"].replace("burn", ""))
        ax_m.set_ylabel(f"Burn {burn_no:02d}", fontsize=_FS - 1)

        # Mass column.
        portal = load_portal_burn(unit, rec["burn"])
        if portal is not None and "pm25" in portal.columns:
            xp = _hours_since(portal["timestamp"], t0)
            mp = (xp >= x_lo) & (xp <= x_hi)
            ax_m.semilogy(xp[mp], portal.loc[mp, "pm25"].where(portal.loc[mp, "pm25"] > 0),
                          color=color, lw=1.2)
        if df_at is not None:
            day = df_at[df_at["Date and Time"].dt.date == t0.date()]
            xa = _hours_since(day["Date and Time"], t0)
            ma = (xa >= x_lo) & (xa <= x_hi)
            pm3 = day.loc[ma, "PM3 (µg/m³)"]
            ax_m.semilogy(xa[ma], pm3.where(pm3 > 0), color=AEROTRAK_COLOR, lw=1.1, ls="--")

        # Number column.
        df = rec["_df"]
        xo = _hours_since(df["timestamp"], t0)
        mo = (xo >= x_lo) & (xo <= x_hi)
        b0 = _smooth_1min(df["bin0"])[mo]
        ax_n.semilogy(xo[mo], b0.where(b0 > 0), color=color, lw=1.2)
        if rec.get("_at_ts") is not None:
            xc = _hours_since(rec["_at_ts"], t0)
            mc = (xc >= x_lo) & (xc <= x_hi)
            ch1 = pd.to_numeric(rec["_at_vals"][mc], errors="coerce")
            ax_n.semilogy(xc[mc], ch1.where(ch1 > 0), color=AEROTRAK_COLOR, lw=1.1, ls="--")

        for ax in (ax_m, ax_n):
            _event_lines(ax, rec, t0)
            ax.tick_params(labelsize=_FS - 2)
        ax_n.text(0.98, 0.94, "saturated" if rec.get("saturated") else "not saturated",
                  transform=ax_n.transAxes, ha="right", va="top", fontsize=_FS - 4)

    axes[0, 0].set_title("PM mass (µg·m⁻³)", fontsize=_FS, fontweight="bold")
    axes[0, 1].set_title("Number concentration (particles·cm⁻³)", fontsize=_FS, fontweight="bold")
    axes[-1, 0].set_xlim(left=x_lo)
    fig.supxlabel("Hours since garage-door closure", fontsize=_FS)
    handles = [
        Line2D([0], [0], color=color, lw=1.2,
               label="MODULAIR-PM2: portal PM2.5 (left), OPC-N3 bin 0 (right)"),
        Line2D([0], [0], color=AEROTRAK_COLOR, lw=1.1, ls="--",
               label="AeroTrak2: PM3 (left), Ch1 0.3 µm to 0.5 µm (right)"),
        Line2D([0], [0], color=EVENT_COLOR, lw=0.9, ls=GARAGE_LS, label="Garage closed"),
        Line2D([0], [0], color=EVENT_COLOR, lw=0.9, ls=PAC_LS, label="PAC on"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=_FS - 3, frameon=False,
               bbox_to_anchor=(0.5, 1.0 + 0.75 / (1.6 * nrows + 0.8)))
    return _save_pdf_png_tight(
        fig, fig_dir / f"{date.today().isoformat()}_modulair_5sec_post_peak_morning_room",
        overwrite=overwrite,
    )


def _mpl_bedroom_smps_v2(results: list, fig_dir: Path, overwrite: bool = False) -> list:
    """
    SI figure: Bedroom 2 post-peak number concentration of MODULAIR-PM1 OPC-N3
    bin 0 (0.35-0.46 um), the co-located SMPS 300-437 nm sum, and AeroTrak1 Ch1
    (0.3-0.5 um), one panel per burn, versus hours since garage-door closure
    (t_ref) on a log axis. All three are particles/cm3, so the panels show
    absolute levels. Vertical lines mark garage closure (solid), portable air
    cleaner activation (dashed), and t_ref + 2 h (grey dotted), the time used
    for the OPC-N3 to SMPS comparison in the text.
    """
    from matplotlib.lines import Line2D

    unit = "MODULAIR-PM1"
    pairs = [r for r in results
             if r["unit"] == unit and r.get("data_present") and pd.notna(r.get("t_ref"))]
    pairs = sorted(pairs, key=lambda r: int(r["burn"].replace("burn", "")))
    if not pairs:
        print("    [mpl] no Bedroom 2 pairs for SMPS comparison.")
        return []
    color = UNIT_COLOR[unit]
    x_lo = -0.5

    ncols = 3
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize("double")[0], 2.3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    for ax, rec in zip(axes, pairs):
        t0, t_end = rec["t_ref"], rec["_t_end"]
        x_hi = _hours_since(pd.Series([t_end]), t0).iloc[0] if pd.notna(t_end) else MAX_WIN_HR
        df = rec["_df"]
        xo = _hours_since(df["timestamp"], t0)
        mo = (xo >= x_lo) & (xo <= x_hi)
        b0 = _smooth_1min(df["bin0"])[mo]
        ax.semilogy(xo[mo], b0.where(b0 > 0), color=color, lw=1.2)
        if rec.get("_smps_ts") is not None:
            xs = _hours_since(rec["_smps_ts"], t0)
            ms = (xs >= x_lo) & (xs <= x_hi)
            sv = pd.to_numeric(rec["_smps_vals"][ms], errors="coerce")
            ax.semilogy(xs[ms], sv.where(sv > 0), color=SMPS_COLOR, lw=1.1,
                        marker="o", ms=2.5)
        if rec.get("_at_ts") is not None:
            xc = _hours_since(rec["_at_ts"], t0)
            mc = (xc >= x_lo) & (xc <= x_hi)
            ch1 = pd.to_numeric(rec["_at_vals"][mc], errors="coerce")
            ax.semilogy(xc[mc], ch1.where(ch1 > 0), color=AEROTRAK_COLOR, lw=1.0, ls="--")
        _event_lines(ax, rec, t0)
        ax.axvline(2.0, color=SHADE, lw=0.8, ls=":")
        burn_no = int(rec["burn"].replace("burn", ""))
        sealed = " (sealed)" if rec["burn"] in ("burn5", "burn6") else ""
        ax.set_title(f"Burn {burn_no:02d}{sealed}", fontsize=_FS - 1)
        ax.tick_params(labelsize=_FS - 2)
    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.supxlabel("Hours since garage-door closure", fontsize=_FS)
    fig.supylabel("Number concentration (particles·cm⁻³)", fontsize=_FS)
    handles = [
        Line2D([0], [0], color=color, lw=1.2, label="MODULAIR-PM1 OPC-N3 bin 0 (0.35 µm to 0.46 µm)"),
        Line2D([0], [0], color=SMPS_COLOR, lw=1.1, marker="o", ms=3, label="SMPS 300 nm to 437 nm"),
        Line2D([0], [0], color=AEROTRAK_COLOR, lw=1.0, ls="--", label="AeroTrak1 Ch1 (0.3 µm to 0.5 µm)"),
        Line2D([0], [0], color=EVENT_COLOR, lw=0.9, ls=GARAGE_LS, label="Garage closed"),
        Line2D([0], [0], color=EVENT_COLOR, lw=0.9, ls=PAC_LS, label="PAC on"),
        Line2D([0], [0], color=SHADE, lw=0.8, ls=":", label="Garage closure + 2 h"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=_FS - 3,
               frameon=True, bbox_to_anchor=(0.98, 0.04))
    return _save_pdf_png_tight(
        fig, fig_dir / f"{date.today().isoformat()}_modulair_5sec_post_peak_bedroom_smps",
        overwrite=overwrite,
    )


# ==============================================================================
# MARKDOWN OUTPUTS
# ==============================================================================


def _fmt(val, fmt: str = ".1f") -> str:
    """Format a number, or '[no data]' for NaN/None."""
    try:
        return f"{val:{fmt}}" if val is not None and not np.isnan(val) else "[no data]"
    except (TypeError, ValueError):
        return "[no data]"


def _write_summary_md(results: list, summary: dict, out_dir: Path) -> None:
    """Plain-language synthesis with a per-pair table and flagged conditions."""
    present = [r for r in results if r.get("data_present") and pd.notna(r.get("t_ref"))]
    skipped = [r for r in results if not (r.get("data_present") and pd.notna(r.get("t_ref")))]

    lines = ["# MODULAIR-PM 5 s post-peak OPC-N3 inversion - summary", ""]
    lines.append("## Plain-language synthesis\n")
    lines.append(
        f"Of {summary['n_pairs_tested']} burn-unit pairs with a garage-closure "
        f"time (t_ref), {summary['n_inversion_present']} show a post-peak "
        f"OPC-N3 small-bin inversion: during the decay phase one or more of the "
        f"three smallest bins (bin0 {BIN0_LABEL}, bin1 {BIN1_LABEL}, bin2 "
        f"{BIN2_LABEL}) rises by more than {int(INVERSION_FRAC * 100)}% above its "
        f"count at garage closure, opposite to the smoke decay."
    )
    # Name the unsaturated records that actually invert, so the prose does not
    # imply the non-inverting fallback-windowed units contribute to the count.
    unsat_inv = [
        f"{r['burn']} {r['unit']}"
        for r in present
        if not r.get("saturated") and r.get("inversion_present")
    ]
    if not unsat_inv:
        unsat_clause = "none of the non-saturated records"
    elif len(unsat_inv) == 1:
        unsat_clause = f"the one non-saturated case being {unsat_inv[0]}"
    else:
        unsat_clause = "the non-saturated cases being " + ", ".join(unsat_inv)

    lines.append(
        f"\nThe inversion is not conditional on prior nephelometer saturation. "
        f"It is present in {summary['n_saturated_with_inversion']} of "
        f"{summary['n_saturated']} saturated records (all MODULAIR-PM2) and in "
        f"{summary['n_unsaturated_with_inversion']} of "
        f"{summary['n_unsaturated']} non-saturated records ({unsat_clause}), so "
        f"it occurs in the OPC-N3 channel even where the peak window never "
        f"saturated the nephelometer."
    )
    lines.append(
        f"\nAcross the {summary['bin0_n_inversions']} pairs with a bin-0 "
        f"inversion, the bin-0 count rises to a median of "
        f"{_fmt(summary['bin0_peak_ratio_median'], '.1f')} times its garage-closure "
        f"value (range {_fmt(summary['bin0_peak_ratio_min'], '.1f')} to "
        f"{_fmt(summary['bin0_peak_ratio_max'], '.1f')}). The inversion persists "
        f"for a median of {_fmt(summary['bin0_inversion_duration_median_h'], '.2f')} "
        f"hours (range {_fmt(summary['bin0_inversion_duration_min_h'], '.2f')} to "
        f"{_fmt(summary['bin0_inversion_duration_max_h'], '.2f')}); durations that "
        f"equal the analysis-window length are right-censored lower bounds."
    )
    lines.append(
        f"\nManuscript checks. Morning Room: the portal PM2.5 to AeroTrak2 PM3 "
        f"ratio changed by less than {int(PM_RATIO_STABLE_FRAC * 100)} % between "
        f"t_ref + {PM_RATIO_HOURS[0]} h and t_ref + {PM_RATIO_HOURS[1]} h in "
        f"{summary['mr_pm25_pm3_n_stable']} of {summary['mr_pm25_pm3_n']} burns "
        f"(largest absolute change {_fmt(summary['mr_pm25_pm3_change_abs_max'], '.2f')}). "
        f"Bedroom 2: at t_ref + 2 h, MODULAIR-PM1 OPC-N3 bin 0 was "
        f"{_fmt(summary['bd_bin0_over_smps_2h_min'], '.2f')} to "
        f"{_fmt(summary['bd_bin0_over_smps_2h_max'], '.2f')} times the SMPS "
        f"300-437 nm concentration ({summary['bd_bin0_over_smps_2h_n']} burns)."
    )
    lines.append(
        f"\nSpearman rho between bin-0 inversion duration and co-located peak "
        f"AeroTrak PM3 mass is "
        f"{_fmt(summary['spearman_rho_dur_vs_peakmass'], '.2f')} "
        f"(p = {_fmt(summary['spearman_p_dur_vs_peakmass'], '.3f')}). The "
        f"saturation-vs-inversion cross-tabulation is: saturated and inverted "
        f"{summary['sat_and_inv']}, saturated and not "
        f"{summary['sat_and_no_inv']}, unsaturated and inverted "
        f"{summary['unsat_and_inv']}, unsaturated and not "
        f"{summary['unsat_and_no_inv']}."
    )

    # Per-pair table.
    lines.append("\n## Per-pair results\n")
    lines.append(
        "| burn | unit | saturated | inversion | bin0 peak ratio | "
        "bin0 dur (h) | OPC/AeroTrak +2h | PM2.5/PM3 +0.5h | PM2.5/PM3 +2h | "
        "bin0/SMPS +2h | SMPS avail |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in present:
        lines.append(
            f"| {r['burn']} | {r['unit']} | {r.get('saturated')} | "
            f"{r.get('inversion_present')} | "
            f"{_fmt(r.get('bin0_peak_inversion_ratio'), '.1f')} | "
            f"{_fmt(r.get('bin0_inversion_duration_hours'), '.2f')} | "
            f"{_fmt(r.get('opc_ratio_vs_aerotrak_2.0h'), '.1f')} | "
            f"{_fmt(r.get('pm25_over_pm3_0.5h'), '.2f')} | "
            f"{_fmt(r.get('pm25_over_pm3_2.0h'), '.2f')} | "
            f"{_fmt(r.get('bin0_over_smps_2.0h'), '.2f')} | "
            f"{r.get('smps_available')} |"
        )

    # Flagged conditions.
    lines.append("\n## Flagged conditions\n")
    flagged = False
    for r in skipped:
        reason = r.get("notes") or ("5 s data missing" if not r.get("data_present")
                                    else "no garage-closure time")
        lines.append(f"- {r['burn']} {r['unit']}: {reason} (not imputed).")
        flagged = True
    for r in present:
        if r.get("location") == "morning_room":
            continue
        if not r.get("smps_available"):
            lines.append(
                f"- {r['burn']} {r['unit']}: SMPS data unavailable; SMPS "
                f"cross-check skipped for this burn."
            )
            flagged = True
    if not flagged:
        lines.append("- None.")

    path = out_dir / "modulair_5sec_post_peak_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    [MD] {path.name}")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Run the full post-peak inversion analysis: load, analyze, write outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace same-day ISO-dated figures instead of raising",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    apply_est_style()
    print("\n" + "=" * 70)
    print("MODULAIR-PM 5 s post-peak OPC-N3 inversion  -  Burns 4-10")
    print("=" * 70)

    out_dir = get_common_file("quantaq_analysis")
    fig_dir = get_common_file("quantaq_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saturation flags, events, AeroTrak, peak-mass...")
    pw = _load_peak_windows()
    events = load_event_times()
    df_at_by_loc = {
        "bedroom2": _load_aerotrak_all("AeroTrak1"),
        "morning_room": _load_aerotrak_all("AeroTrak2"),
    }
    peak_mass = _load_aerotrak_peak_mass()
    smps_cache: dict = {}

    results = []
    for burn_id in BURNS:
        print(f"--- {burn_id} ---")
        for unit in UNITS:
            rec = analyze_pair(burn_id, unit, pw, events, df_at_by_loc, smps_cache)
            results.append(rec)
            if rec.get("data_present") and pd.notna(rec.get("t_ref")):
                print(
                    f"    {unit}: sat={rec.get('saturated')} "
                    f"inversion={rec.get('inversion_present')} "
                    f"bin0_peak_ratio={_fmt(rec.get('bin0_peak_inversion_ratio'), '.1f')}"
                )

    print("\nWriting CSV outputs...")
    _write_per_burn_csv(results, out_dir)
    summary = _write_cross_burn_csv(results, peak_mass, out_dir)

    print("\nWriting Bokeh per-pair figures...")
    for rec in results:
        _bokeh_pair(rec, fig_dir)

    print("\nWriting matplotlib figures...")
    # The legacy normalized small multiples, overlay, and +2 h ratio bars are
    # no longer manuscript figures (each divides by a near-zero reference
    # count), so they are not regenerated.
    _mpl_post_peak_pm25_v2(results, df_at_by_loc, fig_dir, overwrite=args.overwrite)
    _mpl_bedroom_smps_v2(results, fig_dir, overwrite=args.overwrite)

    print("\nWriting markdown outputs...")
    _write_summary_md(results, summary, out_dir)

    print("\nDone. Check quantaq_analysis/ and quantaq_figures/ for outputs.")


if __name__ == "__main__":
    main()
