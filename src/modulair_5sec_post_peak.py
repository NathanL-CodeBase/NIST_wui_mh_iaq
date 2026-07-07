"""
MODULAIR-PM 5 s post-peak OPC-N3 small-bin inversion analysis (SI for 3.2.3).

During the post-peak decay phase the smallest OPC-N3 bins of the MODULAIR-PM
units increase in count rather than decaying with the smoke. This is distinct
from the peak-window suppression characterized in modulair_5sec_peak_window.py
(Section 3.2.3). This script quantifies the inversion per burn-unit pair and
contrasts the OPC-N3 bin-0 decay trajectory against the co-located AeroTrak Ch1
count and (Bedroom 2 only) the SMPS 300-437 nm number concentration, both of
which decay monotonically.

The decay window begins at t_peak_end (from modulair_5sec_peak_per_burn.csv,
column t_peak_end) and runs to ignition + 4 h or PM3 return to baseline,
whichever is earlier (consistent with the Prompt 1 / coincidence definition).

Scope: the inversion test runs on every record that has a t_peak_end (all
fallback-windowed MODULAIR-PM1 records plus the saturated MODULAIR-PM2
records), so whether the inversion occurs WITHOUT prior nephelometer saturation
can be reported. Headline counts are also reported split by saturation state.

The script stays descriptive: it does not propose a mechanism and does not cite
any OPC-N3 coincidence threshold (no published value exists for the OPC-N3).

All paths resolve through data_config.json via src/modulair_5sec_io.py. No raw
5 s data are written or committed. Missing data are flagged, not imputed.

Outputs (under common_folders quantaq_analysis / quantaq_figures):
    modulair_5sec_post_peak_per_burn.csv
    modulair_5sec_post_peak_cross_burn_summary.csv
    modulair_5sec_post_peak_<burn>_<unit>.html       (Bokeh, one per pair)
    modulair_5sec_post_peak_smallmultiples.png        (matplotlib, SI)
    modulair_5sec_post_peak_overlay.png               (matplotlib, main/SI)
    modulair_5sec_pm25_bias.png                       (matplotlib, SI)
    modulair_5sec_post_peak_summary.md
    modulair_5sec_post_peak_manuscript_sentences.md

Author: Nathan Lima
Created: 2026-06-26
"""

import sys
import warnings
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
    REF_LINE,
    ROLE_COLORS,
    UNIT_COLORS,
    apply_est_style,
    figsize,
    save_fig,
)
from src.modulair_5sec_io import (  # noqa: E402
    BURN_DATES,
    OPC_BIN_EDGES_UM,
    UNIT_CONFIG,
    load_5sec_burn,
    load_event_times,
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

# Reference time points after t_peak_end (hours) for the trajectory comparison.
REF_HOURS = [0.5, 1.0, 2.0, 4.0]
# A subset used for the inversion-presence test (per the prompt: +0.5/1/2 h).
INVERSION_TEST_HOURS = [0.5, 1.0, 2.0]

# Inversion classification: any bin count at a test hour exceeds the t_peak_end
# value by more than this fraction.
INVERSION_FRAC = 0.50  # > 50% increase
# Return-to-baseline persistence for the inversion-end / duration test.
RETURN_PERSIST_MIN = 15.0  # must stay below t_peak_end value this long (min)

# Minimum t_peak_end count for a ratio (peak_inversion_ratio,
# opc_ratio_vs_aerotrak) to be reported. Some OPC-N3 small-bin counts at
# t_peak_end sit near zero (e.g. burn10 MODULAIR-PM2 bin0 ~0.01 counts), so a
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

# Input CSV from the peak-window analysis (carries t_peak_end and saturated).
PEAK_CSV = "modulair_5sec_peak_per_burn.csv"

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

    t0 is t_peak_end (start of decay). The bin count at t0 is the reference;
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
    or no t_peak_end is flagged and skipped, never imputed.
    """
    location = UNIT_LOCATION[unit]
    pwin = _peak_window_lookup(pw, burn_id, unit)
    t0 = pwin["t_peak_end"]
    saturated = pwin["saturated"]

    ev = events.loc[burn_id] if burn_id in events.index else None
    ignition = ev["ignition"] if ev is not None else pd.NaT
    garage = ev["garage_closed"] if ev is not None else pd.NaT
    pac_on = ev["pac_on"] if ev is not None else pd.NaT
    burn_date = pd.Timestamp(BURN_DATES[burn_id])

    base = dict(
        burn=burn_id, unit=unit, location=location, saturated=saturated,
        peak_window_method=pwin["peak_window_method"], t_peak_end=t0,
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
        base["notes"] = "no t_peak_end (no peak window); inversion test skipped"
        print(f"    [{burn_id}|{unit}] no t_peak_end - skipped.")
        return base

    df_at = df_at_by_loc.get(location)
    t_end = _decay_window_end(df_at, burn_date, ignition)
    if pd.isna(t_end) or t_end <= t0:
        # Fall back to a fixed 4 h decay window anchored on t_peak_end.
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
    # ratio is only reported when the bin-0 count at t_peak_end clears
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
            "peak_window_method", "t_peak_end", "inversion_present"]
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
    SMPS at +2 h, both relative to their t_peak_end value.

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
        if pd.isna(r.get("t_peak_end")):
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
    present = [r for r in results if r.get("data_present") and pd.notna(r.get("t_peak_end"))]
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
    """Vertical lines: ignition, garage closed, PAC on, t_peak_end."""
    for t_ev, color, dash in [
        (rec["_ignition"], AEROTRAK_COLOR, "solid"),
        (rec["_garage"], AEROTRAK_COLOR, "dashed"),
        (rec["_pac_on"], AEROTRAK_COLOR, "dotted"),
        (rec["t_peak_end"], "#1f77b4", "solid"),
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
        middle = OPC-N3 bins 0-2 raw counts,
        bottom = co-located AeroTrak Ch1 (0.3-0.5 um) count concentration.
    All panels carry event lines and a shaded post-peak decay band.

    Output: quantaq_figures/modulair_5sec_post_peak_<burn>_<unit>.html
    """
    if not rec.get("data_present") or pd.isna(rec.get("t_peak_end")):
        return
    df = rec["_df"]
    t0 = rec["t_peak_end"]
    t_end = rec["_t_end"]

    # Display from 30 min before t_peak_end through the decay-window end + 15 min.
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
    p_top.yaxis.axis_label = "Neph bin0 raw"
    _decay_band(p_top, t0, t_end)
    _event_spans(p_top, rec)

    # --- middle: OPC-N3 bins 0-2 ---
    p_mid = figure(x_axis_type="datetime", width=1300, height=300, x_range=p_top.x_range,
                   title="OPC-N3 bins 0-2 raw counts", tools=_BOKEH_TOOLS)
    items = []
    for b in OPC_SMALL_BINS:
        r = p_mid.line(sub["timestamp"], pd.to_numeric(sub[b], errors="coerce"),
                       color=_OPC_FIG_COLORS[b], line_width=1.4)
        items.append(LegendItem(label=f"{b} ({_OPC_FIG_LABELS[b]})", renderers=[r]))
    p_mid.yaxis.axis_label = "OPC raw counts"
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
# MATPLOTLIB - SMALL MULTIPLES (SI)
# ==============================================================================


def _normalized_trace(ts, vals, t0, t_end):
    """
    Return (hours_since_t0, value / value_at_t0) over [t0, t_end], masking
    non-positive values so the log axis does not break. value_at_t0 uses the
    same local-mean sampler as the analysis.
    """
    if ts is None or vals is None or pd.isna(t0):
        return None, None
    ref = _value_at(ts, vals, t0)
    if not (np.isfinite(ref) and ref >= MIN_COUNT_FOR_RATIO):
        return None, None
    m = (ts >= t0) & (ts <= t_end)
    x = (ts[m] - t0).dt.total_seconds() / 3600.0
    y = pd.to_numeric(vals[m], errors="coerce") / ref
    y = y.where(y > 0)
    return x.to_numpy(dtype=float), y.to_numpy(dtype=float)


def _mpl_small_multiples(results: list, fig_dir: Path) -> None:
    """
    Figure S4 (SI): one panel per burn-unit pair with an inversion. OPC-N3 bin 0
    normalized to its value at t_peak_end (solid) vs the co-located AeroTrak Ch1
    normalized to its value at t_peak_end (dashed). x = hours since t_peak_end,
    log y. Shared bottom x-label and left y-label; 12 pt fonts via the shared
    style; one figure-level legend.
    """
    pairs = [r for r in results if r.get("inversion_present")]
    pairs = sorted(pairs, key=lambda r: (int(r["burn"].replace("burn", "")), r["unit"]))
    if not pairs:
        print("    [mpl] no inversion pairs for small multiples.")
        return

    ncols = 3
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize("double")[0], 2.4 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for ax_idx, rec in enumerate(pairs):
        ax = axes[ax_idx]
        t0, t_end = rec["t_peak_end"], rec["_t_end"]
        df = rec["_df"]
        # OPC-N3 bin 0.
        x_opc, y_opc = _normalized_trace(df["timestamp"], df["bin0"], t0, t_end)
        if x_opc is not None:
            ax.semilogy(x_opc, y_opc, color=UNIT_COLOR[rec["unit"]], lw=1.1,
                        label="OPC-N3 bin0", alpha=0.9)
        # AeroTrak Ch1.
        x_at, y_at = _normalized_trace(rec["_at_ts"], rec["_at_vals"], t0, t_end)
        if x_at is not None:
            ax.semilogy(x_at, y_at, color=AEROTRAK_COLOR, lw=1.1, ls="--",
                        label="AeroTrak Ch1", alpha=0.9)
        ax.axhline(1.0, color=REF_LINE, lw=0.7, ls=":")
        ax.tick_params(labelsize=_FS - 2)
        tag = "Bdrm" if rec["unit"] == "MODULAIR-PM1" else "MR"
        sat = "sat" if rec["saturated"] else "no-sat"
        ax.set_title(f"{rec['burn']} {tag} ({sat})", fontsize=_FS - 1)

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.supxlabel("Hours since end of peak window", fontsize=_FS)
    fig.supylabel("count / count at t_peak_end", fontsize=_FS)

    # Legend swatches must match the plotted OPC-N3 trace colors, which are
    # per unit (Bedroom 2 PM1 blue, Morning Room PM2 vermillion), not a single
    # gray. The AeroTrak Ch1 companion is dashed black.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=UNIT_COLOR["MODULAIR-PM1"], lw=1.5,
               label="OPC-N3 bin0 - Bedroom 2 (norm.)"),
        Line2D([0], [0], color=UNIT_COLOR["MODULAIR-PM2"], lw=1.5,
               label="OPC-N3 bin0 - Morning Room (norm.)"),
        Line2D([0], [0], color=AEROTRAK_COLOR, lw=1.5, ls="--",
               label="AeroTrak Ch1 (normalized)"),
        Line2D([0], [0], color=REF_LINE, lw=0.7, ls=":", label="ratio = 1"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               fontsize=_FS - 2, frameon=True, bbox_to_anchor=(0.5, 1.04))

    save_fig(fig, fig_dir / "modulair_5sec_post_peak_smallmultiples.png")


# ==============================================================================
# MATPLOTLIB - OVERLAY (main text / SI)
# ==============================================================================


def _mpl_overlay(results: list, fig_dir: Path) -> None:
    """
    Figure 3 (main text, double column): MODULAIR-PM2 (Morning Room) OPC-N3 bin 0
    normalized to its value at t_peak_end, overlaid across all qualifying
    (inversion-present) burns. x = hours since t_peak_end, log y, dashed
    reference at y = 1.
    """
    pairs = [r for r in results
             if r["unit"] == "MODULAIR-PM2" and r.get("inversion_present")]
    pairs = sorted(pairs, key=lambda r: int(r["burn"].replace("burn", "")))
    if not pairs:
        print("    [mpl] no MODULAIR-PM2 inversion pairs for overlay.")
        return

    fig, ax = plt.subplots(figsize=figsize("double", aspect=0.55))
    cmap = plt.get_cmap("viridis", max(len(pairs), 2))
    for i, rec in enumerate(pairs):
        x, y = _normalized_trace(rec["_df"]["timestamp"], rec["_df"]["bin0"],
                                 rec["t_peak_end"], rec["_t_end"])
        if x is not None:
            ax.semilogy(x, y, color=cmap(i), lw=1.4, alpha=0.85, label=rec["burn"])

    ax.axhline(1.0, color=REF_LINE, lw=1.0, ls="--", label="value at t_peak_end")
    ax.set_xlabel("Hours since end of peak window", fontsize=_FS)
    ax.set_ylabel("OPC-N3 bin0 count / t_peak_end value", fontsize=_FS)
    ax.tick_params(labelsize=_FS)
    ax.legend(fontsize=_FS - 2, ncol=1, loc="lower right")

    save_fig(fig, fig_dir / "modulair_5sec_post_peak_overlay.png")


# ==============================================================================
# MATPLOTLIB - PM2.5 BIAS (SI)
# ==============================================================================


def _mpl_pm25_bias(results: list, fig_dir: Path) -> None:
    """
    Figure S5 (SI), multi-burn: for every Bedroom 2 pair with a co-located SMPS,
    the OPC-N3 bin-0 +2 h count ratio (count / count at t_peak_end) beside the
    SMPS 300-437 nm +2 h ratio. The SMPS is the only same-size reference that is
    mechanistically independent of the OPC scattering artifact (the co-located
    AeroTrak Ch1 shares the same OPC reversal behavior and is not used as a
    reference). A dashed line at ratio = 1 marks the end-of-peak value; the OPC /
    SMPS factor is annotated above each pair.

    The figure conveys that the OPC-N3 small-bin count stays elevated at +2 h
    relative to the independent SMPS reference, so the portal PM2.5 (derived in
    part from the OPC-N3 counts) is biased high post-peak. Morning Room has no
    co-located SMPS and therefore no valid reference, so the bias is quantified
    in Bedroom 2 only. No mechanism is proposed.
    """
    present = [r for r in results
               if r.get("data_present") and pd.notna(r.get("t_peak_end"))]
    rows = _bias_factor_rows(present)
    rows = sorted(rows, key=lambda d: int(d["burn"].replace("burn", "")))
    if not rows:
        print("    [mpl] no Bedroom 2 SMPS pairs for PM2.5 bias chart.")
        return

    labels = [f"{d['burn']}\nPM1 / Bdrm" for d in rows]
    opc_r = [d["opc_ratio"] for d in rows]
    smps_r = [d["ref_ratio"] for d in rows]
    factors = [d["factor"] for d in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(figsize("double")[0], 3.6))

    opc_color = UNIT_COLOR["MODULAIR-PM1"]
    ax.bar(x - w / 2, opc_r, w, color=opc_color, alpha=0.9,
           label="OPC-N3 bin0 (count / t_peak_end)")
    smps_bars = ax.bar(x + w / 2, smps_r, w, color=SMPS_COLOR, alpha=0.85,
                       label="SMPS 300-437 nm (count / t_peak_end)")

    # Annotate the OPC / SMPS elevation factor above each pair.
    tops = [max(o, s) for o, s in zip(opc_r, smps_r)]
    for xi, top, fac in zip(x, tops, factors):
        ax.annotate(f"{fac:.0f}x", xy=(xi, top), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=_FS - 3)

    ax.axhline(1.0, color=REF_LINE, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_FS - 2)
    ax.set_ylabel("Count ratio at +2 h (vs t_peak_end)", fontsize=_FS)
    ax.set_title("Bedroom 2: OPC-N3 small-bin vs SMPS at +2 h post-peak",
                 fontsize=_FS)
    ax.tick_params(labelsize=_FS - 1)
    ax.legend(fontsize=_FS - 2, ncol=1, loc="upper right")

    save_fig(fig, fig_dir / "modulair_5sec_pm25_bias.png")


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
    present = [r for r in results if r.get("data_present") and pd.notna(r.get("t_peak_end"))]
    skipped = [r for r in results if not (r.get("data_present") and pd.notna(r.get("t_peak_end")))]

    lines = ["# MODULAIR-PM 5 s post-peak OPC-N3 inversion - summary", ""]
    lines.append("## Plain-language synthesis\n")
    lines.append(
        f"Of {summary['n_pairs_tested']} burn-unit pairs with a defined peak "
        f"window (t_peak_end), {summary['n_inversion_present']} show a post-peak "
        f"OPC-N3 small-bin inversion: during the decay phase one or more of the "
        f"three smallest bins (bin0 {BIN0_LABEL}, bin1 {BIN1_LABEL}, bin2 "
        f"{BIN2_LABEL}) rises by more than {int(INVERSION_FRAC * 100)}% above its "
        f"count at t_peak_end, opposite to the smoke decay."
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
        f"{_fmt(summary['bin0_peak_ratio_median'], '.1f')} times its t_peak_end "
        f"value (range {_fmt(summary['bin0_peak_ratio_min'], '.1f')} to "
        f"{_fmt(summary['bin0_peak_ratio_max'], '.1f')}). The inversion persists "
        f"for a median of {_fmt(summary['bin0_inversion_duration_median_h'], '.2f')} "
        f"hours (range {_fmt(summary['bin0_inversion_duration_min_h'], '.2f')} to "
        f"{_fmt(summary['bin0_inversion_duration_max_h'], '.2f')}); durations that "
        f"equal the analysis-window length are right-censored lower bounds."
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
        "bin0 dur (h) | OPC/AeroTrak +2h | SMPS avail |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in present:
        lines.append(
            f"| {r['burn']} | {r['unit']} | {r.get('saturated')} | "
            f"{r.get('inversion_present')} | "
            f"{_fmt(r.get('bin0_peak_inversion_ratio'), '.1f')} | "
            f"{_fmt(r.get('bin0_inversion_duration_hours'), '.2f')} | "
            f"{_fmt(r.get('opc_ratio_vs_aerotrak_2.0h'), '.1f')} | "
            f"{r.get('smps_available')} |"
        )

    # Flagged conditions.
    lines.append("\n## Flagged conditions\n")
    flagged = False
    for r in skipped:
        reason = r.get("notes") or ("5 s data missing" if not r.get("data_present")
                                    else "no t_peak_end")
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


def _write_manuscript_md(results: list, summary: dict, out_dir: Path) -> None:
    """
    Write the brief main-text mention and the full SI subsection for the
    post-peak OPC-N3 inversion. All numbers are derived from the analysis;
    no mechanism is proposed and no OPC-N3 coincidence threshold is cited.
    """
    present = [r for r in results if r.get("data_present") and pd.notna(r.get("t_peak_end"))]
    sat = [r for r in present if r.get("saturated")]
    sat_inv = [r for r in sat if r.get("inversion_present")]
    n_sat, n_sat_inv = len(sat), len(sat_inv)

    pr_med = summary["bin0_peak_ratio_median"]
    pr_min = summary["bin0_peak_ratio_min"]
    pr_max = summary["bin0_peak_ratio_max"]
    dur_med = summary["bin0_inversion_duration_median_h"]
    dur_min = summary["bin0_inversion_duration_min_h"]
    dur_max = summary["bin0_inversion_duration_max_h"]

    # Multi-burn PM2.5-bias factor at +2 h: OPC-N3 bin0 ratio over the SMPS
    # 300-437 nm ratio, Bedroom 2 pairs with a co-located SMPS. The SMPS is the
    # only mechanistically independent same-size reference; the AeroTrak Ch1
    # shares the OPC artifact and is not used. This supersedes the earlier
    # single-Bedroom-2-burn framing and is independent of inversion presence.
    bias_rows = _bias_factor_rows(present)
    factors = np.array([row["factor"] for row in bias_rows], dtype=float)
    factors = factors[np.isfinite(factors)]
    n_smps = len(bias_rows)
    bias = dict(
        n=int(factors.size),
        median=float(np.median(factors)) if factors.size else np.nan,
        min=float(np.min(factors)) if factors.size else np.nan,
        max=float(np.max(factors)) if factors.size else np.nan,
    )
    bias_ref = "co-located SMPS 300-437 nm reference"

    # Phrase the bias factor honestly: one value for a single qualifying pair,
    # a median over a range for several.
    if bias["n"] == 1:
        bias_clause = (
            f"by a factor of approximately {_fmt(bias['median'], '.0f')} "
            f"(the single qualifying Bedroom 2 pair)"
        )
    elif bias["n"] >= 2:
        bias_clause = (
            f"by a factor of approximately {_fmt(bias['median'], '.0f')} "
            f"(median across {bias['n']} Bedroom 2 pairs with a co-located SMPS; "
            f"range {_fmt(bias['min'], '.0f')} to {_fmt(bias['max'], '.0f')})"
        )
    else:
        bias_clause = "by an undetermined factor (no qualifying SMPS reference data)"

    # Opening observation (saturated framing, per the prompt template).
    s_open = (
        f"Examination of the 5 s OPC-N3 records during the post-peak decay phase "
        f"reveals an anomalous behavior: in {n_sat_inv} of {n_sat} MODULAIR-PM "
        f"burn-unit pairs where the peak window produced nephelometer saturation, "
        f"the smallest OPC-N3 bins ({BIN0_LABEL} and {BIN1_LABEL}) report count "
        f"concentrations that increase during the decay phase, in the opposite "
        f"direction from the same-size-range counts simultaneously reported by "
        f"the co-located AeroTrak."
    )
    s_quant = (
        f"The inversion persists for a median of {_fmt(dur_med, '.1f')} hours "
        f"(range {_fmt(dur_min, '.1f')} to {_fmt(dur_max, '.1f')}) after the peak "
        f"window, during which the OPC-N3 bin 0 count rises to a median of "
        f"{_fmt(pr_med, '.0f')} times its value at the end of the peak window "
        f"(range {_fmt(pr_min, '.0f')} to {_fmt(pr_max, '.0f')})."
    )
    s_impl = (
        f"During the inversion phase, the portal-delivered PM2.5 product likely "
        f"overestimates true small-particle concentration; at the 2-hour mark, "
        f"the OPC-N3-implied bin 0 count is elevated {bias_clause} relative to "
        f"the {bias_ref}, the only co-located reference that is mechanistically "
        f"independent of the OPC scattering artifact."
    )
    s_distinct = (
        f"This post-peak behavior is observationally distinct from the bin "
        f"reversal artifact documented for the AeroTrak in Section 3.2.2: the "
        f"OPC-N3 inversion persists for hours rather than minutes and produces an "
        f"INCREASE rather than a decrease in small-bin counts. It is confined to "
        f"the optical particle counters: the SMPS, which classifies particles by "
        f"electrical mobility and is mechanistically independent of optical "
        f"artifacts, shows no corresponding rise over the same post-peak window."
    )

    # Mechanism statement, with the saturation-association clause only if the
    # cross-tabulation supports it (every saturated record also inverts).
    assoc = (summary["sat_and_inv"] == summary["n_saturated"]
             and summary["n_saturated"] > 0)
    if assoc:
        mech_assoc = (
            " The phenomenon is consistently associated with nephelometer "
            "saturation during the preceding peak window, suggesting a connection "
            "between peak-window loading and the post-peak behavior, but the "
            "causal pathway is not established."
        )
    else:
        mech_assoc = (
            " The phenomenon is not confined to records with prior nephelometer "
            "saturation; it also appears in records whose peak window never "
            "saturated, so it is not uniquely tied to saturation."
        )
    s_mech = (
        f"The physical mechanism for the OPC-N3 inversion is not determined from "
        f"the present data.{mech_assoc} Users of MODULAIR-PM instruments in "
        f"high-concentration fire smoke environments should be aware that the "
        f"portal-delivered PM2.5 data product may be elevated relative to true "
        f"concentrations for an extended period following the peak event."
    )

    # Brief main-text paragraph (about 90-130 words).
    n_inv = summary["n_inversion_present"]
    n_tested = summary["n_pairs_tested"]
    para_main = (
        f"The 5 s OPC-N3 records reveal a second anomaly during the post-peak "
        f"decay phase: in {n_inv} of {n_tested} burn-unit pairs (all Morning "
        f"Room) the smallest OPC-N3 bins ({BIN0_LABEL} and {BIN1_LABEL}) increase "
        f"in count as the smoke clears. The bin 0 count "
        f"rises to a median of {_fmt(pr_med, '.0f')} times its end-of-peak value "
        f"and the inversion persists for a median of {_fmt(dur_med, '.1f')} hours. "
        f"In Bedroom 2, where a co-located SMPS provides a same-size reference "
        f"that is independent of the OPC scattering artifact, the OPC-N3 bin 0 "
        f"reads elevated {bias_clause} relative to the SMPS at + 2 h. "
        f"Because the portal PM2.5 product is derived in part from these OPC-N3 "
        f"counts, the delivered PM2.5 likely overestimates true small-particle "
        f"concentration for an extended period after the peak. The full "
        f"characterization is given in the Supporting Information."
    )

    # Full SI subsection (about 250-400 words).
    rho = summary["spearman_rho_dur_vs_peakmass"]
    pval = summary["spearman_p_dur_vs_peakmass"]
    si = (
        f"Post-peak OPC-N3 small-bin inversion. The raw 5 s OPC-N3 records were "
        f"examined over the post-peak decay phase, defined from the end of the "
        f"peak window (t_peak_end) to ignition + 4 h or the AeroTrak PM3 return "
        f"to within 10% of the pre-burn baseline, whichever is earlier. A pair "
        f"was classified as showing an inversion when any of the three smallest "
        f"OPC-N3 bins (bin0 {BIN0_LABEL}, bin1 {BIN1_LABEL}, bin2 {BIN2_LABEL}) "
        f"rose more than {int(INVERSION_FRAC * 100)}% above its t_peak_end count "
        f"at + 0.5, + 1, or + 2 h. Of {n_tested} burn-unit pairs with a defined "
        f"peak window, {n_inv} showed an inversion: "
        f"{summary['n_saturated_with_inversion']} of {summary['n_saturated']} "
        f"records with prior nephelometer saturation and "
        f"{summary['n_unsaturated_with_inversion']} of "
        f"{summary['n_unsaturated']} records without it, so the inversion is not "
        f"confined to saturated peaks. Across the {summary['bin0_n_inversions']} "
        f"pairs with a bin-0 inversion the count rose to a median of "
        f"{_fmt(pr_med, '.0f')} times its t_peak_end value (range "
        f"{_fmt(pr_min, '.0f')} to {_fmt(pr_max, '.0f')}) and the inversion "
        f"persisted for a median of {_fmt(dur_med, '.1f')} hours (range "
        f"{_fmt(dur_min, '.1f')} to {_fmt(dur_max, '.1f')}). The co-located "
        f"AeroTrak Ch1 (0.3-0.5 um) is an OPC that shares the same scattering "
        f"artifact characterized in Section 3.2.2 and is therefore shown as a "
        f"qualitative companion (Figure S4) rather than an independent reference. "
        f"The only same-size reference that is mechanistically independent of the "
        f"OPC artifact is the SMPS 300-437 nm channel, which is deployed in "
        f"Bedroom 2 only. Across the {n_smps} Bedroom 2 pairs with a co-located "
        f"SMPS, the OPC-N3 bin 0 + 2 h count ratio was elevated {bias_clause} "
        f"relative to the SMPS 300-437 nm ratio (Figure S5), so the portal PM2.5 "
        f"reads high relative to the independent reference. Morning Room has no "
        f"co-located SMPS and therefore no valid reference for the bias. "
        f"Spearman rho between bin-0 inversion duration and "
        f"co-located peak AeroTrak PM3 mass was {_fmt(rho, '.2f')} "
        f"(p = {_fmt(pval, '.3f')}). The inversion is observationally distinct "
        f"from the AeroTrak bin reversal of Section 3.2.2 (hours not minutes, an "
        f"increase not a decrease, and confined to the OPC-N3 channel). The "
        f"physical mechanism is not determined from the present data. See SI "
        f"figures modulair_5sec_post_peak_<burn>_<unit>, "
        f"modulair_5sec_post_peak_smallmultiples, modulair_5sec_post_peak_overlay, "
        f"and modulair_5sec_pm25_bias."
    )

    text = (
        "# MODULAIR-PM 5 s post-peak inversion - manuscript sentences\n\n"
        "_All values derived from data. Brief main-text mention belongs at the "
        "end of Section 3.2.3 (NOT a standalone 3.2.4); full detail goes to the "
        "Supporting Information. Descriptive only; no mechanism proposed; no "
        "OPC-N3 coincidence threshold cited._\n\n---\n\n"
        f"**Opening observation:** \"{s_open}\"\n\n"
        f"**Quantification:** \"{s_quant}\"\n\n"
        f"**Implication:** \"{s_impl}\"\n\n"
        f"**Distinction from AeroTrak reversal:** \"{s_distinct}\"\n\n"
        f"**Mechanism statement:** \"{s_mech}\"\n\n"
        "---\n\n## Brief main-text paragraph (end of Section 3.2.3)\n\n"
        f"{para_main}\n\n"
        "---\n\n## Full SI subsection\n\n"
        f"{si}\n"
    )
    path = out_dir / "modulair_5sec_post_peak_manuscript_sentences.md"
    path.write_text(text, encoding="utf-8")
    print(f"    [MD] {path.name}")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Run the full post-peak inversion analysis: load, analyze, write outputs."""
    warnings.filterwarnings("ignore")
    apply_est_style()
    print("\n" + "=" * 70)
    print("MODULAIR-PM 5 s post-peak OPC-N3 inversion  -  Burns 4-10")
    print("=" * 70)

    out_dir = get_common_file("quantaq_analysis")
    fig_dir = get_common_file("quantaq_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading peak windows, events, AeroTrak, peak-mass...")
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
            if rec.get("data_present") and pd.notna(rec.get("t_peak_end")):
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
    _mpl_small_multiples(results, fig_dir)
    _mpl_overlay(results, fig_dir)
    _mpl_pm25_bias(results, fig_dir)

    print("\nWriting markdown outputs...")
    _write_summary_md(results, summary, out_dir)
    _write_manuscript_md(results, summary, out_dir)

    print("\nDone. Check quantaq_analysis/ and quantaq_figures/ for outputs.")


if __name__ == "__main__":
    main()
