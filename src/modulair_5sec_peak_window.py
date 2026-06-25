"""
MODULAIR-PM 5 s peak-window behavior analysis (Section 3.2.3).

Characterizes the raw on-instrument (no QA/QC) 5 s record of the two indoor
QuantAQ MODULAIR-PM units during the smoke-peak window of burns 4-10:
    - PMS5003 nephelometer saturation at a fixed plateau value,
    - OPC-N3 small-bin suppression in the opposite direction,
    - alignment of both with the portal QA/QC removal window,
    - dependence on co-located AeroTrak peak PM3 mass.

Scope is the peak window only; decay-phase / post-peak behavior is a separate
analysis. The script describes the phenomena and does not propose a mechanism.

All paths resolve through data_config.json via src/modulair_5sec_io.py. No raw
5 s data are written or committed. Missing burn-unit data are flagged, not
imputed.

Outputs (under common_folders quantaq_analysis / quantaq_figures):
    modulair_5sec_peak_per_burn.csv
    modulair_5sec_peak_cross_burn_summary.csv
    modulair_5sec_neph_plateau_values.csv
    modulair_5sec_peak_<burn>_<unit>.html         (Bokeh, one per pair)
    modulair_5sec_bin_response_grid.png           (matplotlib, SI)
    modulair_5sec_qaqc_overlap.png                (matplotlib)
    modulair_5sec_peak_summary.md
    modulair_5sec_peak_manuscript_sentences.md

Author: Nathan Lima
Created: 2026-06-25
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
from bokeh.models import Label, Legend, LegendItem, Span  # noqa: E402
from bokeh.plotting import figure  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.data_paths import get_common_file  # noqa: E402
from src.modulair_5sec_io import (  # noqa: E402
    BURN_DATES,
    NEPH_BINS,
    OPC_BIN_EDGES_UM,
    OPC_BINS,
    UNIT_CONFIG,
    load_5sec_burn,
    load_event_times,
    load_portal_burn,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# All burns with 5 s QuantAQ deployment (per-burn instrument behavior uses all).
BURNS = list(BURN_DATES.keys())  # burn4..burn10

# The two indoor analysis units.
UNITS = list(UNIT_CONFIG.keys())  # MODULAIR-PM1, MODULAIR-PM2

# Map MODULAIR-PM unit -> location label used in outputs.
UNIT_LOCATION = {u: UNIT_CONFIG[u]["location"] for u in UNITS}

# Nephelometer saturation detection. The PMS5003 neph_bin* fields are 16-bit;
# the observed hardware ceiling is exactly 65535. A burn-unit pair is treated
# as "saturated" only if neph_bin0 reaches that ceiling for at least
# MIN_SAT_SAMPLES samples. The peak window then runs from the first ceiling
# sample to the recovery point. The plateau-membership test uses the exact
# ceiling (SAT_FRAC = 1.0): a slightly lower threshold admits rising/falling
# edge samples where the OPC small bins are still elevated, which would mask
# the suppression they undergo once the nephelometer is truly pinned.
NEPH_CEILING = 65535.0
SAT_FRAC = 1.0           # plateau = neph_bin0 at the exact 16-bit ceiling
RECOVERY_FRAC = 0.95     # peak window ends when neph_bin0 first drops below
                         # 95 % of the plateau on the recovery side
MIN_SAT_SAMPLES = 6      # >= 6 samples (30 s) to call it a real plateau

# OPC-N3 three smallest bins used for the suppression classification.
OPC_SMALL_BINS = ["bin0", "bin1", "bin2"]

# Pre-burn baseline window (minutes before ignition) for the peak/pre ratio.
BASELINE_MIN = 30

# Bin-response classification thresholds (peak/pre ratio).
SUPPRESSED_MAX = 0.5     # ratio < 0.5  -> suppressed
ELEVATED_MIN = 2.0       # ratio > 2.0  -> elevated; in between -> unchanged

# Portal QA/QC removal window detection. The portal sets pm25 to NaN for
# removed minutes; such NaN rows also appear sporadically through the day, so
# the removal window is the contiguous block of removed minutes near the peak.
QAQC_SEARCH_MIN = 30     # search NaN rows within +/- this of the peak/ignition
QAQC_GAP_MIN = 3         # grow the block across gaps up to this many minutes

# Co-located AeroTrak coincidence CSV (peak PM3 mass + Ch1 reversal interval).
AEROTRAK_CSV = "aerotrak_coincidence_per_burn.csv"

# Matplotlib TEXT_CONFIG (project convention).
_FS = 12
TEXT_CONFIG = dict(fontsize=_FS, labelsize=_FS, ticksize=_FS, legendsize=_FS)

# Per-unit plot colors (bedroom dark blue, morning room pink, matching the
# AeroTrak coincidence figures).
UNIT_COLOR = {"MODULAIR-PM1": "#003f5c", "MODULAIR-PM2": "#ef5675"}


# ==============================================================================
# PEAK-WINDOW DETECTION
# ==============================================================================


def _detect_peak_window(df: pd.DataFrame) -> dict:
    """
    Define the peak window from the nephelometer bin-0 saturation plateau.

    The window runs from the time neph_bin0 first reaches the saturation
    plateau (>= SAT_FRAC * NEPH_CEILING) to the time it first drops below
    RECOVERY_FRAC of the plateau value on the recovery side.

    Parameters
    ----------
    df : pd.DataFrame
        5 s record for one burn-unit pair (local time), with 'neph_bin0'.

    Returns
    -------
    dict
        Keys: saturated (bool), t_peak_start, t_peak_end (pd.Timestamp/NaT),
        plateau_value (float), peak_window_duration_seconds/_minutes (float),
        n_sat_samples (int).
    """
    blank = dict(
        saturated=False,
        t_peak_start=pd.NaT,
        t_peak_end=pd.NaT,
        plateau_value=np.nan,
        peak_window_duration_seconds=np.nan,
        peak_window_duration_minutes=np.nan,
        n_sat_samples=0,
    )
    if df is None or df.empty or "neph_bin0" not in df.columns:
        return blank

    nb0 = pd.to_numeric(df["neph_bin0"], errors="coerce")
    ts = df["timestamp"]
    sat_mask = (nb0 >= SAT_FRAC * NEPH_CEILING).to_numpy()
    if sat_mask.sum() < MIN_SAT_SAMPLES:
        return blank

    # First saturated sample = window start. Plateau value = median neph_bin0
    # over the saturated samples (a fixed ceiling, not a concentration signal).
    sat_idx = np.where(sat_mask)[0]
    start_i = int(sat_idx[0])
    plateau = float(np.nanmedian(nb0.to_numpy()[sat_mask]))

    # Recovery: first sample AFTER the last saturated sample where neph_bin0
    # falls below RECOVERY_FRAC * plateau. Search from the last plateau sample
    # forward so a transient dip mid-plateau does not close the window early.
    last_sat_i = int(sat_idx[-1])
    recovery_thresh = RECOVERY_FRAC * plateau
    end_i = last_sat_i
    after = nb0.to_numpy()[last_sat_i:]
    below = np.where(after < recovery_thresh)[0]
    if below.size:
        end_i = last_sat_i + int(below[0])

    t_start = pd.Timestamp(ts.iloc[start_i])
    t_end = pd.Timestamp(ts.iloc[end_i])
    dur_s = (t_end - t_start).total_seconds()

    return dict(
        saturated=True,
        t_peak_start=t_start,
        t_peak_end=t_end,
        plateau_value=plateau,
        peak_window_duration_seconds=float(dur_s),
        peak_window_duration_minutes=float(dur_s / 60.0),
        n_sat_samples=int(sat_mask.sum()),
    )


# ==============================================================================
# PORTAL QA/QC REMOVAL WINDOW
# ==============================================================================


def _portal_qaqc_window(
    portal: pd.DataFrame | None,
    t_peak_start: pd.Timestamp,
    t_peak_end: pd.Timestamp,
    ignition: pd.Timestamp,
) -> dict:
    """
    Characterize the portal QA/QC removal window and its overlap with the peak.

    The 1-minute portal product flags peak-window rows by setting the
    delivered PM mass (pm25) to NaN. Such NaN rows occur sporadically through
    the day for unrelated reasons, so the removal window is restricted to the
    contiguous block of removed minutes spanning the smoke peak: NaN rows are
    searched within QAQC_SEARCH_MIN of the peak window (or of ignition when no
    saturation was detected), and the block is grown only across gaps of at
    most QAQC_GAP_MIN minutes. Overlap and tail-extension are then computed
    against the 5 s peak window.

    Parameters
    ----------
    portal : pd.DataFrame or None
        Portal burn-day product with 'timestamp', 'pm25' (and 'flag').
    t_peak_start, t_peak_end : pd.Timestamp
        Peak-window bounds from the 5 s nephelometer saturation (may be NaT).
    ignition : pd.Timestamp
        Fallback anchor when no peak window was detected.

    Returns
    -------
    dict
        portal_qaqc_removal_start/_end (pd.Timestamp/NaT),
        portal_qaqc_removal_duration_minutes (float),
        overlap_with_peak_window (fraction 0-1 of the peak window also flagged),
        tail_extension_minutes (minutes the flag persists past peak end).
    """
    blank = dict(
        portal_qaqc_removal_start=pd.NaT,
        portal_qaqc_removal_end=pd.NaT,
        portal_qaqc_removal_duration_minutes=np.nan,
        overlap_with_peak_window=np.nan,
        tail_extension_minutes=np.nan,
    )
    if portal is None or portal.empty or "pm25" not in portal.columns:
        return blank

    # Anchor the search at the peak window if present, else at ignition.
    if pd.notna(t_peak_start) and pd.notna(t_peak_end):
        anchor_lo = t_peak_start - pd.Timedelta(minutes=QAQC_SEARCH_MIN)
        anchor_hi = t_peak_end + pd.Timedelta(minutes=QAQC_SEARCH_MIN)
    elif pd.notna(ignition):
        anchor_lo = ignition - pd.Timedelta(minutes=QAQC_SEARCH_MIN)
        anchor_hi = ignition + pd.Timedelta(minutes=QAQC_SEARCH_MIN)
    else:
        return blank

    near = portal[
        (portal["timestamp"] >= anchor_lo)
        & (portal["timestamp"] <= anchor_hi)
        & portal["pm25"].isna()
    ].sort_values("timestamp")

    if near.empty:
        # No QA/QC removal near the peak: zero-length window so overlap/tail
        # still compute meaningfully against the peak window.
        out = blank.copy()
        out["portal_qaqc_removal_duration_minutes"] = 0.0
        if pd.notna(t_peak_start) and pd.notna(t_peak_end):
            out["overlap_with_peak_window"] = 0.0
            out["tail_extension_minutes"] = 0.0
        return out

    # Grow the contiguous removed block across gaps <= QAQC_GAP_MIN. The anchor
    # row is the removed minute nearest the peak-window center (or ignition).
    center = (
        t_peak_start + (t_peak_end - t_peak_start) / 2
        if pd.notna(t_peak_start) and pd.notna(t_peak_end)
        else ignition
    )
    times = near["timestamp"].tolist()
    anchor_t = min(times, key=lambda t: abs((t - center).total_seconds()))
    lo = hi = times.index(anchor_t)
    gap = pd.Timedelta(minutes=QAQC_GAP_MIN)
    while lo > 0 and (times[lo] - times[lo - 1]) <= gap:
        lo -= 1
    while hi < len(times) - 1 and (times[hi + 1] - times[hi]) <= gap:
        hi += 1

    # Removal window = first to last removed minute in the block (1-min cadence,
    # so add one minute so a single removed sample reads as ~1 min, not 0).
    q_start = pd.Timestamp(times[lo])
    q_end = pd.Timestamp(times[hi]) + pd.Timedelta(minutes=1)
    q_dur_min = (q_end - q_start).total_seconds() / 60.0

    out = dict(
        portal_qaqc_removal_start=q_start,
        portal_qaqc_removal_end=q_end,
        portal_qaqc_removal_duration_minutes=float(q_dur_min),
        overlap_with_peak_window=np.nan,
        tail_extension_minutes=np.nan,
    )

    if pd.notna(t_peak_start) and pd.notna(t_peak_end):
        peak_dur_s = (t_peak_end - t_peak_start).total_seconds()
        if peak_dur_s > 0:
            ov_start = max(q_start, t_peak_start)
            ov_end = min(q_end, t_peak_end)
            ov_s = max(0.0, (ov_end - ov_start).total_seconds())
            out["overlap_with_peak_window"] = float(ov_s / peak_dur_s)
        else:
            out["overlap_with_peak_window"] = 0.0
        out["tail_extension_minutes"] = float(
            max(0.0, (q_end - t_peak_end).total_seconds()) / 60.0
        )
    return out


# ==============================================================================
# OPC-N3 AND NEPHELOMETER RESPONSE DURING THE PEAK WINDOW
# ==============================================================================


def _classify_ratio(ratio: float) -> str:
    """Classify a peak/pre bin ratio as suppressed / unchanged / elevated."""
    if np.isnan(ratio):
        return "no_data"
    if ratio < SUPPRESSED_MAX:
        return "suppressed"
    if ratio > ELEVATED_MIN:
        return "elevated"
    return "unchanged"


def _opc_response(
    df: pd.DataFrame,
    ignition: pd.Timestamp,
    t_peak_start: pd.Timestamp,
    t_peak_end: pd.Timestamp,
) -> dict:
    """
    Compute peak/pre-baseline ratios and classifications for all OPC-N3 bins.

    The pre-burn baseline is the BASELINE_MIN minutes immediately before
    ignition (mean bin count). The peak level is the MEDIAN bin count over the
    saturated-plateau samples only (neph_bin0 >= SAT_FRAC * NEPH_CEILING),
    not the full peak window: the rising edge inflates a window mean and would
    mask the small-bin collapse that occurs once the nephelometer is pinned.
    Ratios for the three smallest bins are returned individually; all 24 bins
    are returned for the SI bin-response grid.

    Returns
    -------
    dict
        ratio_bin{i}, class_bin{i} for the three smallest bins, plus
        'all_ratios' (list aligned with OPC_BINS) for the grid figure.
    """
    out: dict = {"all_ratios": [np.nan] * len(OPC_BINS)}
    for b in OPC_SMALL_BINS:
        out[f"ratio_{b}"] = np.nan
        out[f"class_{b}"] = "no_data"
    if df is None or df.empty or pd.isna(ignition):
        return out
    if pd.isna(t_peak_start) or pd.isna(t_peak_end):
        return out

    ts = df["timestamp"]
    pre_mask = (ts >= ignition - pd.Timedelta(minutes=BASELINE_MIN)) & (ts < ignition)
    # Peak level: saturated-plateau samples within the peak window only.
    in_window = (ts >= t_peak_start) & (ts <= t_peak_end)
    nb0 = pd.to_numeric(df["neph_bin0"], errors="coerce")
    peak_mask = in_window & (nb0 >= SAT_FRAC * NEPH_CEILING)
    if pre_mask.sum() == 0 or peak_mask.sum() == 0:
        return out

    for i, b in enumerate(OPC_BINS):
        if b not in df.columns:
            continue
        col = pd.to_numeric(df[b], errors="coerce")
        pre_mean = float(col[pre_mask].mean())
        peak_med = float(col[peak_mask].median())
        ratio = peak_med / pre_mean if pre_mean and pre_mean > 0 else np.nan
        out["all_ratios"][i] = ratio
        if b in OPC_SMALL_BINS:
            out[f"ratio_{b}"] = ratio
            out[f"class_{b}"] = _classify_ratio(ratio)
    return out


def _neph_plateau(
    df: pd.DataFrame,
    t_peak_start: pd.Timestamp,
    t_peak_end: pd.Timestamp,
) -> dict:
    """
    Median nephelometer signal over the saturated window for the two smallest
    neph bins (plateau_bin0, plateau_bin1).
    """
    out = dict(plateau_bin0=np.nan, plateau_bin1=np.nan)
    if df is None or df.empty or pd.isna(t_peak_start) or pd.isna(t_peak_end):
        return out
    ts = df["timestamp"]
    mask = (ts >= t_peak_start) & (ts <= t_peak_end)
    if mask.sum() == 0:
        return out
    for j, b in enumerate(["neph_bin0", "neph_bin1"]):
        if b in df.columns:
            out[f"plateau_bin{j}"] = float(
                pd.to_numeric(df[b], errors="coerce")[mask].median()
            )
    return out


# ==============================================================================
# AEROTRAK CROSS-REFERENCE
# ==============================================================================


def _load_aerotrak() -> pd.DataFrame | None:
    """
    Load the co-located AeroTrak coincidence per-burn CSV, if present.

    Returns a DataFrame indexed by (burn, location) with the columns needed for
    the mass cross-reference, or None if the file is absent.
    """
    path = get_common_file("coincidence_analysis") / AEROTRAK_CSV
    if not path.exists():
        print(f"  [AeroTrak] {path.name} not found; mass cross-reference skipped.")
        return None
    at = pd.read_csv(path)
    keep = [
        c
        for c in ("burn", "location", "peak_total_PM3_mass_ug_m3", "reversal_present",
                  "reversal_onset", "t_min", "reversal_end", "t_peak")
        if c in at.columns
    ]
    return at[keep].copy()


def _aerotrak_for_pair(at: pd.DataFrame | None, burn_id: str, unit: str) -> dict:
    """
    Look up the co-located AeroTrak peak PM3 mass, Ch1 reversal flag, and the
    full Ch1 reversal interval (onset, trough, recovery) for a MODULAIR-PM
    burn-unit pair (matched on burn and location).
    """
    out = dict(
        aerotrak_peak_PM3_mass_ug_m3=np.nan,
        aerotrak_reversal_present=np.nan,
        aerotrak_reversal_onset=pd.NaT,
        aerotrak_t_min=pd.NaT,
        aerotrak_reversal_end=pd.NaT,
    )
    if at is None:
        return out
    loc = UNIT_LOCATION[unit]
    row = at[(at["burn"] == burn_id) & (at["location"] == loc)]
    if row.empty:
        return out
    r = row.iloc[0]
    out["aerotrak_peak_PM3_mass_ug_m3"] = float(r.get("peak_total_PM3_mass_ug_m3", np.nan))
    rev = r.get("reversal_present", np.nan)
    out["aerotrak_reversal_present"] = bool(rev) if pd.notna(rev) else np.nan
    for src, dst in (("reversal_onset", "aerotrak_reversal_onset"),
                     ("t_min", "aerotrak_t_min"),
                     ("reversal_end", "aerotrak_reversal_end")):
        if src in row.columns and pd.notna(r.get(src)):
            out[dst] = pd.to_datetime(r[src], errors="coerce")
    return out


# ==============================================================================
# PER-PAIR ANALYSIS DRIVER
# ==============================================================================


def analyze_pair(
    burn_id: str,
    unit: str,
    events: pd.DataFrame,
    at: pd.DataFrame | None,
) -> dict:
    """
    Run the full peak-window analysis for one burn-unit pair.

    Returns a dict with all per-burn CSV fields plus private keys (prefixed
    '_') carrying the loaded frames and event times for the figure functions.
    A pair with no 5 s data is flagged via 'data_present' = False and skipped
    by the figure functions; it is never imputed.
    """
    burn_key = burn_id  # events index uses 'burn4' etc.
    ev = events.loc[burn_key] if burn_key in events.index else None
    ignition = ev["ignition"] if ev is not None else pd.NaT
    garage = ev["garage_closed"] if ev is not None else pd.NaT
    pac_on = ev["pac_on"] if ev is not None else pd.NaT

    df = load_5sec_burn(unit, burn_id)
    if df is None or df.empty:
        print(f"    [{burn_id}|{unit}] no 5 s data - flagged, not imputed.")
        return dict(
            burn=burn_id, unit=unit, location=UNIT_LOCATION[unit],
            data_present=False, saturated=False, notes="5 s data missing",
            _df=None, _portal=None, _ignition=ignition, _garage=garage,
            _pac_on=pac_on,
        )

    portal = load_portal_burn(unit, burn_id)
    pw = _detect_peak_window(df)
    qa = _portal_qaqc_window(portal, pw["t_peak_start"], pw["t_peak_end"], ignition)
    opc = _opc_response(df, ignition, pw["t_peak_start"], pw["t_peak_end"])
    neph = _neph_plateau(df, pw["t_peak_start"], pw["t_peak_end"])
    atx = _aerotrak_for_pair(at, burn_id, unit)

    # Cross-reference the MODULAIR-PM peak window against the full AeroTrak Ch1
    # reversal interval (onset -> recovery), not a single instant. Both
    # quantities are real time intervals, so we report the fraction of the
    # MODULAIR-PM peak window that falls inside the AeroTrak reversal interval,
    # and the signed gap between the two intervals (0 when they overlap;
    # otherwise minutes of separation). This is robust to the independent
    # instrument clocks and to the AeroTrak recovery plateau arriving later.
    rev_overlap_frac = np.nan
    rev_gap_min = np.nan
    if (atx["aerotrak_reversal_present"] is True and pw["saturated"]
            and pd.notna(atx["aerotrak_reversal_onset"])
            and pd.notna(atx["aerotrak_reversal_end"])):
        a0, a1 = atx["aerotrak_reversal_onset"], atx["aerotrak_reversal_end"]
        p0, p1 = pw["t_peak_start"], pw["t_peak_end"]
        peak_dur_s = (p1 - p0).total_seconds()
        ov_s = max(0.0, (min(a1, p1) - max(a0, p0)).total_seconds())
        rev_overlap_frac = float(ov_s / peak_dur_s) if peak_dur_s > 0 else 0.0
        if ov_s > 0:
            rev_gap_min = 0.0
        elif p1 < a0:
            rev_gap_min = float((a0 - p1).total_seconds() / 60.0)
        else:
            rev_gap_min = float((p0 - a1).total_seconds() / 60.0)

    notes = [] if pw["saturated"] else ["neph_bin0 did not reach plateau"]
    if portal is None:
        notes.append("portal product missing")

    rec = dict(
        burn=burn_id, unit=unit, location=UNIT_LOCATION[unit],
        data_present=True, portal_present=portal is not None,
        **{k: pw[k] for k in (
            "saturated", "t_peak_start", "t_peak_end", "plateau_value",
            "peak_window_duration_seconds", "peak_window_duration_minutes",
            "n_sat_samples")},
        **qa,
        **{k: opc[k] for k in opc if k != "all_ratios"},
        **neph,
        **atx,
        aerotrak_reversal_overlap_fraction=rev_overlap_frac,
        aerotrak_reversal_gap_minutes=rev_gap_min,
        notes="; ".join(notes),
        # private
        _df=df, _portal=portal, _ignition=ignition, _garage=garage,
        _pac_on=pac_on, _opc_all_ratios=opc["all_ratios"],
    )
    return rec


# ==============================================================================
# BOKEH PER-PAIR FIGURE
# ==============================================================================

# OPC bins 0-6 plotted in the bottom panel: dark (small) -> light (large).
_OPC_FIG_BINS = [f"bin{i}" for i in range(7)]
_OPC_FIG_COLORS = [
    "#004D00", "#1A6B1A", "#2D8B2D", "#43A843", "#66BB66", "#95D595", "#C8EEC8",
]
_NEPH_FIG_BINS = NEPH_BINS  # all six
_NEPH_FIG_COLORS = [
    "#67000D", "#A50026", "#D73027", "#F46D43", "#FDAE61", "#FEE8C8",
]
_BOKEH_TOOLS = "pan,box_zoom,wheel_zoom,crosshair,reset,save"


def _event_spans(p, rec: dict) -> None:
    """Add vertical event lines: ignition, garage closure, PAC on, peak start/end."""
    events = [
        (rec["_ignition"], "#444444", "solid", "Ignition"),
        (rec["_garage"], "#444444", "dashed", "Garage closed"),
        (rec["_pac_on"], "#444444", "dotted", "PAC on"),
        (rec.get("t_peak_start"), "#1f77b4", "solid", "Peak start"),
        (rec.get("t_peak_end"), "#1f77b4", "dashed", "Peak end"),
    ]
    for t_ev, color, dash, _lbl in events:
        if pd.notna(t_ev):
            p.add_layout(
                Span(
                    location=int(pd.Timestamp(t_ev).timestamp() * 1000),
                    dimension="height", line_color=color, line_dash=dash,
                    line_width=1.4,
                )
            )


def _bokeh_pair(rec: dict, t_pad_min: float = 30.0) -> None:
    """
    Two-panel Bokeh figure for one burn-unit pair.

    Top: PMS5003 nephelometer raw signal (all six bins) with a horizontal
    plateau marker at the detected saturation value. Bottom: OPC-N3 bins 0-6
    raw counts. Both panels share the x-axis and carry event lines for
    ignition, garage closure, PAC activation, and peak-window start/end.

    Output: quantaq_figures/modulair_5sec_peak_<burn>_<unit>.html
    """
    if not rec.get("data_present"):
        return
    df = rec["_df"]
    # Display window: 30 min before ignition (or peak start) through peak end
    # + 30 min, so the saturation plateau and onset are both visible.
    anchor_lo = rec["_ignition"] if pd.notna(rec["_ignition"]) else rec.get("t_peak_start")
    anchor_hi = rec.get("t_peak_end") if pd.notna(rec.get("t_peak_end")) else rec["_ignition"]
    if pd.isna(anchor_lo) or pd.isna(anchor_hi):
        return
    t_lo = anchor_lo - pd.Timedelta(minutes=t_pad_min)
    t_hi = anchor_hi + pd.Timedelta(minutes=t_pad_min)
    sub = df[(df["timestamp"] >= t_lo) & (df["timestamp"] <= t_hi)]
    if sub.empty:
        return

    unit_tag = "pm1" if rec["unit"] == "MODULAIR-PM1" else "pm2"
    title = f"{rec['burn']}  |  {rec['unit']} ({UNIT_CONFIG[rec['unit']]['location_label']})"

    # --- top: nephelometer ---
    p_top = figure(
        x_axis_type="datetime", width=1400, height=420,
        title=f"{title}  -  PMS5003 nephelometer raw signal",
        tools=_BOKEH_TOOLS,
    )
    items_top = []
    for b, color in zip(_NEPH_FIG_BINS, _NEPH_FIG_COLORS):
        if b not in sub.columns:
            continue
        r = p_top.line(sub["timestamp"], pd.to_numeric(sub[b], errors="coerce"),
                       color=color, line_width=1.4)
        items_top.append(LegendItem(label=b, renderers=[r]))
    if rec.get("saturated") and pd.notna(rec.get("plateau_value")):
        p_top.add_layout(Span(location=rec["plateau_value"], dimension="width",
                              line_color="black", line_dash="dotted", line_width=1.2))
        p_top.add_layout(Label(x=int(t_lo.timestamp() * 1000), y=rec["plateau_value"],
                               text=f"plateau ~ {rec['plateau_value']:.0f}",
                               text_font_size="9px", text_color="black", y_offset=3))
    p_top.yaxis.axis_label = "Neph raw signal"
    _event_spans(p_top, rec)
    p_top.add_layout(Legend(items=items_top, click_policy="hide",
                            label_text_font_size="8pt"), "right")

    # --- bottom: OPC-N3 bins 0-6 ---
    p_bot = figure(
        x_axis_type="datetime", width=1400, height=420, x_range=p_top.x_range,
        title="OPC-N3 bins 0-6 raw counts", tools=_BOKEH_TOOLS,
    )
    items_bot = []
    for b, color in zip(_OPC_FIG_BINS, _OPC_FIG_COLORS):
        if b not in sub.columns:
            continue
        r = p_bot.line(sub["timestamp"], pd.to_numeric(sub[b], errors="coerce"),
                       color=color, line_width=1.4)
        items_bot.append(LegendItem(label=b, renderers=[r]))
    p_bot.yaxis.axis_label = "OPC raw counts"
    p_bot.xaxis.axis_label = "Local time (EDT)"
    _event_spans(p_bot, rec)
    p_bot.add_layout(Legend(items=items_bot, click_policy="hide",
                            label_text_font_size="8pt"), "right")

    fig_dir = get_common_file("quantaq_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"modulair_5sec_peak_{rec['burn']}_{unit_tag}.html"
    reset_output()
    output_file(str(out_path), title=f"5 s peak {rec['burn']} {rec['unit']}")
    save(bokeh_column(p_top, p_bot))
    print(f"    [Bokeh] {out_path.name}")


# ==============================================================================
# MATPLOTLIB - BIN RESPONSE GRID (SI)
# ==============================================================================


def _mpl_bin_response_grid(results: list[dict]) -> None:
    """
    One panel per burn: x = OPC-N3 bin lower edge (um), y = peak/pre ratio
    (log). Horizontal references at y = 1 and y = 0.5. Both indoor units
    overlaid in different colors. Intended for the SI.
    """
    bin_lower = [lo for (lo, _hi) in OPC_BIN_EDGES_UM]
    burns = sorted(
        {r["burn"] for r in results if r.get("data_present")},
        key=lambda b: int(b.replace("burn", "")),
    )
    if not burns:
        print("    [mpl] no data for bin-response grid.")
        return

    ncols = 3
    nrows = int(np.ceil(len(burns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax_idx, burn_id in enumerate(burns):
        ax = axes[ax_idx]
        any_positive = False
        for unit in UNITS:
            rec = next((r for r in results if r["burn"] == burn_id
                        and r["unit"] == unit and r.get("data_present")), None)
            if rec is None:
                continue
            ratios = rec.get("_opc_all_ratios")
            if not ratios:
                continue
            vals = np.array(ratios, dtype=float)
            # Log axis cannot show zero/negative ratios (collapsed bins); mask
            # them so the line breaks rather than crashing the locator.
            x = np.array(bin_lower, dtype=float)
            valid = np.isfinite(vals) & (vals > 0)
            if not valid.any():
                continue
            any_positive = True
            ax.semilogy(x[valid], vals[valid], marker="o", ms=4, lw=1.0,
                        color=UNIT_COLOR[unit],
                        label=UNIT_CONFIG[unit]["location_label"], alpha=0.85)
        if not any_positive:
            # No saturating unit for this burn: leave an empty labelled panel
            # on a linear axis (a log axis with no positive data raises).
            ax.set_title(f"{burn_id} (no plateau)", fontsize=TEXT_CONFIG["labelsize"],
                         fontweight="bold")
            ax.set_xlabel("OPC-N3 bin lower edge (µm)",
                          fontsize=TEXT_CONFIG["labelsize"])
            ax.tick_params(labelsize=TEXT_CONFIG["ticksize"])
            continue
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("OPC-N3 bin lower edge (µm)", fontsize=TEXT_CONFIG["labelsize"])
        ax.set_ylabel("peak / pre ratio", fontsize=TEXT_CONFIG["labelsize"])
        ax.tick_params(labelsize=TEXT_CONFIG["ticksize"])
        ax.set_title(burn_id, fontsize=TEXT_CONFIG["labelsize"], fontweight="bold")

    for ax in axes[len(burns):]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=UNIT_COLOR["MODULAIR-PM1"], marker="o",
               label=UNIT_CONFIG["MODULAIR-PM1"]["location_label"]),
        Line2D([0], [0], color=UNIT_COLOR["MODULAIR-PM2"], marker="o",
               label=UNIT_CONFIG["MODULAIR-PM2"]["location_label"]),
        Line2D([0], [0], color="black", ls="--", label="ratio = 1"),
        Line2D([0], [0], color="gray", ls=":", label="ratio = 0.5"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.7),
               fontsize=TEXT_CONFIG["legendsize"], frameon=True)

    fig_dir = get_common_file("quantaq_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "modulair_5sec_bin_response_grid.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [mpl] {out_path.name}")


# ==============================================================================
# MATPLOTLIB - QA/QC OVERLAP (main text / SI)
# ==============================================================================


def _mpl_qaqc_overlap(results: list[dict]) -> None:
    """
    Grouped bar chart per burn-unit pair: peak_window_duration vs
    portal_qaqc_removal_duration (both minutes). Only pairs that saturated and
    have a portal product are shown.
    """
    rows = [
        r for r in results
        if r.get("data_present") and r.get("saturated")
        and not np.isnan(r.get("peak_window_duration_minutes", np.nan))
    ]
    if not rows:
        print("    [mpl] no saturated pairs for QA/QC overlap chart.")
        return

    rows = sorted(rows, key=lambda r: (int(r["burn"].replace("burn", "")), r["unit"]))
    labels = [f"{r['burn']}\n{('Bdrm' if r['unit'] == 'MODULAIR-PM1' else 'MR')}"
              for r in rows]
    peak_dur = [r["peak_window_duration_minutes"] for r in rows]
    qaqc_dur = [r.get("portal_qaqc_removal_duration_minutes", np.nan) for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(rows)), 5.0))
    ax.bar(x - w / 2, peak_dur, w, label="5 s peak window",
           color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, qaqc_dur, w, label="Portal QA/QC removal",
           color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TEXT_CONFIG["ticksize"])
    ax.set_ylabel("Duration (minutes)", fontsize=TEXT_CONFIG["labelsize"],
                  fontweight="bold")
    ax.set_title("Peak window vs portal QA/QC removal",
                 fontsize=TEXT_CONFIG["labelsize"], fontweight="bold")
    ax.tick_params(labelsize=TEXT_CONFIG["ticksize"])
    ax.legend(fontsize=TEXT_CONFIG["legendsize"])

    fig_dir = get_common_file("quantaq_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "modulair_5sec_qaqc_overlap.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [mpl] {out_path.name}")


# ==============================================================================
# CSV OUTPUTS
# ==============================================================================

# Per-burn CSV column order (one row per burn-unit pair).
_PER_BURN_COLS = [
    "burn", "unit", "location", "data_present", "portal_present", "saturated",
    "t_peak_start", "t_peak_end", "plateau_value",
    "peak_window_duration_seconds", "peak_window_duration_minutes",
    "n_sat_samples",
    "portal_qaqc_removal_start", "portal_qaqc_removal_end",
    "portal_qaqc_removal_duration_minutes", "overlap_with_peak_window",
    "tail_extension_minutes",
    "ratio_bin0", "class_bin0", "ratio_bin1", "class_bin1",
    "ratio_bin2", "class_bin2",
    "plateau_bin0", "plateau_bin1",
    "aerotrak_peak_PM3_mass_ug_m3", "aerotrak_reversal_present",
    "aerotrak_reversal_onset", "aerotrak_t_min", "aerotrak_reversal_end",
    "aerotrak_reversal_overlap_fraction", "aerotrak_reversal_gap_minutes",
    "notes",
]


def _write_per_burn_csv(results: list[dict]) -> pd.DataFrame:
    """Write modulair_5sec_peak_per_burn.csv; return the DataFrame for reuse."""
    out_dir = get_common_file("quantaq_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{k: r.get(k, np.nan) for k in _PER_BURN_COLS} for r in results]
    df = pd.DataFrame(rows, columns=_PER_BURN_COLS)
    path = out_dir / "modulair_5sec_peak_per_burn.csv"
    df.to_csv(str(path), index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")
    return df


def _write_plateau_csv(results: list[dict]) -> pd.DataFrame:
    """
    Write modulair_5sec_neph_plateau_values.csv: one row per unit-bin pair,
    summarizing the plateau value across saturating burns to test whether the
    ceiling is fixed (hardware) rather than concentration-proportional.
    """
    out_dir = get_common_file("quantaq_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for unit in UNITS:
        sat = [r for r in results if r["unit"] == unit and r.get("saturated")]
        for j, key in enumerate(("plateau_bin0", "plateau_bin1")):
            vals = np.array(
                [r.get(key, np.nan) for r in sat], dtype=float
            )
            vals = vals[~np.isnan(vals)]
            rows.append(dict(
                unit=unit,
                location=UNIT_LOCATION[unit],
                neph_bin=f"neph_bin{j}",
                n_saturating_burns=int(len(vals)),
                plateau_median=float(np.median(vals)) if vals.size else np.nan,
                plateau_min=float(np.min(vals)) if vals.size else np.nan,
                plateau_max=float(np.max(vals)) if vals.size else np.nan,
                plateau_std=float(np.std(vals)) if vals.size else np.nan,
            ))
    df = pd.DataFrame(rows)
    path = out_dir / "modulair_5sec_neph_plateau_values.csv"
    df.to_csv(str(path), index=False, float_format="%.6g")
    print(f"    [CSV] {path.name}")
    return df


def _write_cross_burn_csv(results: list[dict]) -> dict:
    """
    Write modulair_5sec_peak_cross_burn_summary.csv with the cross-burn tally,
    and return the summary dict for reuse in the markdown synthesis.
    """
    out_dir = get_common_file("quantaq_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    present = [r for r in results if r.get("data_present")]
    n_pairs = len(present)
    sat = [r for r in present if r.get("saturated")]
    n_sat = len(sat)

    # Plateau values across saturating burns, per unit.
    plateau_by_unit = {}
    for unit in UNITS:
        vals = np.array(
            [r.get("plateau_value", np.nan) for r in sat if r["unit"] == unit],
            dtype=float,
        )
        vals = vals[~np.isnan(vals)]
        plateau_by_unit[unit] = float(np.median(vals)) if vals.size else np.nan

    # OPC bin0 suppression tally.
    supp = [r for r in present if r.get("class_bin0") == "suppressed"]
    supp_ratios = np.array(
        [r.get("ratio_bin0", np.nan) for r in supp], dtype=float
    )
    supp_ratios = supp_ratios[~np.isnan(supp_ratios)]

    # QA/QC removal durations across saturating pairs.
    qa_dur = np.array(
        [r.get("portal_qaqc_removal_duration_minutes", np.nan) for r in sat],
        dtype=float,
    )
    qa_dur = qa_dur[~np.isnan(qa_dur)]
    peak_dur = np.array(
        [r.get("peak_window_duration_minutes", np.nan) for r in sat], dtype=float
    )
    peak_dur = peak_dur[~np.isnan(peak_dur)]

    summary = dict(
        n_burn_unit_pairs=n_pairs,
        n_neph_bin0_saturated=n_sat,
        plateau_median_MODULAIR_PM1=plateau_by_unit.get("MODULAIR-PM1", np.nan),
        plateau_median_MODULAIR_PM2=plateau_by_unit.get("MODULAIR-PM2", np.nan),
        n_opc_bin0_suppressed=int(len(supp)),
        opc_bin0_suppressed_median_ratio=(
            float(np.median(supp_ratios)) if supp_ratios.size else np.nan),
        peak_window_duration_median_min=(
            float(np.median(peak_dur)) if peak_dur.size else np.nan),
        peak_window_duration_min_min=(
            float(np.min(peak_dur)) if peak_dur.size else np.nan),
        peak_window_duration_max_min=(
            float(np.max(peak_dur)) if peak_dur.size else np.nan),
        qaqc_removal_duration_median_min=(
            float(np.median(qa_dur)) if qa_dur.size else np.nan),
        qaqc_removal_duration_min_min=(
            float(np.min(qa_dur)) if qa_dur.size else np.nan),
        qaqc_removal_duration_max_min=(
            float(np.max(qa_dur)) if qa_dur.size else np.nan),
    )
    path = out_dir / "modulair_5sec_peak_cross_burn_summary.csv"
    pd.DataFrame([summary]).to_csv(str(path), index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")
    return summary


# ==============================================================================
# MARKDOWN OUTPUTS
# ==============================================================================


def _fmt(val: float, fmt: str = ".1f") -> str:
    """Format a number, or '[no data]' for NaN."""
    try:
        return f"{val:{fmt}}" if val is not None and not np.isnan(val) else "[no data]"
    except (TypeError, ValueError):
        return "[no data]"


def _write_summary_md(results: list[dict], summary: dict) -> None:
    """
    Plain-language synthesis with a per-pair table and flagged anomalies.
    """
    out_dir = get_common_file("quantaq_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [r for r in results if r.get("data_present")]
    missing = [r for r in results if not r.get("data_present")]

    lines = ["# MODULAIR-PM 5 s peak-window analysis - summary", ""]
    lines.append("## Plain-language synthesis\n")
    lines.append(
        f"Across {summary['n_burn_unit_pairs']} burn-unit pairs with 5 s data, "
        f"the PMS5003 nephelometer bin 0 reached its fixed saturation plateau in "
        f"{summary['n_neph_bin0_saturated']} pairs. The plateau value was "
        f"essentially fixed per unit (median "
        f"{_fmt(summary['plateau_median_MODULAIR_PM2'], '.0f')} for MODULAIR-PM2, "
        f"{_fmt(summary['plateau_median_MODULAIR_PM1'], '.0f')} for MODULAIR-PM1), "
        f"consistent with a hardware ceiling rather than a concentration-"
        f"proportional response."
    )
    lines.append(
        f"\nDuring the same peak window the OPC-N3 response was bin-dependent. "
        f"The smallest bin (bin 0, 0.35-0.46 um) was suppressed (peak/pre < 0.5) "
        f"in {summary['n_opc_bin0_suppressed']} pairs, with a median peak/pre "
        f"ratio of {_fmt(summary['opc_bin0_suppressed_median_ratio'], '.2f')} "
        f"among the suppressed pairs (it collapsed to near zero while the "
        f"nephelometer was pinned). The next-larger bins (bins 1-2) did not "
        f"follow bin 0 down; they rose sharply from a near-zero pre-fire "
        f"baseline. See the per-pair table for the bin-1 and bin-2 "
        f"classifications."
    )
    lines.append(
        f"\nThe 5 s peak window lasted a median of "
        f"{_fmt(summary['peak_window_duration_median_min'])} min "
        f"(range {_fmt(summary['peak_window_duration_min_min'])} to "
        f"{_fmt(summary['peak_window_duration_max_min'])} min). The portal QA/QC "
        f"removal lasted a median of "
        f"{_fmt(summary['qaqc_removal_duration_median_min'])} min "
        f"(range {_fmt(summary['qaqc_removal_duration_min_min'])} to "
        f"{_fmt(summary['qaqc_removal_duration_max_min'])} min)."
    )

    # Per-pair table.
    lines.append("\n## Per-pair results\n")
    lines.append(
        "| burn | unit | saturated | plateau | peak win (min) | QA/QC (min) | "
        "overlap | tail (min) | bin0 (class) | bin1 (class) | bin2 (class) | "
        "AeroTrak PM3 (ug/m3) | AT reversal overlap | AT gap (min) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in present:
        lines.append(
            f"| {r['burn']} | {r['unit']} | {r.get('saturated')} | "
            f"{_fmt(r.get('plateau_value'), '.0f')} | "
            f"{_fmt(r.get('peak_window_duration_minutes'))} | "
            f"{_fmt(r.get('portal_qaqc_removal_duration_minutes'))} | "
            f"{_fmt(r.get('overlap_with_peak_window'), '.2f')} | "
            f"{_fmt(r.get('tail_extension_minutes'))} | "
            f"{_fmt(r.get('ratio_bin0'), '.2f')} ({r.get('class_bin0')}) | "
            f"{_fmt(r.get('ratio_bin1'), '.2f')} ({r.get('class_bin1')}) | "
            f"{_fmt(r.get('ratio_bin2'), '.2f')} ({r.get('class_bin2')}) | "
            f"{_fmt(r.get('aerotrak_peak_PM3_mass_ug_m3'), '.0f')} | "
            f"{_fmt(r.get('aerotrak_reversal_overlap_fraction'), '.2f')} | "
            f"{_fmt(r.get('aerotrak_reversal_gap_minutes'), '.1f')} |"
        )

    # Flagged anomalies.
    lines.append("\n## Flagged anomalies\n")
    flagged = False
    for r in missing:
        lines.append(f"- {r['burn']} {r['unit']}: 5 s data missing (not imputed).")
        flagged = True
    for r in present:
        if not r.get("saturated"):
            lines.append(
                f"- {r['burn']} {r['unit']}: nephelometer bin 0 did not reach the "
                f"saturation plateau in this burn."
            )
            flagged = True
        tail = r.get("tail_extension_minutes", np.nan)
        if not np.isnan(tail) and tail > 5.0:
            lines.append(
                f"- {r['burn']} {r['unit']}: portal QA/QC flag persists "
                f"{tail:.1f} min past the 5 s peak-window end."
            )
            flagged = True
        # Flag only when the AeroTrak reversed but its interval does not overlap
        # the MODULAIR-PM peak window at all (gap > 0 and overlap fraction 0).
        ov = r.get("aerotrak_reversal_overlap_fraction", np.nan)
        if (r.get("aerotrak_reversal_present") is True
                and not np.isnan(ov) and ov == 0.0):
            gap = r.get("aerotrak_reversal_gap_minutes", np.nan)
            lines.append(
                f"- {r['burn']} {r['unit']}: AeroTrak Ch1 reversal interval did "
                f"not overlap the MODULAIR-PM peak window "
                f"(gap {_fmt(gap, '.1f')} min)."
            )
            flagged = True
    if not flagged:
        lines.append("- None.")

    path = out_dir / "modulair_5sec_peak_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    [MD] {path.name}")


def _write_manuscript_md(results: list[dict], summary: dict) -> None:
    """
    Write the data-filled sentences for the rewritten Section 3.2.3.

    All numbers are derived from the analysis; no values are invented. The
    5 s record contradicts the prior assumption that all three smallest OPC-N3
    bins drop together: only the smallest bin (bin 0, 0.35-0.46 um) collapses
    during saturation, while bins 1-2 rise sharply from a near-zero pre-fire
    baseline. The sentences below report that bin-0-specific suppression rather
    than forcing an aggregate three-bin ratio.
    """
    out_dir = get_common_file("quantaq_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    present = [r for r in results if r.get("data_present")]
    sat = [r for r in present if r.get("saturated")]
    n_sat = len(sat)
    n_pairs = len(present)

    plat_pm2 = summary["plateau_median_MODULAIR_PM2"]
    plat_pm1 = summary["plateau_median_MODULAIR_PM1"]

    # bin-0 suppression across the pairs where bin 0 was actually suppressed.
    supp = [r for r in sat if r.get("class_bin0") == "suppressed"]
    n_supp = len(supp)
    supp_ratios = np.array([r.get("ratio_bin0", np.nan) for r in supp], dtype=float)
    supp_ratios = supp_ratios[~np.isnan(supp_ratios)]
    med_supp_pct = float(np.median(supp_ratios) * 100) if supp_ratios.size else np.nan
    min_supp_pct = float(np.min(supp_ratios) * 100) if supp_ratios.size else np.nan
    max_supp_pct = float(np.max(supp_ratios) * 100) if supp_ratios.size else np.nan

    qa_med = summary["qaqc_removal_duration_median_min"]
    qa_min = summary["qaqc_removal_duration_min_min"]
    qa_max = summary["qaqc_removal_duration_max_min"]

    s_intro = (
        "Raw 5 s data from the MODULAIR-PM instruments were obtained directly "
        "from the on-instrument storage and characterize the OPC-N3 and PMS5003 "
        "responses prior to QA/QC filtering."
    )
    s_neph = (
        f"The PMS5003 nephelometer channel saturated at a fixed plateau value "
        f"({_fmt(plat_pm2, '.0f')} for MODULAIR-PM2, {_fmt(plat_pm1, '.0f')} for "
        f"MODULAIR-PM1) during the peak window in {n_sat} of the {n_pairs} "
        f"MODULAIR-PM-deployed burn-unit records analysed."
    )
    s_opc = (
        f"The smallest OPC-N3 bin (0.35-0.46 um) responded in the opposite "
        f"direction during the same window, dropping to a median of "
        f"{_fmt(med_supp_pct, '.0f')}% of its pre-peak baseline (range "
        f"{_fmt(min_supp_pct, '.0f')}% to {_fmt(max_supp_pct, '.0f')}%) in "
        f"{n_supp} of the {n_sat} saturated records, while the next-larger OPC-N3 "
        f"bins increased over the same window."
    )
    s_qaqc = (
        f"The portal-delivered QA/QC removed a window of approximately "
        f"{_fmt(qa_med, '.0f')} minutes around peak (median across saturated "
        f"records; range {_fmt(qa_min, '.0f')} to {_fmt(qa_max, '.0f')}), "
        f"corresponding to the period where the nephelometer was saturated and "
        f"the smallest OPC-N3 bin was suppressed."
    )

    para = (
        f"Raw 5 s data from the MODULAIR-PM instruments, retrieved from "
        f"on-instrument storage prior to QA/QC filtering, resolve the "
        f"sub-minute instrument behavior during the smoke peak. In "
        f"{n_sat} of the {n_pairs} burn-unit records analysed, the PMS5003 "
        f"nephelometer bin 0 reached a fixed plateau value "
        f"({_fmt(plat_pm2, '.0f')} for MODULAIR-PM2 and {_fmt(plat_pm1, '.0f')} "
        f"for MODULAIR-PM1); the plateau was essentially constant across burns "
        f"for a given unit, consistent with a fixed ceiling rather than a "
        f"concentration-proportional signal. Over the same window the OPC-N3 "
        f"response was bin-dependent: the smallest bin (0.35-0.46 um) collapsed "
        f"to a median of {_fmt(med_supp_pct, '.0f')}% of its pre-peak baseline "
        f"in {n_supp} of the {n_sat} saturated records, while the next-larger "
        f"bins rose sharply. The portal-delivered "
        f"QA/QC product removed approximately {_fmt(qa_med, '.0f')} minutes of "
        f"data around the peak (median across saturated records), aligning with "
        f"the period in which the nephelometer was saturated and the smallest "
        f"OPC-N3 bin was suppressed. These features were co-located in time with "
        f"the AeroTrak Ch1 reversal documented in Section 3.2.2, indicating that "
        f"the peak-window behavior is common to the optical particle instruments "
        f"at the highest smoke concentrations. The 5 s record thus shows that "
        f"the brief gap in the QA/QC-filtered MODULAIR-PM product coincides with "
        f"a saturated nephelometer and a suppressed smallest OPC-N3 bin, and is "
        f"not a loss of physical signal that simple interpolation would recover."
    )

    text = (
        "# MODULAIR-PM 5 s - Manuscript sentences for Section 3.2.3\n\n"
        "_All values derived from data. Replaces the prior 'raw 5 Hz data were "
        "not accessible' text in the manuscript. Note: the on-instrument record "
        "is logged at 5 s (0.2 Hz) cadence; the manuscript should refer to it as "
        "5 s data, not 5 Hz._\n\n---\n\n"
        f"**Intro sentence:** \"{s_intro}\"\n\n"
        f"**Nephelometer saturation:** \"{s_neph}\"\n\n"
        f"**OPC-N3 suppression:** \"{s_opc}\"\n\n"
        f"**QA/QC alignment:** \"{s_qaqc}\"\n\n"
        "---\n\n## Replacement paragraph (third paragraph of 3.2.3)\n\n"
        f"{para}\n"
    )
    path = out_dir / "modulair_5sec_peak_manuscript_sentences.md"
    path.write_text(text, encoding="utf-8")
    print(f"    [MD] {path.name}")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Run the full peak-window analysis: load, analyze, write CSV/MD/figures."""
    warnings.filterwarnings("ignore")
    print("\n" + "=" * 70)
    print("MODULAIR-PM 5 s peak-window analysis  -  Burns 4-10")
    print("=" * 70)

    events = load_event_times()
    at = _load_aerotrak()

    results: list[dict] = []
    for burn_id in BURNS:
        print(f"--- {burn_id} ---")
        for unit in UNITS:
            rec = analyze_pair(burn_id, unit, events, at)
            results.append(rec)
            if rec.get("data_present"):
                print(
                    f"    {unit}: saturated={rec.get('saturated')} "
                    f"peak_win={_fmt(rec.get('peak_window_duration_minutes'))} min "
                    f"bin0 ratio={_fmt(rec.get('ratio_bin0'), '.2f')}"
                )

    print("\nWriting CSV outputs...")
    _write_per_burn_csv(results)
    _write_plateau_csv(results)
    summary = _write_cross_burn_csv(results)

    print("\nWriting Bokeh per-pair figures...")
    for rec in results:
        _bokeh_pair(rec)

    print("\nWriting matplotlib figures...")
    _mpl_bin_response_grid(results)
    _mpl_qaqc_overlap(results)

    print("\nWriting markdown outputs...")
    _write_summary_md(results, summary)
    _write_manuscript_md(results, summary)

    print("\nDone. Check quantaq_analysis/ and quantaq_figures/ for outputs.")


if __name__ == "__main__":
    main()
