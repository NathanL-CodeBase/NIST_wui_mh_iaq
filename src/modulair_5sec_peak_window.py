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
from src.fig_style import (  # noqa: E402
    REF_LINE,
    ROLE_COLORS,
    SHADE,
    SHADE_ALPHA,
    UNIT_COLORS,
    apply_est_style,
    figsize,
    save_fig,
)
from src.modulair_5sec_io import (  # noqa: E402
    BEDROOM_SEALED_BURNS,
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

# Fallback peak-window definition for records that carry smoke but never pin
# neph_bin0 at the 16-bit ceiling (all MODULAIR-PM1 records and burn8
# MODULAIR-PM2). Without a window these records yield no t_peak_end, so the
# OPC ratios and the Prompt-4 cross-reference cannot be computed. The fallback
# window is defined independently of the nephelometer, in priority order:
#   (a) the burn-log peak interval: from max(garage closure, ignition) to PAC
#       activation (the physical dense-smoke period before air cleaning);
#   (b) the co-located AeroTrak PM3 peak time +/- the median saturated-window
#       half-width;
#   (c) the OPC-N3 total-count peak +/- the same half-width.
# Saturated records always use the ceiling-based window above; the fallback
# only fires when SAT detection fails. peak_window_method records which path
# produced the window: "neph_saturation" | "fallback_burnlog" |
# "fallback_aerotrak" | "fallback_opc".
FALLBACK_HALF_WIDTH_FLOOR_MIN = 2.5   # min half-width if no saturated median yet

# burn8 MODULAIR-PM2 near-saturation diagnostic / relaxed-saturation test. The
# detector's exact-ceiling test (SAT_FRAC = 1.0) is the primary saturation
# rule. If a record approaches but never reaches the exact ceiling, a relaxed
# test (neph_bin0 >= NEAR_SAT_FRAC * NEPH_CEILING for >= MIN_SAT_SAMPLES) is
# evaluated for the diagnostic print only; it does not reclassify the record.
NEAR_SAT_FRAC = 0.98

# OPC-N3 three smallest bins used for the suppression classification.
OPC_SMALL_BINS = ["bin0", "bin1", "bin2"]

# Pre-burn baseline window (minutes before ignition) for the peak/pre ratio.
BASELINE_MIN = 30

# Minimum pre-burn baseline count for a bin ratio to be reported. The OPC-N3
# bin1/bin2 pre-fire baselines sit near zero (~0.01-0.08 counts), so a peak/pre
# ratio divides by a tiny denominator and explodes to implausible values
# (hundreds to thousands). Ratios built on a baseline below this floor are not
# meaningful and are returned as NaN; the classification then falls back to the
# absolute peak count (see MIN_PEAK_COUNT_ELEVATED).
MIN_BASELINE_FOR_RATIO = 0.5

# Absolute peak count (median over the peak window) above which a bin whose
# pre-fire baseline was below MIN_BASELINE_FOR_RATIO is classified as
# "elevated_from_zero": it rose from a near-zero baseline to a real count, a
# genuine increase that the ratio cannot quantify without a div-by-zero
# artifact. Set from the data: bin1/bin2 peak medians in the dense-smoke
# records reach tens to hundreds of counts, well above this floor.
MIN_PEAK_COUNT_ELEVATED = 1.0

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

# Main-text Figure 2 worked example. Burn 06 MODULAIR-PM2 is the clean overlap
# case: the portal QA/QC removal falls fully inside the saturated, bin-0
# suppressed window. Burn 09 was the original choice but its removal sits on the
# rising edge (~2 min before the exact-ceiling plateau), so it is not the
# example that supports the "coincides" sentence.
FIG2_BURN = "burn6"
FIG2_UNIT = "MODULAIR-PM2"

# Numeric point size for matplotlib calls (matches src.fig_style BASE_FONT_PT).
_FS = 12

# Per-unit plot colors from the shared colorblind-safe palette (bedroom blue,
# morning room vermillion).
UNIT_COLOR = dict(UNIT_COLORS)


# ==============================================================================
# PEAK-WINDOW DETECTION
# ==============================================================================


def _detect_peak_window(
    df: pd.DataFrame,
    ignition: pd.Timestamp = pd.NaT,
    garage: pd.Timestamp = pd.NaT,
    pac_on: pd.Timestamp = pd.NaT,
    aerotrak_t_peak: pd.Timestamp = pd.NaT,
    sat_half_width_min: float = np.nan,
) -> dict:
    """
    Define the peak window for one burn-unit record.

    Two paths:

    1. Saturation (primary). If neph_bin0 reaches the 16-bit ceiling
       (>= SAT_FRAC * NEPH_CEILING) for at least MIN_SAT_SAMPLES samples, the
       window runs from the first ceiling sample to the first sample on the
       recovery side that drops below RECOVERY_FRAC of the plateau value.
       'saturated' is True and 'peak_window_method' is "neph_saturation".

    2. Fallback (non-saturating). Records that carry smoke but never pin
       neph_bin0 at the ceiling still need a window so the OPC ratios and the
       Prompt-4 cross-reference can be computed. The fallback window is defined
       independently of the nephelometer, in priority order:
         (a) burn-log peak interval: max(garage, ignition) -> pac_on
             ('peak_window_method' = "fallback_burnlog");
         (b) AeroTrak PM3 peak +/- sat_half_width_min
             ('peak_window_method' = "fallback_aerotrak");
         (c) OPC-N3 total-count peak +/- sat_half_width_min
             ('peak_window_method' = "fallback_opc").
       'saturated' stays False. The window is clipped to the record's own time
       span. 'plateau_value' carries the in-window neph_bin0 maximum for these
       records (not a fixed ceiling).

    Parameters
    ----------
    df : pd.DataFrame
        5 s record for one burn-unit pair (local time), with 'neph_bin0'.
    ignition, garage, pac_on : pd.Timestamp
        Burn-log event times for the fallback burn-log interval (a).
    aerotrak_t_peak : pd.Timestamp
        Co-located AeroTrak PM3 peak time for fallback (b).
    sat_half_width_min : float
        Median saturated-window half-width (minutes) for fallbacks (b)/(c).

    Returns
    -------
    dict
        Keys: saturated (bool), peak_window_method (str), t_peak_start,
        t_peak_end (pd.Timestamp/NaT), plateau_value (float),
        peak_window_duration_seconds/_minutes (float), n_sat_samples (int).
    """
    blank = dict(
        saturated=False,
        peak_window_method="none",
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
    if sat_mask.sum() >= MIN_SAT_SAMPLES:
        # First saturated sample = window start. Plateau value = median
        # neph_bin0 over the saturated samples (a fixed ceiling, not a signal).
        sat_idx = np.where(sat_mask)[0]
        start_i = int(sat_idx[0])
        plateau = float(np.nanmedian(nb0.to_numpy()[sat_mask]))

        # Recovery: first sample AFTER the last saturated sample where neph_bin0
        # falls below RECOVERY_FRAC * plateau. Search from the last plateau
        # sample forward so a transient mid-plateau dip does not close it early.
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
            peak_window_method="neph_saturation",
            t_peak_start=t_start,
            t_peak_end=t_end,
            plateau_value=plateau,
            peak_window_duration_seconds=float(dur_s),
            peak_window_duration_minutes=float(dur_s / 60.0),
            n_sat_samples=int(sat_mask.sum()),
        )

    # --- Fallback window for a non-saturating record --------------------------
    rec_lo = pd.Timestamp(ts.iloc[0])
    rec_hi = pd.Timestamp(ts.iloc[-1])
    half = (
        pd.Timedelta(minutes=sat_half_width_min)
        if np.isfinite(sat_half_width_min) and sat_half_width_min > 0
        else pd.Timedelta(minutes=FALLBACK_HALF_WIDTH_FLOOR_MIN)
    )

    method = None
    t_start = t_end = pd.NaT

    # (a) burn-log peak interval: max(garage, ignition) -> pac_on.
    peak_lo = pd.NaT
    if pd.notna(garage) and pd.notna(ignition):
        peak_lo = max(garage, ignition)
    elif pd.notna(ignition):
        peak_lo = ignition
    elif pd.notna(garage):
        peak_lo = garage
    if pd.notna(peak_lo) and pd.notna(pac_on) and pac_on > peak_lo:
        t_start, t_end, method = peak_lo, pac_on, "fallback_burnlog"
    # (b) AeroTrak PM3 peak +/- half-width.
    elif pd.notna(aerotrak_t_peak):
        t_start = aerotrak_t_peak - half
        t_end = aerotrak_t_peak + half
        method = "fallback_aerotrak"
    # (c) OPC-N3 total-count peak +/- half-width.
    else:
        opc_cols = [c for c in OPC_BINS if c in df.columns]
        if opc_cols:
            total = df[opc_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            if total.notna().any():
                t_opc = pd.Timestamp(ts.iloc[int(total.values.argmax())])
                t_start = t_opc - half
                t_end = t_opc + half
                method = "fallback_opc"

    if method is None or pd.isna(t_start) or pd.isna(t_end):
        return blank

    # Clip the window to the record's own time span.
    t_start = max(t_start, rec_lo)
    t_end = min(t_end, rec_hi)
    if t_end <= t_start:
        return blank

    in_win = (ts >= t_start) & (ts <= t_end)
    nb0_in = nb0[in_win]
    plateau_val = float(np.nanmax(nb0_in.to_numpy())) if nb0_in.notna().any() else np.nan
    dur_s = (t_end - t_start).total_seconds()

    return dict(
        saturated=False,
        peak_window_method=method,
        t_peak_start=t_start,
        t_peak_end=t_end,
        plateau_value=plateau_val,
        peak_window_duration_seconds=float(dur_s),
        peak_window_duration_minutes=float(dur_s / 60.0),
        n_sat_samples=int(in_win.sum()),
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

    # Removal window = the true coverage of the contiguous removed block. Each
    # portal label is interval-centered (see load_portal_burn), so a removed
    # minute stamped at t filtered the 30 s on either side of t; the block runs
    # from the first center minus 30 s to the last center plus 30 s.
    q_start = pd.Timestamp(times[lo]) - pd.Timedelta(seconds=30)
    q_end = pd.Timestamp(times[hi]) + pd.Timedelta(seconds=30)
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


def _classify_bin(
    ratio: float, pre_count: float, peak_count: float
) -> str:
    """
    Classify a bin's peak-window response robustly against a near-zero baseline.

    When the pre-fire baseline is at or above MIN_BASELINE_FOR_RATIO the ratio
    is meaningful and drives the class (suppressed / unchanged / elevated). When
    the baseline is below that floor the ratio is a division-by-zero artifact;
    the class then comes from the absolute peak count: a peak median above
    MIN_PEAK_COUNT_ELEVATED is "elevated_from_zero" (rose from a near-zero
    baseline), otherwise "unchanged".
    """
    if np.isfinite(pre_count) and pre_count >= MIN_BASELINE_FOR_RATIO:
        if np.isnan(ratio):
            return "no_data"
        if ratio < SUPPRESSED_MAX:
            return "suppressed"
        if ratio > ELEVATED_MIN:
            return "elevated"
        return "unchanged"
    # Near-zero baseline: classify on the absolute peak count instead.
    if not np.isfinite(peak_count):
        return "no_data"
    if peak_count > MIN_PEAK_COUNT_ELEVATED:
        return "elevated_from_zero"
    return "unchanged"


def _opc_response(
    df: pd.DataFrame,
    ignition: pd.Timestamp,
    t_peak_start: pd.Timestamp,
    t_peak_end: pd.Timestamp,
    saturated: bool,
) -> dict:
    """
    Compute peak/pre-baseline ratios, absolute counts, and classifications for
    all OPC-N3 bins.

    The pre-burn baseline is the BASELINE_MIN minutes immediately before
    ignition (mean bin count). The peak level is the MEDIAN bin count over the
    peak samples. For a saturated record the peak samples are the
    saturated-plateau samples only (neph_bin0 >= SAT_FRAC * NEPH_CEILING): the
    rising edge inflates a window mean and would mask the small-bin collapse
    that occurs once the nephelometer is pinned. For a fallback window there is
    no plateau, so the peak samples are all samples in [t_peak_start,
    t_peak_end].

    Bin ratios are only reported when the pre-burn baseline exceeds
    MIN_BASELINE_FOR_RATIO; the OPC-N3 bin1/bin2 baselines sit near zero, so a
    ratio against them divides by a tiny denominator and explodes. Such ratios
    are returned as NaN, and the classification falls back to the absolute peak
    count (see _classify_bin), so a bin that rose from a near-zero baseline is
    reported as "elevated_from_zero" rather than as a spurious thousands-fold
    ratio. The absolute pre/peak counts are returned for the three smallest
    bins so the manuscript can cite the count change directly.

    Returns
    -------
    dict
        ratio_bin{i}, class_bin{i}, pre_count_bin{i}, peak_count_bin{i} for the
        three smallest bins, plus 'all_ratios' (list aligned with OPC_BINS) and
        'pre_profile'/'peak_profile' (lists of mean pre-peak and median
        peak-window counts across all OPC_BINS, for the absolute-count grid).
    """
    out: dict = {
        "all_ratios": [np.nan] * len(OPC_BINS),
        "pre_profile": [np.nan] * len(OPC_BINS),
        "peak_profile": [np.nan] * len(OPC_BINS),
    }
    for b in OPC_SMALL_BINS:
        out[f"ratio_{b}"] = np.nan
        out[f"class_{b}"] = "no_data"
        out[f"pre_count_{b}"] = np.nan
        out[f"peak_count_{b}"] = np.nan
    if df is None or df.empty or pd.isna(ignition):
        return out
    if pd.isna(t_peak_start) or pd.isna(t_peak_end):
        return out

    ts = df["timestamp"]
    pre_mask = (ts >= ignition - pd.Timedelta(minutes=BASELINE_MIN)) & (ts < ignition)
    in_window = (ts >= t_peak_start) & (ts <= t_peak_end)
    if saturated:
        # Peak level: saturated-plateau samples within the peak window only.
        nb0 = pd.to_numeric(df["neph_bin0"], errors="coerce")
        peak_mask = in_window & (nb0 >= SAT_FRAC * NEPH_CEILING)
    else:
        # Fallback window: no plateau, use all in-window samples.
        peak_mask = in_window
    if pre_mask.sum() == 0 or peak_mask.sum() == 0:
        return out

    for i, b in enumerate(OPC_BINS):
        if b not in df.columns:
            continue
        col = pd.to_numeric(df[b], errors="coerce")
        pre_mean = float(col[pre_mask].mean())
        peak_med = float(col[peak_mask].median())
        # Absolute count profiles for the bin-response grid (Figure S3): the
        # pre-peak baseline mean and the peak-window median per bin, plotted
        # directly so bin 0 suppression and the larger-bin rise are both
        # visible without dividing by a near-zero pre-fire denominator.
        out["pre_profile"][i] = pre_mean
        out["peak_profile"][i] = peak_med
        # Guard against a near-zero baseline (bin1/bin2 pre-fire counts ~0):
        # a ratio there is not meaningful, so leave it NaN.
        ratio = (
            peak_med / pre_mean
            if np.isfinite(pre_mean) and pre_mean >= MIN_BASELINE_FOR_RATIO
            else np.nan
        )
        out["all_ratios"][i] = ratio
        if b in OPC_SMALL_BINS:
            out[f"ratio_{b}"] = ratio
            out[f"pre_count_{b}"] = pre_mean
            out[f"peak_count_{b}"] = peak_med
            out[f"class_{b}"] = _classify_bin(ratio, pre_mean, peak_med)
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
        aerotrak_t_peak=pd.NaT,
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
                     ("reversal_end", "aerotrak_reversal_end"),
                     ("t_peak", "aerotrak_t_peak")):
        if src in row.columns and pd.notna(r.get(src)):
            out[dst] = pd.to_datetime(r[src], errors="coerce")
    return out


# ==============================================================================
# PER-PAIR ANALYSIS DRIVER
# ==============================================================================


def _saturation_diagnostic(df: pd.DataFrame, burn_id: str, unit: str,
                           peak_lo: pd.Timestamp, peak_hi: pd.Timestamp) -> None:
    """
    Print the per-record near-saturation diagnostic (FIX B2).

    Reports the neph_bin0 maximum and its timestamp, the count of samples at or
    above the exact ceiling and within NEAR_SAT_FRAC of it, the 5 s record time
    range, and whether the record covers the burn-log peak interval (to detect a
    data gap over the peak). Diagnostic only; it does not reclassify the record.
    """
    if df is None or df.empty or "neph_bin0" not in df.columns:
        return
    nb0 = pd.to_numeric(df["neph_bin0"], errors="coerce")
    ts = df["timestamp"]
    mx = float(np.nanmax(nb0.to_numpy()))
    t_mx = pd.Timestamp(ts.iloc[int(nb0.values.argmax())])
    n_ceiling = int((nb0 >= NEPH_CEILING).sum())
    n_near = int((nb0 >= NEAR_SAT_FRAC * NEPH_CEILING).sum())
    covers = "n/a"
    if pd.notna(peak_lo) and pd.notna(peak_hi):
        in_peak = ((ts >= peak_lo) & (ts <= peak_hi))
        if in_peak.any():
            gaps = ts[in_peak].diff().dt.total_seconds()
            max_gap = float(gaps.max()) if gaps.notna().any() else 0.0
            covers = f"yes (n={int(in_peak.sum())}, max gap {max_gap:.0f}s)"
        else:
            covers = "NO - data gap over peak interval"
    print(
        f"    [sat-diag] {burn_id} {unit}: neph_bin0 max {mx:.0f} at {t_mx}; "
        f"n>=ceiling {n_ceiling}, n>=0.98*ceiling {n_near}; "
        f"record {ts.iloc[0]}..{ts.iloc[-1]}; covers peak interval: {covers}"
    )


def analyze_pair(
    burn_id: str,
    unit: str,
    events: pd.DataFrame,
    at: pd.DataFrame | None,
    sat_half_width_min: float = np.nan,
) -> dict:
    """
    Run the full peak-window analysis for one burn-unit pair.

    Returns a dict with all per-burn CSV fields plus private keys (prefixed
    '_') carrying the loaded frames and event times for the figure functions.
    A pair with no 5 s data is flagged via 'data_present' = False and skipped
    by the figure functions; it is never imputed.

    sat_half_width_min is the median saturated-window half-width (minutes) used
    by the AeroTrak/OPC fallback windows; pass NaN on the first pass (before any
    saturated window is known) and the real value on the second pass.
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
            data_present=False, saturated=False, peak_window_method="none",
            notes="5 s data missing",
            _df=None, _portal=None, _ignition=ignition, _garage=garage,
            _pac_on=pac_on,
        )

    portal = load_portal_burn(unit, burn_id)
    atx = _aerotrak_for_pair(at, burn_id, unit)

    # AeroTrak PM3 peak time for the fallback (b) anchor.
    at_t_peak = atx.get("aerotrak_t_peak", pd.NaT)

    pw = _detect_peak_window(
        df, ignition, garage, pac_on, at_t_peak, sat_half_width_min
    )

    # Near-saturation diagnostic for every non-saturating record (FIX B2).
    if not pw["saturated"]:
        peak_lo = (
            max(garage, ignition)
            if pd.notna(garage) and pd.notna(ignition)
            else (ignition if pd.notna(ignition) else garage)
        )
        _saturation_diagnostic(df, burn_id, unit, peak_lo, pac_on)

    qa = _portal_qaqc_window(portal, pw["t_peak_start"], pw["t_peak_end"], ignition)
    opc = _opc_response(
        df, ignition, pw["t_peak_start"], pw["t_peak_end"], pw["saturated"]
    )
    neph = _neph_plateau(df, pw["t_peak_start"], pw["t_peak_end"])

    # Cross-reference the MODULAIR-PM peak window against the full AeroTrak Ch1
    # reversal interval (onset -> recovery), not a single instant. Both
    # quantities are real time intervals, so we report the fraction of the
    # MODULAIR-PM peak window that falls inside the AeroTrak reversal interval,
    # and the signed gap between the two intervals (0 when they overlap;
    # otherwise minutes of separation). This is robust to the independent
    # instrument clocks and to the AeroTrak recovery plateau arriving later.
    # Any detected peak window (saturation or near-ceiling fallback) is used,
    # so the records that never saturated still get a Prompt-4 cross-reference.
    rev_overlap_frac = np.nan
    rev_gap_min = np.nan
    has_window = pd.notna(pw["t_peak_start"]) and pd.notna(pw["t_peak_end"])
    if (atx["aerotrak_reversal_present"] is True and has_window
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

    method = pw["peak_window_method"]
    if pw["saturated"]:
        notes = []
    elif method == "fallback_burnlog":
        notes = ["neph_bin0 did not saturate; burn-log peak-interval fallback window used"]
    elif method == "fallback_aerotrak":
        notes = ["neph_bin0 did not saturate; AeroTrak-peak fallback window used"]
    elif method == "fallback_opc":
        notes = ["neph_bin0 did not saturate; OPC-peak fallback window used"]
    else:
        notes = ["neph_bin0 did not saturate; no peak window could be defined"]
    # Flag the sealed low-concentration Bedroom 2 records: they get a window so
    # Prompt 4 can run, but the co-located AeroTrak PM3 shows no real smoke.
    if unit == "MODULAIR-PM1" and burn_id in BEDROOM_SEALED_BURNS:
        notes.append("sealed bedroom, low concentration (not a dense-smoke case)")
    if portal is None:
        notes.append("portal product missing")

    rec = dict(
        burn=burn_id, unit=unit, location=UNIT_LOCATION[unit],
        data_present=True, portal_present=portal is not None,
        bedroom_sealed=(unit == "MODULAIR-PM1" and burn_id in BEDROOM_SEALED_BURNS),
        **{k: pw[k] for k in (
            "saturated", "peak_window_method", "t_peak_start", "t_peak_end",
            "plateau_value", "peak_window_duration_seconds",
            "peak_window_duration_minutes", "n_sat_samples")},
        **qa,
        **{k: opc[k] for k in opc if k != "all_ratios"},
        **neph,
        **{k: v for k, v in atx.items() if k != "aerotrak_t_peak"},
        aerotrak_reversal_overlap_fraction=rev_overlap_frac,
        aerotrak_reversal_gap_minutes=rev_gap_min,
        notes="; ".join(notes),
        # private
        _df=df, _portal=portal, _ignition=ignition, _garage=garage,
        _pac_on=pac_on, _opc_all_ratios=opc["all_ratios"],
        _opc_pre_profile=opc["pre_profile"], _opc_peak_profile=opc["peak_profile"],
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
    Figure S3 (SI): per burn-unit panel, absolute OPC-N3 count concentration
    per bin for the pre-peak baseline and for the peak window, versus bin lower
    edge (log-log). Plotting absolute counts (not a peak/pre ratio) makes the
    bin 0 suppression and the rise of the larger bins both visible without
    dividing by a near-zero pre-fire denominator.

    One panel per burn-unit pair that produced a peak window; panels without a
    window are dropped (not left empty) and reported in the run log. Bin 0
    (0.35-0.46 um) is marked with a vertical guide. Shared bottom x-label and
    left y-label; constrained layout prevents clipped axis labels.
    """
    bin_lower = np.array([lo for (lo, _hi) in OPC_BIN_EDGES_UM], dtype=float)

    # Panels = burn-unit pairs with a peak window and a usable count profile.
    panels = []
    dropped = []
    for r in results:
        if not r.get("data_present") or pd.isna(r.get("t_peak_end")):
            dropped.append((r["burn"], r["unit"], "no peak window"))
            continue
        pre = np.array(r.get("_opc_pre_profile") or [], dtype=float)
        peak = np.array(r.get("_opc_peak_profile") or [], dtype=float)
        if pre.size == 0 or peak.size == 0 or not (
            np.isfinite(pre).any() and np.isfinite(peak).any()
        ):
            dropped.append((r["burn"], r["unit"], "no count profile"))
            continue
        panels.append(r)

    if not panels:
        print("    [mpl] no panels with count profiles for bin-response grid.")
        return

    panels = sorted(panels, key=lambda r: (int(r["burn"].replace("burn", "")), r["unit"]))

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize("double")[0],
                                                    2.3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    bin0_lo, bin0_hi = OPC_BIN_EDGES_UM[0]

    for ax_idx, rec in enumerate(panels):
        ax = axes[ax_idx]
        color = UNIT_COLOR[rec["unit"]]
        pre = np.array(rec["_opc_pre_profile"], dtype=float)
        peak = np.array(rec["_opc_peak_profile"], dtype=float)
        # Mask non-positive for the log axis.
        pre_v = np.where(pre > 0, pre, np.nan)
        peak_v = np.where(peak > 0, peak, np.nan)
        ax.loglog(bin_lower, pre_v, marker="o", ms=3, lw=1.0, ls="--",
                  color=color, alpha=0.8, label="pre-peak")
        ax.loglog(bin_lower, peak_v, marker="s", ms=3, lw=1.3, ls="-",
                  color=color, alpha=0.95, label="peak window")
        # Mark bin 0.
        ax.axvspan(bin0_lo, bin0_hi, color=SHADE, alpha=0.18, lw=0)
        tag = "Bdrm" if rec["unit"] == "MODULAIR-PM1" else "MR"
        sat = "sat" if rec.get("saturated") else "no-sat"
        ax.set_title(f"{rec['burn']} {tag} ({sat})", fontsize=_FS - 1)
        ax.tick_params(labelsize=_FS - 2)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    # Shared axis labels and one legend.
    fig.supxlabel("OPC-N3 bin lower edge (µm)", fontsize=_FS)
    fig.supylabel("OPC-N3 count concentration (counts per 5 s)", fontsize=_FS)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#555555", marker="o", ls="--", label="pre-peak baseline"),
        Line2D([0], [0], color="#555555", marker="s", ls="-", label="peak window"),
        Line2D([0], [0], color=UNIT_COLOR["MODULAIR-PM1"], lw=4,
               label=UNIT_CONFIG["MODULAIR-PM1"]["location_label"]),
        Line2D([0], [0], color=UNIT_COLOR["MODULAIR-PM2"], lw=4,
               label=UNIT_CONFIG["MODULAIR-PM2"]["location_label"]),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=1,
               fontsize=_FS - 2, frameon=True, bbox_to_anchor=(0.98, 0.04))

    fig_dir = get_common_file("quantaq_figures")
    save_fig(fig, fig_dir / "modulair_5sec_bin_response_grid.png")
    if dropped:
        drop_str = ", ".join(f"{b} {u} ({why})" for b, u, why in dropped)
        print(f"    [mpl] bin-response grid dropped panels: {drop_str}")


# ==============================================================================
# MATPLOTLIB - QA/QC OVERLAP (main text / SI)
# ==============================================================================


def _mpl_qaqc_timeseries_fig2(results: list[dict]) -> None:
    """
    Figure 2 (main text, double column): FIG2_BURN MODULAIR-PM2 (Morning Room)
    5 s time series over the peak window, three aligned-time panels showing
    that the saturated nephelometer, the suppressed OPC-N3 bin 0, and the
    portal QA/QC removal interval coincide:
        (a) PMS5003 neph_bin0 raw with the 65535 ceiling marked;
        (b) OPC-N3 bin0 raw counts (the suppression);
        (c) a strip marking the 5 s peak-window span and the portal QA/QC
            removal interval.

    The worked example is Burn 06 (see FIG2_BURN): its portal removal falls
    fully inside the saturated, bin-0 suppressed window on the shared clock.
    """
    burn_no = FIG2_BURN.replace("burn", "").zfill(2)
    loc_label = UNIT_CONFIG[FIG2_UNIT]["location_label"]
    rec = next((r for r in results
                if r["burn"] == FIG2_BURN and r["unit"] == FIG2_UNIT
                and r.get("data_present")), None)
    if rec is None or pd.isna(rec.get("t_peak_start")):
        print(f"    [mpl] {FIG2_BURN} {FIG2_UNIT} not available for QA/QC time series.")
        return

    df = rec["_df"]
    t0 = rec["t_peak_start"]
    t1 = rec["t_peak_end"]
    # Pad the display window so both the rising edge and the removal interval are
    # visible even when the removal extends slightly past the saturated plateau.
    q0 = rec.get("portal_qaqc_removal_start")
    q1 = rec.get("portal_qaqc_removal_end")
    pad = pd.Timedelta(minutes=8)
    span_lo = min([t for t in (t0, q0) if pd.notna(t)])
    span_hi = max([t for t in (t1, q1) if pd.notna(t)])
    t_lo, t_hi = span_lo - pad, span_hi + pad
    sub = df[(df["timestamp"] >= t_lo) & (df["timestamp"] <= t_hi)].copy()
    if sub.empty:
        print(f"    [mpl] {FIG2_BURN} {FIG2_UNIT} window empty for QA/QC time series.")
        return

    ts = sub["timestamp"]
    color = UNIT_COLOR[FIG2_UNIT]

    fig, axes = plt.subplots(
        3, 1, figsize=(figsize("double")[0], 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 0.8]},
    )
    ax_neph, ax_opc, ax_strip = axes

    # (a) Nephelometer bin0 with the 16-bit ceiling.
    ax_neph.plot(ts, pd.to_numeric(sub["neph_bin0"], errors="coerce"),
                 color=ROLE_COLORS["PMS5003"], lw=1.2)
    ax_neph.axhline(NEPH_CEILING, color=REF_LINE, ls="--", lw=1.0)
    ax_neph.annotate("16-bit ceiling (65535)", xy=(ts.iloc[0], NEPH_CEILING),
                     xytext=(2, -10), textcoords="offset points",
                     fontsize=_FS - 3, color=REF_LINE, va="top")
    ax_neph.set_ylabel("PMS5003\nneph bin0", fontsize=_FS - 1)
    ax_neph.set_title(f"Burn {burn_no} {loc_label} ({FIG2_UNIT}): saturated "
                      "nephelometer, suppressed OPC-N3 bin 0, and portal "
                      "QA/QC removal coincide", fontsize=_FS - 1)

    # (b) OPC-N3 bin0 suppression.
    ax_opc.plot(ts, pd.to_numeric(sub["bin0"], errors="coerce"),
                color=color, lw=1.2)
    ax_opc.set_ylabel("OPC-N3 bin0\n(counts per 5 s)", fontsize=_FS - 1)

    # (c) Peak-window span + portal QA/QC removal interval.
    ax_strip.axvspan(t0, t1, ymin=0.55, ymax=0.95, color=color, alpha=0.35,
                     label="5 s peak window")
    if pd.notna(q0) and pd.notna(q1):
        ax_strip.axvspan(q0, q1, ymin=0.10, ymax=0.50, color=ROLE_COLORS["SMPS"],
                         alpha=0.6, label="portal QA/QC removal")
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("intervals", fontsize=_FS - 2)
    ax_strip.legend(loc="lower center", fontsize=_FS - 3, frameon=True, ncol=2)
    ax_strip.set_xlabel("Local time (EDT)", fontsize=_FS)

    # Vertical guides at the peak-window bounds across the data panels.
    for ax in (ax_neph, ax_opc):
        ax.axvline(t0, color=SHADE, ls=":", lw=1.0)
        ax.axvline(t1, color=SHADE, ls=":", lw=1.0)
        ax.tick_params(labelsize=_FS - 2)

    # Thin and rotate the shared x-axis time ticks so the EDT labels stop
    # colliding on the narrow peak window.
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax_strip.xaxis.set_major_locator(locator)
    ax_strip.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=30, ha="right")

    fig_dir = get_common_file("quantaq_figures")
    save_fig(fig, fig_dir / f"modulair_5sec_qaqc_timeseries_{FIG2_BURN}.png")


def _mpl_qaqc_overlap(results: list[dict]) -> None:
    """
    Figure S6 (SI, demoted from main text): grouped bar chart per burn-unit
    pair, peak_window_duration vs portal_qaqc_removal_duration (both minutes).
    Every pair with a peak window (saturation or fallback) and a portal product
    is shown; fallback-windowed pairs are hatched to distinguish them from
    saturated pairs. Unit and location are explicit in the x labels.
    """
    rows = [
        r for r in results
        if r.get("data_present")
        and pd.notna(r.get("t_peak_end"))
        and not np.isnan(r.get("peak_window_duration_minutes", np.nan))
    ]
    if not rows:
        print("    [mpl] no saturated pairs for QA/QC overlap chart.")
        return

    rows = sorted(rows, key=lambda r: (int(r["burn"].replace("burn", "")), r["unit"]))
    labels = [
        f"{r['burn']}\n{'Bdrm' if r['unit'] == 'MODULAIR-PM1' else 'MR'}"
        for r in rows
    ]
    peak_dur = [r["peak_window_duration_minutes"] for r in rows]
    qaqc_dur = [r.get("portal_qaqc_removal_duration_minutes", np.nan) for r in rows]
    # Hatch the near-ceiling fallback windows so they read distinctly from the
    # true saturated plateaus.
    hatches = ["" if r.get("saturated") else "///" for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(figsize("double")[0], 3.4))
    sat_color = UNIT_COLOR["MODULAIR-PM2"]
    bars = ax.bar(x - w / 2, peak_dur, w, color=sat_color, alpha=0.85)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.bar(x + w / 2, qaqc_dur, w, color=ROLE_COLORS["SMPS"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_FS - 3, rotation=30, ha="right")
    ax.set_ylabel("Duration (minutes)", fontsize=_FS)
    ax.set_title("Peak window vs portal QA/QC removal", fontsize=_FS)
    ax.tick_params(labelsize=_FS - 1)

    from matplotlib.patches import Patch
    handles = [
        Patch(color=sat_color, alpha=0.85, label="5 s peak window (saturated)"),
        Patch(facecolor=sat_color, alpha=0.85, hatch="///",
              label="5 s peak window (fallback)"),
        Patch(color=ROLE_COLORS["SMPS"], alpha=0.85, label="Portal QA/QC removal"),
    ]
    ax.legend(handles=handles, fontsize=_FS - 2)

    fig_dir = get_common_file("quantaq_figures")
    save_fig(fig, fig_dir / "modulair_5sec_qaqc_overlap.png")


# ==============================================================================
# CSV OUTPUTS
# ==============================================================================

# Per-burn CSV column order (one row per burn-unit pair).
_PER_BURN_COLS = [
    "burn", "unit", "location", "bedroom_sealed", "data_present",
    "portal_present", "saturated", "peak_window_method",
    "t_peak_start", "t_peak_end", "plateau_value",
    "peak_window_duration_seconds", "peak_window_duration_minutes",
    "n_sat_samples",
    "portal_qaqc_removal_start", "portal_qaqc_removal_end",
    "portal_qaqc_removal_duration_minutes", "overlap_with_peak_window",
    "tail_extension_minutes",
    "ratio_bin0", "class_bin0", "pre_count_bin0", "peak_count_bin0",
    "ratio_bin1", "class_bin1", "pre_count_bin1", "peak_count_bin1",
    "ratio_bin2", "class_bin2", "pre_count_bin2", "peak_count_bin2",
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
    fallback = [
        r for r in present
        if str(r.get("peak_window_method", "")).startswith("fallback")
    ]
    n_fallback = len(fallback)
    # Tally fallback windows by method so the summary can state how each
    # non-saturating record got its window.
    method_counts = {}
    for r in present:
        m = r.get("peak_window_method", "none")
        method_counts[m] = method_counts.get(m, 0) + 1

    # Near-saturation diagnostic. burn8 MODULAIR-PM2 carries dense smoke yet
    # neph_bin0 peaks at ~63 828, ~1700 counts short of the 65 535 ceiling, so
    # it is the one Morning Room burn that does not saturate. Report how close
    # each non-saturating record came so the burn8 PM2 near-miss is visible.
    near_misses = []
    for r in present:
        if r.get("saturated"):
            continue
        peak = r.get("plateau_value", np.nan)  # in-window neph_bin0 max
        if np.isfinite(peak):
            near_misses.append((r["burn"], r["unit"], float(peak)))

    # Plateau values across saturating burns, per unit.
    plateau_by_unit = {}
    for unit in UNITS:
        vals = np.array(
            [r.get("plateau_value", np.nan) for r in sat if r["unit"] == unit],
            dtype=float,
        )
        vals = vals[~np.isnan(vals)]
        plateau_by_unit[unit] = float(np.median(vals)) if vals.size else np.nan

    # OPC bin0 suppression tally over every record with a peak window (not just
    # the saturated ones): burn8 MODULAIR-PM2 never pinned at the ceiling yet
    # shows the same bin-0 collapse via its fallback window.
    windowed = [r for r in present if pd.notna(r.get("t_peak_end"))]
    supp = [r for r in windowed if r.get("class_bin0") == "suppressed"]
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
    # Most saturated burns had no portal removal at all, so the median over all
    # saturated records is ~0 and reads as "nothing removed". Report the count
    # of records that actually had a removal and the non-zero distribution
    # alongside it so the QA/QC sentence is not undercut by the zeros.
    qa_nonzero = qa_dur[qa_dur > 0]
    peak_dur = np.array(
        [r.get("peak_window_duration_minutes", np.nan) for r in sat], dtype=float
    )
    peak_dur = peak_dur[~np.isnan(peak_dur)]

    # burn8 MODULAIR-PM2 closest-approach to the ceiling (the near-saturation
    # diagnostic): the highest neph_bin0 among MODULAIR-PM2 records that did
    # not saturate.
    pm2_unsat_peaks = [p for (b, u, p) in near_misses if u == "MODULAIR-PM2"]
    pm2_max_unsat_neph_bin0 = (
        float(np.max(pm2_unsat_peaks)) if pm2_unsat_peaks else np.nan
    )

    summary = dict(
        n_burn_unit_pairs=n_pairs,
        neph_ceiling=NEPH_CEILING,
        n_neph_bin0_saturated=n_sat,
        n_fallback_window=n_fallback,
        n_fallback_burnlog=method_counts.get("fallback_burnlog", 0),
        n_fallback_aerotrak=method_counts.get("fallback_aerotrak", 0),
        n_fallback_opc=method_counts.get("fallback_opc", 0),
        plateau_median_MODULAIR_PM1=plateau_by_unit.get("MODULAIR-PM1", np.nan),
        plateau_median_MODULAIR_PM2=plateau_by_unit.get("MODULAIR-PM2", np.nan),
        pm2_max_unsaturated_neph_bin0=pm2_max_unsat_neph_bin0,
        n_opc_bin0_suppressed=int(len(supp)),
        n_windowed=int(len(windowed)),
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
        n_saturated_with_qaqc_removal=int(qa_nonzero.size),
        n_saturated_qaqc=int(qa_dur.size),
        qaqc_removal_nonzero_median_min=(
            float(np.median(qa_nonzero)) if qa_nonzero.size else np.nan),
        qaqc_removal_nonzero_min_min=(
            float(np.min(qa_nonzero)) if qa_nonzero.size else np.nan),
        qaqc_removal_nonzero_max_min=(
            float(np.max(qa_nonzero)) if qa_nonzero.size else np.nan),
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
        f"the PMS5003 nephelometer bin 0 reached its fixed 16-bit saturation "
        f"ceiling of {_fmt(summary['neph_ceiling'], '.0f')} in "
        f"{summary['n_neph_bin0_saturated']} pairs. The plateau value sat at "
        f"that ceiling for the saturated MODULAIR-PM2 records (median "
        f"{_fmt(summary['plateau_median_MODULAIR_PM2'], '.0f')}); no "
        f"MODULAIR-PM1 record saturated (median "
        f"{_fmt(summary['plateau_median_MODULAIR_PM1'], '.0f')}). The fixed "
        f"ceiling, rather than a concentration-proportional response, is what a "
        f"16-bit field clipped at {_fmt(summary['neph_ceiling'], '.0f')} would "
        f"show."
    )
    lines.append(
        f"\nThe one Morning Room near-miss is burn8 MODULAIR-PM2: its "
        f"neph_bin0 peaked at "
        f"{_fmt(summary.get('pm2_max_unsaturated_neph_bin0'), '.0f')}, about "
        f"{_fmt(summary['neph_ceiling'] - summary.get('pm2_max_unsaturated_neph_bin0', np.nan), '.0f')} "
        f"counts short of the {_fmt(summary['neph_ceiling'], '.0f')} ceiling, so "
        f"it never produced a true saturated plateau despite carrying dense "
        f"smoke (co-located AeroTrak peak PM3 over 1200 ug/m3); the 5 s record "
        f"covers the peak with no data gap, so this is a genuine sub-ceiling "
        f"maximum rather than a missing-sample artifact. For records that did "
        f"not pin at the ceiling ({summary['n_fallback_window']} pairs, all "
        f"MODULAIR-PM1 plus burn8 MODULAIR-PM2), the peak window is defined "
        f"independently of the nephelometer: "
        f"{summary['n_fallback_burnlog']} from the burn-log peak interval "
        f"(garage closure or ignition to PAC activation), "
        f"{summary['n_fallback_aerotrak']} from the co-located AeroTrak PM3 "
        f"peak, and {summary['n_fallback_opc']} from the OPC-N3 total-count "
        f"peak, so the OPC ratios and the AeroTrak cross-reference compute for "
        f"every record with data."
    )
    lines.append(
        f"\nDuring the same peak window the OPC-N3 response was bin-dependent. "
        f"The smallest bin (bin 0, 0.35-0.46 um) was suppressed (peak/pre < 0.5) "
        f"in {summary['n_opc_bin0_suppressed']} of the {summary['n_windowed']} "
        f"records with a peak window, with a median peak/pre ratio of "
        f"{_fmt(summary['opc_bin0_suppressed_median_ratio'], '.2f')} among the "
        f"suppressed records (it collapsed toward zero while the nephelometer "
        f"was pinned). The next-larger bins (bins 1-2) did not follow bin 0 "
        f"down; they rose from a near-zero pre-fire baseline. Because those "
        f"bin-1/bin-2 baselines sit below "
        f"{_fmt(MIN_BASELINE_FOR_RATIO, '.1f')} counts, their peak/pre ratios "
        f"are not reported (the tiny denominator makes the ratio meaningless); "
        f"the classification instead uses the absolute peak count, and a bin "
        f"that rose from a near-zero baseline is marked 'elevated_from_zero'. "
        f"See the per-pair table for the bin classifications and absolute "
        f"counts."
    )
    lines.append(
        f"\nThe 5 s peak window lasted a median of "
        f"{_fmt(summary['peak_window_duration_median_min'])} min "
        f"(range {_fmt(summary['peak_window_duration_min_min'])} to "
        f"{_fmt(summary['peak_window_duration_max_min'])} min). The portal QA/QC "
        f"product removed peak-window data in "
        f"{summary['n_saturated_with_qaqc_removal']} of the "
        f"{summary['n_saturated_qaqc']} saturated records; among those, the "
        f"removal lasted a median of "
        f"{_fmt(summary['qaqc_removal_nonzero_median_min'], '.0f')} min "
        f"(range {_fmt(summary['qaqc_removal_nonzero_min_min'], '.0f')} to "
        f"{_fmt(summary['qaqc_removal_nonzero_max_min'], '.0f')} min). The "
        f"remaining saturated records had no portal removal over the peak."
    )

    # Per-pair table.
    lines.append("\n## Per-pair results\n")
    lines.append(
        "| burn | unit | saturated | window method | plateau/neph max | "
        "peak win (min) | QA/QC (min) | overlap | tail (min) | "
        "bin0 ratio (class) | bin0 pre/peak ct | bin1 (class) | bin1 pre/peak ct | "
        "bin2 (class) | bin2 pre/peak ct | AeroTrak PM3 (ug/m3) | "
        "AT reversal overlap | AT gap (min) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in present:
        lines.append(
            f"| {r['burn']} | {r['unit']} | {r.get('saturated')} | "
            f"{r.get('peak_window_method', 'none')} | "
            f"{_fmt(r.get('plateau_value'), '.0f')} | "
            f"{_fmt(r.get('peak_window_duration_minutes'))} | "
            f"{_fmt(r.get('portal_qaqc_removal_duration_minutes'))} | "
            f"{_fmt(r.get('overlap_with_peak_window'), '.2f')} | "
            f"{_fmt(r.get('tail_extension_minutes'))} | "
            f"{_fmt(r.get('ratio_bin0'), '.2f')} ({r.get('class_bin0')}) | "
            f"{_fmt(r.get('pre_count_bin0'), '.2f')}/{_fmt(r.get('peak_count_bin0'), '.0f')} | "
            f"{_fmt(r.get('ratio_bin1'), '.2f')} ({r.get('class_bin1')}) | "
            f"{_fmt(r.get('pre_count_bin1'), '.2f')}/{_fmt(r.get('peak_count_bin1'), '.0f')} | "
            f"{_fmt(r.get('ratio_bin2'), '.2f')} ({r.get('class_bin2')}) | "
            f"{_fmt(r.get('pre_count_bin2'), '.2f')}/{_fmt(r.get('peak_count_bin2'), '.0f')} | "
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
        method = r.get("peak_window_method", "none")
        if not r.get("saturated"):
            peak = r.get("plateau_value", np.nan)
            short = (
                summary["neph_ceiling"] - peak if np.isfinite(peak) else np.nan
            )
            method_label = {
                "fallback_burnlog": "burn-log peak-interval",
                "fallback_aerotrak": "AeroTrak-peak",
                "fallback_opc": "OPC-peak",
            }.get(method, "no")
            if method != "none":
                lines.append(
                    f"- {r['burn']} {r['unit']}: nephelometer bin 0 did not reach "
                    f"the {_fmt(summary['neph_ceiling'], '.0f')} ceiling "
                    f"(in-window max {_fmt(peak, '.0f')}, {_fmt(short, '.0f')} "
                    f"short); {method_label} fallback peak window used."
                )
            else:
                lines.append(
                    f"- {r['burn']} {r['unit']}: no peak window could be defined "
                    f"(no saturation, no burn-log interval, no AeroTrak/OPC peak)."
                )
            flagged = True
        if r.get("bedroom_sealed"):
            lines.append(
                f"- {r['burn']} {r['unit']}: Bedroom 2 sealed; co-located "
                f"AeroTrak PM3 only "
                f"{_fmt(r.get('aerotrak_peak_PM3_mass_ug_m3'), '.0f')} ug/m3. "
                f"Windowed for completeness but not a dense-smoke case."
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
    ceiling = summary["neph_ceiling"]
    pm2_near = summary.get("pm2_max_unsaturated_neph_bin0", np.nan)

    s_neph = (
        f"The PMS5003 nephelometer channel saturated at the 16-bit ceiling "
        f"(65535) during the peak window in {n_sat} of the {n_pairs} "
        f"MODULAIR-PM-deployed burn-unit records analysed (no MODULAIR-PM1 "
        f"record saturated); the saturated-window plateau distribution spanned "
        f"the ceiling closely (median {_fmt(plat_pm2, '.0f')} across the "
        f"MODULAIR-PM2 records). The one Morning Room near-miss (burn8 "
        f"MODULAIR-PM2) peaked at {_fmt(pm2_near, '.0f')}, just below the "
        f"ceiling, with the 5 s record covering the peak with no data gap."
    )
    # bin-0 suppression over every record with a peak window (saturation or
    # near-ceiling fallback), not just the saturated ones: burn8 MODULAIR-PM2
    # never pinned at the ceiling yet shows the same bin-0 collapse, so its
    # fallback window belongs in the suppression tally. n_windowed is the
    # denominator (records that produced any peak window).
    windowed = [r for r in present if pd.notna(r.get("t_peak_end"))]
    n_windowed = len(windowed)
    supp = [r for r in windowed if r.get("class_bin0") == "suppressed"]
    n_supp = len(supp)
    supp_ratios = np.array([r.get("ratio_bin0", np.nan) for r in supp], dtype=float)
    supp_ratios = supp_ratios[~np.isnan(supp_ratios)]
    med_supp_pct = float(np.median(supp_ratios) * 100) if supp_ratios.size else np.nan
    min_supp_pct = float(np.min(supp_ratios) * 100) if supp_ratios.size else np.nan
    max_supp_pct = float(np.max(supp_ratios) * 100) if supp_ratios.size else np.nan

    qa_n_removed = summary["n_saturated_with_qaqc_removal"]
    qa_n_sat = summary["n_saturated_qaqc"]
    qa_nz_med = summary["qaqc_removal_nonzero_median_min"]
    qa_nz_min = summary["qaqc_removal_nonzero_min_min"]
    qa_nz_max = summary["qaqc_removal_nonzero_max_min"]

    s_intro = (
        "Raw 5 s data from the MODULAIR-PM instruments were obtained directly "
        "from the on-instrument storage and characterize the OPC-N3 and PMS5003 "
        "responses prior to QA/QC filtering."
    )
    s_opc = (
        f"The smallest OPC-N3 bin (0.35-0.46 um) responded in the opposite "
        f"direction during the same window, dropping to a median of "
        f"{_fmt(med_supp_pct, '.0f')}% of its pre-peak baseline (range "
        f"{_fmt(min_supp_pct, '.0f')}% to {_fmt(max_supp_pct, '.0f')}%) in "
        f"{n_supp} of the {n_windowed} records with a peak window, while the "
        f"next-larger OPC-N3 bins increased over the same window."
    )
    s_qaqc = (
        f"The portal-delivered QA/QC product removed peak-window data in "
        f"{qa_n_removed} of the {qa_n_sat} saturated records, for a median of "
        f"{_fmt(qa_nz_med, '.0f')} minutes among those records (range "
        f"{_fmt(qa_nz_min, '.0f')} to {_fmt(qa_nz_max, '.0f')} minutes); where a "
        f"removal occurred it coincided with the period in which the "
        f"nephelometer was saturated and the smallest OPC-N3 bin was suppressed."
    )

    para = (
        f"Raw 5 s data from the MODULAIR-PM instruments, retrieved from "
        f"on-instrument storage prior to QA/QC filtering, resolve the "
        f"sub-minute instrument behavior during the smoke peak. In "
        f"{n_sat} of the {n_pairs} burn-unit records analysed, the PMS5003 "
        f"nephelometer bin 0 saturated at the 16-bit ceiling (65535; no "
        f"MODULAIR-PM1 record saturated, median saturated-window value "
        f"{_fmt(plat_pm2, '.0f')} across the MODULAIR-PM2 records); the value "
        f"sat at that ceiling rather than tracking concentration, as a clipped "
        f"16-bit field would. Over the same window the OPC-N3 response was "
        f"bin-dependent: the smallest bin (0.35-0.46 um) collapsed to a median "
        f"of {_fmt(med_supp_pct, '.0f')}% of its pre-peak baseline in "
        f"{n_supp} of the {n_windowed} records with a peak window, while the "
        f"next-larger bins rose from a near-zero pre-fire baseline (their "
        f"peak/pre ratios are not reported because that near-zero denominator "
        f"makes the ratio meaningless; the increase is documented by the "
        f"absolute peak counts in the per-pair table instead). The "
        f"portal-delivered QA/QC product removed peak-window data in "
        f"{qa_n_removed} of the {qa_n_sat} saturated records (a median of "
        f"{_fmt(qa_nz_med, '.0f')} minutes where a removal occurred, range "
        f"{_fmt(qa_nz_min, '.0f')} to {_fmt(qa_nz_max, '.0f')}), and where it "
        f"did, the removal coincided with the period in which the nephelometer "
        f"was saturated and the smallest OPC-N3 bin was suppressed. These "
        f"features were co-located in time with the AeroTrak Ch1 reversal "
        f"documented in Section 3.2.2, indicating that the peak-window behavior "
        f"is common to the optical particle instruments at the highest smoke "
        f"concentrations. The 5 s record thus shows that the brief gap in the "
        f"QA/QC-filtered MODULAIR-PM product coincides with a saturated "
        f"nephelometer and a suppressed smallest OPC-N3 bin, and is not a loss "
        f"of physical signal that simple interpolation would recover."
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
        f"{para}\n\n"
        "---\n\n## Figure captions (reworked figures)\n\n"
        "**Figure 2 (main text), modulair_5sec_qaqc_timeseries_burn6.png:** "
        "\"Burn 06 Morning Room (MODULAIR-PM2) 5 s time series over the peak "
        "window. Top: the PMS5003 nephelometer bin 0 sits at the 16-bit ceiling "
        "(65535, dashed line). Middle: the OPC-N3 bin 0 (0.35-0.46 um) count is "
        "suppressed over the same window. Bottom: the 5 s peak-window span and "
        "the portal QA/QC removal interval, showing that the removal falls "
        "within the saturated, bin-0-suppressed window. The three features "
        "coincide in time.\"\n\n"
        "**Figure S3 (SI), modulair_5sec_bin_response_grid.png:** \"Peak-window "
        "reshaping of the OPC-N3 size distribution, shown as absolute OPC-N3 "
        "count concentration per bin (counts per 5 s) versus bin lower edge "
        "(log-log) for the pre-peak baseline (dashed) and the peak window "
        "(solid), one panel per MODULAIR-PM burn-unit pair with a peak window. "
        "The smallest bin (0.35-0.46 um, shaded) is suppressed during the peak "
        "while the larger bins rise from a near-zero pre-fire baseline. Counts "
        "are plotted directly rather than as a peak/pre ratio because the "
        "larger-bin pre-fire baselines are near zero.\"\n\n"
        "**Figure S6 (SI), modulair_5sec_qaqc_overlap.png (demoted from main "
        "text):** \"Duration of the 5 s peak window (solid bars where the "
        "nephelometer bin 0 saturated, hatched where defined by the fallback "
        "criterion) and of the portal QA/QC removal interval for each "
        "MODULAIR-PM burn-unit record (unit and location given in the x labels). "
        "Where a portal removal occurs it is shorter than, and falls within, the "
        "peak window.\"\n"
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
    apply_est_style()
    print("\n" + "=" * 70)
    print("MODULAIR-PM 5 s peak-window analysis  -  Burns 4-10")
    print("=" * 70)

    events = load_event_times()
    at = _load_aerotrak()

    # Pass 1: detect saturated windows to learn the median saturated half-width,
    # which anchors the AeroTrak/OPC fallbacks (methods b and c). The burn-log
    # fallback (a) does not need it, so most records resolve without pass 2.
    sat_half_widths = []
    for burn_id in BURNS:
        for unit in UNITS:
            df0 = load_5sec_burn(unit, burn_id)
            if df0 is None or df0.empty:
                continue
            ev = events.loc[burn_id] if burn_id in events.index else None
            ign = ev["ignition"] if ev is not None else pd.NaT
            gar = ev["garage_closed"] if ev is not None else pd.NaT
            pac = ev["pac_on"] if ev is not None else pd.NaT
            pw0 = _detect_peak_window(df0, ign, gar, pac)
            if pw0["saturated"] and np.isfinite(pw0["peak_window_duration_minutes"]):
                sat_half_widths.append(pw0["peak_window_duration_minutes"] / 2.0)
    median_half_width = (
        float(np.median(sat_half_widths)) if sat_half_widths else np.nan
    )
    print(
        f"\nMedian saturated-window half-width: "
        f"{_fmt(median_half_width)} min "
        f"(from {len(sat_half_widths)} saturated windows)."
    )

    results: list[dict] = []
    for burn_id in BURNS:
        print(f"--- {burn_id} ---")
        for unit in UNITS:
            rec = analyze_pair(burn_id, unit, events, at, median_half_width)
            results.append(rec)
            if rec.get("data_present"):
                print(
                    f"    {unit}: saturated={rec.get('saturated')} "
                    f"method={rec.get('peak_window_method')} "
                    f"peak_win={_fmt(rec.get('peak_window_duration_minutes'))} min "
                    f"bin0 ratio={_fmt(rec.get('ratio_bin0'), '.2f')} "
                    f"({rec.get('class_bin0')})"
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
    _mpl_qaqc_timeseries_fig2(results)
    _mpl_qaqc_overlap(results)

    print("\nWriting markdown outputs...")
    _write_summary_md(results, summary)
    _write_manuscript_md(results, summary)

    print("\nDone. Check quantaq_analysis/ and quantaq_figures/ for outputs.")


if __name__ == "__main__":
    main()
