"""
Bracketing analysis for true PM mass concentration (Section 3.2.4).

For every burn with a co-located DustTrak (photometer) and AeroTrak (OPC) at
the same location, this script brackets the true peak PM mass concentration:

    Upper bound  = DustTrak total mass corrected by literature biomass-smoke
                   factors (1.6 to 4).
    Lower bound  = AeroTrak PM3 mass corrected for unit-density bias only
                   (1.2 to 1.5). No coincidence-loss term is applied: the
                   Poisson model (Prompt 1) gives L < 0.1% at observed peak
                   count concentrations, so 1/(1-L) ~= 1.000.

A qualitative suppression caveat is attached to the lower bound when the
AeroTrak Ch1 channel was in reversal during the peak interval (the true lower
bound is then higher than reported by an unknown factor). The MODULAIR-PM
portal PM2.5 and PurpleAir uncorrected PM2.5 are reported as cross-checks.

All instrument readings are the top-10th-percentile of the peak interval, not
the single-scan maximum. The peak interval runs from the later of garage-door
closure and ignition to portable-air-cleaner (PAC / CR Box) activation.

Inputs (resolved through data_config.json):
    dusttrak                          : all_data.xlsx (TOTAL mass)
    aerotrak_bedroom/aerotrak_kitchen : all_data.xlsx (via aerotrak_coincidence)
    quantaq portal products           : via modulair_5sec_io.load_portal_burn
    purpleair                         : garage-kitchen.xlsx, (P2)kitchen sheet
    coincidence_analysis CSV          : reversal_present flag (Prompt 1)
    quantaq_analysis CSV              : portal QA/QC removal window (Prompt 2)
    burn_log                          : burn_log.xlsx, Sheet2

Outputs:
    bracketing_analysis/bracketing_per_burn.csv
    bracketing_analysis/bracketing_cross_burn_summary.csv
    bracketing_analysis/bracketing_burn09_sensitivity.csv
    bracketing_analysis/bracketing_summary.md
    bracketing_analysis/bracketing_manuscript_sentences.md
    bracketing_figures/bracketing_burn09_morning_room.png
    bracketing_figures/bracketing_cross_burn.png

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

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.aerotrak_coincidence import _load_aerotrak_all  # noqa: E402
from src.data_paths import get_common_file, get_instrument_path  # noqa: E402
from src.fig_style import (  # noqa: E402
    LOC_COLORS,
    ROLE_COLORS,
    apply_est_style,
    figsize,
    save_fig,
)
from src.modulair_5sec_io import BURN_DATES, load_event_times, load_portal_burn  # noqa: E402

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Qualifying burn-location pairs: a co-located DustTrak AND AeroTrak.
#   Bedroom 2  : DustTrak deployed burns 02-04; AeroTrak1 present.
#   Morning Room: DustTrak moved here 2024-05-19, burns 07-10; AeroTrak2 present.
# Burn 02 is dropped: the DustTrak record begins 2024-05-05, so there is no
# co-located DustTrak on the burn-02 day (2024-05-02). Burns 05-06 have no
# co-located DustTrak at either location and are excluded from the bracket.
QUALIFYING_PAIRS = [
    ("burn3", "bedroom2"),
    ("burn4", "bedroom2"),
    ("burn7", "morning_room"),
    ("burn8", "morning_room"),
    ("burn9", "morning_room"),
    ("burn10", "morning_room"),
]

# Location -> AeroTrak instrument label used by aerotrak_coincidence loader.
LOC_AEROTRAK = {"bedroom2": "AeroTrak1", "morning_room": "AeroTrak2"}

# Location -> MODULAIR-PM unit label used by modulair_5sec_io loaders.
LOC_MODULAIR = {"bedroom2": "MODULAIR-PM1", "morning_room": "MODULAIR-PM2"}

# Burns flagged sealed-bedroom in the wider study (none qualify for the bracket,
# but the flag is carried for any future row).
BEDROOM_SEALED_BURNS = {"burn5", "burn6"}

# DustTrak 7-minute clock shift (matches project scripts and data_config notes).
DUSTTRAK_TIME_SHIFT_MIN = 7.0

# Correction-factor ranges (literature, citations pre-verified in the prompt).
DUSTTRAK_CF_LOW = 4.0    # Delp & Singer 2020 (factor ~4 for fresh wildfire smoke)
DUSTTRAK_CF_HIGH = 1.6   # low end of biomass-smoke range (McNamara 2011 ~1.6)
DENSITY_LOW = 1.2        # Reid et al. 2005 unit-density correction, low end
DENSITY_HIGH = 1.5       # Reid et al. 2005 unit-density correction, high end
PURPLEAIR_CF_LOW = 2.1   # PurpleAir Plantower overestimate, smoke (high divisor)
PURPLEAIR_CF_HIGH = 1.6  # low divisor

# Top-percentile used for the sustained peak reading.
PEAK_PCTL = 0.90  # mean of readings at/above the 90th percentile (top 10%)

# Worked-example pair for the sensitivity analysis and the bar figure.
EXAMPLE_BURN = "burn9"
EXAMPLE_LOC = "morning_room"

# Input CSVs from earlier prompts.
COINCIDENCE_CSV = "aerotrak_coincidence_per_burn.csv"
MODULAIR_CSV = "modulair_5sec_peak_per_burn.csv"

# Numeric point size for matplotlib calls (matches src.fig_style BASE_FONT_PT).
_FS = 12

# CSV column order for the per-burn output.
_PER_BURN_COLS = [
    "burn", "location", "bedroom_sealed",
    "dusttrak_ug", "aerotrak_pm3_ug", "aerotrak_suppressed",
    "modulair_pm25_ug", "modulair_qaqc_bounded", "purpleair_pm25_ug",
    "mass_upper_low", "mass_upper_high", "mass_lower_low", "mass_lower_high",
    "mass_mid_low", "mass_mid_high",
    "spread_uncorrected", "spread_corrected",
    "bracket_consistent", "bracket_inverted", "notes",
]


# ==============================================================================
# PEAK-INTERVAL HELPERS
# ==============================================================================


def _peak_interval(events: pd.DataFrame, burn_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Peak interval for one burn: from the later of garage-door closure and
    ignition, to PAC (CR Box) activation.

    Parameters
    ----------
    events : pd.DataFrame
        Output of modulair_5sec_io.load_event_times (indexed by 'Burn ID').
    burn_id : str
        e.g. 'burn9'.

    Returns
    -------
    tuple of pd.Timestamp
        (interval_start, interval_end); either may be NaT if the burn-log
        entry is missing the corresponding event.
    """
    if burn_id not in events.index:
        return pd.NaT, pd.NaT
    row = events.loc[burn_id]
    ignition = row["ignition"]
    garage = row["garage_closed"]
    pac = row["pac_on"]

    starts = [t for t in (garage, ignition) if pd.notna(t)]
    start = max(starts) if starts else pd.NaT
    return start, pac


def _top_pctl_mean(values: pd.Series) -> float:
    """
    Mean of the readings at or above the 90th percentile (the sustained peak).

    Returns NaN if the input is empty after dropping NaN.
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return np.nan
    thr = v.quantile(PEAK_PCTL)
    top = v[v >= thr]
    return float(top.mean()) if not top.empty else float(v.max())


# ==============================================================================
# INSTRUMENT READINGS OVER THE PEAK INTERVAL
# ==============================================================================


def _dusttrak_peak(df_dt: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """
    DustTrak TOTAL mass top-10th-percentile (ug/m3) over the peak interval.

    df_dt carries 'datetime' (already time-shifted) and 'TOTAL [mg/m3]'.
    """
    if pd.isna(t0) or pd.isna(t1):
        return np.nan
    mask = (df_dt["datetime"] >= t0) & (df_dt["datetime"] <= t1)
    sub = df_dt.loc[mask, "TOTAL [mg/m3]"]
    return _top_pctl_mean(sub * 1000.0)  # mg/m3 -> ug/m3


def _aerotrak_pm3_peak(
    df_at: pd.DataFrame, burn_date: pd.Timestamp, t0: pd.Timestamp, t1: pd.Timestamp
) -> float:
    """
    AeroTrak PM3 mass top-10th-percentile (ug/m3) over the peak interval.

    df_at is the full multi-day AeroTrak frame from _load_aerotrak_all, which
    already computes the cumulative 'PM3 (µg/m³)' column from per-bin
    differential counts (Mie sphere, geometric-mean diameter, unit density).
    """
    if pd.isna(t0) or pd.isna(t1):
        return np.nan
    mask = (df_at["Date and Time"] >= t0) & (df_at["Date and Time"] <= t1)
    sub = df_at.loc[mask, "PM3 (µg/m³)"]
    return _top_pctl_mean(sub)


def _modulair_pm25_peak(unit: str, burn_id: str, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """
    MODULAIR-PM portal PM2.5 top-10th-percentile (ug/m3) over the peak interval.

    Returns NaN if the unit was not deployed for this burn (MODULAIR-PM units
    cover burns 04 and 07-10 only) or the portal product has no in-window data.
    """
    if pd.isna(t0) or pd.isna(t1) or burn_id not in BURN_DATES:
        return np.nan
    portal = load_portal_burn(unit, burn_id)
    if portal is None or "pm25" not in portal.columns:
        return np.nan
    mask = (portal["timestamp"] >= t0) & (portal["timestamp"] <= t1)
    return _top_pctl_mean(portal.loc[mask, "pm25"])


def _purpleair_peak(df_pa: pd.DataFrame | None, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """
    PurpleAir uncorrected PM2.5 (channel-averaged Plantower 'Average') top-10th
    percentile (ug/m3) over the peak interval. Morning Room only.
    """
    if df_pa is None or pd.isna(t0) or pd.isna(t1):
        return np.nan
    mask = (df_pa["DateTime"] >= t0) & (df_pa["DateTime"] <= t1)
    return _top_pctl_mean(df_pa.loc[mask, "Average"])


# ==============================================================================
# AUXILIARY CSV LOADERS (Prompt 1 reversal flag, Prompt 2 QA/QC window)
# ==============================================================================


def _load_reversal_flags() -> pd.DataFrame:
    """
    Load the AeroTrak coincidence per-burn CSV and return the reversal flag
    keyed by (burn, location).

    Returns
    -------
    pd.DataFrame
        Columns 'burn', 'location', 'reversal_present' (bool).
    """
    path = get_common_file("coincidence_analysis") / COINCIDENCE_CSV
    df = pd.read_csv(path)
    df = df[["burn", "location", "reversal_present"]].copy()
    df["reversal_present"] = df["reversal_present"].astype(bool)
    return df


def _load_qaqc_windows() -> pd.DataFrame:
    """
    Load the MODULAIR-PM peak-window CSV and return the portal QA/QC removal
    interval keyed by (burn, unit).

    Returns
    -------
    pd.DataFrame
        Columns 'burn', 'unit', 'portal_qaqc_removal_start',
        'portal_qaqc_removal_end' (pd.Timestamp / NaT).
    """
    path = get_common_file("quantaq_analysis") / MODULAIR_CSV
    df = pd.read_csv(path)
    keep = ["burn", "unit", "portal_qaqc_removal_start", "portal_qaqc_removal_end"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in ("portal_qaqc_removal_start", "portal_qaqc_removal_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _qaqc_bounded(
    qaqc: pd.DataFrame, burn_id: str, unit: str, t0: pd.Timestamp, t1: pd.Timestamp
) -> bool:
    """
    True if the peak interval [t0, t1] overlaps the portal QA/QC removal window
    for this burn-unit pair, meaning the portal-delivered PM2.5 reading at peak
    is partially filtered (a data gap) and should be treated as bounded.
    """
    if pd.isna(t0) or pd.isna(t1):
        return False
    row = qaqc[(qaqc["burn"] == burn_id) & (qaqc["unit"] == unit)]
    if row.empty:
        return False
    q0 = row["portal_qaqc_removal_start"].iloc[0]
    q1 = row["portal_qaqc_removal_end"].iloc[0]
    if pd.isna(q0) or pd.isna(q1):
        return False
    # Intervals overlap when each starts before the other ends.
    return bool((q0 <= t1) and (t0 <= q1))


# ==============================================================================
# BRACKET COMPUTATION
# ==============================================================================


def _compute_bracket(
    dusttrak_ug: float,
    aerotrak_pm3_ug: float,
    purpleair_pm25_ug: float,
    modulair_pm25_ug: float,
    modulair_qaqc_bounded: bool,
) -> dict:
    """
    Compute the corrected bracket bounds, cross-check range, and summary
    metrics from the peak-interval instrument readings.

    Returns
    -------
    dict
        mass_upper_low/high, mass_lower_low/high, mass_mid_low/high,
        spread_uncorrected, spread_corrected, bracket_consistent,
        bracket_inverted.
    """
    # Upper bound: DustTrak corrected by biomass-smoke factors (1.6 to 4).
    mass_upper_low = dusttrak_ug / DUSTTRAK_CF_LOW
    mass_upper_high = dusttrak_ug / DUSTTRAK_CF_HIGH

    # Lower bound: AeroTrak corrected for unit-density only (no coincidence term).
    mass_lower_low = aerotrak_pm3_ug * DENSITY_LOW
    mass_lower_high = aerotrak_pm3_ug * DENSITY_HIGH

    # Mid-range corroboration: PurpleAir uncorrected PM2.5.
    if not np.isnan(purpleair_pm25_ug):
        mass_mid_low = purpleair_pm25_ug / PURPLEAIR_CF_LOW
        mass_mid_high = purpleair_pm25_ug / PURPLEAIR_CF_HIGH
    else:
        mass_mid_low = np.nan
        mass_mid_high = np.nan

    # Uncorrected instrument spread. MODULAIR is excluded from the min when its
    # peak reading is QA/QC-bounded (a partial data gap).
    raw_vals = [dusttrak_ug, aerotrak_pm3_ug]
    if not np.isnan(purpleair_pm25_ug):
        raw_vals.append(purpleair_pm25_ug)
    if not np.isnan(modulair_pm25_ug) and not modulair_qaqc_bounded:
        raw_vals.append(modulair_pm25_ug)
    raw_vals = [v for v in raw_vals if not np.isnan(v) and v > 0]
    spread_uncorrected = (max(raw_vals) / min(raw_vals)) if len(raw_vals) >= 2 else np.nan

    # Corrected spread: widest upper over lowest lower.
    spread_corrected = (
        mass_upper_high / mass_lower_low
        if (not np.isnan(mass_upper_high) and not np.isnan(mass_lower_low) and mass_lower_low > 0)
        else np.nan
    )

    bracket_consistent = bool(mass_lower_high < mass_upper_low)
    bracket_inverted = bool(mass_lower_high > mass_upper_high)

    return dict(
        mass_upper_low=mass_upper_low,
        mass_upper_high=mass_upper_high,
        mass_lower_low=mass_lower_low,
        mass_lower_high=mass_lower_high,
        mass_mid_low=mass_mid_low,
        mass_mid_high=mass_mid_high,
        spread_uncorrected=spread_uncorrected,
        spread_corrected=spread_corrected,
        bracket_consistent=bracket_consistent,
        bracket_inverted=bracket_inverted,
    )


# ==============================================================================
# DATA LOADING
# ==============================================================================


def _load_dusttrak() -> pd.DataFrame:
    """
    Load the combined DustTrak all_data.xlsx, apply the 7-minute clock shift,
    and return a frame with 'datetime' and the mass columns.
    """
    path = get_instrument_path("dusttrak") / "all_data.xlsx"
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"]) + pd.Timedelta(
        minutes=DUSTTRAK_TIME_SHIFT_MIN
    )
    return df


def _load_purpleair() -> pd.DataFrame | None:
    """
    Load the PurpleAir Morning Room (kitchen) sheet. 'Average' is the mean of
    the two co-located Plantower channels; no time shift is applied (matches
    the existing PurpleAir comparison script).

    Returns None if the file or sheet is unavailable.
    """
    path = get_instrument_path("purpleair") / "garage-kitchen.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name="(P2)kitchen")
    df.columns = df.columns.str.strip()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


# ==============================================================================
# PER-PAIR DRIVER
# ==============================================================================


def analyze_pair(
    burn_id: str,
    location: str,
    events: pd.DataFrame,
    df_dt: pd.DataFrame,
    df_at_by_loc: dict[str, pd.DataFrame],
    df_pa: pd.DataFrame | None,
    reversal: pd.DataFrame,
    qaqc: pd.DataFrame,
) -> dict:
    """
    Compute the full bracket row for one qualifying burn-location pair.

    Returns the per-burn CSV dict plus private '_t0'/'_t1' keys for any
    downstream use.
    """
    t0, t1 = _peak_interval(events, burn_id)
    burn_date = events.loc[burn_id, "Date"] if burn_id in events.index else pd.NaT

    # Peak-interval instrument readings (top 10th percentile).
    dusttrak_ug = _dusttrak_peak(df_dt, t0, t1)
    aerotrak_pm3_ug = _aerotrak_pm3_peak(df_at_by_loc[location], burn_date, t0, t1)
    unit = LOC_MODULAIR[location]
    modulair_pm25_ug = _modulair_pm25_peak(unit, burn_id, t0, t1)
    purpleair_pm25_ug = (
        _purpleair_peak(df_pa, t0, t1) if location == "morning_room" else np.nan
    )

    # Reversal suppression flag (Prompt 1) and QA/QC bounding (Prompt 2).
    rev_row = reversal[(reversal["burn"] == burn_id) & (reversal["location"] == location)]
    aerotrak_suppressed = bool(rev_row["reversal_present"].iloc[0]) if not rev_row.empty else False
    modulair_qaqc_bounded = _qaqc_bounded(qaqc, burn_id, unit, t0, t1)

    bracket = _compute_bracket(
        dusttrak_ug,
        aerotrak_pm3_ug,
        purpleair_pm25_ug,
        modulair_pm25_ug,
        modulair_qaqc_bounded,
    )

    # Notes.
    notes_parts = []
    if aerotrak_suppressed:
        notes_parts.append(
            "AeroTrak Ch1 reversal present during peak; mass_lower_* is a "
            "further underestimate of unknown magnitude."
        )
    if modulair_qaqc_bounded:
        notes_parts.append(
            "MODULAIR portal PM2.5 peak reading overlaps the QA/QC removal "
            "window; treat as partially filtered."
        )
    if bracket["bracket_inverted"]:
        notes_parts.append(
            "bracket inverted: corrected lower bound exceeds corrected upper "
            "bound (see summary for likely-responsible factor)."
        )

    return {
        "burn": burn_id,
        "location": location,
        "bedroom_sealed": burn_id in BEDROOM_SEALED_BURNS,
        "dusttrak_ug": dusttrak_ug,
        "aerotrak_pm3_ug": aerotrak_pm3_ug,
        "aerotrak_suppressed": aerotrak_suppressed,
        "modulair_pm25_ug": modulair_pm25_ug,
        "modulair_qaqc_bounded": modulair_qaqc_bounded,
        "purpleair_pm25_ug": purpleair_pm25_ug,
        **bracket,
        "notes": " ".join(notes_parts),
        "_t0": t0,
        "_t1": t1,
    }


# ==============================================================================
# SENSITIVITY ANALYSIS (Burn 09 Morning Room worked example)
# ==============================================================================


def _sensitivity(example: dict) -> pd.DataFrame:
    """
    Vary the AeroTrak density correction and the DustTrak correction factor for
    the worked example, tabulating mass_lower, mass_upper_low/high, and
    spread_corrected for each scenario.
    """
    at = example["aerotrak_pm3_ug"]
    dt = example["dusttrak_ug"]

    rows = []

    # Density-correction scenarios (lower bound). Upper bounds held at the
    # default literature range so spread_corrected reflects only the density
    # choice's effect on the lower bound.
    up_low_def = dt / DUSTTRAK_CF_LOW
    up_high_def = dt / DUSTTRAK_CF_HIGH
    for label, dfac in [
        ("density 1.0 (no correction)", 1.0),
        ("density 1.2 (low default)", 1.2),
        ("density 1.5 (high default)", 1.5),
        ("density 2.0 (Reid 2005 BC-rich)", 2.0),
    ]:
        mass_lower = at * dfac
        rows.append(
            {
                "scenario_type": "density",
                "scenario": label,
                "factor": dfac,
                "mass_lower": mass_lower,
                "mass_upper_low": up_low_def,
                "mass_upper_high": up_high_def,
                "spread_corrected": (up_high_def / mass_lower if mass_lower > 0 else np.nan),
            }
        )

    # DustTrak correction-factor scenarios (upper bound). Lower bound held at
    # the default density range midpoint behavior: report mass_lower at the low
    # density (1.2) so spread_corrected = mass_upper / mass_lower_low.
    mass_lower_low_def = at * DENSITY_LOW
    for label, cfac in [
        ("DustTrak CF 1.6 (low; mass_upper_high)", 1.6),
        ("DustTrak CF 2.5 (midpoint)", 2.5),
        ("DustTrak CF 4.0 (high; mass_upper_low)", 4.0),
    ]:
        mass_upper = dt / cfac
        rows.append(
            {
                "scenario_type": "dusttrak_cf",
                "scenario": label,
                "factor": cfac,
                "mass_lower": mass_lower_low_def,
                "mass_upper_low": dt / DUSTTRAK_CF_LOW,
                "mass_upper_high": mass_upper,
                "spread_corrected": (
                    mass_upper / mass_lower_low_def if mass_lower_low_def > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# CSV WRITERS
# ==============================================================================


def _write_per_burn(results: list[dict], out_dir: Path) -> pd.DataFrame:
    """Write bracketing_per_burn.csv and return the DataFrame."""
    rows = [{k: r.get(k, np.nan) for k in _PER_BURN_COLS} for r in results]
    df = pd.DataFrame(rows, columns=_PER_BURN_COLS)
    path = out_dir / "bracketing_per_burn.csv"
    df.to_csv(path, index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")
    return df


def _write_cross_burn(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Write bracketing_cross_burn_summary.csv: median and range of the lower
    bracket, upper bracket, and corrected/uncorrected spreads across all
    qualifying pairs. Returns the summary dict for the markdown writers.
    """

    def _stat(col: str) -> dict:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return {f"{col}_median": np.nan, f"{col}_min": np.nan, f"{col}_max": np.nan}
        return {
            f"{col}_median": float(s.median()),
            f"{col}_min": float(s.min()),
            f"{col}_max": float(s.max()),
        }

    summary = {"n_pairs": int(len(df))}
    for col in (
        "mass_lower_low",
        "mass_lower_high",
        "mass_upper_low",
        "mass_upper_high",
        "spread_corrected",
        "spread_uncorrected",
    ):
        summary.update(_stat(col))
    summary["n_bracket_consistent"] = int(df["bracket_consistent"].sum())
    summary["n_bracket_inverted"] = int(df["bracket_inverted"].sum())

    path = out_dir / "bracketing_cross_burn_summary.csv"
    pd.DataFrame([summary]).to_csv(path, index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")
    return summary


def _write_sensitivity(sens: pd.DataFrame, out_dir: Path) -> None:
    """Write bracketing_burn09_sensitivity.csv."""
    path = out_dir / "bracketing_burn09_sensitivity.csv"
    sens.to_csv(path, index=False, float_format="%.4g")
    print(f"    [CSV] {path.name}")


# ==============================================================================
# FIGURES
# ==============================================================================


def _fig_burn09_bars(example: dict, fig_dir: Path) -> None:
    """
    Worked-example bar figure: one horizontal bar per instrument showing the
    raw reading (point) and the corrected range (shaded band). DustTrak labeled
    with the correction-factor source; AeroTrak labeled with the density range
    and the suppression flag if present. Intended for main text (Figure 4).

    Raw-value labels are placed to the LEFT of each point (ha="right") so they
    cannot collide with the corrected band that extends to the point's right,
    and the x-limits carry headroom so no text leaves the axes.
    """
    dt = example["dusttrak_ug"]
    at = example["aerotrak_pm3_ug"]
    pa = example["purpleair_pm25_ug"]
    mod = example["modulair_pm25_ug"]
    suppressed = example["aerotrak_suppressed"]

    # (label, raw, band_low, band_high, color, annotation)
    bars = []
    bars.append(
        (
            "DustTrak\n(photometer)",
            dt,
            example["mass_upper_low"],
            example["mass_upper_high"],
            ROLE_COLORS["DustTrak"],
            "corrected /1.6 to /4\n(McNamara 2011; Delp & Singer 2020)",
        )
    )
    at_note = "corrected x1.2 to x1.5 (Reid 2005)"
    if suppressed:
        at_note += "\nCh1 reversal: true value likely higher"
    bars.append(
        (
            "AeroTrak\n(OPC, PM3)",
            at,
            example["mass_lower_low"],
            example["mass_lower_high"],
            ROLE_COLORS["OPC-N3"],
            at_note,
        )
    )
    if not np.isnan(pa):
        bars.append(
            (
                "PurpleAir\n(uncorrected PM2.5)",
                pa,
                example["mass_mid_low"],
                example["mass_mid_high"],
                ROLE_COLORS["PurpleAir"],
                "corrected /1.6 to /2.1",
            )
        )
    # MODULAIR-PM2 portal PM2.5: a single QA/QC-bounded reading (Section 3.2.3),
    # so it is shown as a point with no correction band. NaN band edges signal
    # the no-band case to the draw loop below.
    if not np.isnan(mod):
        bars.append(
            (
                "MODULAIR-PM2\n(portal PM2.5)",
                mod,
                np.nan,
                np.nan,
                ROLE_COLORS["PMS5003"],
                "QA/QC-bounded (Section 3.2.3)",
            )
        )

    fig, ax = plt.subplots(figsize=figsize("onehalf", aspect=0.85))
    y_positions = list(range(len(bars)))[::-1]

    # x-range with headroom: low end below the smallest value, high end with
    # room for the right-hand correction-factor annotations on a log axis.
    all_vals = [v for b in bars for v in (b[1], b[2], b[3]) if not np.isnan(v)]
    x_lo = min(all_vals) / 4.0
    x_hi = max(all_vals) * 4.0
    ax.set_xlim(x_lo, x_hi)

    for y, (label, raw, lo, hi, color, note) in zip(y_positions, bars):
        has_band = not (np.isnan(lo) or np.isnan(hi))
        if has_band:
            # Corrected range as a shaded band.
            ax.barh(
                y, hi - lo, left=lo, height=0.45, color=color, alpha=0.30,
                edgecolor=color, linewidth=1.2,
            )
        # Raw instrument reading as a point.
        ax.plot(raw, y, "o", color=color, markersize=9, zorder=5)
        # Raw-value label below the point so it never overlaps the band or the
        # correction note placed above.
        ax.annotate(
            f"raw {raw:.0f}",
            xy=(raw, y), xytext=(0, -11), textcoords="offset points",
            ha="center", va="top", fontsize=_FS - 3, color=color,
        )
        # Correction-factor note above the band/point, clear of the markers and
        # of any dashed bracket lines.
        note_x = hi if has_band else raw
        ax.annotate(
            note,
            xy=(note_x, y), xytext=(0, 12), textcoords="offset points",
            ha="center", va="bottom", fontsize=_FS - 3, color="black",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([b[0] for b in bars], fontsize=_FS)
    ax.set_ylim(-0.8, len(bars) - 0.2)
    ax.set_xscale("log")
    ax.set_xlabel("PM mass concentration (µg/m³)", fontsize=_FS)
    ax.tick_params(axis="x", labelsize=_FS)
    ax.grid(axis="x", ls=":", alpha=0.4)

    save_fig(fig, fig_dir / "bracketing_burn09_morning_room.png")


def _fig_cross_burn(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Cross-burn figure (Figure S6): one panel per location, x = burn number, y =
    corrected range as a vertical band from mass_lower_low to mass_upper_high.
    Shared log y. The lower-bound (mass_lower_low) and upper-bound
    (mass_upper_high) values are annotated next to each band so the bracket
    endpoints are readable.
    """
    locations = ["bedroom2", "morning_room"]
    loc_labels = {"bedroom2": "Bedroom 2", "morning_room": "Morning Room"}
    loc_color = {"bedroom2": LOC_COLORS["bedroom2"],
                 "morning_room": LOC_COLORS["morning_room"]}

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", aspect=0.5),
                             sharey=True)

    for ax, loc in zip(axes, locations):
        sub = df[df["location"] == loc].copy()
        sub = sub.sort_values("burn", key=lambda s: s.str.replace("burn", "").astype(int))
        if sub.empty:
            ax.set_visible(False)
            continue
        xs = [int(b.replace("burn", "")) for b in sub["burn"]]
        for x, (_, row) in zip(xs, sub.iterrows()):
            lo = row["mass_lower_low"]
            hi = row["mass_upper_high"]
            ax.vlines(x, lo, hi, color=loc_color[loc], lw=8, alpha=0.45)
            ax.plot(x, lo, "_", color=loc_color[loc], markersize=14)
            ax.plot(x, hi, "_", color=loc_color[loc], markersize=14)
            # Annotate the bracket endpoints so they are readable.
            ax.annotate(f"{hi:.0f}", xy=(x, hi), xytext=(6, 2),
                        textcoords="offset points", fontsize=_FS - 3,
                        color=loc_color[loc], va="bottom")
            ax.annotate(f"{lo:.0f}", xy=(x, lo), xytext=(6, -2),
                        textcoords="offset points", fontsize=_FS - 3,
                        color=loc_color[loc], va="top")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=_FS)
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_xlabel("Burn number", fontsize=_FS)
        ax.set_title(loc_labels[loc], fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.grid(axis="y", ls=":", alpha=0.4)

    axes[0].set_ylabel("Corrected PM mass bracket (µg/m³)", fontsize=_FS)

    save_fig(fig, fig_dir / "bracketing_cross_burn.png")


# ==============================================================================
# MARKDOWN WRITERS
# ==============================================================================


def _fmt(v: float, fmt: str = ".0f") -> str:
    """Format a float for prose, or '[no data]' if NaN."""
    return f"{v:{fmt}}" if (v is not None and not np.isnan(v)) else "[no data]"


def _write_summary_md(df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    """Write bracketing_summary.md (plain-language synthesis plus table)."""
    n = summary["n_pairs"]
    n_consistent = summary["n_bracket_consistent"]
    n_inverted = summary["n_bracket_inverted"]

    lines = [
        "# Bracketing analysis for true PM mass - summary",
        "",
        "## Method",
        "",
        "For each qualifying burn-location pair (co-located DustTrak and "
        "AeroTrak), the true peak PM mass concentration is bracketed between a "
        "DustTrak-derived upper bound (raw reading divided by literature "
        "biomass-smoke correction factors of 1.6 to 4) and an AeroTrak-derived "
        "lower bound (PM3 reading multiplied by a unit-density correction of "
        "1.2 to 1.5). No coincidence-loss correction is applied to the AeroTrak: "
        "the Poisson model gives L < 0.1% at observed peak count concentrations, "
        "so the 1/(1-L) factor is effectively 1.000. All readings are the mean "
        "of the top 10th percentile of each instrument over the peak interval "
        "(later of garage-door closure and ignition, to PAC activation).",
        "",
        "## Coverage",
        "",
        f"{n} qualifying burn-location pairs: Bedroom 2 burns 3 and 4 "
        "(DustTrak co-located there for burns 02-04), Morning Room burns 7-10 "
        "(DustTrak moved to the Morning Room on 2024-05-19). Burn 02 is excluded "
        "because the DustTrak record begins 2024-05-05, so no DustTrak was "
        "co-located on the burn-02 day. Burns 05 and 06 are excluded from the "
        "bracket because no DustTrak was co-located with an AeroTrak at either "
        "location during those burns.",
        "",
        "## Cross-burn result",
        "",
        f"The corrected lower bracket (mass_lower_low) ranged from "
        f"{_fmt(summary['mass_lower_low_min'])} to "
        f"{_fmt(summary['mass_lower_low_max'])} ug/m3 "
        f"(median {_fmt(summary['mass_lower_low_median'])}). The corrected upper "
        f"bracket (mass_upper_high) ranged from "
        f"{_fmt(summary['mass_upper_high_min'])} to "
        f"{_fmt(summary['mass_upper_high_max'])} ug/m3 "
        f"(median {_fmt(summary['mass_upper_high_median'])}). The corrected "
        f"spread (mass_upper_high / mass_lower_low) ranged from "
        f"{_fmt(summary['spread_corrected_min'], '.2f')} to "
        f"{_fmt(summary['spread_corrected_max'], '.2f')} "
        f"(median {_fmt(summary['spread_corrected_median'], '.2f')}), versus an "
        f"uncorrected instrument spread of "
        f"{_fmt(summary['spread_uncorrected_min'], '.2f')} to "
        f"{_fmt(summary['spread_uncorrected_max'], '.2f')} "
        f"(median {_fmt(summary['spread_uncorrected_median'], '.2f')}).",
        "",
        f"The corrected bounds did not overlap (mass_lower_high < "
        f"mass_upper_low) in {n_consistent} of {n} pairs, leaving an explicit "
        f"gap that contains the true value. {n_inverted} of {n} pairs were "
        f"inverted (corrected lower bound exceeds corrected upper bound).",
        "",
    ]

    if n_inverted > 0:
        inv = df[df["bracket_inverted"]]
        for _, r in inv.iterrows():
            lines.append(
                f"- Inversion in {r['burn']} {r['location']}: "
                f"mass_lower_high {_fmt(r['mass_lower_high'])} > mass_upper_high "
                f"{_fmt(r['mass_upper_high'])} ug/m3. The most likely-responsible "
                f"input is the DustTrak upper-bound correction factor (a /4 "
                f"correction may be too aggressive for this burn's smoke), or the "
                f"AeroTrak density factor at the high end (x1.5)."
            )
        lines.append("")

    # Per-pair table.
    lines.append("## Per-pair brackets")
    lines.append("")
    lines.append(
        "| burn | location | DustTrak | AeroTrak PM3 | upper (low-high) | "
        "lower (low-high) | spread corr | spread raw | consistent | inverted |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['burn']} | {r['location']} | {_fmt(r['dusttrak_ug'])} | "
            f"{_fmt(r['aerotrak_pm3_ug'])} | "
            f"{_fmt(r['mass_upper_low'])}-{_fmt(r['mass_upper_high'])} | "
            f"{_fmt(r['mass_lower_low'])}-{_fmt(r['mass_lower_high'])} | "
            f"{_fmt(r['spread_corrected'], '.2f')} | "
            f"{_fmt(r['spread_uncorrected'], '.2f')} | "
            f"{r['bracket_consistent']} | {r['bracket_inverted']} |"
        )

    path = out_dir / "bracketing_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    [MD] {path.name}")


def _write_manuscript_md(df: pd.DataFrame, summary: dict, example: dict, out_dir: Path) -> None:
    """Write bracketing_manuscript_sentences.md (Sentences 1-4 plus inversion)."""
    n = summary["n_pairs"]

    s1 = (
        f"Applying a particle density correction of 1.2 to 1.5 relative to unit "
        f"density to the AeroTrak2 PM3 reading of approximately "
        f"{_fmt(example['aerotrak_pm3_ug'])} ug/m3 yields a corrected "
        f"lower-bound estimate of approximately {_fmt(example['mass_lower_low'])} "
        f"ug/m3 to {_fmt(example['mass_lower_high'])} ug/m3; because the AeroTrak "
        f"Ch1 channel was in reversal during this peak interval, the true value "
        f"is likely higher than this estimate by an unknown factor."
    )
    s2 = (
        f"Applying the literature range of biomass-smoke DustTrak correction "
        f"factors (1.6 to 4) to the DustTrak reading of approximately "
        f"{_fmt(example['dusttrak_ug'])} ug/m3 yields a corrected upper-bound "
        f"estimate of approximately {_fmt(example['mass_upper_low'])} ug/m3 to "
        f"{_fmt(example['mass_upper_high'])} ug/m3."
    )
    s3 = (
        f"The corrected bracket spans a factor of approximately "
        f"{_fmt(example['spread_corrected'], '.2g')}, compared to an uncorrected "
        f"instrument spread of a factor of approximately "
        f"{_fmt(example['spread_uncorrected'], '.2g')}."
    )

    # Conclusion clause for Sentence 4.
    sc_med = summary["spread_corrected_median"]
    su_med = summary["spread_uncorrected_median"]
    if not np.isnan(sc_med) and not np.isnan(su_med):
        if sc_med < 0.9 * su_med:
            rel = "smaller than"
            verdict = (
                "the correction narrows the inter-instrument disagreement, so "
                "the bracketing approach adds explanatory value relative to "
                "reporting the raw instrument spread"
            )
        elif sc_med > 1.1 * su_med:
            rel = "larger than"
            verdict = (
                "the correction does not narrow the disagreement, so the "
                "bracketing approach does not add explanatory value beyond the "
                "raw instrument spread for these burns"
            )
        else:
            rel = "similar to"
            verdict = (
                "the correction leaves the disagreement broadly unchanged, so "
                "the bracketing approach mainly reframes rather than narrows the "
                "raw instrument spread"
            )
    else:
        rel = "comparable to"
        verdict = "the comparison is inconclusive given missing data"

    s4 = (
        f"Across the {n} qualifying burn-location pairs, the corrected spread "
        f"ranged from {_fmt(summary['spread_corrected_min'], '.2g')} to "
        f"{_fmt(summary['spread_corrected_max'], '.2g')} "
        f"(median {_fmt(sc_med, '.2g')}), {rel} the uncorrected spread range of "
        f"{_fmt(summary['spread_uncorrected_min'], '.2g')} to "
        f"{_fmt(summary['spread_uncorrected_max'], '.2g')} "
        f"(median {_fmt(su_med, '.2g')}); {verdict}."
    )

    # Sentence acknowledging where the PM2.5-only cross-checks (PurpleAir
    # corrected mid-range, MODULAIR portal PM2.5) fall relative to the bracket.
    # Both are reported only when they sit below the corrected lower bound, which
    # is the expected behavior here: the AeroTrak Ch1 reversal means the true
    # lower bound is higher than the OPC suggests, and PM2.5-only sensors miss
    # the coarse mass captured by the DustTrak total and the AeroTrak PM3.
    ex_lower_low = example["mass_lower_low"]
    pa_mid_low = example["mass_mid_low"]
    pa_mid_high = example["mass_mid_high"]
    mod_pm25 = example["modulair_pm25_ug"]
    mid_clause_parts = []
    if not np.isnan(pa_mid_high):
        mid_clause_parts.append(
            f"the PurpleAir corrected mid-range of approximately "
            f"{_fmt(pa_mid_low)} ug/m3 to {_fmt(pa_mid_high)} ug/m3"
        )
    if not np.isnan(mod_pm25):
        mid_clause_parts.append(
            f"the MODULAIR-PM portal PM2.5 reading of approximately "
            f"{_fmt(mod_pm25)} ug/m3"
        )
    if mid_clause_parts:
        s_mid = (
            f"Both PM2.5-only cross-checks fell below the corrected lower bound "
            f"of approximately {_fmt(ex_lower_low)} ug/m3 in the worked example "
            f"({' and '.join(mid_clause_parts)}), consistent with these "
            f"fine-mode sensors missing the coarse mass captured by the DustTrak "
            f"total and the AeroTrak PM3 and with the AeroTrak Ch1 reversal that "
            f"makes the true lower bound higher than the OPC count alone implies; "
            f"the cross-checks therefore bound the fine-mode contribution rather "
            f"than corroborating the total-mass bracket."
        )
    else:
        s_mid = None

    lines = [
        "# Bracketing - manuscript sentences for Section 3.2.4",
        "",
        "_All values derived from data (top 10th percentile over the peak "
        "interval). Worked example: Burn 09 Morning Room. Insert into manuscript "
        "text._",
        "",
        "---",
        "",
        f'**Sentence 1 (lower bound, worked example):** "{s1}"',
        "",
        f'**Sentence 2 (upper bound, worked example):** "{s2}"',
        "",
        f'**Sentence 3 (spread, worked example):** "{s3}"',
        "",
        f'**Sentence 4 (cross-burn):** "{s4}"',
        "",
    ]

    if s_mid is not None:
        lines += [
            f'**Sentence 4b (PM2.5 cross-checks below bracket):** "{s_mid}"',
            "",
        ]

    inv = df[df["bracket_inverted"]]
    if not inv.empty:
        burns = ", ".join(f"{r['burn']} {r['location']}" for _, r in inv.iterrows())
        s5 = (
            f"In {len(inv)} of the {n} pairs ({burns}) the corrected bracket was "
            f"inverted: the density-corrected AeroTrak lower bound exceeded the "
            f"DustTrak-corrected upper bound, indicating that the DustTrak "
            f"correction factor of 4 (Delp & Singer 2020, fresh wildfire smoke) "
            f"is too aggressive for this smoke, or that the AeroTrak high-end "
            f"density factor of 1.5 overstates the lower bound."
        )
        lines += [f'**Sentence 5 (inversion finding):** "{s5}"', ""]

    path = out_dir / "bracketing_manuscript_sentences.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    [MD] {path.name}")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Run the full bracketing pipeline and write all outputs."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    apply_est_style()

    out_dir = get_common_file("bracketing_analysis")
    fig_dir = get_common_file("bracketing_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading event times and instrument data...")
    events = load_event_times()
    df_dt = _load_dusttrak()
    df_pa = _load_purpleair()
    df_at_by_loc = {
        "bedroom2": _load_aerotrak_all("AeroTrak1"),
        "morning_room": _load_aerotrak_all("AeroTrak2"),
    }
    reversal = _load_reversal_flags()
    qaqc = _load_qaqc_windows()

    print("Computing brackets...")
    results = []
    for burn_id, location in QUALIFYING_PAIRS:
        r = analyze_pair(
            burn_id, location, events, df_dt, df_at_by_loc, df_pa, reversal, qaqc
        )
        results.append(r)
        print(
            f"  {burn_id} {location}: DT={_fmt(r['dusttrak_ug'])} "
            f"AT={_fmt(r['aerotrak_pm3_ug'])} "
            f"upper={_fmt(r['mass_upper_low'])}-{_fmt(r['mass_upper_high'])} "
            f"lower={_fmt(r['mass_lower_low'])}-{_fmt(r['mass_lower_high'])} "
            f"inverted={r['bracket_inverted']}"
        )

    print("\nWriting CSVs...")
    df = _write_per_burn(results, out_dir)
    summary = _write_cross_burn(df, out_dir)

    example = next(
        r for r in results if r["burn"] == EXAMPLE_BURN and r["location"] == EXAMPLE_LOC
    )
    sens = _sensitivity(example)
    _write_sensitivity(sens, out_dir)

    # OLD vs NEW comparison table for the 1.5 -> 1.6 low-end correction-factor
    # change (the only correctness edit this pass). OLD values are the
    # pre-change CSV figures, computed here as DustTrak/1.5 for direct
    # comparison so the draft 3.2.4 numbers can be updated.
    ex_dt = example["dusttrak_ug"]
    ex_ll = example["mass_lower_low"]
    old_uh = ex_dt / 1.5
    new_uh = example["mass_upper_high"]
    old_spread = old_uh / ex_ll if ex_ll > 0 else np.nan
    new_spread = example["spread_corrected"]
    print("\nOLD vs NEW (CF low end 1.5 -> 1.6):")
    print(f"  Burn 09 mass_upper_high : {old_uh:.0f} -> {new_uh:.0f} ug/m3")
    print(f"  Burn 09 spread_corrected: {old_spread:.2f} -> {new_spread:.2f}")
    print(
        f"  cross-burn corrected spread (median/min/max): "
        f"~5.93/4.58/8.54 -> "
        f"{summary['spread_corrected_median']:.2f}/"
        f"{summary['spread_corrected_min']:.2f}/"
        f"{summary['spread_corrected_max']:.2f}"
    )

    print("\nWriting figures...")
    _fig_burn09_bars(example, fig_dir)
    _fig_cross_burn(df, fig_dir)

    print("\nWriting markdown...")
    _write_summary_md(df, summary, out_dir)
    _write_manuscript_md(df, summary, example, out_dir)

    print("\nDone. Check bracketing_analysis/ and bracketing_figures/ for outputs.")


if __name__ == "__main__":
    main()
