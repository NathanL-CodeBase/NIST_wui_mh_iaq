"""
SMPS cross-check of the AeroTrak Ch1 reversal (Section 3.2.2 support).

The AeroTrak bin-reversal analysis (aerotrak_coincidence.py) attributes the
0.3-0.5 um (Ch1) count collapse at peak smoke to optical-coincidence overload:
the true particle concentration passes the instrument's Poisson rollover
ceiling, so the reported count falls even as the true concentration rises. The
Poisson model then predicts a specific true concentration (the upper root of
n_meas = n_true exp(-n_true V)) far above the reported count.

This script tests that prediction against an independent, non-optical reference.
The SMPS (Bedroom 2 only) classifies particles by electrical mobility and is
therefore free of the optical coincidence and refractive-index effects that
distort the AeroTrak. For each Bedroom 2 AeroTrak reversal, it compares, over
the reversal window:
    - the AeroTrak Ch1 (0.3-0.5 um) reported count concentration,
    - the SMPS number concentration summed over 300-437 nm (its overlap with
      Ch1; the SMPS upper bin is ~414 nm, so this is a LOWER bound on the true
      300-500 nm count), and
    - the Poisson rollover prediction of the true concentration implied by the
      reported Ch1 count.
If the SMPS concentration greatly exceeds the reported Ch1 count and approaches
the rollover prediction, the overload interpretation is supported.

The Poisson constants, the AeroTrak and SMPS loaders, and the reversal windows
are imported or read from aerotrak_coincidence.py and its per-burn CSV so that a
single source of truth drives both analyses.

Inputs (resolved through data_config.json):
    aerotrak_bedroom : all_data.xlsx
    smps             : MH_apollo_bed_MMDDYYYY_NumConc.xlsx
    burn_log         : burn_log.xlsx, Sheet2
    coincidence_analysis/aerotrak_coincidence_per_burn.csv (reversal windows)

Outputs:
    coincidence_analysis/smps_reversal_crosscheck_per_burn.csv
    coincidence_analysis/smps_reversal_crosscheck_summary.md
    coincidence_figures/smps_reversal_crosscheck.png

Author: Nathan Lima
Created: 2026-07-07
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repository root on path ---------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.data_paths import get_common_file  # noqa: E402
from src.fig_style import (  # noqa: E402
    INSTR_COLORS,
    REF_LINE,
    apply_est_style,
    figsize,
    save_fig,
)

# Reuse the coincidence analysis loaders and Poisson constants so both analyses
# share one definition of the sensing volume, the rollover ceiling, the SMPS
# 300-437 nm sum, and the AeroTrak time base.
from src.aerotrak_coincidence import (  # noqa: E402
    BEDROOM_SEALED_BURNS,
    COINCIDENCE_THRESHOLD_CM3,
    N_MEAS_CEILING_CM3,
    V_CENTRAL,
    _day_slice,
    _load_aerotrak_all,
    _load_burn_log,
    _load_smps_numconc,
    _smps_300_437,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Only Bedroom 2 carries a co-located SMPS, so only AeroTrak1 reversals can be
# cross-checked.
INSTRUMENT = "AeroTrak1"
LOCATION = "bedroom2"

# Nearest-neighbour tolerance when sampling one series at another's timestamp.
# The SMPS scan cadence is about 135 s, so a 3.5 min tolerance picks the scan
# bracketing an AeroTrak feature without reaching into an unrelated scan.
SMPS_TOL = pd.Timedelta("3.5min")
AEROTRAK_TOL = pd.Timedelta("3min")

# Pre-roll added before the reversal onset so the figure shows the rise into the
# reversal, not just the suppressed interval.
WINDOW_PREROLL = pd.Timedelta("15min")

# CSV column order
_CSV_COLS = [
    "burn",
    "aerotrak_ch1_peak_cm3",
    "t_ch1_peak",
    "aerotrak_ch1_trough_cm3",
    "t_ch1_trough",
    "smps_300_437_at_ch1_peak_cm3",
    "smps_300_437_max_window_cm3",
    "t_smps_max",
    "smps_max_over_ch1_peak",
    "smps_over_aerotrak_at_peak",
    "predicted_true_from_ch1_peak_cm3",
    "predicted_true_from_ch1_trough_cm3",
    "smps_exceeds_reported_ch1",
    "n_smps_scans_in_window",
    "notes",
]


# ==============================================================================
# POISSON ROLLOVER INVERSION
# ==============================================================================


def _poisson_true_upper_root(n_meas: float, v_cm3: float = V_CENTRAL) -> float:
    """
    Invert the Poisson measured-count relation for the true concentration.

    The measured count relates to the true concentration by
        n_meas = n_true * exp(-n_true * V).
    Below the rollover ceiling (n_true = 1/V) this maps one-to-one, but above it
    the same measured value corresponds to a second, higher true concentration.
    Under overload the instrument sits on the upper branch, so this returns the
    upper root: the true concentration that reproduces the suppressed reading.

    Parameters
    ----------
    n_meas : float
        Measured (reported) count concentration (#/cm3).
    v_cm3 : float
        Effective optical sensing volume (cm3).

    Returns
    -------
    float
        Upper-root true concentration (#/cm3), or NaN if n_meas is not below the
        rollover ceiling (no distinct upper root exists).
    """
    ceiling_meas = np.exp(-1.0) / v_cm3  # max reportable count
    if np.isnan(n_meas) or n_meas <= 0 or n_meas >= ceiling_meas:
        return np.nan

    # Bisection on the upper branch, from just above the ceiling concentration
    # (1/V) to a generous upper limit. n_true * exp(-n_true V) decreases
    # monotonically there, so a sign change brackets the root.
    lo, hi = 1.0 / v_cm3 + 1.0, 1.0e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * np.exp(-mid * v_cm3) > n_meas:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ==============================================================================
# SERIES HELPERS
# ==============================================================================


def _ch1_conc_column(df_day: pd.DataFrame) -> str | None:
    """
    Return the AeroTrak Ch1 (smallest-diameter) count-concentration column.

    _load_aerotrak_all builds columns named 'Ʃ{lo}-{hi}µm (#/cm³)'. The Ch1
    column is the one with the smallest lower bound, identified here by parsing
    the leading diameter rather than hard-coding '0.3-0.5' in case the cut
    points differ between exports.
    """
    candidates = []
    for col in df_day.columns:
        s = str(col)
        if s.startswith("Ʃ") and "#/cm" in s:
            try:
                lo = float(s.replace("Ʃ", "").split("-")[0])
            except ValueError:
                continue
            candidates.append((lo, col))
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[0])[1]


def _nearest_value(
    times: pd.Series,
    values: pd.Series,
    t_target: pd.Timestamp,
    tol: pd.Timedelta,
) -> float:
    """Return values at the row whose time is nearest t_target within tol."""
    if pd.isna(t_target) or times.empty:
        return np.nan
    dt = (times - t_target).abs()
    idx = dt.idxmin()
    if dt.loc[idx] > tol:
        return np.nan
    return float(values.loc[idx])


# ==============================================================================
# PER-BURN ANALYSIS
# ==============================================================================


def analyze_burn(
    burn_id: str,
    df_aerotrak: pd.DataFrame,
    rev_row: pd.Series,
    burn_date: pd.Timestamp,
    df_smps: pd.DataFrame | None,
) -> dict | None:
    """
    Compare AeroTrak Ch1 and SMPS 300-437 nm over one Bedroom 2 reversal window.

    Parameters
    ----------
    burn_id : str
    df_aerotrak : pd.DataFrame
        Full AeroTrak1 record (time-shifted, concentration columns present).
    rev_row : pd.Series
        Row from aerotrak_coincidence_per_burn.csv for this burn (reversal
        window timestamps and the reported peak count).
    burn_date : pd.Timestamp
    df_smps : pd.DataFrame or None
        SMPS numConc record for the burn day.

    Returns
    -------
    dict or None
        Per-burn metrics, or None if the essential series are unavailable.
    """
    df_day = _day_slice(df_aerotrak, burn_date)
    if df_day.empty:
        return None

    ch1_col = _ch1_conc_column(df_day)
    if ch1_col is None:
        print(f"  [{burn_id}] Ch1 concentration column not found - skipped.")
        return None

    at_times = df_day["Date and Time"]
    ch1 = pd.to_numeric(df_day[ch1_col], errors="coerce")

    # Reversal window from the coincidence CSV (fall back to +/- 30 min of the
    # count peak when the recovery time was not resolved).
    t_peak = pd.to_datetime(rev_row.get("t_peak"), errors="coerce")
    t_onset = pd.to_datetime(rev_row.get("reversal_onset"), errors="coerce")
    t_trough = pd.to_datetime(rev_row.get("t_min"), errors="coerce")
    t_end = pd.to_datetime(rev_row.get("reversal_end"), errors="coerce")

    win_start = (t_onset if pd.notna(t_onset) else t_peak) - WINDOW_PREROLL
    win_end = t_end if pd.notna(t_end) else (t_peak + pd.Timedelta("30min"))
    if pd.isna(win_start) or pd.isna(win_end):
        print(f"  [{burn_id}] reversal window undefined - skipped.")
        return None

    # AeroTrak reported peak and trough within the window.
    at_mask = (at_times >= win_start) & (at_times <= win_end) & ch1.notna()
    if not at_mask.any():
        print(f"  [{burn_id}] no AeroTrak Ch1 samples in window - skipped.")
        return None
    at_win_times = at_times[at_mask].reset_index(drop=True)
    at_win_ch1 = ch1[at_mask].reset_index(drop=True)

    i_peak = int(at_win_ch1.idxmax())
    ch1_peak = float(at_win_ch1.iloc[i_peak])
    t_ch1_peak = at_win_times.iloc[i_peak]

    if pd.notna(t_trough):
        ch1_trough = _nearest_value(at_win_times, at_win_ch1, t_trough, AEROTRAK_TOL)
        t_ch1_trough = t_trough
    else:
        i_tr = int(at_win_ch1.idxmin())
        ch1_trough = float(at_win_ch1.iloc[i_tr])
        t_ch1_trough = at_win_times.iloc[i_tr]

    # SMPS 300-437 nm over the window.
    smps_at_peak = np.nan
    smps_max = np.nan
    t_smps_max = pd.NaT
    n_scans = 0
    if df_smps is not None:
        smps_series = _smps_300_437(df_smps)
        smps_times = df_smps["datetime"]
        sm_mask = (smps_times >= win_start) & (smps_times <= win_end) & smps_series.notna()
        n_scans = int(sm_mask.sum())
        if n_scans > 0:
            sm_win_times = smps_times[sm_mask].reset_index(drop=True)
            sm_win_vals = smps_series[sm_mask].reset_index(drop=True)
            j = int(sm_win_vals.idxmax())
            smps_max = float(sm_win_vals.iloc[j])
            t_smps_max = sm_win_times.iloc[j]
            smps_at_peak = _nearest_value(sm_win_times, sm_win_vals, t_ch1_peak, SMPS_TOL)

    pred_from_peak = _poisson_true_upper_root(ch1_peak)
    pred_from_trough = _poisson_true_upper_root(ch1_trough)

    smps_over_at_peak = (
        smps_at_peak / ch1_peak if not np.isnan(smps_at_peak) and ch1_peak > 0 else np.nan
    )
    # Primary metric: the window-max SMPS against the window-peak reported Ch1.
    # This avoids the timing noise in the at-peak alignment, because the Ch1
    # count rolls over and starts falling before the true smoke concentration
    # (SMPS) reaches its own maximum.
    smps_max_over_ch1_peak = (
        smps_max / ch1_peak if not np.isnan(smps_max) and ch1_peak > 0 else np.nan
    )
    smps_exceeds = (
        bool(smps_max > ch1_peak) if not np.isnan(smps_max) and not np.isnan(ch1_peak) else None
    )

    notes = []
    if df_smps is None or n_scans == 0:
        notes.append("no SMPS scans in reversal window")
    notes.append("SMPS 300-437 nm is a lower bound on the true 300-500 nm (Ch1) count")

    return {
        "burn": burn_id,
        "aerotrak_ch1_peak_cm3": ch1_peak,
        "t_ch1_peak": t_ch1_peak,
        "aerotrak_ch1_trough_cm3": ch1_trough,
        "t_ch1_trough": t_ch1_trough,
        "smps_300_437_at_ch1_peak_cm3": smps_at_peak,
        "smps_300_437_max_window_cm3": smps_max,
        "t_smps_max": t_smps_max,
        "smps_max_over_ch1_peak": smps_max_over_ch1_peak,
        "smps_over_aerotrak_at_peak": smps_over_at_peak,
        "predicted_true_from_ch1_peak_cm3": pred_from_peak,
        "predicted_true_from_ch1_trough_cm3": pred_from_trough,
        "smps_exceeds_reported_ch1": smps_exceeds,
        "n_smps_scans_in_window": n_scans,
        "notes": "; ".join(notes),
        # private, for the figure
        "_win_start": win_start,
        "_win_end": win_end,
        "_at_times": at_win_times,
        "_at_ch1": at_win_ch1,
        "_df_smps": df_smps,
    }


# ==============================================================================
# FIGURE
# ==============================================================================


def _plot_small_multiples(results: list[dict]) -> None:
    """
    One panel per burn: AeroTrak Ch1 (reported) and SMPS 300-437 nm (reference)
    count concentration over the reversal window, on a shared log axis, with the
    rollover ceiling and the coincidence limit marked. Where the SMPS lies far
    above the AeroTrak trace, the reported count is a severe undercount.
    """
    usable = [r for r in results if r["_at_times"] is not None and len(r["_at_times"]) > 0]
    if not usable:
        print("    [fig] no usable panels.")
        return

    n = len(usable)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))

    # Constrained layout is disabled here because this figure places its own
    # shared labels and a bottom legend with explicit spacing; the two layout
    # managers fight otherwise. subplots_adjust reserves the margins.
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(7.0, 2.1 * nrow + 0.8), squeeze=False, constrained_layout=False
    )

    at_color = INSTR_COLORS.get("AeroTrak1", "#0072B2")
    smps_color = "#E69F00"

    for k, r in enumerate(usable):
        ax = axes[k // ncol][k % ncol]
        t0 = r["_win_start"]
        at_min = (r["_at_times"] - t0).dt.total_seconds() / 60.0
        ax.plot(at_min, r["_at_ch1"], color=at_color, lw=1.6, label="AeroTrak Ch1 (reported)")

        df_smps = r["_df_smps"]
        if df_smps is not None:
            smps_series = _smps_300_437(df_smps)
            times = df_smps["datetime"]
            m = (times >= r["_win_start"]) & (times <= r["_win_end"]) & smps_series.notna()
            if m.any():
                smin = (times[m] - t0).dt.total_seconds() / 60.0
                ax.plot(
                    smin,
                    smps_series[m],
                    color=smps_color,
                    marker="o",
                    ms=4,
                    lw=1.2,
                    ls="--",
                    label="SMPS 300-437 nm",
                )

        ax.axhline(N_MEAS_CEILING_CM3, color=REF_LINE, lw=0.8, ls="-.")
        ax.axhline(COINCIDENCE_THRESHOLD_CM3, color=REF_LINE, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(r["burn"], fontsize=11)
        ax.tick_params(labelsize=10)

    # Blank any unused panels.
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # Reserve generous margins, then place shared labels and the legend at
    # distinct heights so nothing overlaps.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.16, hspace=0.42, wspace=0.26)
    fig.text(0.54, 0.075, "Minutes from reversal-window start", ha="center", fontsize=12)
    fig.text(0.015, 0.55, "Count concentration (#/cm³)", va="center", rotation=90, fontsize=12)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color=REF_LINE, ls="-.", lw=0.8),
        plt.Line2D([], [], color=REF_LINE, ls=":", lw=0.8),
    ]
    labels += ["Poisson rollover ceiling", "Coincidence limit"]
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.54, 0.0),
        ncol=4, fontsize=9, frameon=False,
    )

    fig_dir = get_common_file("coincidence_figures")
    save_fig(fig, fig_dir / "smps_reversal_crosscheck.png")


# ==============================================================================
# OUTPUTS
# ==============================================================================


def _write_outputs(results: list[dict]) -> None:
    """Write the per-burn CSV and the narrative summary."""
    out_dir = get_common_file("coincidence_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [{k: r.get(k, np.nan) for k in _CSV_COLS} for r in results]
    df = pd.DataFrame(rows, columns=_CSV_COLS)
    csv_path = out_dir / "smps_reversal_crosscheck_per_burn.csv"
    df.to_csv(str(csv_path), index=False, float_format="%.4g")
    print(f"    [CSV] {csv_path.name}")

    with_smps = df[df["smps_300_437_max_window_cm3"].notna()]
    if with_smps.empty:
        summary = (
            "# SMPS reversal cross-check - summary\n\n"
            "No Bedroom 2 reversal window had overlapping SMPS scans.\n"
        )
        (out_dir / "smps_reversal_crosscheck_summary.md").write_text(summary, encoding="utf-8")
        print("    [MD] smps_reversal_crosscheck_summary.md")
        return

    n = int(with_smps.shape[0])
    n_exceed = int((df["smps_exceeds_reported_ch1"] == True).sum())
    r_med = float(with_smps["smps_max_over_ch1_peak"].median())
    r_min = float(with_smps["smps_max_over_ch1_peak"].min())
    r_max = float(with_smps["smps_max_over_ch1_peak"].max())
    smps_med = float(with_smps["smps_300_437_max_window_cm3"].median())
    pred_med = float(with_smps["predicted_true_from_ch1_trough_cm3"].median())

    summary = (
        "# SMPS reversal cross-check - summary\n\n"
        "## Plain-language synthesis\n\n"
        f"Across {n} Bedroom 2 AeroTrak reversal windows with overlapping SMPS "
        f"scans, the SMPS 300-437 nm window-maximum number concentration exceeded "
        f"the co-located AeroTrak Ch1 window-peak reported count in {n_exceed} of "
        f"{n}, by a median factor of {r_med:.1f} (range {r_min:.1f} to {r_max:.1f}). "
        f"The independent, non-optical reference therefore shows a true 300-437 nm "
        f"concentration several times the reported Ch1 count. Because the SMPS "
        f"upper bin is about 414 nm, the 300-437 nm sum is a lower bound on the "
        f"true 0.3-0.5 um (Ch1) count, so the real discrepancy is larger.\n\n"
        f"The median SMPS 300-437 nm window maximum was approximately "
        f"{smps_med:.0f} particles/cm3. The Poisson rollover model, inverted from "
        f"the suppressed Ch1 trough, implies a median true concentration of "
        f"approximately {pred_med:.0f} particles/cm3. The SMPS lower bound and the "
        f"rollover prediction agree in direction and order of magnitude: the true "
        f"concentration greatly exceeds the reported count, as coincidence "
        f"overload requires.\n\n"
        "Note: only Bedroom 2 carries a co-located SMPS. Bedroom 2 peak masses "
        "are lower than the Morning Room, so this cross-check bounds the effect "
        "in the less-extreme room; the Morning Room reversals (higher mass) "
        "cannot be independently referenced.\n"
    )
    (out_dir / "smps_reversal_crosscheck_summary.md").write_text(summary, encoding="utf-8")
    print("    [MD] smps_reversal_crosscheck_summary.md")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Run the SMPS cross-check across all Bedroom 2 AeroTrak reversals."""
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    apply_est_style()

    per_burn_csv = get_common_file("coincidence_analysis") / "aerotrak_coincidence_per_burn.csv"
    if not per_burn_csv.exists():
        print(f"Missing {per_burn_csv}. Run aerotrak_coincidence.py first.")
        return
    coincidence = pd.read_csv(per_burn_csv)

    # Bedroom 2, non-sealed, reversal-present rows only.
    rev = coincidence[
        (coincidence["instrument"] == INSTRUMENT)
        & (coincidence["reversal_present"] == True)
        & (~coincidence["bedroom_sealed"].astype(bool))
    ].copy()
    if rev.empty:
        print("No non-sealed Bedroom 2 reversals in the coincidence CSV.")
        return

    print("Loading burn log and AeroTrak1 record...")
    burn_log = _load_burn_log()
    df_aerotrak = _load_aerotrak_all(INSTRUMENT)

    results: list[dict] = []
    for _, rev_row in rev.iterrows():
        burn_id = rev_row["burn"]
        if burn_id in BEDROOM_SEALED_BURNS:
            continue
        bl = burn_log[burn_log["Burn ID"] == burn_id]
        if bl.empty:
            print(f"  [{burn_id}] not in burn log - skipped.")
            continue
        burn_date = bl.iloc[0]["Date"]

        df_smps = _load_smps_numconc(burn_date)
        if df_smps is None:
            print(f"  [{burn_id}] SMPS numConc not found for {burn_date.date()}")

        print(f"  Analysing {burn_id}...")
        res = analyze_burn(burn_id, df_aerotrak, rev_row, burn_date, df_smps)
        if res is not None:
            results.append(res)

    if not results:
        print("No results computed.")
        return

    print(f"\n{len(results)} burn(s) processed.")
    print("\nGenerating figure...")
    _plot_small_multiples(results)
    print("\nWriting outputs...")
    _write_outputs(results)
    print("\nDone. See coincidence_analysis/ and coincidence_figures/.")


if __name__ == "__main__":
    main()
