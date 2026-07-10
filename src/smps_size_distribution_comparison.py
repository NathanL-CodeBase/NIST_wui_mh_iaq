"""
SMPS particle size distribution comparison: WUI smoke vs. KCl challenge aerosol.

Purpose: Generate SI figure for ea-2026-00137g comparing normalized dN/dlogDp
         distributions of WUI mixed-fuel smoke (Burn 01, 2024-04-26) and the KCl
         aerosol used for ASTM CADR derivation in Link et al. (2024). Supports the
         claim that KCl-derived CADRs predict WUI smoke removal performance.
Author:  Nathan Lima
Created: 2026-06-04
Updates:
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Make src/ importable so the pre/post-PAC helpers can reach data_paths and
# fig_style regardless of the caller's working directory.
sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

config_path = Path(__file__).parent.parent / "data_config.json"
if not config_path.exists():
    sys.exit("data_config.json not found. Copy data_config.template.json and fill in paths.")

with open(config_path) as f:
    cfg = json.load(f)

smps_dir = Path(cfg["instruments"]["smps"]["path"])
kcl_path = Path(cfg["common_folders"]["kcl_cadr_smps"])
output_dir = Path(cfg["common_folders"]["output_figures"])
output_dir.mkdir(parents=True, exist_ok=True)

xlsx_path = smps_dir / "MH_apollo_bed_04262024_numConc.xlsx"

for p in (xlsx_path, kcl_path):
    if not p.exists():
        sys.exit(f"File not found: {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Publication text config
# ──────────────────────────────────────────────────────────────────────────────

TEXT_CONFIG = {
    "font_size": 12,
    "title_font_size": 12,
    "axis_label_font_size": 12,
    "axis_tick_font_size": 12,
    "legend_font_size": 12,
    "plot_font_style": "bold",  # axis labels
    "font_style": "normal",  # ticks, legend
}


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────


def parse_tsi_txt(filepath):
    """Parse a TSI SMPS text export (tab-delimited, Concentration DW / Number).

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    diameters : ndarray, shape (n_bins,)
        Diameter midpoints, nm.
    data : DataFrame, shape (n_bins, n_scans)
        dN/dlogDp per scan.  Columns are 1-based integer scan numbers.
    total_conc : ndarray, shape (n_scans,)
        Total number concentration (#/cm³) per scan from the footer row.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # Scan-number labels from "Sample #" row
    sample_line = next(ln for ln in lines if ln.startswith("Sample #"))
    scan_nums = [int(s) for s in sample_line.strip().split("\t")[1:] if s.strip()]
    n_scans = len(scan_nums)

    # Locate the row AFTER "Diameter Midpoint" -- that is where data begins
    diam_header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Diameter Midpoint"))

    # Read size-bin rows until the first column is no longer a float
    diameters, rows = [], []
    for ln in lines[diam_header_idx + 1 :]:
        parts = ln.rstrip("\n").split("\t")
        try:
            d = float(parts[0])
        except (ValueError, IndexError):
            break
        vals = []
        for v in parts[1 : n_scans + 1]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(0.0)
        while len(vals) < n_scans:
            vals.append(0.0)
        diameters.append(d)
        rows.append(vals)

    data = pd.DataFrame(rows, index=diameters, columns=scan_nums, dtype=float)

    # Total concentration from footer (used only for peak-scan selection)
    total_conc_line = next((ln for ln in lines if ln.startswith("Total Concentration")), None)
    if total_conc_line:
        parts = total_conc_line.strip().split("\t")[1:]
        total_conc = np.array([float(v) if v.strip() else 0.0 for v in parts[:n_scans]])
    else:
        total_conc = data.sum(axis=0).values

    return np.array(diameters, dtype=float), data, total_conc


def parse_wui_xlsx(filepath):
    """Parse WUI smoke SMPS xlsx export (all_data sheet).

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    diameters : ndarray, shape (n_bins,)
        Diameter midpoints, nm.
    data : DataFrame, shape (n_scans, n_bins)
        dN/dlogDp per scan.  Columns are float diameter midpoints.
    total_conc : ndarray, shape (n_scans,)
        Total number concentration (#/cm³) per scan.
    """
    df = pd.read_excel(filepath, sheet_name="all_data", header=0)
    # Strip whitespace from string column names; leave numeric columns alone
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Total-concentration column (contains "Total" and "Conc")
    total_col = next(c for c in df.columns if isinstance(c, str) and "Total" in c and "Conc" in c)
    total_conc = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0).values

    # Size-bin columns: column name is a positive float (the diameter midpoint)
    bin_cols = [c for c in df.columns if not isinstance(c, str) and float(c) > 0]
    diameters = np.array([float(c) for c in bin_cols])
    data = df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return diameters, data, total_conc


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────


def peak_normalize(dist):
    """Normalize to unit peak.

    To switch to unit-area normalization replace with:
        area = np.trapz(dist, np.log(diameters))
        return dist / area
    and pass diameters as an argument.
    """
    m = np.nanmax(dist)
    return dist / m if m > 0 else dist


def compute_gmd_gsd(diameters, dndlogdp):
    """Compute geometric mean diameter and geometric standard deviation.

    Bins are log-uniform (constant dlogDp), so dN weights are proportional to
    dN/dlogDp values; dlogDp cancels in the moment integrals.

    Parameters
    ----------
    diameters : array-like
        Midpoint diameters, nm.
    dndlogdp : array-like
        dN/dlogDp values (#/cm³).

    Returns
    -------
    gmd : float  (nm)
    gsd : float  (dimensionless, >= 1)
    """
    d = np.asarray(diameters, dtype=float)
    n = np.maximum(np.asarray(dndlogdp, dtype=float), 0.0)
    total = n.sum()
    if total == 0.0:
        return np.nan, np.nan
    f = n / total
    ln_gmd = np.dot(f, np.log(d))
    gmd = np.exp(ln_gmd)
    gsd = np.exp(np.sqrt(np.dot(f, (np.log(d) - ln_gmd) ** 2)))
    return float(gmd), float(gsd)


# ──────────────────────────────────────────────────────────────────────────────
# Pre/post-PAC averaged spectra (Section 3.4)
# ──────────────────────────────────────────────────────────────────────────────

# Burn date -> SMPS file suffix (Bedroom 2 is the only SMPS location).
_BURN_SUFFIX = {
    "2024-04-26": "04262024",
    "2024-05-02": "05022024",
    "2024-05-06": "05062024",
    "2024-05-09": "05092024",
    "2024-05-13": "05132024",
    "2024-05-17": "05172024",
    "2024-05-20": "05202024",
    "2024-05-23": "05232024",
    "2024-05-28": "05282024",
    "2024-05-31": "05312024",
}

# Coarse band edges (nm) shared with the decay analysis and Section 3.2.2.
_BAND_EDGES_NM = [9, 100, 200, 300, 437]


def parse_wui_xlsx_timed(filepath):
    """Parse a WUI SMPS xlsx export, returning per-scan timestamps as well.

    Same source as :func:`parse_wui_xlsx` but keeps the scan datetime so callers
    can average dN/dlogDp over explicit time windows (e.g. pre/post air-cleaner).

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    diameters : ndarray, shape (n_bins,)
        Diameter midpoints, nm.
    data : DataFrame, shape (n_scans, n_bins)
        dN/dlogDp per scan; columns are float diameter midpoints.
    total_conc : ndarray, shape (n_scans,)
        Total number concentration (#/cm³) per scan.
    times : Series of Timestamp, shape (n_scans,)
        Scan start datetime (Date + Start Time).
    """
    df = pd.read_excel(filepath, sheet_name="all_data", header=0)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    total_col = next(
        c for c in df.columns if isinstance(c, str) and "Total" in c and "Conc" in c
    )
    total_conc = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0).values

    bin_cols = [c for c in df.columns if not isinstance(c, str) and float(c) > 0]
    diameters = np.array([float(c) for c in bin_cols])
    data = df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # The "Date" cell can carry a spurious 00:00:00 time component; take only
    # the date part before combining with the scan Start Time.
    date_part = df["Date"].astype(str).str.split(" ").str[0]
    times = pd.to_datetime(
        date_part + " " + df["Start Time"].astype(str), errors="coerce"
    )

    return diameters, data, total_conc, times


def _burn_event_times(burn_date, burn_log):
    """Return (garage_closed, pac_on) Timestamps for a burn from the burn log.

    Parameters
    ----------
    burn_date : str
        'YYYY-MM-DD'.
    burn_log : DataFrame
        Burn log sheet with 'Date', 'garage closed', 'CR Box on' columns.

    Returns
    -------
    tuple of (Timestamp, Timestamp or None)
        Garage-closed datetime and CR Box (PAC) on datetime. PAC is None if the
        burn had no air cleaner (e.g. the baseline burns).
    """
    d = pd.to_datetime(burn_date).date()
    row = burn_log[pd.to_datetime(burn_log["Date"]).dt.date == d]
    if row.empty:
        raise ValueError(f"No burn-log row for {burn_date}")
    row = row.iloc[0]

    garage = pd.Timestamp.combine(d, pd.to_datetime(str(row["garage closed"])).time())
    pac_raw = row["CR Box on"]
    pac = (
        None
        if pd.isna(pac_raw)
        else pd.Timestamp.combine(d, pd.to_datetime(str(pac_raw)).time())
    )
    return garage, pac


def _decay_fit_window(burn_date, burn_log, band="Total Concentration (µg/m³)"):
    """Return the post-PAC decay-fit window (start, end) as Timestamps.

    Reads the decay start/end offsets (hours since garage close) that the CADR
    script already fit, so the post-PAC spectrum uses the same interval as the
    reported decay rates. No refitting.

    Parameters
    ----------
    burn_date : str
        'YYYY-MM-DD'.
    burn_log : DataFrame
    band : str
        Which decay row to read the window from; the total-concentration row
        gives the common decay interval used across bands.

    Returns
    -------
    tuple of (Timestamp, Timestamp)
    """
    from data_paths import get_common_file

    xlsx = get_common_file("burn_calcs") / "SMPS_decay_and_CADR.xlsx"
    if not xlsx.exists():
        sys.exit(
            f"Not found: {xlsx}\n"
            "Run clean_air_delivery_rates_pmsizes.py with dataset='SMPS' first."
        )
    decay = pd.read_excel(xlsx)

    burn_num = _BURN_SUFFIX  # noqa: F841  (kept for clarity of mapping origin)
    # Burn id in the decay file is 'burnN'; derive N from the ordered suffix map.
    burn_id = f"burn{list(_BURN_SUFFIX).index(burn_date) + 1}"
    rows = decay[(decay["burn"] == burn_id) & (decay["pollutant"] == band)]
    if rows.empty:
        raise ValueError(f"No decay window for {burn_id} / {band} in {xlsx.name}")
    r = rows.iloc[0]

    garage, _ = _burn_event_times(burn_date, burn_log)
    start = garage + pd.Timedelta(hours=float(r["decay_start_time"]))
    end = garage + pd.Timedelta(hours=float(r["decay_end_time"]))
    return start, end


def _average_spectrum(data, times, t0, t1):
    """Average dN/dlogDp over scans whose start time is in [t0, t1].

    Parameters
    ----------
    data : DataFrame, shape (n_scans, n_bins)
    times : Series of Timestamp
    t0, t1 : Timestamp

    Returns
    -------
    mean_spectrum : ndarray, shape (n_bins,)
    n_scans : int
    """
    mask = (times >= t0) & (times <= t1)
    n = int(mask.sum())
    if n == 0:
        raise ValueError(f"No SMPS scans in window {t0} to {t1}")
    return data[mask.values].mean(axis=0).values.astype(float), n


def plot_pre_post_pac_spectra(burn_date="2024-05-31", data_type="numConc"):
    """Averaged pre- and post-PAC dN/dlogDp spectra for a burn (Bedroom 2).

    Pre-PAC window: garage close to PAC (CR Box) activation, characterizing the
    smoke that infiltrated the home. Post-PAC window: the decay-fit interval the
    CADR script used, showing the size-dependent removal by filtration.

    Two-panel figure: (1) absolute averaged dN/dlogDp (the concentration drop),
    (2) peak-normalized overlay (the shape change). GMD and GSD are annotated per
    period; the four coarse band edges (9, 100, 200, 300, 437 nm) are drawn as
    light vertical guides so the spectra connect to the decay bands.

    Parameters
    ----------
    burn_date : str
        'YYYY-MM-DD'. Default '2024-05-31' (Burn 10).
    data_type : str
        SMPS product suffix, 'numConc' (number) is the default.

    Returns
    -------
    dict
        Summary numbers for the Section 3.4 text (GMD/GSD pre and post, mode
        diameter, window scan counts, and the fractional removal per band).
    """
    import matplotlib as mpl

    from fig_style import OKABE_ITO, apply_est_style, figsize

    if burn_date not in _BURN_SUFFIX:
        raise ValueError(f"Unknown burn date {burn_date}; add it to _BURN_SUFFIX.")

    burn_log = pd.read_excel(cfg["common_folders"]["burn_log"], sheet_name="Sheet2")
    garage, pac = _burn_event_times(burn_date, burn_log)
    if pac is None:
        sys.exit(f"Burn {burn_date} has no CR Box; a pre/post-PAC split is undefined.")

    suffix = _BURN_SUFFIX[burn_date]
    fpath = smps_dir / f"MH_apollo_bed_{suffix}_{data_type}.xlsx"
    if not fpath.exists():
        sys.exit(f"File not found: {fpath}")

    diameters, data, total_conc, times = parse_wui_xlsx_timed(fpath)

    # Data-quality gate: the burn must have scans on both sides of PAC with no
    # long gap across the activation instant.
    day_mask = times.dt.date == pd.to_datetime(burn_date).date()
    data = data[day_mask.values].reset_index(drop=True)
    times = times[day_mask.values].reset_index(drop=True)
    total_conc = np.asarray(total_conc)[day_mask.values]
    order = np.argsort(times.values)
    data = data.iloc[order].reset_index(drop=True)
    times = times.iloc[order].reset_index(drop=True)
    total_conc = total_conc[order]

    peak_idx = int(np.argmax(total_conc))
    print(f"Burn {burn_date} SMPS Bedroom 2 ({data_type})")
    print(f"  Number peak     : {total_conc[peak_idx]:.0f} #/cm³ at {times[peak_idx]}")
    print(f"  Garage closed   : {garage}")
    print(f"  PAC (CR Box) on : {pac}")

    around = times[(times >= garage) & (times <= pac + pd.Timedelta(minutes=30))]
    max_gap_s = around.diff().dt.total_seconds().max()
    print(f"  Max scan gap across PAC window: {max_gap_s:.0f} s")
    if max_gap_s > 600:  # a >10 min hole would corrupt a window average
        sys.exit(
            f"Data-quality stop: {max_gap_s:.0f} s gap across the PAC window for "
            f"{burn_date}. Inspect the record before plotting."
        )

    pre_spec, n_pre = _average_spectrum(data, times, garage, pac)
    post_start, post_end = _decay_fit_window(burn_date, burn_log)
    post_spec, n_post = _average_spectrum(data, times, post_start, post_end)
    print(f"  Pre-PAC window  : {garage} to {pac}  ({n_pre} scans)")
    print(f"  Post-PAC window : {post_start} to {post_end}  ({n_post} scans)")

    gmd_pre, gsd_pre = compute_gmd_gsd(diameters, pre_spec)
    gmd_post, gsd_post = compute_gmd_gsd(diameters, post_spec)
    mode_pre = float(diameters[int(np.argmax(pre_spec))])
    mode_post = float(diameters[int(np.argmax(post_spec))])
    print(f"  Pre  GMD={gmd_pre:.1f} nm  GSD={gsd_pre:.2f}  mode={mode_pre:.1f} nm")
    print(f"  Post GMD={gmd_post:.1f} nm  GSD={gsd_post:.2f}  mode={mode_post:.1f} nm")

    # Fractional removal per coarse band (1 - post/pre of the band integral).
    removal = {}
    for lo, hi in zip(_BAND_EDGES_NM[:-1], _BAND_EDGES_NM[1:]):
        bmask = (diameters >= lo) & (diameters < hi)
        pre_sum = pre_spec[bmask].sum()
        post_sum = post_spec[bmask].sum()
        frac = 1.0 - post_sum / pre_sum if pre_sum > 0 else np.nan
        removal[f"{lo}-{hi} nm"] = frac
    print("  Fractional removal (1 - post/pre) by band:")
    for band, frac in removal.items():
        print(f"    {band:>10}: {frac * 100:.1f} %")
    best_band = max(removal, key=lambda k: removal[k])
    print(f"  Largest fractional removal: {best_band} ({removal[best_band] * 100:.1f} %)")

    # ── Figure ────────────────────────────────────────────────────────────────
    apply_est_style()
    c_pre = OKABE_ITO["vermillion"]
    c_post = OKABE_ITO["blue"]

    w, h = figsize("double", aspect=0.42)
    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(w, h))

    def _band_guides(ax):
        for edge in _BAND_EDGES_NM:
            ax.axvline(edge, color="0.75", linewidth=0.6, linestyle=":", zorder=0)

    # Panel 1: absolute averaged spectra.
    _band_guides(ax_abs)
    ax_abs.plot(diameters, pre_spec, color=c_pre, linewidth=1.8, label="Pre-PAC")
    ax_abs.plot(
        diameters, post_spec, color=c_post, linewidth=1.8, linestyle="--",
        label="Post-PAC",
    )
    ax_abs.set_xscale("log")
    ax_abs.set_xlim(9, 437)
    ax_abs.set_xlabel("Particle diameter (nm)")
    ax_abs.set_ylabel(r"d$N$/d$\log D_\mathrm{p}$ (#/cm³)")
    ax_abs.set_title("(a) Averaged spectra")
    ax_abs.legend(frameon=True, framealpha=0.9, edgecolor="0.7", loc="upper right")
    ax_abs.text(
        0.03, 0.97,
        f"Pre: GMD {gmd_pre:.0f} nm, GSD {gsd_pre:.2f}\n"
        f"Post: GMD {gmd_post:.0f} nm, GSD {gsd_post:.2f}",
        transform=ax_abs.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    # Panel 2: peak-normalized overlay (shape change).
    _band_guides(ax_norm)
    ax_norm.plot(
        diameters, peak_normalize(pre_spec), color=c_pre, linewidth=1.8, label="Pre-PAC"
    )
    ax_norm.plot(
        diameters, peak_normalize(post_spec), color=c_post, linewidth=1.8,
        linestyle="--", label="Post-PAC",
    )
    ax_norm.set_xscale("log")
    ax_norm.set_xlim(9, 437)
    ax_norm.set_ylim(0, 1.08)
    ax_norm.set_xlabel("Particle diameter (nm)")
    ax_norm.set_ylabel(r"Normalized d$N$/d$\log D_\mathrm{p}$ (–)")
    ax_norm.set_title("(b) Peak-normalized")
    ax_norm.legend(frameon=True, framealpha=0.9, edgecolor="0.7", loc="upper right")

    for ax in (ax_abs, ax_norm):
        ticks = [10, 20, 50, 100, 200, 400]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    burn_id = f"burn{list(_BURN_SUFFIX).index(burn_date) + 1}"
    out_pdf = output_dir / f"smps_pre_post_pac_spectra_{burn_id}.pdf"
    out_png = output_dir / f"smps_pre_post_pac_spectra_{burn_id}.png"
    for out in (out_pdf, out_png):
        if out.exists():
            import datetime as _dt

            out = out.with_name(f"{out.stem}_{_dt.date.today().isoformat()}{out.suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Figure saved: {out}")
    plt.close(fig)

    return {
        "burn_date": burn_date,
        "gmd_pre_nm": gmd_pre,
        "gsd_pre": gsd_pre,
        "mode_pre_nm": mode_pre,
        "gmd_post_nm": gmd_post,
        "gsd_post": gsd_post,
        "mode_post_nm": mode_post,
        "n_pre_scans": n_pre,
        "n_post_scans": n_post,
        "fractional_removal": removal,
        "largest_removal_band": best_band,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Load data and select peak scans
# ──────────────────────────────────────────────────────────────────────────────

d_wui, df_wui, tc_wui = parse_wui_xlsx(xlsx_path)
peak_idx_wui = int(np.argmax(tc_wui))
dist_wui_raw = df_wui.iloc[peak_idx_wui].values.astype(float)
gmd_wui, gsd_wui = compute_gmd_gsd(d_wui, dist_wui_raw)
dist_wui = peak_normalize(dist_wui_raw)

print("WUI smoke (Burn 01)")
print(f"  Peak scan row   : {peak_idx_wui} of {len(tc_wui)}")
print(f"  Total conc      : {tc_wui[peak_idx_wui]:.1f} #/cm³")
print(f"  GMD             : {gmd_wui:.1f} nm")
print(f"  GSD             : {gsd_wui:.3f}")

d_kcl, df_kcl, tc_kcl = parse_tsi_txt(kcl_path)
# Ensure we have a numpy array for thresholding
tc_kcl = np.asarray(tc_kcl)

# The file contains two KCl injection cycles (generator on → peak → natural decay).
# We want the second cycle.  Strategy: find all contiguous blocks of scans
# above a high-concentration threshold, then take the peak within the second block.
_KCL_EVENT_THRESHOLD = 1000.0  # #/cm³ -- well above background, well below event peak
# Find indices where concentration exceeds threshold
above = tc_kcl >= _KCL_EVENT_THRESHOLD
# Label contiguous blocks
block_ids = np.diff(above.astype(int), prepend=0)
block_starts = np.where(block_ids == 1)[0]
block_ends = np.where(block_ids == -1)[0] - 1
# If the signal ends while still above threshold, adjust last block
if above[-1]:
    block_ends = np.append(block_ends, len(tc_kcl) - 1)
# Ensure we have at least two blocks
if len(block_starts) < 2:
    sys.exit("Expected at least two KCl events; check data or threshold.")
# Use the second block (index 1)
second_start = block_starts[1]
second_end = block_ends[1]
# Find peak within second block
peak_idx_kcl = second_start + int(np.argmax(tc_kcl[second_start : second_end + 1]))
dist_kcl_raw = df_kcl.iloc[:, peak_idx_kcl].values.astype(float)
gmd_kcl, gsd_kcl = compute_gmd_gsd(d_kcl, dist_kcl_raw)
dist_kcl = peak_normalize(dist_kcl_raw)

print("\nKCl (Link et al. 2024) - second event")
print(f"  Second event    : scans {df_kcl.columns[second_start]}–{df_kcl.columns[second_end]}")
print(f"  Peak scan number: {df_kcl.columns[peak_idx_kcl]}")
print(f"  Total conc      : {tc_kcl[peak_idx_kcl]:.1f} #/cm³")
print(f"  GMD             : {gmd_kcl:.1f} nm")
print(f"  GSD             : {gsd_kcl:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

TC = TEXT_CONFIG

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(
    d_wui,
    dist_wui,
    color="#D55E00",
    linewidth=1.8,
    label="WUI smoke, Burn 01",
)
ax.plot(
    d_kcl,
    dist_kcl,
    color="#0072B2",
    linewidth=1.8,
    linestyle="--",
    label="KCl, Link et al. (2024)",
)

ax.set_xscale("log")
ax.set_xlim(4, 500)
ax.set_ylim(0, 1.08)

# Clean log x-axis ticks
major_ticks = [5, 10, 20, 50, 100, 200, 500]
ax.set_xticks(major_ticks)
ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(t) for t in major_ticks]))
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.set_xlabel(
    "Particle diameter (nm)",
    fontsize=TC["axis_label_font_size"],
    fontweight=TC["plot_font_style"],
)
ax.set_ylabel(
    r"Normalized d$N$/d$\log D_\mathrm{p}$ (–)",
    fontsize=TC["axis_label_font_size"],
    fontweight=TC["plot_font_style"],
)

ax.tick_params(axis="both", labelsize=TC["axis_tick_font_size"])
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight(TC["font_style"])

legend = ax.legend(
    fontsize=TC["legend_font_size"],
    loc='lower right',
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
)
for txt in legend.get_texts():
    txt.set_fontweight(TC["font_style"])

ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5, color="0.6")
ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3, color="0.6")

fig.tight_layout()

out_pdf = output_dir / "SI_smps_size_distribution_comparison.pdf"
out_png = output_dir / "SI_smps_size_distribution_comparison.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nFigure saved: {out_pdf}")
print(f"PNG preview:  {out_png}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3.4 pre/post-PAC spectra (Burn 10 by default)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 78)
    plot_pre_post_pac_spectra(burn_date="2024-05-31")
