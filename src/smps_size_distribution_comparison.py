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
