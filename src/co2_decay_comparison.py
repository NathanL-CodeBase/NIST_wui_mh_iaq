"""
CO2 concentration comparison and exponential decay analysis.

Purpose: Plot CO2 traces from Bedroom, Entry, and MH Outside sensors for a
         defined time window, and fit exponential decay to the bedroom CO2
         over two specified intervals to quantify decay rates with uncertainty.
Author:  Nathan Lima
Created: 2026-06-10
"""

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

config_path = Path(__file__).parent.parent / "data_config.json"
if not config_path.exists():
    sys.exit("data_config.json not found. Copy data_config.template.json and fill in paths.")

with open(config_path) as f:
    cfg = json.load(f)

co2_dir = Path(cfg["instruments"]["co2"]["path"])
output_dir = Path(cfg["common_folders"]["output_figures"])
output_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Analysis window and decay period definitions
# ──────────────────────────────────────────────────────────────────────────────

PLOT_START = pd.Timestamp("2026-06-10 11:01:00")
PLOT_END = pd.Timestamp("2026-06-10 11:25:00")

# Decay periods for bedroom CO2 (inclusive bounds)
DECAY_PERIODS = [
    {
        "label": "Cleaner air space",
        "start": pd.Timestamp("2026-06-10 11:01:00"),
        "end": pd.Timestamp("2026-06-10 11:15:00"),
        "color": "#D55E00",
    },
    {
        "label": "Door open",
        "start": pd.Timestamp("2026-06-10 11:17:00"),
        "end": pd.Timestamp("2026-06-10 11:24:00"),
        "color": "#0072B2",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Publication text config
# ──────────────────────────────────────────────────────────────────────────────

TEXT_CONFIG = {
    "font_size": 12,
    "title_font_size": 12,
    "axis_label_font_size": 12,
    "axis_tick_font_size": 12,
    "legend_font_size": 12,
    "plot_font_style": "bold",
    "font_style": "normal",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

# Timestamp format used by Aranet4 export: DD/MM/YYYY h:mm:ss AM/PM
_DT_FORMAT = "%d/%m/%Y %I:%M:%S %p"


def load_co2_csv(path: Path) -> pd.Series:
    """Load an Aranet4 CO2 CSV and return a Series of CO2 (ppm) indexed by datetime.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.Series
        CO2 concentration in ppm, datetime index.
    """
    df = pd.read_csv(path, quotechar='"')
    df.columns = [c.strip() for c in df.columns]

    time_col = df.columns[0]
    co2_col = next(c for c in df.columns if "Carbon dioxide" in c or "CO2" in c.upper())

    dt = pd.to_datetime(df[time_col].str.strip('"'), format=_DT_FORMAT)
    co2 = pd.to_numeric(df[co2_col].astype(str).str.strip('"'), errors="coerce")

    s = pd.Series(co2.values, index=dt, name="co2_ppm", dtype=float)
    return s.sort_index()


def find_sensor_file(directory: Path, prefix: str) -> Path:
    """Return the first CSV whose name starts with prefix (case-insensitive)."""
    matches = sorted(directory.glob("*.csv"))
    for p in matches:
        if p.name.lower().startswith(prefix.lower()):
            return p
    sys.exit(f"No CSV file starting with '{prefix}' found in {directory}")


bedroom_path = find_sensor_file(co2_dir, "Bedroom")
entry_path = find_sensor_file(co2_dir, "Entry")
outside_path = find_sensor_file(co2_dir, "Mh Outside")

print(f"Bedroom : {bedroom_path.name}")
print(f"Entry   : {entry_path.name}")
print(f"Outside : {outside_path.name}")

s_bed = load_co2_csv(bedroom_path)
s_ent = load_co2_csv(entry_path)
s_out = load_co2_csv(outside_path)

# Crop to plot window (use a small buffer to ensure boundary points are included)
_buf = pd.Timedelta("30s")


def crop(s: pd.Series) -> pd.Series:
    return s[(s.index >= PLOT_START - _buf) & (s.index <= PLOT_END + _buf)]


s_bed_plot = crop(s_bed)
s_ent_plot = crop(s_ent)
s_out_plot = crop(s_out)

print(f"\nBedroom points in window : {len(s_bed_plot)}")
print(f"Entry points in window   : {len(s_ent_plot)}")
print(f"Outside points in window : {len(s_out_plot)}")

# ──────────────────────────────────────────────────────────────────────────────
# Exponential decay fitting (bedroom only)
# ──────────────────────────────────────────────────────────────────────────────


def exp_decay(t, a, k):
    """C(t) = a * exp(-k * t), t in minutes."""
    return a * np.exp(-k * t)


def fit_decay(series: pd.Series, t_start: pd.Timestamp, t_end: pd.Timestamp):
    """Fit C(t) = A exp(-k t) to the bedroom CO2 in [t_start, t_end].

    Parameters
    ----------
    series : pd.Series
        CO2 (ppm) with datetime index.
    t_start, t_end : pd.Timestamp
        Inclusive time bounds.

    Returns
    -------
    dict with keys: t_min (array), co2 (array), t_fine (array), fit (array),
                    k (float, min^-1), k_err (float, min^-1), a (float), a_err (float)
    """
    mask = (series.index >= t_start) & (series.index <= t_end)
    seg = series[mask].dropna()

    if len(seg) < 3:
        sys.exit(
            f"Only {len(seg)} data points in decay window {t_start}–{t_end}. "
            "Need at least 3 to fit 2 parameters."
        )

    # Time in minutes relative to start of period
    t_min = np.array([(ts - t_start).total_seconds() / 60.0 for ts in seg.index], dtype=float)
    co2 = seg.values.astype(float)

    a0 = co2[0]
    k0 = 0.05

    popt, pcov = curve_fit(exp_decay, t_min, co2, p0=[a0, k0], maxfev=10_000)
    perr = np.sqrt(np.diag(pcov))

    t_fine = np.linspace(t_min[0], t_min[-1], 200)
    fit_curve = exp_decay(t_fine, *popt)

    # Convert t_fine back to datetimes for plotting
    dt_fine = [t_start + pd.Timedelta(minutes=float(tv)) for tv in t_fine]

    return {
        "t_min": t_min,
        "co2": co2,
        "dt_fine": dt_fine,
        "fit": fit_curve,
        "k": popt[1] * 60.0,  # convert min^-1 to hr^-1
        "k_err": perr[1] * 60.0,
        "a": popt[0],
        "a_err": perr[0],
        "n_pts": len(seg),
    }


fit_results = []
for period in DECAY_PERIODS:
    result = fit_decay(s_bed, period["start"], period["end"])
    result["label"] = period["label"]
    result["color"] = period["color"]
    fit_results.append(result)
    print(
        f"\n{period['label']} ({period['start'].strftime('%H:%M')}–{period['end'].strftime('%H:%M')})"
        f"  n={result['n_pts']}"
    )
    print(f"  k = {result['k']:.4g} +/- {result['k_err']:.4g} hr^-1")
    print(f"  A = {result['a']:.1f} ± {result['a_err']:.1f} ppm")

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

TC = TEXT_CONFIG

SENSOR_COLORS = {
    "Bedroom": "#333333",
    "Entry": "#E69F00",
    "Outside": "#009E73",
}

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Shade decay windows
shade_alpha = 0.10
for period, res in zip(DECAY_PERIODS, fit_results):
    ax.axvspan(
        period["start"],
        period["end"],
        color=res["color"],
        alpha=shade_alpha,
        zorder=0,
        label=f"_shade_{period['label']}",
    )

# Plot sensor traces as markers
ax.plot(
    s_out_plot.index,
    s_out_plot.values,
    color=SENSOR_COLORS["Outside"],
    linestyle="none",
    marker="o",
    markersize=5,
    label="Outside",
    zorder=2,
)
ax.plot(
    s_ent_plot.index,
    s_ent_plot.values,
    color=SENSOR_COLORS["Entry"],
    linestyle="none",
    marker="s",
    markersize=5,
    label="Entry",
    zorder=3,
)
ax.plot(
    s_bed_plot.index,
    s_bed_plot.values,
    color=SENSOR_COLORS["Bedroom"],
    linestyle="none",
    marker="^",
    markersize=5,
    label="Bedroom",
    zorder=4,
)

# Overlay fitted decay curves as solid lines
for res in fit_results:
    ax.plot(
        res["dt_fine"],
        res["fit"],
        color=res["color"],
        linewidth=1.8,
        linestyle="-",
        zorder=5,
        label=f"_fit_{res['label']}",
    )

# Annotate decay rates inside their shaded bands
annotation_offsets = [0.82, 0.48]  # y-position as fraction of axes height
for res, y_frac in zip(fit_results, annotation_offsets):
    x_mid = res["dt_fine"][len(res["dt_fine"]) // 2]
    k_val = res["k"]
    k_err = res["k_err"]
    ax.annotate(
        f"{res['label']}\n$k$ = {k_val:.2g} ± {k_err:.2g} hr$^{{-1}}$",
        xy=(x_mid, 0),
        xycoords=("data", "axes fraction"),
        xytext=(x_mid, y_frac),
        textcoords=("data", "axes fraction"),
        ha="center",
        fontsize=10,
        color=res["color"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=res["color"], alpha=0.85, lw=0.8),
    )

# Axis formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 1)))

ax.set_xlim(PLOT_START - pd.Timedelta("30s"), PLOT_END + pd.Timedelta("30s"))
ax.set_ylim(bottom=0)

ax.set_xlabel(
    "Time (HH:MM)",
    fontsize=TC["axis_label_font_size"],
    fontweight=TC["plot_font_style"],
)
ax.set_ylabel(
    "CO$_2$ concentration (ppm)",
    fontsize=TC["axis_label_font_size"],
    fontweight=TC["plot_font_style"],
)

ax.tick_params(axis="both", labelsize=TC["axis_tick_font_size"])
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight(TC["font_style"])

legend = ax.legend(
    fontsize=TC["legend_font_size"],
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.7",
)
for txt in legend.get_texts():
    txt.set_fontweight(TC["font_style"])

ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5, color="0.6")
ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3, color="0.6")

fig.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

out_pdf = output_dir / "co2_decay_comparison.pdf"
out_png = output_dir / "co2_decay_comparison.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nFigure saved: {out_pdf}")
print(f"PNG preview:  {out_png}")
