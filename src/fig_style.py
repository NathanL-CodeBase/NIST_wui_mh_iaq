"""
Shared ES&T Air figure styling for the Section 3.2 instrument figures.

Single source of truth for matplotlib styling, column widths, and the
colorblind-safe (Okabe-Ito) categorical palette used across the bracketing,
MODULAIR-PM peak-window, MODULAIR-PM post-peak, and AeroTrak coincidence
figures. Import this module and call apply_est_style() once per script so the
whole Section 3.2 figure set is visually consistent at ES&T Air column widths.

Why Okabe-Ito: the previous per-script palette (dark blue #003f5c, pink
#ef5675, orange #ffa600) was not colorblind-safe. The roles below keep the same
semantic mapping (Bedroom 2 / MODULAIR-PM1 = blue, Morning Room / MODULAIR-PM2 =
vermillion, SMPS = orange) but draw the actual colors from the Okabe-Ito set so
the figures remain legible to readers with color-vision deficiency.

Author: Nathan Lima
Created: 2026-06-29
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ==============================================================================
# COLUMN WIDTHS (ES&T Air)
# ==============================================================================

COL_SINGLE_IN = 3.33   # single-column figure width (inches)
COL_ONEHALF_IN = 5.0   # ~1.5-column figure width (inches)
COL_DOUBLE_IN = 7.0    # full (double) column figure width (inches)

_WIDTHS = {
    "single": COL_SINGLE_IN,
    "onehalf": COL_ONEHALF_IN,
    "double": COL_DOUBLE_IN,
}


def figsize(width: str = "single", aspect: float = 0.62) -> tuple[float, float]:
    """
    Figure size for a declared column width.

    Parameters
    ----------
    width : str
        "single", "onehalf", or "double".
    aspect : float
        Height / width ratio. Default 0.62 (close to golden ratio) gives a
        compact landscape panel; pass a larger value for taller figures.

    Returns
    -------
    tuple of float
        (width_inches, height_inches).
    """
    w = _WIDTHS.get(width, COL_SINGLE_IN)
    return (w, w * aspect)


# ==============================================================================
# COLORBLIND-SAFE PALETTE (Okabe-Ito) AND SEMANTIC ROLES
# ==============================================================================

# Okabe-Ito base colors.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Per-unit colors. Bedroom 2 / MODULAIR-PM1 = blue; Morning Room /
# MODULAIR-PM2 = vermillion. Kept stable across all Section 3.2 figures.
UNIT_COLORS = {
    "MODULAIR-PM1": OKABE_ITO["blue"],
    "MODULAIR-PM2": OKABE_ITO["vermillion"],
}

# Location colors (alias the per-unit mapping so AeroTrak / bracketing figures
# that key on location share the same blue/vermillion convention).
LOC_COLORS = {
    "bedroom2": OKABE_ITO["blue"],
    "morning_room": OKABE_ITO["vermillion"],
}

# AeroTrak instrument-label colors used by the coincidence loaders.
INSTR_COLORS = {
    "AeroTrak1": OKABE_ITO["blue"],
    "AeroTrak2": OKABE_ITO["vermillion"],
}

# Reference-instrument roles for the cross-instrument figures.
ROLE_COLORS = {
    "AeroTrak": OKABE_ITO["black"],
    "OPC-N3": OKABE_ITO["blue"],
    "SMPS": OKABE_ITO["orange"],
    "PMS5003": OKABE_ITO["vermillion"],
    "DustTrak": OKABE_ITO["vermillion"],
    "PurpleAir": OKABE_ITO["orange"],
}

# Reference-line and shading colors.
REF_LINE = OKABE_ITO["black"]
SHADE = "#999999"
SHADE_ALPHA = 0.25

# Numeric base font size (pt) for any call site that still passes an explicit
# fontsize. apply_est_style() sets the same value as the rcParams default.
BASE_FONT_PT = 12

# Project TEXT_CONFIG dict, retained so call sites that read the old keys keep
# working after the refactor. Mirrors the ES&T Air convention: bold titles and
# axis labels, normal-weight ticks and legend.
TEXT_CONFIG = {
    "font_size": "12pt",
    "title_font_size": "12pt",
    "axis_label_font_size": "12pt",
    "axis_tick_font_size": "12pt",
    "legend_font_size": "12pt",
    "plot_font_style": "bold",   # titles, axis labels
    "font_style": "normal",      # ticks, legend, data labels
}


# ==============================================================================
# RCPARAMS
# ==============================================================================


def apply_est_style() -> None:
    """
    Set matplotlib rcParams for ES&T Air submission-quality static figures.

    12 pt base font, bold figure/panel titles and axis labels, normal-weight
    tick labels and legend, 300 dpi save default, and constrained-layout plus
    tight bbox so bold labels render without clipping when bbox_inches="tight".
    """
    mpl.rcParams.update({
        # Fonts.
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT,
        "ytick.labelsize": BASE_FONT_PT,
        "legend.fontsize": BASE_FONT_PT,
        "figure.titlesize": BASE_FONT_PT,
        # Weights: bold titles + axis labels, normal ticks + legend.
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.titleweight": "bold",
        # Layout / save defaults.
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        # Cleaner axes.
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, path) -> None:
    """
    Save a figure at the ES&T Air defaults (300 dpi, tight bbox) and close it.

    Creates parent directories if needed and prints a one-line run-log entry.
    """
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [fig] {path.name}")


def _as_path(path):
    """Coerce str/Path to Path without importing pathlib at module top twice."""
    from pathlib import Path
    return path if isinstance(path, Path) else Path(path)
