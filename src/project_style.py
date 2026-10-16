"""
Project-specific plot colors and labels for the WUI manufactured-home IAQ figures.

Purpose:
    Map this project's rooms and instruments to colors from the shared palette
    in plot_style.py, so a room or instrument keeps one color in every figure
    built on the new style. plot_style.py is identical across projects; this
    file is the only style file that differs. Figures not yet migrated still
    use src/fig_style.py.

Author: Nathan Lima
Created: 2026-10-16

Update log:
    2026-10-16: Initial version, for the poster and manuscript figure update
        (burn 9 Morning Room comparison, AeroTrak coincidence overlay,
        arrival lag, MODULAIR-PM QA/QC time series).
"""

from contextlib import contextmanager
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from src.plot_style import OKABE_ITO, STYLE_FILE, save_figure

LOCATION_COLORS = {
    "bedroom2": OKABE_ITO["blue"],
    "morning_room": OKABE_ITO["vermillion"],
}

# Morning Room instruments share one figure (Burn 09 comparison), so each gets
# its own color and line style. AeroTrak2 keeps vermillion to match the
# coincidence overlay. Black is left for event lines.
INSTRUMENT_COLORS = {
    "AeroTrak2": OKABE_ITO["vermillion"],
    "MODULAIR-PM2": OKABE_ITO["blue"],
    "DustTrak": OKABE_ITO["green"],
    "PurpleAir": OKABE_ITO["purple"],
}
INSTRUMENT_LINESTYLES = {
    "DustTrak": "-",
    "PurpleAir": "--",
    "AeroTrak2": "-.",
    "MODULAIR-PM2": ":",
}

REF_LINE = OKABE_ITO["black"]
NEUTRAL = "#999999"

# Axis labels carry units. Keep them here so every script writes them the same way.
AXIS_LABELS = {
    "pm": "PM concentration (µg/m³)",
    "pm_mass": "PM mass concentration (µg/m³)",
    "pm1": "PM1 (µg/m³)",
    "aerotrak_ch1": "0.3 µm to 0.5 µm (#/cm³)",
    "opc_bin0": "OPC-N3 bin 0 (#/cm³)",
    # QuantAQ documents neph_bin0 only as "particles"; the PMS5003 native field
    # is per 0.1 L, but the QuantAQ firmware scaling is not documented.
    "neph_bin0": "PMS5003 bin 0 (particles)",
    "time_from_ignition_min": "Time from ignition (min)",
    "time_since_garage_h": "Time since garage closed (h)",
    "local_time": "Local time (EDT)",
    "arrival_lag_min": "Arrival lag (min)",
}


@contextmanager
def journal_style():
    """Apply journal.mplstyle inside a block only.

    Resets to matplotlib defaults first, so rcParams set by the legacy
    fig_style.apply_est_style() (tight bounding box, 12 pt bold labels) do not
    leak into new-style figures, and restores them on exit for any legacy
    figure drawn afterward in the same run.
    """
    with plt.style.context(["default", str(STYLE_FILE)]):
        yield


def dated_stem(out_dir, name):
    """Return out_dir / 'YYYY-MM-DD_name' (no suffix) using today's date.

    Parameters
    ----------
    out_dir : str or Path
        Output folder.
    name : str
        File name without date or suffix.

    Returns
    -------
    Path
    """
    return Path(out_dir) / f"{date.today().isoformat()}_{name}"


def save_pdf_png(fig, stem, overwrite=False):
    """Save a figure as vector PDF and 600 dpi PNG, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    stem : Path
        Output path without suffix (see dated_stem).
    overwrite : bool
        Passed to save_figure; existing files raise unless True.

    Returns
    -------
    list of Path
    """
    stem = Path(stem)
    pdf = save_figure(fig, stem.with_suffix(".pdf"), overwrite=overwrite, close=False)
    png = save_figure(fig, stem.with_suffix(".png"), overwrite=overwrite, close=True)
    return [pdf, png]
