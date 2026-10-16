"""
Shared plot style for manuscript figures (matplotlib) and interactive or slide plots (Bokeh).

Purpose:
    Hold the settings every project shares: the Wong (Okabe-Ito) palette, the
    loader for journal.mplstyle, ES&T Air column widths, a save function that
    enforces the 600 dpi floor, and the Bokeh and slide text settings.
    journal.mplstyle sits next to this file and is found from this file's own
    location, so the loader works from scripts/, notebooks/, or the project root.

    This file is the same in every project. Colors tied to a project's rooms,
    instruments, or test conditions go in project_style.py, not here.

Author: Nathan Lima
Created: 2026-10-10

Update log:
    2026-10-10: Initial version.
    2026-10-10: Added figsize(), series_style(), save_figure(), named palette
        entries, and apply_bokeh_text() so one module replaces the per-project
        fig_style.py and plot_style.py variants.
    2026-10-10: Reordered the palette by contrast against white.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------------------------------------------------------------
# Palette
# ------------------------------------------------------------------------------

# Wong (2011) colorblind-safe categorical palette, also published as Okabe-Ito.
# The eight colors are unchanged. They are ordered by contrast against a white
# background, highest first, so the first five meet the 3:1 ratio ACS asks for
# non-text elements (21.00, 5.19, 3.87, 3.42, 3.06). Sky blue (2.31), orange
# (2.25), and yellow (1.32) fall below it and come last.
# journal.mplstyle lists the same colors in the same order without the "#".
OKABE_ITO = {
    "black": "#000000",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "orange": "#E69F00",
    "yellow": "#F0E442",
}
WONG = list(OKABE_ITO.values())

# Number of leading colors in WONG that meet 3:1 against white.
N_HIGH_CONTRAST = 5

# Second channel for every series, so a figure still reads in greyscale.
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# ------------------------------------------------------------------------------
# Figure geometry (ACS ES&T Air, final printed size in inches)
# ------------------------------------------------------------------------------

SINGLE_COLUMN_IN = 3.33
DOUBLE_COLUMN_MIN_IN = 4.17
DOUBLE_COLUMN_MAX_IN = 7.00
MAX_DEPTH_IN = 9.17            # includes the caption
CAPTION_LINE_IN = 0.17         # subtract from MAX_DEPTH_IN for each caption line

_WIDTHS = {
    "single": SINGLE_COLUMN_IN,
    "onehalf": 5.00,
    "double": DOUBLE_COLUMN_MAX_IN,
}

MIN_RASTER_DPI = 600
ACCEPTED_FONTS = ("Helvetica", "Arial")
STYLE_FILE = Path(__file__).resolve().parent / "journal.mplstyle"

# ------------------------------------------------------------------------------
# Bokeh and slide text
# ------------------------------------------------------------------------------

# Screen and slide text sizes. These are not journal sizes: manuscript figures
# take their 8 pt text from journal.mplstyle.
TEXT_CONFIG = {
    "font_size": "12pt",
    "title_font_size": "12pt",
    "axis_label_font_size": "12pt",
    "axis_tick_font_size": "12pt",
    "legend_font_size": "12pt",
    "plot_font_style": "bold",    # titles, axis labels
    "font_style": "normal",       # ticks, legend, data labels
}


def use_journal_style():
    """Apply journal.mplstyle to the current matplotlib session.

    Raises
    ------
    FileNotFoundError
        If journal.mplstyle is not next to this module.
    """
    if not STYLE_FILE.is_file():
        raise FileNotFoundError(
            f"{STYLE_FILE} not found. journal.mplstyle belongs in the same "
            "folder as plot_style.py; copy it from the manuscript-figures skill "
            "assets or the repository template."
        )
    plt.style.use(STYLE_FILE)


def figsize(width="single", aspect=0.65):
    """Return a figure size at final printed width.

    Parameters
    ----------
    width : str or float
        "single" (3.33 in), "onehalf" (5.00 in), "double" (7.00 in), or a
        width in inches.
    aspect : float
        Height divided by width. Use 0.6 to 0.7 for wide trend plots and 1.0
        for square panels.

    Returns
    -------
    tuple of float
        (width, height) in inches.

    Raises
    ------
    ValueError
        If the width name is unknown or the size falls outside the ES&T Air
        limits.
    """
    if isinstance(width, str):
        if width not in _WIDTHS:
            raise ValueError(f"Unknown width '{width}'. Use one of {sorted(_WIDTHS)} or inches.")
        w = _WIDTHS[width]
    else:
        w = float(width)
    h = w * aspect
    if w > DOUBLE_COLUMN_MAX_IN:
        raise ValueError(f"Width {w:.2f} in exceeds the {DOUBLE_COLUMN_MAX_IN:.2f} in maximum.")
    if h > MAX_DEPTH_IN:
        raise ValueError(f"Height {h:.2f} in exceeds the {MAX_DEPTH_IN:.2f} in maximum depth.")
    return (w, h)


def series_style(index):
    """Return color, line style, and marker for the series at a position.

    Each series differs from its neighbors in two channels, so no pair is
    separated by color alone. Positions 0 to 4 meet 3:1 contrast against
    white. Positions 5 to 7 (sky blue, orange, yellow) do not, so a figure
    with more than five series needs direct labels or a different encoding.

    Parameters
    ----------
    index : int
        Position of the series, starting at 0.

    Returns
    -------
    dict
        Keyword arguments for ``Axes.plot``: ``color``, ``linestyle``, ``marker``.
    """
    return {
        "color": WONG[index % len(WONG)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "marker": MARKERS[index % len(MARKERS)],
    }


def resolved_font():
    """Return the name of the font matplotlib will draw sans-serif text with.

    matplotlib falls back to DejaVu Sans without stopping when neither
    Helvetica nor Arial is installed, so the name is checked before export.

    Returns
    -------
    str
        Font family name of the resolved font file.
    """
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=mpl.rcParams["font.family"])
    )
    return font_manager.FontProperties(fname=font_path).get_name()


def check_font():
    """Print the resolved font and report whether a journal accepts it.

    Returns
    -------
    bool
        True if the resolved font is Helvetica or Arial.
    """
    name = resolved_font()
    accepted = name in ACCEPTED_FONTS
    status = "accepted" if accepted else "NOT accepted, do not export"
    print(f"Resolved font: {name} ({status})")
    return accepted


def save_figure(fig, path, dpi=MIN_RASTER_DPI, overwrite=False, close=True):
    """Save a figure without changing its size.

    The format follows the file suffix: .pdf and .svg are vector, .png and
    .tif are raster. No tight bounding box is applied, because it changes the
    figure dimensions set at creation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Output file. Parent folders are created.
    dpi : int
        Raster resolution. Values under 600 are refused.
    overwrite : bool
        If False, an existing file raises instead of being replaced.
    close : bool
        Close the figure after saving.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``dpi`` is under 600.
    FileExistsError
        If the file exists and ``overwrite`` is False.
    """
    if dpi < MIN_RASTER_DPI:
        raise ValueError(f"dpi={dpi} is under the {MIN_RASTER_DPI} dpi floor.")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists. Use a new ISO-dated or versioned name, or pass overwrite=True."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)
    print(f"[fig] {path.name}")
    return path


def apply_bokeh_text(plot, config=None):
    """Apply TEXT_CONFIG to a Bokeh figure.

    Sets title, axis label, tick label, and legend text in one call, so
    scripts do not repeat the same property assignments for every plot.

    Parameters
    ----------
    plot : bokeh.plotting.figure
        Figure to style. A figure with no legend yet is left without one.
    config : dict, optional
        Replacement for TEXT_CONFIG with the same keys.

    Returns
    -------
    bokeh.plotting.figure
        The same figure.
    """
    cfg = config or TEXT_CONFIG
    if plot.title is not None:
        plot.title.text_font_size = cfg["title_font_size"]
        plot.title.text_font_style = cfg["plot_font_style"]
    for axis in (*plot.xaxis, *plot.yaxis):
        axis.axis_label_text_font_size = cfg["axis_label_font_size"]
        axis.axis_label_text_font_style = cfg["plot_font_style"]
        axis.major_label_text_font_size = cfg["axis_tick_font_size"]
        axis.major_label_text_font_style = cfg["font_style"]
    for legend in plot.legend:
        legend.label_text_font_size = cfg["legend_font_size"]
        legend.label_text_font_style = cfg["font_style"]
    return plot
