#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantAQ MODULAIR-PM 5 Hz Time Series by Burn
=============================================

Generates one interactive Bokeh HTML figure per burn (burns 4–10) from the
5 Hz SD-card data logged by the two QuantAQ MODULAIR-PM sensors deployed in
the WUI manufactured-home smoke experiments.

Key Channels Plotted:
    - OPC bin0–bin6 (p/cm³): solid lines, Alphasense R2 optical particle counter
    - neph_bin0–neph_bin5 (p/cm³): dashed lines, nephelometer particle counts
    Kitchen sensor (MOD-PM-00197): shades of red, darkest = smallest particle bin
    Bedroom sensor (MOD-PM-00194): shades of green, darkest = smallest particle bin

Analysis Features:
    - Time axis: hours since garage door closed (window: −1 to +2 hours)
    - Time-shift corrections applied per sensor (bedroom: −2.97 min; kitchen: 0 min)
    - Vertical lines mark garage-closed (solid) and CR Box power-on (dashed)
    - Semi-transparent colored bands mark data quality flag periods (bitmask decoded)
    - Flag summary embedded in per-figure metadata footer

Data Quality Flags (bitmask in 'flag' column):
    Bit 1    — Startup (device power-on period)
    Bit 2    — OPC Fault (OPC data failed to transfer)
    Bit 4    — Neph Fault (nephelometer transfer error)
    Bit 8    — RH/T Fault (humidity/temperature sensor failure)
    Bit 4096 — OPC Overheat
    Bit 8192 — SD Card Fault

Methodology:
    1. Load DATA_YYYYMMDD.csv from each sensor's path_5hz directory.
    2. Skip the 3-row device-header block; parse timestamp_iso as local time.
    3. Apply per-sensor time-shift correction.
    4. Compute time since garage closed; filter to [−1, +2] hours.
    5. Decode flag bitmask into contiguous time spans per flag bit.
    6. Build Bokeh figure: flag bands → event lines → data lines → legend.
    7. Append metadata footer (script name, run date, flag summary).
    8. Save as HTML to output_figures/quantaq_5hz_timeseries/.

Output Files:
    - quantaq_5hz_burn4.html through quantaq_5hz_burn10.html

Applications:
    - Visual QA of high-resolution particle size data during burn experiments
    - Identifying sensor faults, startup transients, and smoke event timing

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, Div, Legend, LegendItem, Span
from bokeh.plotting import figure

warnings.filterwarnings("ignore")

# Add repository root to path for portable imports
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.data_loaders import load_burn_log
from scripts.metadata_utils import get_flag_metadata, get_script_metadata
from src.data_paths import get_common_file, resolver

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

BURNS = [f"burn{i}" for i in range(4, 11)]

OPC_BINS = [f"bin{i}" for i in range(7)]          # bin0–bin6
NEPH_BINS = [f"neph_bin{i}" for i in range(6)]    # neph_bin0–neph_bin5

# Hours relative to garage-closed to display
PLOT_WINDOW = (-1.0, 2.0)

SENSOR_CONFIG = {
    "kitchen": {
        "config_key": "quantaq_kitchen",
        "time_shift_min": 0.0,
        "display_label": "Kitchen (MOD-PM-00197)",
        "legend_prefix": "K",
    },
    "bedroom": {
        "config_key": "quantaq_bedroom",
        "time_shift_min": -2.97,
        "display_label": "Bedroom (MOD-PM-00194)",
        "legend_prefix": "B",
    },
}

# Flag bitmask definitions: bit_value → (label, fill_color, fill_alpha)
FLAG_DEFS = {
    1:    ("Startup",      "#FFD700", 0.20),
    2:    ("OPC Fault",    "#FF4444", 0.25),
    4:    ("Neph Fault",   "#FF8C00", 0.25),
    8:    ("RH/T Fault",   "#4169E1", 0.20),
    4096: ("OPC Overheat", "#9400D3", 0.25),
    8192: ("SD Fault",     "#808080", 0.25),
}

# Color palettes: index 0 = smallest particle (bin0/neph_bin0, darkest)
#                 index 6 = largest particle (bin6, lightest)
# Kitchen → shades of red; Bedroom → shades of green
KITCHEN_COLORS = [
    "#67000D",  # bin0 / neph_bin0
    "#A50026",
    "#D73027",
    "#F46D43",
    "#FDAE61",
    "#FDD49E",
    "#FEE8C8",  # bin6
]
BEDROOM_COLORS = [
    "#00441B",  # bin0 / neph_bin0
    "#006D2C",
    "#238B45",
    "#41AB5D",
    "#74C476",
    "#A1D99B",
    "#D9F0A3",  # bin6
]
PALETTE = {"kitchen": KITCHEN_COLORS, "bedroom": BEDROOM_COLORS}

OUTPUT_SUBDIR = "quantaq_5hz_timeseries"
FIGURE_WIDTH = 1200
FIGURE_HEIGHT = 560


# ============================================================================
# DATA LOADING
# ============================================================================

def get_sensor_5hz_path(config_key):
    """Return the path_5hz directory for an instrument key from data_config.json.

    Parameters:
        config_key (str): Key under 'instruments' in data_config.json
            (e.g., 'quantaq_kitchen').

    Returns:
        Path: Resolved path to the 5 Hz SD-card data directory.
    """
    instr = resolver.config.get("instruments", {}).get(config_key, {})
    raw = instr.get("path_5hz")
    if raw is None:
        raise KeyError(
            f"No 'path_5hz' key for instrument '{config_key}' in data_config.json."
        )
    return Path(raw)


def load_5hz_data(sensor_path, burn_date):
    """Load the DATA_YYYYMMDD.csv file for a given burn date.

    The CSV has a 3-row device-header block (deviceModel, deviceID, deviceSN)
    before the column-header row, so skiprows=3 is required.

    Parameters:
        sensor_path (Path): Directory containing DATA_*.csv files.
        burn_date (datetime-like): Burn date used to build the filename.

    Returns:
        pd.DataFrame: Raw data, or empty DataFrame if file missing/unreadable.
    """
    date_str = pd.to_datetime(burn_date).strftime("%Y%m%d")
    file_path = sensor_path / f"DATA_{date_str}.csv"
    if not file_path.exists():
        print(f"    [SKIP] File not found: {file_path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, skiprows=3, low_memory=False)
        print(f"    Loaded {file_path.name}  ({len(df):,} rows)")
        return df
    except Exception as exc:
        print(f"    [ERROR] {file_path.name}: {str(exc)[:100]}")
        return pd.DataFrame()


def prepare_sensor_data(df_raw, time_shift_min, garage_time):
    """Parse timestamps, apply time shift, add time axis, and filter to plot window.

    Parameters:
        df_raw (pd.DataFrame): Raw 5 Hz data from load_5hz_data().
        time_shift_min (float): Clock correction in minutes (negative = shift earlier).
        garage_time (pd.Timestamp): Garage-closed reference time.

    Returns:
        pd.DataFrame: Processed data with 't_hrs' column, filtered to PLOT_WINDOW.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp_iso"].str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    ).dt.tz_localize(None)

    if time_shift_min != 0:
        df["timestamp"] += pd.Timedelta(minutes=time_shift_min)

    df["t_hrs"] = (df["timestamp"] - garage_time).dt.total_seconds() / 3600

    in_window = (df["t_hrs"] >= PLOT_WINDOW[0]) & (df["t_hrs"] <= PLOT_WINDOW[1])
    df = df.loc[in_window].copy()
    return df


# ============================================================================
# FLAG ANALYSIS
# ============================================================================

def extract_flag_spans(df, t_col="t_hrs", flag_col="flag"):
    """Decode the 'flag' bitmask column into contiguous time spans per flag bit.

    Parameters:
        df (pd.DataFrame): Sensor data within the burn window.
        t_col (str): Column name for the time axis (hours since garage closed).
        flag_col (str): Column name for the bitmask flag integer.

    Returns:
        dict: {bit_value: [(t_start, t_end), ...]} for each bit that fired.
    """
    spans = {}
    if flag_col not in df.columns or df.empty:
        return spans

    flag_int = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int).values
    t_vals = df[t_col].values

    for bit in FLAG_DEFS:
        mask = (flag_int & bit).astype(bool)
        if not mask.any():
            continue
        # Pad with False at both ends to cleanly detect rising and falling edges
        padded = np.concatenate([[False], mask, [False]])
        rise = np.where(~padded[:-1] & padded[1:])[0]   # index in mask where flag starts
        fall = np.where(padded[:-1] & ~padded[1:])[0]    # index in mask after last True

        bit_spans = [
            (t_vals[s], t_vals[e - 1])
            for s, e in zip(rise, fall)
            if s < len(t_vals) and e > 0
        ]
        if bit_spans:
            spans[bit] = bit_spans

    return spans


def merge_sensor_spans(spans_list):
    """Union-merge flag spans from multiple sensors so each flag bit has one set of bands.

    Parameters:
        spans_list (list of dict): One extract_flag_spans() result per sensor.

    Returns:
        dict: {bit_value: [(t_start, t_end), ...]} with overlapping spans merged.
    """
    merged = {}
    all_bits = set().union(*(s.keys() for s in spans_list))
    for bit in all_bits:
        combined = sorted(
            [sp for s in spans_list for sp in s.get(bit, [])],
            key=lambda x: x[0],
        )
        if not combined:
            continue
        result = [list(combined[0])]
        for t_start, t_end in combined[1:]:
            if t_start <= result[-1][1]:
                result[-1][1] = max(result[-1][1], t_end)
            else:
                result.append([t_start, t_end])
        merged[bit] = [(s, e) for s, e in result]
    return merged


# ============================================================================
# PLOTTING
# ============================================================================

def add_flag_bands(p, merged_spans):
    """Add semi-transparent BoxAnnotations (rendered before data lines) for each flag.

    Parameters:
        p (figure): Bokeh figure to annotate.
        merged_spans (dict): Output of merge_sensor_spans().
    """
    for bit, spans in merged_spans.items():
        _, color, alpha = FLAG_DEFS[bit]
        for t_start, t_end in spans:
            p.add_layout(BoxAnnotation(
                left=t_start, right=t_end,
                fill_color=color, fill_alpha=alpha,
                line_color=None,
            ))


def add_event_lines(p, cr_box_hrs):
    """Add vertical reference lines for garage-closed (t=0) and CR Box on.

    Parameters:
        p (figure): Bokeh figure.
        cr_box_hrs (float or None): Time of CR Box power-on in hours since garage closed.
    """
    p.add_layout(Span(
        location=0, dimension="height",
        line_color="#444444", line_width=2.0, line_dash="solid",
    ))
    if cr_box_hrs is not None:
        p.add_layout(Span(
            location=cr_box_hrs, dimension="height",
            line_color="#444444", line_width=2.0, line_dash="dashed",
        ))


def add_sensor_lines(p, df, sensor):
    """Plot OPC bin lines (solid) and neph bin lines (dashed) for one sensor.

    Parameters:
        p (figure): Bokeh figure.
        df (pd.DataFrame): Prepared sensor data with 't_hrs' column.
        sensor (str): 'kitchen' or 'bedroom'.

    Returns:
        list of LegendItem: One item per rendered line for the figure legend.
    """
    colors = PALETTE[sensor]
    prefix = SENSOR_CONFIG[sensor]["legend_prefix"]
    t = df["t_hrs"].values
    items = []

    for i, col in enumerate(OPC_BINS):
        if col not in df.columns:
            continue
        r = p.line(
            t, df[col].values,
            line_color=colors[i], line_width=1.4,
            line_dash="solid", alpha=0.88,
        )
        items.append(LegendItem(label=f"{prefix} OPC {col}", renderers=[r]))

    for i, col in enumerate(NEPH_BINS):
        if col not in df.columns:
            continue
        r = p.line(
            t, df[col].values,
            line_color=colors[i], line_width=1.4,
            line_dash="dashed", alpha=0.88,
        )
        # Trim "neph_bin" → "nb" for compact legend labels
        items.append(LegendItem(label=f"{prefix} neph b{col[-1]}", renderers=[r]))

    return items


def build_flag_key_html(merged_spans):
    """Build an inline HTML color-key string for the flag bands on the figure.

    Parameters:
        merged_spans (dict): Output of merge_sensor_spans().

    Returns:
        str: HTML fragment, or empty string if no flags present.
    """
    if not merged_spans:
        return ""
    parts = ["<b>Flag bands:</b>&nbsp;"]
    for bit in sorted(merged_spans):
        label, color, _ = FLAG_DEFS[bit]
        parts.append(
            f"<span style='display:inline-block;width:18px;height:10px;"
            f"background:{color};opacity:0.7;border:1px solid #777;vertical-align:middle'>"
            f"</span>&nbsp;{label}&nbsp;&nbsp;"
        )
    return "".join(parts)


def create_burn_figure(burn_id, data_by_sensor, timing, flag_meta_str):
    """Assemble the complete Bokeh Column layout (figure + metadata footer) for one burn.

    Parameters:
        burn_id (str): Burn identifier, e.g. 'burn4'.
        data_by_sensor (dict): {'kitchen': df, 'bedroom': df} prepared data.
        timing (dict): Burn timing info from get_burn_timing().
        flag_meta_str (str): Flag summary string from get_flag_metadata().

    Returns:
        bokeh.layouts.column: Layout ready for save().
    """
    burn_date_str = timing["burn_date"].strftime("%Y-%m-%d")
    cr_hrs = timing["cr_box_hrs"]

    title_parts = [
        f"QuantAQ 5 Hz  —  {burn_id}  ({burn_date_str})",
        "garage closed at t = 0 h (solid line)",
    ]
    if cr_hrs is not None:
        title_parts.append(f"CR Box on at t = {cr_hrs:.3f} h (dashed line)")
    title = "  |  ".join(title_parts)

    p = figure(
        title=title,
        x_axis_label="Time Since Garage Closed (hours)",
        y_axis_label="Particle Count Density (p/cm³)",
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        x_range=PLOT_WINDOW,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    p.title.text_font_size = "11pt"
    p.xaxis.axis_label_text_font_size = "11pt"
    p.yaxis.axis_label_text_font_size = "11pt"
    p.grid.grid_line_alpha = 0.35

    # Flag bands drawn first so they appear behind data lines
    non_empty = [df for df in data_by_sensor.values() if not df.empty]
    spans_list = [extract_flag_spans(df) for df in non_empty]
    merged_spans = merge_sensor_spans(spans_list)
    add_flag_bands(p, merged_spans)

    # Reference event lines
    add_event_lines(p, cr_hrs)

    # Data lines and legend
    legend_items = []
    for sensor, df in data_by_sensor.items():
        if not df.empty:
            legend_items.extend(add_sensor_lines(p, df, sensor))

    if legend_items:
        legend = Legend(
            items=legend_items,
            click_policy="hide",
            label_text_font_size="8pt",
            glyph_height=12,
            glyph_width=22,
            spacing=2,
            padding=6,
        )
        p.add_layout(legend, "right")

    # Footer: flag color key + metadata text
    flag_key_html = build_flag_key_html(merged_spans)
    script_meta = get_script_metadata()
    footer_html = (
        f"<div style='font-size:9px;color:#555;margin-top:4px'>"
        f"{flag_key_html}<br>"
        f"{script_meta}<br>"
        f"{flag_meta_str}"
        f"</div>"
    )
    meta_div = Div(text=footer_html, width=FIGURE_WIDTH)

    return column(p, meta_div)


# ============================================================================
# BURN TIMING
# ============================================================================

def get_burn_timing(burn_log, burn_id):
    """Extract garage-closed and CR Box on times for one burn.

    Parameters:
        burn_log (pd.DataFrame): Loaded burn log (Sheet2).
        burn_id (str): Burn identifier, e.g. 'burn4'.

    Returns:
        dict or None: Keys 'burn_date', 'garage_time', 'cr_box_hrs'.
            Returns None if the burn is missing or has no garage-closed time.
    """
    row = burn_log[burn_log["Burn ID"] == burn_id]
    if row.empty:
        return None

    burn_date = pd.to_datetime(row["Date"].iloc[0])
    date_str = burn_date.strftime("%Y-%m-%d")

    garage_str = row["garage closed"].iloc[0]
    if pd.isna(garage_str):
        return None
    garage_time = pd.to_datetime(f"{date_str} {garage_str}")

    cr_hrs = None
    cr_str = row["CR Box on"].iloc[0]
    if pd.notna(cr_str):
        cr_time = pd.to_datetime(f"{date_str} {cr_str}")
        cr_hrs = (cr_time - garage_time).total_seconds() / 3600

    return {
        "burn_date": burn_date,
        "garage_time": garage_time,
        "cr_box_hrs": cr_hrs,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Load 5 Hz data, build per-burn Bokeh figures, and save HTML files."""
    print("\n" + "=" * 70)
    print("QuantAQ MODULAIR-PM 5 Hz Time Series  —  Burns 4–10")
    print("=" * 70)

    burn_log = load_burn_log(get_common_file("burn_log"))
    output_dir = get_common_file("output_figures") / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Resolve 5 Hz data paths from data_config.json
    sensor_paths = {}
    for sensor, cfg in SENSOR_CONFIG.items():
        try:
            sensor_paths[sensor] = get_sensor_5hz_path(cfg["config_key"])
        except KeyError as exc:
            print(f"  [WARNING] {exc}")

    # ---- per-burn loop ----
    for burn_id in BURNS:
        print(f"--- {burn_id} ---")

        timing = get_burn_timing(burn_log, burn_id)
        if timing is None:
            print(f"  [SKIP] Missing timing in burn log.\n")
            continue

        # Load and prepare both sensors
        data_by_sensor = {}
        flag_meta_parts = []

        for sensor, cfg in SENSOR_CONFIG.items():
            path = sensor_paths.get(sensor)
            if path is None:
                data_by_sensor[sensor] = pd.DataFrame()
                continue

            df_raw = load_5hz_data(path, timing["burn_date"])
            df = prepare_sensor_data(df_raw, cfg["time_shift_min"], timing["garage_time"])
            data_by_sensor[sensor] = df

            if not df.empty:
                print(f"    {cfg['display_label']}: {len(df):,} rows in window")
                spans = extract_flag_spans(df)
                flag_meta_parts.append(
                    get_flag_metadata(spans, FLAG_DEFS, sensor)
                )

        if all(df.empty for df in data_by_sensor.values()):
            print(f"  [SKIP] No data found in burn window.\n")
            continue

        flag_meta_str = "  |  ".join(flag_meta_parts) if flag_meta_parts else "No flag data"

        layout = create_burn_figure(burn_id, data_by_sensor, timing, flag_meta_str)

        out_path = output_dir / f"quantaq_5hz_{burn_id}.html"
        output_file(str(out_path), title=f"QuantAQ 5Hz {burn_id}")
        save(layout)
        print(f"  Saved → {out_path.name}\n")

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
