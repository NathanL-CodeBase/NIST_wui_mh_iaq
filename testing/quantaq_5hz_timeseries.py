#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantAQ MODULAIR-PM 5 Hz Time Series by Burn
=============================================

Generates one interactive Bokeh HTML figure per burn (burns 4–10) from the
5 Hz SD-card data logged by the two QuantAQ MODULAIR-PM sensors deployed in
the WUI manufactured-home smoke experiments.

Each HTML file contains two stacked panels (kitchen on top, bedroom on bottom)
with a linked x-axis so zooming or panning one panel simultaneously updates
the other.

Key Channels Plotted:
    - OPC bin0–bin6 (p/cm³): shades of green, darkest = smallest particle
    - neph_bin0–neph_bin5 (p/cm³): shades of red, darkest = smallest particle
    Sensor identity (kitchen vs. bedroom) is indicated by the panel title only.

Data Quality Flags (bitmask in 'flag' column — values can be summed):
    Bit 1    (value 1)    — Startup: device power-on transient
    Bit 2    (value 2)    — OPC Fault: OPC data transfer failure
    Bit 4    (value 4)    — Neph Fault: nephelometer transfer error
    Bit 8    (value 8)    — RH/T Fault: humidity/temperature sensor failure
    Bit 4096 (value 4096) — OPC Overheat
    Bit 8192 (value 8192) — SD Card Fault
    Example: flag = 6 → bit 2 (OPC Fault) AND bit 4 (Neph Fault) simultaneously.

Methodology:
    1. Load DATA_YYYYMMDD.csv from each sensor's path_5hz directory.
    2. Skip the 3-row device-header block; parse timestamp_iso as UTC.
    3. Convert UTC → EDT by applying a fixed -4 h offset.
    4. Apply per-sensor clock-correction shift (bedroom: -2.97 min; kitchen: 0).
    5. Compute time since garage closed; filter to [-1, +2] hours.
    6. Decode flag bitmask into contiguous time spans per bit (supports compound
       flags: each bit is tested independently via bitwise AND).
    7. Build Bokeh panels per sensor: flag bands → event lines → scatter markers.
    8. Stack panels with linked x-range; append metadata footer.
    9. Save as HTML to output_figures/quantaq_5hz_timeseries/.

Output Files:
    - quantaq_5hz_burn4.html through quantaq_5hz_burn10.html

Interactive Tools per Panel:
    - Pan, box zoom, box select, lasso select, wheel zoom, crosshair, reset, save
    - Hover tooltip: channel name, concentration, time since garage closed,
      local timestamp, and raw flag value

Applications:
    - High-resolution QA of particle size data during burn experiments
    - Identifying sensor faults, startup transients, and smoke event timing
    - Understanding which data points were flagged and why

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
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    Div,
    HoverTool,
    Legend,
    LegendItem,
    Span,
)
from bokeh.plotting import figure

warnings.filterwarnings("ignore")

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

PLOT_WINDOW = (-1.0, 2.0)     # hours relative to garage-closed time
UTC_OFFSET_HRS = -4           # sensor timestamps are UTC; experiments were EDT (UTC-4)

SENSOR_CONFIG = {
    "kitchen": {
        "config_key": "quantaq_kitchen",
        "time_shift_min": 0.0,
        "display_label": "Kitchen  (MOD-PM-00197)",
        "legend_prefix": "K",
    },
    "bedroom": {
        "config_key": "quantaq_bedroom",
        "time_shift_min": -2.97,
        "display_label": "Bedroom  (MOD-PM-00194)",
        "legend_prefix": "B",
    },
}

# Flag bitmask: bit_value → (label, fill_color, fill_alpha)
# Compound flags (e.g., flag=6 = OPC Fault + Neph Fault) are decoded bit-by-bit.
FLAG_DEFS = {
    1:    ("Startup",      "#FFD700", 0.20),
    2:    ("OPC Fault",    "#FF4444", 0.25),
    4:    ("Neph Fault",   "#FF8C00", 0.25),
    8:    ("RH/T Fault",   "#4169E1", 0.20),
    4096: ("OPC Overheat", "#9400D3", 0.25),
    8192: ("SD Fault",     "#808080", 0.25),
}

# OPC bins (bin0–bin6): shades of green, dark (small) → light (large)
OPC_COLORS = [
    "#004D00",  # bin0
    "#1A6B1A",
    "#2D8B2D",
    "#43A843",
    "#66BB66",
    "#95D595",
    "#C8EEC8",  # bin6
]

# Neph bins (neph_bin0–neph_bin5): shades of red, dark (small) → light (large)
NEPH_COLORS = [
    "#67000D",  # neph_bin0
    "#A50026",
    "#D73027",
    "#F46D43",
    "#FDAE61",
    "#FEE8C8",  # neph_bin5
]

TOOLS = "pan,box_zoom,box_select,lasso_select,wheel_zoom,crosshair,reset,save"
OUTPUT_SUBDIR = "quantaq_5hz_timeseries"
FIGURE_WIDTH = 1200
PANEL_HEIGHT = 460
MARKER_SIZE = 3
MARKER_ALPHA = 0.55


# ============================================================================
# DATA LOADING
# ============================================================================

def get_sensor_5hz_path(config_key):
    """Return the path_5hz directory for an instrument key from data_config.json.

    Parameters:
        config_key (str): Key under 'instruments' in data_config.json.

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
        burn_date (datetime-like): Burn date used to construct the filename.

    Returns:
        pd.DataFrame: Raw data, or empty DataFrame if file is missing/unreadable.
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
    """Parse timestamps, apply UTC→EDT conversion and instrument time shift,
    add hours-since-garage-closed axis, and filter to the plot window.

    Sensor timestamps are stored as UTC (ISO 8601 with 'Z' suffix). The
    experiments were conducted in EDT (UTC-4), so UTC_OFFSET_HRS = -4 is
    applied before computing the time axis relative to the burn log times,
    which are already in local (EDT) time.

    Parameters:
        df_raw (pd.DataFrame): Raw 5 Hz data from load_5hz_data().
        time_shift_min (float): Instrument clock correction in minutes.
        garage_time (pd.Timestamp): Garage-closed reference time (local EDT).

    Returns:
        pd.DataFrame: Processed data with 'timestamp' and 't_hrs' columns,
            filtered to PLOT_WINDOW.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # Parse ISO timestamp (UTC) and convert to local EDT
    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp_iso"].str.replace("T", " ").str.replace("Z", ""),
            errors="coerce",
        ).dt.tz_localize(None)
        + pd.Timedelta(hours=UTC_OFFSET_HRS)
    )

    # Apply per-instrument clock-correction shift
    if time_shift_min != 0:
        df["timestamp"] += pd.Timedelta(minutes=time_shift_min)

    # Time axis: hours since garage closed
    df["t_hrs"] = (df["timestamp"] - garage_time).dt.total_seconds() / 3600

    in_window = (df["t_hrs"] >= PLOT_WINDOW[0]) & (df["t_hrs"] <= PLOT_WINDOW[1])
    return df.loc[in_window].copy()


# ============================================================================
# FLAG ANALYSIS
# ============================================================================

def extract_flag_spans(df, t_col="t_hrs", flag_col="flag"):
    """Decode the 'flag' bitmask column into contiguous time spans per flag bit.

    Compound flag values (e.g., flag=6 = OPC Fault bit 2 + Neph Fault bit 4)
    are handled correctly: each bit is tested independently using bitwise AND,
    so a single row with flag=6 contributes to both bit-2 and bit-4 spans.

    Parameters:
        df (pd.DataFrame): Sensor data within the burn window.
        t_col (str): Column name for the time axis (hrs since garage closed).
        flag_col (str): Column name for the bitmask integer.

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
        # Pad with False at both ends for clean edge detection
        padded = np.concatenate([[False], mask, [False]])
        rise = np.where(~padded[:-1] & padded[1:])[0]
        fall = np.where(padded[:-1] & ~padded[1:])[0]
        bit_spans = [
            (t_vals[s], t_vals[e - 1])
            for s, e in zip(rise, fall)
            if s < len(t_vals) and e > 0
        ]
        if bit_spans:
            spans[bit] = bit_spans

    return spans


# ============================================================================
# PLOTTING
# ============================================================================

def add_hover_tool(p):
    """Add a HoverTool that reports channel name, concentration, time, local
    timestamp, and raw flag value for the nearest data point.

    Parameters:
        p (figure): Bokeh figure to attach the tool to.
    """
    hover = HoverTool(
        tooltips=[
            ("Channel",       "@channel"),
            ("Concentration", "@value{0.0000} p/cm³"),
            ("Time (hrs)",    "@t_hrs{0.000}"),
            ("Local time",    "@ts"),
            ("Flag value",    "@flag_val"),
        ],
        point_policy="snap_to_data",
    )
    p.add_tools(hover)


def add_flag_bands(p, flag_spans):
    """Add semi-transparent BoxAnnotations behind data for each active flag bit.

    Parameters:
        p (figure): Bokeh figure.
        flag_spans (dict): Output of extract_flag_spans() for this sensor.
    """
    for bit, spans in flag_spans.items():
        _, color, alpha = FLAG_DEFS[bit]
        for t_start, t_end in spans:
            p.add_layout(BoxAnnotation(
                left=t_start, right=t_end,
                fill_color=color, fill_alpha=alpha,
                line_color=None,
            ))


def add_event_lines(p, cr_box_hrs):
    """Add vertical reference lines: garage-closed at t=0 (solid) and CR Box
    on at t=cr_box_hrs (dashed, if available).

    Parameters:
        p (figure): Bokeh figure.
        cr_box_hrs (float or None): CR Box activation time in hours since
            garage closed.
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


def add_sensor_markers(p, df, sensor):
    """Plot OPC and neph bin scatter markers for one sensor panel.

    Each channel gets its own ColumnDataSource so the HoverTool can report
    channel name, local timestamp, and flag value per point.  Non-selected
    points automatically dim when box_select or lasso_select is active.

    OPC bins (bin0–bin6): green markers (solid circle, no border)
    Neph bins (neph_bin0–5): red markers (solid circle, no border)

    Parameters:
        p (figure): Bokeh figure for this sensor panel.
        df (pd.DataFrame): Prepared sensor data with 't_hrs' and 'timestamp'.
        sensor (str): 'kitchen' or 'bedroom'.

    Returns:
        list of LegendItem: One item per rendered channel for the panel legend.
    """
    prefix = SENSOR_CONFIG[sensor]["legend_prefix"]
    ts_str = df["timestamp"].dt.strftime("%H:%M:%S").values
    flag_str = (
        pd.to_numeric(df.get("flag", pd.Series([0] * len(df))), errors="coerce")
        .fillna(0).astype(int).astype(str).values
    )
    items = []

    for i, col in enumerate(OPC_BINS):
        if col not in df.columns:
            continue
        valid = df[col].notna()
        source = ColumnDataSource(dict(
            t_hrs=df.loc[valid, "t_hrs"].values,
            value=df.loc[valid, col].values,
            channel=[f"{prefix} OPC {col}"] * valid.sum(),
            ts=ts_str[valid.values],
            flag_val=flag_str[valid.values],
        ))
        r = p.circle(
            x="t_hrs", y="value", source=source,
            fill_color=OPC_COLORS[i], fill_alpha=MARKER_ALPHA,
            line_color=None, size=MARKER_SIZE,
        )
        items.append(LegendItem(label=f"{prefix} OPC {col}", renderers=[r]))

    for i, col in enumerate(NEPH_BINS):
        if col not in df.columns:
            continue
        valid = df[col].notna()
        source = ColumnDataSource(dict(
            t_hrs=df.loc[valid, "t_hrs"].values,
            value=df.loc[valid, col].values,
            channel=[f"{prefix} neph b{i}"] * valid.sum(),
            ts=ts_str[valid.values],
            flag_val=flag_str[valid.values],
        ))
        r = p.circle(
            x="t_hrs", y="value", source=source,
            fill_color=NEPH_COLORS[i], fill_alpha=MARKER_ALPHA,
            line_color=None, size=MARKER_SIZE,
        )
        items.append(LegendItem(label=f"{prefix} neph b{i}", renderers=[r]))

    return items


def build_sensor_panel(sensor, df, flag_spans, cr_box_hrs, panel_title,
                       x_range=None):
    """Create a single Bokeh figure panel for one sensor.

    Parameters:
        sensor (str): 'kitchen' or 'bedroom'.
        df (pd.DataFrame): Prepared sensor data.
        flag_spans (dict): extract_flag_spans() result for this sensor.
        cr_box_hrs (float or None): CR Box activation time (hrs since garage).
        panel_title (str): Figure title string.
        x_range: Bokeh Range1d or figure x_range to link to (None = new range).

    Returns:
        figure: Configured Bokeh panel.
    """
    kwargs = dict(x_range=x_range) if x_range is not None else dict(x_range=PLOT_WINDOW)

    p = figure(
        title=panel_title,
        x_axis_label="",
        y_axis_label="Concentration (p/cm³)",
        width=FIGURE_WIDTH,
        height=PANEL_HEIGHT,
        tools=TOOLS,
        **kwargs,
    )
    p.title.text_font_size = "11pt"
    p.yaxis.axis_label_text_font_size = "10pt"
    p.grid.grid_line_alpha = 0.3
    add_hover_tool(p)

    # Flag bands behind everything else
    add_flag_bands(p, flag_spans)

    # Reference event lines
    add_event_lines(p, cr_box_hrs)

    # Scatter markers and legend
    if not df.empty:
        legend_items = add_sensor_markers(p, df, sensor)
        if legend_items:
            legend = Legend(
                items=legend_items,
                click_policy="hide",
                label_text_font_size="8pt",
                glyph_height=10,
                glyph_width=18,
                spacing=1,
                padding=5,
            )
            p.add_layout(legend, "right")

    return p


def build_flag_key_html(all_spans):
    """Build a compact HTML color-key for all flag bands shown in the figure.

    Parameters:
        all_spans (list of dict): One extract_flag_spans() result per sensor.

    Returns:
        str: HTML fragment, or empty string if no flags are present.
    """
    active_bits = set().union(*(s.keys() for s in all_spans))
    if not active_bits:
        return ""
    parts = ["<b>Flag bands:</b>&ensp;"]
    for bit in sorted(active_bits):
        label, color, _ = FLAG_DEFS[bit]
        parts.append(
            f"<span style='display:inline-block;width:16px;height:10px;"
            f"background:{color};opacity:0.75;border:1px solid #888;"
            f"vertical-align:middle'></span>&nbsp;{label}&ensp;"
        )
    return "".join(parts)


def create_burn_figure(burn_id, data_by_sensor, timing, flag_spans_by_sensor,
                       flag_meta_str):
    """Assemble the full Bokeh column layout for one burn: two stacked sensor
    panels with a linked x-axis, plus a metadata/flag-key footer.

    Parameters:
        burn_id (str): Burn identifier (e.g., 'burn4').
        data_by_sensor (dict): {'kitchen': df, 'bedroom': df}.
        timing (dict): From get_burn_timing(); keys 'burn_date', 'cr_box_hrs'.
        flag_spans_by_sensor (dict): {'kitchen': spans_dict, 'bedroom': spans_dict}.
        flag_meta_str (str): Concatenated get_flag_metadata() output for both sensors.

    Returns:
        bokeh.layouts.column: Layout ready for output_file() + save().
    """
    burn_date_str = timing["burn_date"].strftime("%Y-%m-%d")
    cr_hrs = timing["cr_box_hrs"]

    def _subtitle(sensor):
        label = SENSOR_CONFIG[sensor]["display_label"]
        note = f"garage closed t=0 (—)  |  CR Box on t={cr_hrs:.3f} h (- -)" \
               if cr_hrs is not None else "garage closed t=0 (—)"
        return f"{burn_id}  —  {label}  ({burn_date_str})  |  {note}"

    # Kitchen panel: defines the shared x-range
    p_kitchen = build_sensor_panel(
        sensor="kitchen",
        df=data_by_sensor["kitchen"],
        flag_spans=flag_spans_by_sensor["kitchen"],
        cr_box_hrs=cr_hrs,
        panel_title=_subtitle("kitchen"),
    )
    p_kitchen.xaxis.axis_label = ""

    # Bedroom panel: links x-range to kitchen
    p_bedroom = build_sensor_panel(
        sensor="bedroom",
        df=data_by_sensor["bedroom"],
        flag_spans=flag_spans_by_sensor["bedroom"],
        cr_box_hrs=cr_hrs,
        panel_title=_subtitle("bedroom"),
        x_range=p_kitchen.x_range,
    )
    p_bedroom.xaxis.axis_label = "Time Since Garage Closed (hours)"
    p_bedroom.xaxis.axis_label_text_font_size = "10pt"

    # Footer: flag color key + metadata
    flag_key_html = build_flag_key_html(list(flag_spans_by_sensor.values()))
    script_meta = get_script_metadata()
    footer = Div(
        text=(
            f"<div style='font-size:9px;color:#555;margin-top:4px'>"
            f"{flag_key_html}<br>"
            f"{script_meta}<br>"
            f"{flag_meta_str}"
            f"</div>"
        ),
        width=FIGURE_WIDTH,
    )

    return column(p_kitchen, p_bedroom, footer)


# ============================================================================
# BURN TIMING
# ============================================================================

def get_burn_timing(burn_log, burn_id):
    """Extract garage-closed and CR Box on times from the burn log.

    Parameters:
        burn_log (pd.DataFrame): Loaded burn log (Sheet2).
        burn_id (str): Burn identifier (e.g., 'burn4').

    Returns:
        dict or None: Keys 'burn_date' (Timestamp), 'garage_time' (Timestamp),
            'cr_box_hrs' (float or None). Returns None if the burn row is
            missing or has no garage-closed entry.
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
    """Load 5 Hz data, build per-burn two-panel Bokeh figures, save HTML."""
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

    for burn_id in BURNS:
        print(f"--- {burn_id} ---")

        timing = get_burn_timing(burn_log, burn_id)
        if timing is None:
            print(f"  [SKIP] Missing timing in burn log.\n")
            continue

        data_by_sensor = {}
        flag_spans_by_sensor = {}
        flag_meta_parts = []

        for sensor, cfg in SENSOR_CONFIG.items():
            path = sensor_paths.get(sensor)
            if path is None:
                data_by_sensor[sensor] = pd.DataFrame()
                flag_spans_by_sensor[sensor] = {}
                continue

            df_raw = load_5hz_data(path, timing["burn_date"])
            df = prepare_sensor_data(df_raw, cfg["time_shift_min"], timing["garage_time"])
            data_by_sensor[sensor] = df

            spans = extract_flag_spans(df) if not df.empty else {}
            flag_spans_by_sensor[sensor] = spans

            if not df.empty:
                print(f"    {cfg['display_label']}: {len(df):,} rows in window")
                flag_meta_parts.append(
                    get_flag_metadata(spans, FLAG_DEFS, sensor)
                )

        if all(df.empty for df in data_by_sensor.values()):
            print(f"  [SKIP] No data in burn window.\n")
            continue

        flag_meta_str = "  |  ".join(flag_meta_parts) if flag_meta_parts else "No flag data"

        layout = create_burn_figure(
            burn_id, data_by_sensor, timing, flag_spans_by_sensor, flag_meta_str
        )

        out_path = output_dir / f"quantaq_5hz_{burn_id}.html"
        output_file(str(out_path), title=f"QuantAQ 5Hz {burn_id}")
        save(layout)
        print(f"  Saved → {out_path.name}\n")

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
