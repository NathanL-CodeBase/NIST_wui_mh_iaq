"""
WUI SMPS Size Distribution Heatmap Visualization

This script generates heatmap visualizations of particle size distributions measured
by the Scanning Mobility Particle Sizer (SMPS) during wildland-urban interface
smoke experiments. The heatmaps show temporal evolution of particle number and mass
concentrations across the ultrafine to fine particle size range.

Key Features:
    - Time-resolved particle size distributions
    - Number concentration (#/cm³) and mass concentration (µg/m³) heatmaps
    - Size range: ~10 nm to 500 nm (mobility diameter)
    - Log-scale color mapping for wide dynamic range
    - Event timeline annotations (ignition, door closure, CR Box activation)

Visualization Components:
    - 2D heatmap: particle size (y-axis) vs time (x-axis)
    - Color intensity represents concentration (log scale)
    - Vertical lines marking key experimental events
    - Total concentration time series overlay
    - Size-integrated metrics (total PM)

Data Processing:
    - SMPS scan averaging and smoothing
    - Electrical mobility to physical diameter conversion
    - Multiple charge correction
    - Diffusion loss correction
    - Quality control for inverted size distributions

Analysis Capabilities:
    - Mode diameter tracking over time
    - Geometric mean diameter evolution
    - Number-to-mass conversion
    - Size distribution moments calculation
    - Growth/shrinkage rate determination

Outputs:
    - Interactive Bokeh HTML heatmaps
    - PNG exports for publication
    - Size distribution statistics CSV files
    - Annotated figures with experimental timeline

Use Cases:
    - Characterizing particle size evolution during smoke events
    - Evaluating filter performance across particle sizes
    - Understanding nucleation and coagulation processes
    - Comparing fine vs ultrafine particle dynamics

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - bokeh: Interactive heatmap visualization
    - matplotlib: Color mapping utilities

Configuration:
    - SMPS data directory path
    - Burn selection
    - Color scale limits and palette
    - Time resolution for binning

Author: Nathan Lima
Date: 2024-2025
"""

# %%
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so CLI PNG export is headless-safe

import pandas as pd
import numpy as np
import datetime
from bokeh.plotting import figure, show, save
from bokeh.io import output_notebook, output_file, reset_output
from bokeh.models import ColorBar, LinearColorMapper, Range1d, Label, Div, Span, BoxAnnotation
from bokeh.layouts import column
from bokeh.palettes import Turbo256

import sys
from pathlib import Path

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
# Add src/ so fig_style can be imported without the src. prefix
sys.path.insert(0, str(script_dir))

from src.data_paths import get_data_root, get_instrument_path, get_common_file


# Set output to display plots in the notebook only when running under one.
# In a plain script, output_notebook() and show() trigger an IPython display
# hook that raises NotImplementedError, so guard both.
def _in_notebook():
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


_SHOW_INLINE = _in_notebook()

if _SHOW_INLINE:
    output_notebook()


def _maybe_show(obj):
    """Show a Bokeh object only in a notebook; no-op in a plain script."""
    if _SHOW_INLINE:
        show(obj)



# Set the path for the dataset (only using OneDrive path as requested)
data_root = get_data_root()  # Portable path - auto-configured
os.chdir(str(data_root))

# Create directory for figures if it doesn't exist
os.makedirs(str(get_common_file('output_figures')), exist_ok=True)

# Define burn information - mapping between burn dates and descriptions
BURN_INFO = {
    "2024-04-26": {"description": "01-House", "file_suffix": "04262024"},
    "2024-05-02": {"description": "02-House-4-N", "file_suffix": "05022024"},
    "2024-05-06": {"description": "03-House-1-U", "file_suffix": "05062024"},
    "2024-05-09": {"description": "04-House-1-N", "file_suffix": "05092024"},
    "2024-05-13": {"description": "05-Room", "file_suffix": "05132024"},
    "2024-05-17": {"description": "06-Room-1-N", "file_suffix": "05172024"},
    "2024-05-20": {"description": "07-House-2A-N", "file_suffix": "05202024"},
    "2024-05-23": {"description": "08-House-2A-U", "file_suffix": "05232024"},
    "2024-05-28": {"description": "09-House-2-N", "file_suffix": "05282024"},
    "2024-05-31": {"description": "10-House-2-U", "file_suffix": "05312024"},
}

# Load burn log once (kept in case needed for future reference)
burn_log_path = str(get_common_file('burn_log'))
try:
    burn_log = pd.read_excel(burn_log_path, sheet_name="Sheet2")
    print(f"Successfully loaded burn log with {len(burn_log)} entries")
except Exception as e:
    print(f"Error loading burn log: {e}")
    burn_log = None


def get_script_metadata():
    """Return a string with script name and execution timestamp for figure metadata"""
    try:
        import inspect

        script_name = os.path.basename(
            inspect.getmodule(inspect.currentframe()).__file__
        )
    except (NameError, AttributeError, TypeError):
        try:
            script_name = os.path.basename(__file__)
        except NameError:
            script_name = "wui_smps_heatmap_updated.py"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Generated by: {script_name} | Date: {timestamp}"


def prepare_data(file_path, date_str):
    """
    Process SMPS data for heatmap visualization with proper time alignment.

    Args:
        file_path (str): Path to the Excel file
        date_str (str): Date string in 'YYYY-MM-DD' format

    Returns:
        tuple: (data_for_plot, time_hours, size_bins, min_size, max_size)
    """

    try:
        # Read the base Excel file
        print(f"Reading data from {file_path}")
        df = pd.read_excel(file_path)
        print(f"Raw data shape: {df.shape}")

        # Convert to target date
        target_date = pd.to_datetime(date_str).date()
        print(f"Target date: {target_date}")

        # Check the structure of the Excel file (first few rows/columns)
        print("\nExamining data structure:")
        for i in range(min(5, len(df))):
            print(f"Row {i}: {list(df.iloc[i, :5])}")

        # Find Date and Start Time columns - search through more rows to handle deep structure
        date_col = None
        time_col = None
        lower_size_col = None
        upper_size_col = None

        # Increase search depth to 150 rows to find metadata that appears deeper in the file
        max_search_rows = min(150, len(df))

        # Search for key columns in rows
        for i in range(max_search_rows):
            for j in range(min(10, df.shape[1])):
                cell_value = str(df.iloc[i, j]).lower()
                if "date" in cell_value:
                    date_col = j
                    print(f"Found Date column at position ({i}, {j})")
                elif "start time" in cell_value:
                    time_col = j
                    print(f"Found Start Time column at position ({i}, {j})")
                elif "lower size" in cell_value:
                    lower_size_col = j
                    print(f"Found Lower Size column at position ({i}, {j})")
                elif "upper size" in cell_value:
                    upper_size_col = j
                    print(f"Found Upper Size column at position ({i}, {j})")

        # If we couldn't find the columns, try looking at column names directly
        if date_col is None or time_col is None:
            for j, col_name in enumerate(df.columns):
                col_str = str(col_name).lower()
                if "date" in col_str:
                    date_col = j
                    print(f"Found Date column at position {j}")
                elif "start time" in col_str or "time" in col_str:
                    time_col = j
                    print(f"Found Start Time column at position {j}")
                elif "lower size" in col_str:
                    lower_size_col = j
                    print(f"Found Lower Size column at position {j}")
                elif "upper size" in col_str:
                    upper_size_col = j
                    print(f"Found Upper Size column at position {j}")

        # Use defaults if columns still not found
        if date_col is None:
            date_col = 0  # Assume date is in first column
            print("Date column not found, using first column as default")
        if time_col is None:
            time_col = 1  # Assume time is in second column
            print("Time column not found, using second column as default")

        # Find the header row index (where column names are)
        header_row = None
        for i in range(max_search_rows):
            # Check if this row contains "Date" and "Start Time"
            row_values = [str(x).lower() for x in df.iloc[i, :]]
            if "date" in row_values and any("time" in x for x in row_values):
                header_row = i
                print(f"Found header row at index {i}")
                break

        # If we couldn't find the header row, assume it's the first row
        if header_row is None:
            header_row = 0
            print("Header row not found, using first row as default")

        # Transpose the data, using the header row as column names
        df_t = df.iloc[header_row:].transpose()

        # Set the first row (original header row) as the new column names
        df_t.columns = df_t.iloc[0]

        # Remove the first row (now duplicated in column names)
        df_t = df_t.iloc[1:].reset_index()

        # Identify the size bin columns (numeric column names)
        size_bins = []
        size_columns = []

        for col in df_t.columns:
            try:
                size_val = float(col)
                size_bins.append(size_val)
                size_columns.append(col)
            except (ValueError, TypeError):
                pass

        print(f"Found {len(size_bins)} size bin columns")

        # Get size range if available
        min_size = None
        max_size = None

        if lower_size_col is not None and upper_size_col is not None:
            # Find where these values would be in the transposed data
            try:
                lower_size_values = df_t["Lower Size(nm)"].values
                upper_size_values = df_t["Upper Size(nm)"].values

                # Use the first non-NaN value
                for val in lower_size_values:
                    if pd.notna(val):
                        try:
                            min_size = float(val)
                            break
                        except:
                            pass

                for val in upper_size_values:
                    if pd.notna(val):
                        try:
                            max_size = float(val)
                            break
                        except:
                            pass

                print(f"Size range: {min_size} to {max_size} nm")
            except:
                print("Could not extract size range from columns")

        # If we couldn't find min/max size, use the min/max of size bins
        if min_size is None or max_size is None:
            min_size = min(size_bins)
            max_size = max(size_bins)
            print(f"Using size bin range: {min_size} to {max_size} nm")

        # Create datetime by combining Date and Start Time columns
        # Find the date and time columns in the transposed data
        date_column = None
        time_column = None

        for col in df_t.columns:
            col_str = str(col).lower()
            if "date" in col_str and date_column is None:
                date_column = col
            elif ("start time" in col_str or "time" in col_str) and time_column is None:
                time_column = col

        if date_column is None or time_column is None:
            print("Could not find Date or Time columns in transposed data")
            return None, None, None, None, None, None

        # Convert to datetime - with a fix for the format warning
        try:
            # First convert the date and time columns to strings to handle various formats
            df_t["date_str"] = df_t[date_column].astype(str)
            df_t["time_str"] = df_t[time_column].astype(str)

            # Debug some sample date/time values
            print(f"Sample date values: {df_t['date_str'].head(3).values}")
            print(f"Sample time values: {df_t['time_str'].head(3).values}")

            # Fix for the datetime parsing warning - extract just the date part before combining
            def extract_date_part(date_str):
                # If date string has time component like "2024-05-30 00:00:00", extract just "2024-05-30"
                if " " in date_str:
                    return date_str.split(" ")[0]
                return date_str

            # Apply the date extraction
            df_t["date_part"] = df_t["date_str"].apply(extract_date_part)

            # Now combine and parse the datetime
            df_t["datetime"] = pd.to_datetime(
                df_t["date_part"] + " " + df_t["time_str"], errors="coerce"
            )

            # Debug the parsed datetimes
            print(f"Sample parsed datetimes: {df_t['datetime'].head(3).values}")
            print(f"Unique dates found: {sorted(df_t['datetime'].dt.date.unique())}")

            # Filter for target date
            df_t = df_t[df_t["datetime"].dt.date == target_date].copy()

            if df_t.empty:
                print(f"No data found for target date {target_date}")

                # Check if there's data for other dates and provide that information
                if "datetime" in df_t.columns and not pd.isna(df_t["datetime"]).all():
                    available_dates = sorted(df_t["datetime"].dt.date.unique())
                    print(f"Available dates in file: {available_dates}")

                return None, None, None, None, None, None

            # Sort by datetime
            df_t = df_t.sort_values("datetime").reset_index(drop=True)

            # Calculate midpoint times between consecutive measurements
            df_t["next_datetime"] = df_t["datetime"].shift(-1)
            df_t["mid_datetime"] = df_t.apply(
                lambda row: (
                    row["datetime"] + (row["next_datetime"] - row["datetime"]) / 2
                    if pd.notna(row["next_datetime"])
                    else row["datetime"]
                ),
                axis=1,
            )

            # Convert to hours of the day
            day_start = datetime.datetime.combine(target_date, datetime.time.min)
            df_t["hours"] = [
                (t - day_start).total_seconds() / 3600 for t in df_t["mid_datetime"]
            ]

            # Extract the data values for each size bin
            data_array = np.zeros((len(size_bins), len(df_t)))

            for i, size_col in enumerate(size_columns):
                for j, idx in enumerate(df_t.index):
                    try:
                        val = float(df_t.loc[idx, size_col])
                        data_array[i, j] = val if pd.notna(val) and val > 0 else np.nan
                    except (ValueError, TypeError):
                        data_array[i, j] = np.nan

            # Check if we have any valid data
            non_nan_count = np.count_nonzero(~np.isnan(data_array))
            print(
                f"Data array shape: {data_array.shape}, non-NaN values: {non_nan_count} ({non_nan_count/(data_array.size)*100:.1f}% of data)"
            )

            if non_nan_count == 0:
                print("No valid data found in measurement columns")
                return None, None, None, None, None, None

            # Per-scan geometric mean diameter (nm) from the size distribution.
            # Bins are log-uniform (constant dlogDp), so the size-bin values act
            # as dN weights and dlogDp cancels in the moment integral.
            size_arr = np.asarray(size_bins, dtype=float)
            gmd_trace = np.full(data_array.shape[1], np.nan)
            for j in range(data_array.shape[1]):
                col = data_array[:, j]
                w = np.where(np.isnan(col), 0.0, col)
                w = np.maximum(w, 0.0)
                total = w.sum()
                if total > 0:
                    gmd_trace[j] = np.exp(np.dot(w / total, np.log(size_arr)))

            # Convert to log scale
            log_data = np.log10(np.where(data_array > 0, data_array, np.nan))

            # Return the processed data, time hours, size bins, and GMD trace
            return log_data, df_t["hours"].values, size_bins, min_size, max_size, gmd_trace

        except Exception as e:
            print(f"Error creating datetime and processing data: {e}")
            import traceback

            traceback.print_exc()
            return None, None, None, None, None, None

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None, None, None


def get_burn_events(burn_date):
    """Return event times (in hours of the day) for a burn from the burn log.

    Reads ignition, garage-closed, and CR Box (PAC) activation times from the
    burn log for the given date. Does not invent times: any event missing in
    the log is returned as ``None``.

    Parameters
    ----------
    burn_date : str
        'YYYY-MM-DD'.

    Returns
    -------
    dict
        Keys 'ignition', 'garage_closed', 'cr_box_on' mapped to the event hour
        of the day (float) or None if not recorded.
    """
    events = {"ignition": None, "garage_closed": None, "cr_box_on": None}
    if burn_log is None:
        return events

    d = pd.to_datetime(burn_date).date()
    row = burn_log[pd.to_datetime(burn_log["Date"]).dt.date == d]
    if row.empty:
        return events
    row = row.iloc[0]

    def _to_hours(val):
        if pd.isna(val):
            return None
        t = pd.to_datetime(str(val)).time()
        return t.hour + t.minute / 60.0 + t.second / 3600.0

    events["ignition"] = _to_hours(row.get("Ignition"))
    events["garage_closed"] = _to_hours(row.get("garage closed"))
    events["cr_box_on"] = _to_hours(row.get("CR Box on"))
    return events


def get_decay_window_hours(burn_date, band="Total Concentration (µg/m³)"):
    """Return the post-PAC decay-fit window (start, end) in hours of the day.

    Reads the decay start/end offsets (hours since garage close) that
    clean_air_delivery_rates_pmsizes.py already fit, so the shaded post-PAC
    region matches the reported decay interval. No refitting here.

    Parameters
    ----------
    burn_date : str
        'YYYY-MM-DD'.
    band : str
        Which decay row to read the window from; the total-concentration row
        gives the common interval used across bands.

    Returns
    -------
    tuple of (float or None, float or None)
        (start_hour, end_hour) as hours of the day, or (None, None) if the
        decay file or a matching row is missing.
    """
    try:
        xlsx = get_common_file("burn_calcs") / "SMPS_decay_and_CADR.xlsx"
    except Exception:
        return (None, None)
    if not os.path.exists(xlsx):
        return (None, None)

    events = get_burn_events(burn_date)
    gc = events.get("garage_closed")
    if gc is None:
        return (None, None)

    try:
        decay = pd.read_excel(xlsx)
    except Exception:
        return (None, None)

    burn_id = f"burn{list(BURN_INFO).index(burn_date) + 1}"
    rows = decay[(decay["burn"] == burn_id) & (decay["pollutant"] == band)]
    if rows.empty:
        return (None, None)
    r = rows.iloc[0]
    return (gc + float(r["decay_start_time"]), gc + float(r["decay_end_time"]))


def _report_burn_quality(burn_date, description, data_type, time_hours, events):
    """Print peak, PAC time, and gap checks; stop if a check fails.

    Confirms the record spans the PAC window without a large gap and that the
    PAC activation is recorded. Stops (raises) rather than silently substituting
    another burn.
    """
    gc = events.get("garage_closed")
    pac = events.get("cr_box_on")
    print(f"  {description} ({data_type}) event times (hours of day): "
          f"ignition={events.get('ignition')}, garage_closed={gc}, cr_box_on={pac}")

    if pac is None:
        print(f"  NOTE: no CR Box (PAC) time recorded for {burn_date}; "
              "before/after shading limited to available events.")
        return

    th = np.asarray(time_hours, dtype=float)
    around = np.sort(th[(th >= gc) & (th <= pac + 0.5)]) if gc is not None else np.array([])
    if around.size >= 2:
        max_gap_min = float(np.max(np.diff(around)) * 60.0)
        print(f"  Max scan gap across PAC window: {max_gap_min:.0f} s -> "
              f"{max_gap_min:.1f} min")
        if max_gap_min > 10.0:
            raise ValueError(
                f"Data-quality stop for {burn_date}: {max_gap_min:.1f} min gap "
                "across the PAC window. Inspect the record before plotting."
            )
    else:
        print(f"  WARNING: fewer than 2 scans in the garage->PAC window for "
              f"{burn_date}.")


def save_est_png(data_for_plot, time_hours, size_bins, description, data_type,
                 time_start, time_end, min_size, max_size, gmd_trace, events,
                 decay_window):
    """Render an ES&T Air publication PNG heatmap via src/fig_style.py.

    Uses matplotlib pcolormesh so the number and mass heatmaps for the selected
    burn match the Section 3.2 figure set (Okabe-Ito palette, column widths,
    apply_est_style). The Bokeh HTML output is kept separately.
    """
    import datetime as _dt

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless PNG export
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from fig_style import OKABE_ITO, apply_est_style, figsize

    apply_est_style()

    data = np.asarray(data_for_plot, dtype=float)  # log10 values, shape (nbins, ntimes)
    th = np.asarray(time_hours, dtype=float)
    sb = np.asarray(size_bins, dtype=float)

    # Restrict to the burn window: ~1 h before garage close through ~3.5 h after
    # the decay-fit end, so the recovery period after the event is visible
    # instead of only the first half hour.
    gc = events.get("garage_closed") if events else None
    d1 = decay_window[1] if decay_window else None
    if gc is not None:
        x_lo = gc - 1.0
        x_hi = (d1 + 3.5) if d1 is not None else gc + 4.0
        mask = (th >= x_lo) & (th <= x_hi)
        if mask.sum() >= 2:
            th = th[mask]
            data = data[:, mask]
    else:
        x_lo, x_hi = time_start, time_end

    label = ("Log$_{10}$(d$N$/dlog$D_p$) (#/cm³)" if data_type == "numConc"
             else "Log$_{10}$(d$M$/dlog$D_p$) (µg/m³)")
    title = (f"SMPS number size distribution, {description}"
             if data_type == "numConc"
             else f"SMPS mass size distribution, {description}")

    w, h = figsize("double", aspect=0.5)
    fig, ax = plt.subplots(figsize=(w, h))

    # pcolormesh needs cell edges; build midpoint edges in time and log-size.
    def _edges(vals):
        vals = np.asarray(vals, dtype=float)
        mids = (vals[:-1] + vals[1:]) / 2.0
        first = vals[0] - (mids[0] - vals[0])
        last = vals[-1] + (vals[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]])

    t_edges = _edges(th)
    s_edges = _edges(sb)

    # Clip the color range so the empty low tail stops flattening the palette
    # while keeping the loaded event range legible. Number concentration uses a
    # fixed log10 window of -2 to 3.5; mass keeps the data-driven range.
    if data_type == "numConc":
        vmin = -2.0
        vmax = 3.5
    else:
        vmin = float(np.nanmin(data)) if np.any(np.isfinite(data)) else 0.0
        vmax = float(np.nanmax(data)) if np.any(np.isfinite(data)) else vmin + 1.0

    mesh = ax.pcolormesh(
        t_edges, s_edges, np.ma.masked_invalid(data), cmap="turbo",
        shading="auto", vmin=vmin, vmax=vmax,
    )
    ax.set_yscale("log")
    ax.set_ylim(min_size, max_size)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Particle diameter (nm)")
    ax.set_title(title)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(label)

    ax.set_yticks([10, 20, 50, 100, 200, 400])
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in [10, 20, 50, 100, 200, 400]]))

    # Pre-PAC (garage->PAC) and post-PAC (decay window) shading. Labels are
    # rotated 90 deg, centered across each band and anchored near the bottom of
    # the panel so the narrow pre-PAC band does not overlap the post-PAC label.
    if events is not None:
        pac = events.get("cr_box_on")
        y_lab = min_size * 1.15  # label height near the panel floor (log axis)
        if gc is not None and pac is not None and pac > gc:
            ax.axvspan(gc, pac, color="#999999", alpha=0.35, zorder=1)
            ax.text((gc + pac) / 2.0, y_lab, "Pre-PAC", ha="center", va="bottom",
                    rotation=90, fontsize=8, fontweight="bold", color="black",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.7))
        if decay_window is not None:
            d0, d1w = decay_window
            if d0 is not None and d1w is not None and d1w > d0:
                ax.axvspan(d0, d1w, color=OKABE_ITO["blue"], alpha=0.28, zorder=1)
                ax.text((d0 + d1w) / 2.0, y_lab, "Post-PAC decay",
                        ha="center", va="bottom", rotation=90, fontsize=8,
                        fontweight="bold", color="black", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))

    # GMD overlay, aligned to the (possibly trimmed) time axis. Drawn in a
    # distinct color (Okabe-Ito sky blue) so it never blends with the black
    # event lines.
    if gmd_trace is not None:
        g_full = np.asarray(gmd_trace, dtype=float)
        t_full = np.asarray(time_hours, dtype=float)
        gm_mask = (t_full >= th[0]) & (t_full <= th[-1]) & np.isfinite(g_full)
        if gm_mask.any():
            ax.plot(t_full[gm_mask], g_full[gm_mask], color=OKABE_ITO["skyblue"],
                    linewidth=2.2, label="Geometric mean diameter", zorder=5)

    # Event lines: all black, same width, distinguished only by line style.
    if events is not None:
        for key, style, elabel in [
            ("ignition", ":", "Ignition"),
            ("garage_closed", "-", "Garage closed"),
            ("cr_box_on", "--", "CR Box on"),
        ]:
            hh = events.get(key)
            if hh is not None and th[0] <= hh <= th[-1]:
                ax.axvline(hh, color="black", linestyle=style, linewidth=1.6,
                           label=elabel, zorder=4)

    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    output_dir = get_common_file("output_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"SMPS_{description}_{data_type}_heatmap.png"
    if out.exists():
        out = out.with_name(f"{out.stem}_{_dt.date.today().isoformat()}{out.suffix}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ES&T PNG saved: {out}")


def create_heatmap(
    data_for_plot,
    time_hours,
    size_bins,
    description,
    data_min=None,
    data_max=None,
    data_type="numConc",
    time_start=0,
    time_end=24,
    min_size=None,
    max_size=None,
    gmd_trace=None,
    events=None,
    decay_window=None,
):
    """
    Create a heatmap visualization for SMPS data

    Args:
        data_for_plot (numpy.ndarray): Log-transformed data array
        time_hours (numpy.ndarray): Array of time points in hours
        size_bins (list): List of particle size values
        description (str): Description of the burn
        data_min (float, optional): Minimum value for color scale
        data_max (float, optional): Maximum value for color scale
        data_type (str, optional): Type of data ('numConc' or 'MassConc')
        time_start (float, optional): Start time for x-axis in hours (default: 0)
        time_end (float, optional): End time for x-axis in hours (default: 24)
        min_size (float, optional): Minimum size for y-axis
        max_size (float, optional): Maximum size for y-axis
        gmd_trace (numpy.ndarray, optional): Per-scan geometric mean diameter
            (nm) to overlay on the heatmap.
        events (dict, optional): Event hours ('ignition', 'garage_closed',
            'cr_box_on') to mark with vertical Spans.
        decay_window (tuple, optional): (start_hour, end_hour) of the post-PAC
            decay-fit window to shade as the post-PAC period.

    Returns:
        layout: A Bokeh layout object containing the heatmap and metadata
    """
    # Check for NaN values in time_hours and size_bins
    time_hours = np.array(time_hours)
    size_bins = np.array(size_bins)

    # Ensure we have valid time_hours and size_bins
    if len(time_hours) == 0:
        print("Warning: No time points found. Using default time range.")
        time_hours = np.linspace(time_start, time_end, data_for_plot.shape[1])

    if len(size_bins) == 0:
        print("Warning: No size bins found. Using default size range.")
        size_bins = np.logspace(1, 3, data_for_plot.shape[0])  # 10 to 1000 nm

    # If min_size and max_size are not provided, use size_bins
    if min_size is None:
        min_size = np.min(size_bins)
    if max_size is None:
        max_size = np.max(size_bins)

    # Set title and legend label based on data type
    if data_type == "numConc":
        title = f"SMPS Particle Number Concentration for {description}"
        legend_label = "Log₁₀(dN/dlogDp) (#/cm³)"
    else:  # MassConc
        title = f"SMPS Particle Mass Concentration for {description}"
        legend_label = "Log₁₀(dM/dlogDp) (µg/m³)"

    # Get min and max values if not provided
    if data_min is None or np.isnan(data_min):
        data_min = (
            float(np.nanmin(data_for_plot)) if np.any(~np.isnan(data_for_plot)) else 0
        )
    if data_max is None or np.isnan(data_max):
        data_max = (
            float(np.nanmax(data_for_plot)) if np.any(~np.isnan(data_for_plot)) else 1
        )

    # Determine the actual data time range
    min_time = np.min(time_hours)
    max_time = np.max(time_hours)

    # Debug output
    print(f"Data time range: {min_time} to {max_time}")
    print(f"Using fixed display range: {time_start} to {time_end}")
    print(f"Size range: {min_size} to {max_size}")
    print(f"Data range: {data_min} to {data_max}")

    # Create figure with fixed time range
    p = figure(
        width=800,
        height=500,
        title=title,
        x_axis_label="Hours of the day",
        y_axis_label="Diameter (nm)",
        x_range=(time_start, time_end),  # Use fixed time range
        y_range=(min_size, max_size),
        x_axis_type="linear",
        y_axis_type="log",
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )

    # Create color mapper with appropriate color palette
    color_mapper = LinearColorMapper(
        palette=Turbo256,
        low=data_min,
        high=data_max,
        nan_color="white",  # Use gray for NaN values
    )

    # Add heatmap image using the data's actual time range
    p.image(
        image=[data_for_plot],
        x=min_time,
        y=min_size,
        dw=max_time - min_time,
        dh=max_size - min_size,
        color_mapper=color_mapper,
    )

    # Add color bar
    color_bar = ColorBar(
        color_mapper=color_mapper,
        title=legend_label,
        title_text_font_size="10pt",
        title_text_font_style="normal",
        title_standoff=12,
        width=15,
        location=(0, 0),
    )

    p.add_layout(color_bar, "right")

    # Shade the pre-PAC infiltration window (garage close -> PAC on) and the
    # post-PAC decay window, using the event times from the burn log. No new
    # times are invented here.
    if events is not None:
        gc = events.get("garage_closed")
        pac = events.get("cr_box_on")
        if gc is not None and pac is not None and pac > gc:
            p.add_layout(
                BoxAnnotation(
                    left=gc, right=pac, fill_color="#999999", fill_alpha=0.18,
                    level="underlay",
                )
            )
        if decay_window is not None:
            d0, d1 = decay_window
            if d0 is not None and d1 is not None and d1 > d0:
                p.add_layout(
                    BoxAnnotation(
                        left=d0, right=d1, fill_color="#0072B2", fill_alpha=0.12,
                        level="underlay",
                    )
                )

    # Overlay the geometric mean diameter trace so the mode shift across PAC
    # activation is visible.
    if gmd_trace is not None:
        gmd_arr = np.asarray(gmd_trace, dtype=float)
        finite = np.isfinite(gmd_arr) & np.isfinite(np.asarray(time_hours))
        if finite.any():
            p.line(
                np.asarray(time_hours)[finite],
                gmd_arr[finite],
                line_color="black",
                line_width=2,
                legend_label="Geometric mean diameter",
            )

    # Vertical event lines (ignition, garage closed, PAC/CR Box on).
    if events is not None:
        event_styles = [
            ("ignition", "#E69F00", "dotted", "Ignition"),
            ("garage_closed", "black", "solid", "Garage closed"),
            ("cr_box_on", "#D55E00", "dashed", "CR Box on"),
        ]
        for key, color, dash, label in event_styles:
            h = events.get(key)
            if h is not None:
                p.add_layout(
                    Span(
                        location=h, dimension="height", line_color=color,
                        line_dash=dash, line_width=2,
                    )
                )

    if p.legend:
        p.legend.location = "top_right"
        p.legend.label_text_font_size = "9pt"
        p.legend.background_fill_alpha = 0.7


    # Custom y-axis tickers (logarithmic)
    p.yaxis.ticker = [10, 20, 40, 60, 80, 100, 200, 400]

    # Customize appearance
    p.title.text_font = "Calibri"
    p.title.text_font_size = "14pt"
    p.title.align = "center"

    p.xaxis.axis_label_text_font = "Calibri"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font = "Calibri"
    p.xaxis.major_label_text_font_size = "12pt"

    p.yaxis.axis_label_text_font = "Calibri"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font = "Calibri"
    p.yaxis.major_label_text_font_size = "12pt"

    # Add metadata
    metadata = get_script_metadata()
    text_div = Div(text=f"<small>{metadata}</small>", width=800)

    # Create layout
    layout = column(p, text_div)

    return layout


def process_single_burn(burn_date, data_type="numConc", time_start=0, time_end=24):
    """
    Process a single burn and create heatmap

    Args:
        burn_date (str): Date string in 'YYYY-MM-DD' format
        data_type (str): Type of data to process ('numConc' or 'MassConc')
        time_start (float): Start time for x-axis in hours (default: 0)
        time_end (float): End time for x-axis in hours (default: 24)
    """
    # Get burn info
    burn_info = BURN_INFO.get(burn_date)
    if burn_info is None:
        print(f"No burn info found for date {burn_date}")
        return

    description = burn_info["description"]
    file_suffix = burn_info["file_suffix"]

    print(f"\n{'='*80}\nProcessing {description} from {burn_date}...")

    # Construct file path
    file_path = f"./burn_data/smps/MH_apollo_bed_{file_suffix}_{data_type}.xlsx"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    # Prepare data
    data_for_plot, time_hours, size_bins, min_size, max_size, gmd_trace = prepare_data(
        file_path, burn_date
    )

    if data_for_plot is None or time_hours is None or size_bins is None:
        print(f"Skipping {description} due to data preparation error")
        return

    # Debug the data
    print(f"\nData shape: {data_for_plot.shape}")
    non_nan_count = np.count_nonzero(~np.isnan(data_for_plot))
    print(
        f"Non-NaN values: {non_nan_count} ({non_nan_count/(data_for_plot.size)*100:.1f}% of data)"
    )

    if non_nan_count == 0:
        print(f"Skipping {description} - all data values are NaN")
        return

    # Get data range for this burn
    data_min = float(np.nanmin(data_for_plot))
    data_max = float(np.nanmax(data_for_plot))

    print(f"Successfully processed data for {description}")
    print(f"Range: {data_min:.2f} to {data_max:.2f}")
    print(f"Time points: {len(time_hours)}, Size bins: {len(size_bins)}")

    # Event times and post-PAC decay window for the before/after shading.
    events = get_burn_events(burn_date)
    decay_window = get_decay_window_hours(burn_date)
    _report_burn_quality(burn_date, description, data_type, time_hours, events)

    # Reset Bokeh output for this plot
    reset_output()
    if _SHOW_INLINE:
        output_notebook()

    # Create heatmap
    try:
        print(f"\nCreating heatmap for {description}...")

        heatmap = create_heatmap(
            data_for_plot,
            time_hours,
            size_bins,
            description,
            data_min,
            data_max,
            data_type,
            time_start,
            time_end,
            min_size,
            max_size,
            gmd_trace=gmd_trace,
            events=events,
            decay_window=decay_window,
        )

        # First show the figure in the notebook
        _maybe_show(heatmap)

        # Then save the figure to file
        output_file_path = (
            f"./Paper_figures/SMPS_{description}_{data_type}_heatmap.html"
        )
        output_file(output_file_path)
        save(heatmap, filename=output_file_path)
        print(f"Saved figure to {output_file_path}")

        # Publication PNG via the shared ES&T Air style module.
        save_est_png(
            data_for_plot, time_hours, size_bins, description, data_type,
            time_start, time_end, min_size, max_size, gmd_trace, events,
            decay_window,
        )

        print(f"Completed {description} heatmap")
        return True
    except Exception as e:
        print(f"Error creating heatmap for {description}: {e}")
        import traceback

        traceback.print_exc()
        return False


def process_all_burns(data_type="numConc", time_start=0, time_end=24):
    """
    Process all burns and create heatmaps

    Args:
        data_type (str): Type of data to process ('numConc' or 'MassConc')
        time_start (float): Start time for x-axis in hours (default: 0)
        time_end (float): End time for x-axis in hours (default: 24)
    """
    print(f"Processing SMPS {data_type} data for all burns...")
    print(f"Using fixed time range: {time_start} to {time_end} hours")

    # Lists to store processed data
    all_data_for_plot = []
    all_time_hours = []
    all_size_bins = []
    all_descriptions = []
    all_min_sizes = []
    all_max_sizes = []
    all_gmd_traces = []
    all_events = []
    all_decay_windows = []
    all_burn_dates = []
    data_min, data_max = float("inf"), float("-inf")

    # First pass - collect data and determine global range
    for burn_date, burn_info in BURN_INFO.items():
        description = burn_info["description"]
        file_suffix = burn_info["file_suffix"]

        print(f"\n{'='*80}\nProcessing {description} from {burn_date}...")

        # Construct file path
        file_path = f"./burn_data/smps/MH_apollo_bed_{file_suffix}_{data_type}.xlsx"

        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            continue

        # Prepare data
        data_for_plot, time_hours, size_bins, min_size, max_size, gmd_trace = prepare_data(
            file_path, burn_date
        )

        if data_for_plot is None or time_hours is None or size_bins is None:
            print(f"Skipping {description} due to data preparation error")
            continue

        # Debug the data
        print(f"\nData shape: {data_for_plot.shape}")
        non_nan_count = np.count_nonzero(~np.isnan(data_for_plot))
        print(
            f"Non-NaN values: {non_nan_count} ({non_nan_count/(data_for_plot.size)*100:.1f}% of data)"
        )

        if non_nan_count == 0:
            print(f"Skipping {description} - all data values are NaN")
            continue

        # Update min and max values (with explicit check for all-NaN)
        if non_nan_count > 0:
            current_min = float(np.nanmin(data_for_plot))
            current_max = float(np.nanmax(data_for_plot))

            # Only update global min/max if we have real data
            data_min = min(data_min, current_min)
            data_max = max(data_max, current_max)

            # Store processed data
            all_data_for_plot.append(data_for_plot)
            all_time_hours.append(time_hours)
            all_size_bins.append(size_bins)
            all_min_sizes.append(min_size)
            all_max_sizes.append(max_size)
            all_descriptions.append(description)
            all_gmd_traces.append(gmd_trace)
            all_events.append(get_burn_events(burn_date))
            all_decay_windows.append(get_decay_window_hours(burn_date))
            all_burn_dates.append(burn_date)

            print(f"Successfully processed data for {description}")
            print(f"Range: {current_min:.2f} to {current_max:.2f}")
            print(f"Time points: {len(time_hours)}, Size bins: {len(size_bins)}")

    # Check if we have any data to plot
    if not all_data_for_plot:
        print(f"No data could be processed for any burns ({data_type})")
        return

    print(f"\n{'='*80}\nGlobal color scale range: {data_min:.2f} to {data_max:.2f}")

    # Second pass - create heatmaps with global color scale
    for i, (
        data_for_plot,
        time_hours,
        size_bins,
        min_size,
        max_size,
        description,
        gmd_trace,
        events,
        decay_window,
        burn_date,
    ) in enumerate(
        zip(
            all_data_for_plot,
            all_time_hours,
            all_size_bins,
            all_min_sizes,
            all_max_sizes,
            all_descriptions,
            all_gmd_traces,
            all_events,
            all_decay_windows,
            all_burn_dates,
        )
    ):

        # Reset Bokeh output for each new plot
        reset_output()
        if _SHOW_INLINE:
            output_notebook()

        print(f"\nCreating heatmap for {description}...")

        # Create heatmap
        try:
            heatmap = create_heatmap(
                data_for_plot,
                time_hours,
                size_bins,
                description,
                data_min,
                data_max,
                data_type,
                time_start,
                time_end,
                min_size,
                max_size,
                gmd_trace=gmd_trace,
                events=events,
                decay_window=decay_window,
            )

            # First show the plot in the notebook
            _maybe_show(heatmap)

            # Then save to file
            output_file_path = (
                f"./Paper_figures/SMPS_{description}_{data_type}_heatmap.html"
            )
            output_file(output_file_path)
            save(heatmap, filename=output_file_path)
            print(f"Saved figure to {output_file_path}")

            # Publication PNG via the shared ES&T Air style module.
            save_est_png(
                data_for_plot, time_hours, size_bins, description, data_type,
                time_start, time_end, min_size, max_size, gmd_trace, events,
                decay_window,
            )

            print(f"Completed {description} heatmap")
        except Exception as e:
            print(f"Error creating heatmap for {description}: {e}")
            import traceback

            traceback.print_exc()


# Execute main function
if __name__ == "__main__":
    import argparse

    # Define time range to use for all plots (0-24 hours by default)
    time_start = 0  # Start time in hours (e.g., 0 = midnight)
    time_end = 24  # End time in hours (e.g., 24 = midnight next day)

    parser = argparse.ArgumentParser(
        description="SMPS size-distribution heatmaps with event lines, GMD "
        "overlay, and pre/post-PAC shading."
    )
    parser.add_argument(
        "--burn", default="2024-05-31",
        help="Burn date 'YYYY-MM-DD' for the before/after-PAC figure "
        "(default 2024-05-31, Burn 10). Use --all to process every burn.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all burns (keeps the original CLI behavior).",
    )
    args = parser.parse_args()

    if args.all:
        # Original behavior: all burns, both data types.
        process_all_burns(data_type="numConc", time_start=time_start, time_end=time_end)
        process_all_burns(data_type="MassConc", time_start=time_start, time_end=time_end)
    else:
        # Selected burn (default Burn 10): number then mass.
        process_single_burn(args.burn, "numConc", time_start, time_end)
        process_single_burn(args.burn, "MassConc", time_start, time_end)

# %%
