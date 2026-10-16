"""
WUI Peak Concentration Analysis: Multi-Instrument Time Series Processing

Analyzes peak particulate matter concentrations during wildland-urban interface
(WUI) smoke injection experiments across 10 burns. Processes data from multiple
instruments to identify maximum concentrations and generate comparative plots.

Instruments:
    - AeroTrak (Bedroom/Kitchen): Size-resolved PM (0.5-25 µm)
    - DustTrak: PM1, PM2.5, PM4, PM10, PM15
    - MiniAMS: Chemical species (Burns 1-3 only)
    - PurpleAir: PM2.5 (Burns 6-10 only)
    - QuantAQ (Bedroom/Kitchen): PM1, PM2.5, PM10 (Burns 4-10 only)
    - SMPS: Size-resolved particle concentrations

Workflow:
    1. Load and synchronize instrument data with time shift corrections
    2. Filter data by burn date and time windows
    3. Identify peak concentrations during smoke injection
    4. Generate Bokeh plots comparing all instruments for selected burn

Outputs:
    - peak_concentrations_all_instruments.xlsx: Peak values for all burns
    - YYYY-MM-DD_{burn_id}_PM25_comparison_morning_room.html/.pdf/.png:
      Morning Room time series (interactive HTML, journal-style PDF and PNG)

Configuration:
    - BURN_TO_PLOT: Selected burn for visualization (default: "burn9")
    - INSTRUMENT_CONFIG: File paths, time shifts, and pollutant lists

Notes:
    - Times normalized to "garage door closed" event
    - Concentrations in µg/m³ (mass) or #/cm³ (number)
    - Burn 3 uses 5-min rolling average; Burn 6 has custom decay window

Author: Nathan Lima
Date: 2025
Update log:
    2026-10-16: Burn comparison now plots the Morning Room instruments only
        (rooms argument), labels units 1/2 (AeroTrak2, MODULAIR-PM2), uses the
        shared Wong palette with a line style per instrument, saves the HTML
        instead of opening it, and adds a static journal-style PDF/PNG.
"""

# %% IMPORT MODULES
import os
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save
from bokeh.io import output_file
from bokeh.models import Range1d, Div
from bokeh.layouts import column

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_instrument_path, get_common_file
from src.plot_style import apply_bokeh_text, series_style
from src.plot_style import figsize as journal_figsize
from src.project_style import (
    AXIS_LABELS,
    INSTRUMENT_COLORS,
    INSTRUMENT_LINESTYLES,
    REF_LINE,
    dated_stem,
    journal_style,
    save_pdf_png,
)


# Set the absolute path for the dataset
data_root = get_data_root()  # Portable path - auto-configured
os.chdir(str(data_root))

# Import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from scripts import get_script_metadata  # pylint: disable=import-error,wrong-import-position

# Variable to set which burn to plot
BURN_TO_PLOT = "burn9"

# Load burn log once
BURN_LOG_PATH = str(get_common_file('burn_log'))
burn_log = pd.read_excel(BURN_LOG_PATH, sheet_name="Sheet2")

# Define instrument configurations
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "file_path": str(get_instrument_path('aerotrak_bedroom') / 'all_data.xlsx'),
        "process_function": "process_aerotrak_data",
        "time_shift": 2.16,
        "process_pollutants": [
            "PM0.5 (µg/m³)",
            "PM1 (µg/m³)",
            "PM3 (µg/m³)",
            "PM5 (µg/m³)",
            "PM10 (µg/m³)",
            "PM25 (µg/m³)",
        ],
        "datetime_column": "Date and Time",
        "special_cases": {
            "burn3": {"apply_rolling_average": True},
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25},
        },
    },
    "AeroTrakK": {
        "file_path": str(get_instrument_path('aerotrak_kitchen') / 'all_data.xlsx'),
        "process_function": "process_aerotrak_data",
        "time_shift": 5,
        "process_pollutants": [
            "PM0.5 (µg/m³)",
            "PM1 (µg/m³)",
            "PM3 (µg/m³)",
            "PM5 (µg/m³)",
            "PM10 (µg/m³)",
            "PM25 (µg/m³)",
        ],
        "datetime_column": "Date and Time",
        "special_cases": {},
    },
    "DustTrak": {
        "file_path": "./burn_data/dusttrak/all_data.xlsx",
        "process_function": "process_dusttrak_data",
        "time_shift": 7,
        "process_pollutants": [
            "PM1 (µg/m³)",
            "PM2.5 (µg/m³)",
            "PM4 (µg/m³)",
            "PM10 (µg/m³)",
            "PM15 (µg/m³)",
        ],
        "datetime_column": "datetime",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
    "MiniAMS": {
        "file_path": "./burn_data/mini-ams/WUI_AMS_Species.xlsx",
        "process_function": "process_miniams_data",
        "time_shift": 0,
        "process_pollutants": [
            "Organic (µg/m³)",
            "Nitrate (µg/m³)",
            "Sulfate (µg/m³)",
            "Ammonium (µg/m³)",
            "Chloride (µg/m³)",
        ],
        "datetime_column": "DateTime",
        "burn_range": range(1, 4),  # Burns 1-3 only
        "special_cases": {},
    },
    "PurpleAirK": {
        "file_path": "./burn_data/purpleair/garage-kitchen.xlsx",
        "process_function": "process_purpleairk_data",
        "time_shift": 0,
        "process_pollutants": ["PM2.5 (µg/m³)"],
        "datetime_column": "DateTime",
        "burn_range": range(6, 11),
        "special_cases": {},
    },
    "QuantAQB": {
        "file_path": str(get_instrument_path('quantaq_bedroom') / 'MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv'),
        "process_function": "process_quantaq_data",
        "time_shift": -2.97,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
    "QuantAQK": {
        "file_path": str(get_instrument_path('quantaq_kitchen') / 'MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv'),
        "process_function": "process_quantaq_data",
        "time_shift": 0,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "special_cases": {},
    },
    "SMPS": {
        "file_path": "./burn_data/smps",
        "process_function": "process_smps_data",
        "time_shift": 0,
        "process_pollutants": ["Total Concentration (µg/m³)"],
        "datetime_column": "datetime",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
}


# Utility function to create timezone-naive datetime
def create_naive_datetime(date_str, time_str):
    """Create a timezone-naive datetime object from date and time strings"""
    dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
    if pd.notna(dt) and hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


# Modified apply_time_shift function
def apply_time_shift(df, instrument, _burn_id, burn_date):
    """Apply time shift based on instrument configuration"""
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
    datetime_column = INSTRUMENT_CONFIG[instrument].get(
        "datetime_column", "Date and Time"
    )

    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
    burn_date = pd.to_datetime(burn_date).date()

    if time_shift != 0:
        mask = df[datetime_column].dt.date == burn_date
        if mask.any():
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


# Helper function to filter data by burn dates
def filter_by_burn_dates(data, burn_range, datetime_column):
    """Filter data to only include rows from specified burn dates

    Args:
        data: DataFrame to filter
        burn_range: Range of burn numbers to include
        datetime_column: Name of the datetime column in the data

    Returns:
        Filtered DataFrame containing only data from the specified burn dates
    """
    burn_ids = [f"burn{i}" for i in burn_range]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)

    if datetime_column in data.columns:
        data["Date"] = pd.to_datetime(data[datetime_column]).dt.date
        return data[data["Date"].isin(burn_dates.dt.date)]
    else:
        raise KeyError(f"Column '{datetime_column}' not found in the dataset.")


# Function to process AeroTrak data
def process_aerotrak_data(file_path, instrument="AeroTrakB"):
    """Process AeroTrak data with standardized output format"""
    aerotrak_data = pd.read_excel(file_path)
    aerotrak_data.columns = aerotrak_data.columns.str.strip()

    # Define size channels and initialize a dictionary for size values
    size_channels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"]
    size_values = {}

    # Extract size values for each channel
    for channel in size_channels:
        size_col = f"{channel} Size (µm)"
        if size_col in aerotrak_data.columns:
            size_value = aerotrak_data[size_col].iloc[0]
            if pd.notna(size_value):
                size_values[channel] = size_value

    # Check for the volume column and convert it to cm³
    volume_column = "Volume (L)"
    volume_cm = None
    if volume_column in aerotrak_data.columns:
        aerotrak_data["Volume (cm³)"] = aerotrak_data[volume_column] * 1000
        volume_cm = aerotrak_data["Volume (cm³)"]

    def g_mean(x):
        a = np.log(x)
        return np.exp(a.mean())

    # Initialize new columns for mass concentration and calculate values
    pm_columns = []
    for i, channel in enumerate(size_channels):
        if channel in size_values:
            next_channel = size_channels[i + 1] if i < len(size_channels) - 1 else None
            next_size_value = size_values[next_channel] if next_channel else 25

            # Calculate the geometric mean of the size range
            particle_size = g_mean([size_values[channel], next_size_value])
            particle_size_m = particle_size * 1e-6  # Convert size from µm to m

            # Initialize variable for this iteration
            new_diff_col_µg_m3 = f"PM{size_values[channel]}-{next_size_value} Diff (µg/m³)"

            diff_col = f"{channel} Diff (#)"
            if diff_col in aerotrak_data.columns and volume_cm is not None:
                particle_counts = aerotrak_data[diff_col]

                # Calculate the volume of a single particle
                radius_m = particle_size_m / 2
                volume_per_particle = (4 / 3) * np.pi * (radius_m**3)  # Volume in m³

                # Calculate particle mass density (1 g/cm³)
                particle_mass = volume_per_particle * 1e6 * 1e6  # Convert to µg

                # Create new column for mass concentration in µg/m³
                aerotrak_data[new_diff_col_µg_m3] = (
                    particle_counts / (volume_cm * 1e-6)
                ) * (particle_mass)
                pm_columns.append(new_diff_col_µg_m3)

            # Handle cumulative counts for PM concentrations
            cumul_col = f"{channel} Cumul (#)"
            if cumul_col in aerotrak_data.columns and new_diff_col_µg_m3 in aerotrak_data.columns:
                # Create new PM concentration column from the Diff column
                pm_column_name = f"PM{next_size_value} (µg/m³)"
                aerotrak_data[pm_column_name] = aerotrak_data[new_diff_col_µg_m3]

    # Define cumulative PM concentration columns
    cumulative_columns = [
        "PM0.5 (µg/m³)",
        "PM1 (µg/m³)",
        "PM3 (µg/m³)",
        "PM5 (µg/m³)",
        "PM10 (µg/m³)",
        "PM25 (µg/m³)",
    ]

    # Calculate cumulative PM concentrations
    for i, col in enumerate(cumulative_columns):
        if i == 0:
            aerotrak_data[col] = aerotrak_data[pm_columns[i]]
        else:
            aerotrak_data[col] = aerotrak_data[pm_columns[i]].add(
                aerotrak_data[cumulative_columns[i - 1]], fill_value=0
            )

    # Replace invalid entries with NaN for numeric columns only
    status_columns = ["Flow Status", "Laser Status"]
    valid_status = (aerotrak_data[status_columns] == "OK").all(axis=1)
    for col in list(aerotrak_data.columns):
        if pd.api.types.is_numeric_dtype(aerotrak_data[col]) and col not in [
            "Date and Time",
            "Sample Time",
            "Volume (L)",
        ]:
            aerotrak_data.loc[~valid_status, col] = pd.NA

    # Filter based on burn dates
    burn_ids = [f"burn{i}" for i in range(1, 11)]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)

    # Convert 'Date and Time' to date and filter AeroTrak data for the burn dates
    aerotrak_data["Date"] = pd.to_datetime(aerotrak_data["Date and Time"]).dt.date
    filtered_aerotrak_data = aerotrak_data[
        aerotrak_data["Date"].isin(burn_dates.dt.date)
    ]
    filtered_aerotrak_data = filtered_aerotrak_data.copy()

    # Apply time shift for each burn ID in the filtered data
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_aerotrak_data = apply_time_shift(
                filtered_aerotrak_data, instrument, burn_id, burn_date
            )

    # Check if there's a special case for burn3 rolling average
    special_cases = INSTRUMENT_CONFIG[instrument].get("special_cases", {})
    if "burn3" in special_cases and special_cases["burn3"].get(
        "apply_rolling_average", False
    ):
        filtered_aerotrak_data = calculate_rolling_average_burn3(filtered_aerotrak_data)

    return filtered_aerotrak_data


# Function to calculate 5-minute rolling average for burn3
def calculate_rolling_average_burn3(data):
    """Calculate 5-minute rolling average for burn3 data to reduce noise

    Args:
        data: DataFrame containing AeroTrak data for multiple burns

    Returns:
        DataFrame with rolling average applied to burn3 data
    """
    burn3_date = burn_log[burn_log["Burn ID"] == "burn3"]["Date"].values[0]
    burn3_date = pd.to_datetime(burn3_date).date()

    burn3_data = data[data["Date"] == burn3_date]
    if burn3_data.empty:
        return data

    burn3_data = burn3_data.set_index("Date and Time")
    rolling_avg_data = {}

    numeric_columns = burn3_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        rolling_avg_data[col] = (
            burn3_data[col]
            .rolling(pd.Timedelta(minutes=5))
            .mean()
            .astype(burn3_data[col].dtype)
        )

    status_columns = ["Flow Status", "Instrument Status", "Laser Status"]
    for col in status_columns:
        if col in burn3_data.columns:
            rolling_avg_data[col] = burn3_data[col].iloc[0]

    rolling_avg_df = pd.DataFrame(rolling_avg_data, index=burn3_data.index)
    rolling_avg_df.reset_index(inplace=True)

    data.loc[
        data["Date"] == burn3_date,
        rolling_avg_df.columns.difference(["Date", "Date and Time"]),
    ] = rolling_avg_df[
        rolling_avg_df.columns.difference(["Date", "Date and Time"])
    ].values
    return data


# Function to process DustTrak data
def process_dusttrak_data(file_path, instrument="DustTrak"):
    """Process DustTrak data with standardized output format"""
    dusttrak_data = pd.read_excel(file_path)
    dusttrak_data.columns = dusttrak_data.columns.str.strip()

    # Specify the columns that need unit conversion (from [mg/m³] to (µg/m³))
    pm_columns = [
        "PM1 [mg/m3]",
        "PM2.5 [mg/m3]",
        "PM4 [mg/m3]",
        "PM10 [mg/m3]",
        "TOTAL [mg/m3]",
    ]

    for col in pm_columns:
        if col in dusttrak_data.columns:
            if col == "TOTAL [mg/m3]":
                new_col_name = "PM15 (µg/m³)"
            else:
                new_col_name = col.replace("[mg/m3]", "(µg/m³)")
            dusttrak_data[new_col_name] = dusttrak_data[col] * 1000

    # Filter by burn dates using 'datetime' column
    filtered_data = filter_by_burn_dates(dusttrak_data, range(1, 11), "datetime")

    # Apply time shift for each burn
    burn_ids = [f"burn{i}" for i in range(1, 11)]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    return filtered_data


def process_miniams_data(file_path, instrument="MiniAMS"):
    """Process Mini-AMS data with standardized output format"""
    miniams_data = pd.read_excel(file_path)
    miniams_data.columns = miniams_data.columns.str.strip()

    # Rename columns to standard format with units
    column_mapping = {
        "Org": "Organic (µg/m³)",
        "NO3": "Nitrate (µg/m³)",
        "SO4": "Sulfate (µg/m³)",
        "NH4": "Ammonium (µg/m³)",
        "Chl": "Chloride (µg/m³)",
    }
    miniams_data.rename(columns=column_mapping, inplace=True)

    # Convert DateTime to proper datetime format
    miniams_data["DateTime"] = pd.to_datetime(miniams_data["DateTime"], errors="coerce")

    # Get burn range from instrument configuration (burns 1-3)
    burn_range = INSTRUMENT_CONFIG[instrument].get("burn_range", range(1, 4))

    # Filter by burn dates
    filtered_data = filter_by_burn_dates(miniams_data, burn_range, "DateTime")

    # Apply time shift for each burn (0 minutes for Mini-AMS)
    burn_ids = [f"burn{i}" for i in burn_range]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    return filtered_data


# Function to process PurpleAirK data
def process_purpleairk_data(file_path):
    """Process PurpleAirK data with standardized output format"""
    purpleair_data = pd.read_excel(file_path, sheet_name="(P2)kitchen")
    purpleair_data.columns = purpleair_data.columns.str.strip()

    # Rename 'Average' column to 'PM2.5 (µg/m³)'
    purpleair_data.rename(columns={"Average": "PM2.5 (µg/m³)"}, inplace=True)

    # Filter by burn dates using 'DateTime' column (burns 6-10 for PurpleAir)
    return filter_by_burn_dates(purpleair_data, range(6, 11), "DateTime")


# Function to process SMPS data
def process_smps_data(file_path, instrument="SMPS"):
    """Process SMPS data with standardized output format"""
    combined_smps_data = pd.DataFrame()

    # Get burn dates from burn log
    burn_ids = [f"burn{i}" for i in range(1, 11)]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)

    # Initialize dynamic bin ranges
    bin_ranges = []
    bin_columns = []

    # Process each burn date
    for burn_date in burn_dates:
        try:
            date_str = burn_date.strftime("%m%d%Y")
            smps_filename = f"MH_apollo_bed_{date_str}_MassConc.xlsx"
            smps_file_path = os.path.join(file_path, smps_filename)

            if not os.path.exists(smps_file_path):
                continue

            # Read the first sheet of the Excel file
            smps_data = pd.read_excel(smps_file_path, sheet_name=0)

            # Transpose the data if it's not already in the right format
            if (
                "Date" not in smps_data.columns
                and "Start Time" not in smps_data.columns
            ):
                smps_data = smps_data.transpose()
                smps_data.columns = smps_data.iloc[0].values
                smps_data = smps_data.iloc[1:].reset_index(drop=True)

            # Ensure key columns exist
            required_columns = [
                "Date",
                "Start Time",
                "Total Concentration(µg/m³)",
                "Lower Size(nm)",
                "Upper Size(nm)",
            ]

            missing_columns = [
                col for col in required_columns if col not in smps_data.columns
            ]
            if missing_columns:
                continue

            # Rename 'Total Concentration(µg/m³)' to 'Total Concentration (µg/m³)' to add space
            if "Total Concentration(µg/m³)" in smps_data.columns:
                smps_data = smps_data.rename(
                    columns={
                        "Total Concentration(µg/m³)": "Total Concentration (µg/m³)"
                    }
                )

            # Get the minimum and maximum size boundaries
            try:
                min_size = (
                    float(smps_data["Lower Size(nm)"].iloc[0])
                    if "Lower Size(nm)" in smps_data.columns
                    else 9.47
                )
                max_size = (
                    float(smps_data["Upper Size(nm)"].iloc[0])
                    if "Upper Size(nm)" in smps_data.columns
                    else 414.2
                )
            except (ValueError, TypeError):
                min_size = 9.47
                max_size = 414.2

            # Create bin ranges if not already defined
            if not bin_ranges:
                bin_ranges = [(min_size, 100), (100, 200), (200, 300), (300, max_size)]
                bin_columns = [
                    f"Æ©{int(start)}-{int(end)}nm (µg/m³)" for start, end in bin_ranges
                ]
                INSTRUMENT_CONFIG["SMPS"]["process_pollutants"] = bin_columns + [
                    "Total Concentration (µg/m³)"
                ]

            # Convert Date and Start Time to datetime
            try:
                smps_data["Date"] = pd.to_datetime(smps_data["Date"], errors="coerce")
                smps_data["Start Time"] = pd.to_datetime(
                    smps_data["Start Time"], format="%H:%M:%S", errors="coerce"
                )
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

            if smps_data["Date"].isna().all() or smps_data["Start Time"].isna().all():
                continue

            # Create datetime column by combining Date and Start Time
            try:
                # Ensure Date and Start Time are datetime types before using strftime
                date_series = pd.to_datetime(smps_data["Date"], errors="coerce")
                time_series = pd.to_datetime(smps_data["Start Time"], errors="coerce")
                smps_data["datetime"] = pd.to_datetime(
                    date_series.dt.strftime("%Y-%m-%d")
                    + " "
                    + time_series.dt.strftime("%H:%M:%S"),
                    errors="coerce",
                )
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

            invalid_rows = smps_data["datetime"].isna().sum()
            if invalid_rows > 0:
                smps_data = smps_data.dropna(subset=["datetime"])

            if smps_data.empty:
                continue

            # Sort by datetime and calculate mid-time
            smps_data = smps_data.sort_values("datetime").reset_index(drop=True)
            smps_data["next_datetime"] = smps_data["datetime"].shift(-1)

            try:
                smps_data["mid_datetime"] = pd.to_datetime(
                    smps_data.apply(
                        lambda row: (
                            row["datetime"]
                            + (row["next_datetime"] - row["datetime"]) / 2
                            if pd.notna(row["next_datetime"])
                            else row["datetime"]
                        ),
                        axis=1,
                    )
                )
                smps_data["datetime"] = smps_data["mid_datetime"]
            except (ValueError, TypeError, KeyError, AttributeError):
                pass

            smps_data = smps_data.drop(
                ["next_datetime", "mid_datetime"], axis=1, errors="ignore"
            )

            # Get numeric columns (size bins)
            known_non_numeric = [
                "Date",
                "Start Time",
                "datetime",
                "Lower Size(nm)",
                "Upper Size(nm)",
                "Total Concentration (µg/m³)",
            ]

            numeric_cols = []
            for col in smps_data.columns:
                if col not in known_non_numeric:
                    try:
                        float(col)
                        numeric_cols.append(col)
                    except (ValueError, TypeError):
                        pass

            if not numeric_cols:
                continue

            # Ensure Total Concentration is numeric
            try:
                smps_data["Total Concentration (µg/m³)"] = pd.to_numeric(
                    smps_data["Total Concentration (µg/m³)"], errors="coerce"
                )
            except (ValueError, TypeError, KeyError):
                pass

            # Create a new DataFrame to avoid fragmentation warning
            new_data = {}

            # Sum columns within each bin range
            for i, (start, end) in enumerate(bin_ranges):
                try:
                    bin_cols = [
                        col for col in numeric_cols if start <= float(col) <= end
                    ]
                except (ValueError, TypeError):
                    bin_cols = []

                if bin_cols:
                    bin_name = bin_columns[i]

                    try:
                        numeric_data = {}
                        for col in bin_cols:
                            numeric_data[col] = pd.to_numeric(
                                smps_data[col], errors="coerce"
                            )
                        new_data[bin_name] = pd.DataFrame(numeric_data).sum(axis=1)
                    except (ValueError, TypeError, KeyError):
                        new_data[bin_name] = pd.Series(np.nan, index=smps_data.index)

            # Get the burn_id for this date
            burn_id_row = burn_log[burn_log["Date"] == burn_date]
            if not burn_id_row.empty:
                burn_id = burn_id_row["Burn ID"].iloc[0]

                # Ensure datetime column is properly typed
                datetime_col = pd.to_datetime(smps_data["datetime"], errors="coerce")

                result_df = pd.DataFrame(
                    {
                        "datetime": datetime_col,
                        "Date": datetime_col.dt.date,
                        "burn_id": burn_id,
                        "Total Concentration (µg/m³)": smps_data[
                            "Total Concentration (µg/m³)"
                        ],
                    }
                )

                # Add bin columns
                for bin_name in bin_columns:
                    if bin_name in new_data:
                        result_df[bin_name] = new_data[bin_name]

                # Apply time shift if needed
                time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
                if time_shift != 0:
                    result_df["datetime"] += pd.Timedelta(minutes=time_shift)

                # Make sure data types are consistent
                result_df["datetime"] = pd.to_datetime(
                    result_df["datetime"], errors="coerce"
                )
                result_df["Total Concentration (µg/m³)"] = pd.to_numeric(
                    result_df["Total Concentration (µg/m³)"], errors="coerce"
                )

                for bin_name in bin_columns:
                    if bin_name in result_df:
                        result_df[bin_name] = pd.to_numeric(
                            result_df[bin_name], errors="coerce"
                        )

                combined_smps_data = pd.concat(
                    [combined_smps_data, result_df], ignore_index=True
                )

        except (ValueError, TypeError, KeyError, AttributeError, FileNotFoundError, OSError) as e:
            print(f"Error processing SMPS file for date {burn_date}: {str(e)}")

    # Final check of data quality
    if not combined_smps_data.empty:
        combined_smps_data = combined_smps_data.replace([np.inf, -np.inf], np.nan)

        concentration_cols = ["Total Concentration (µg/m³)"] + bin_columns
        existing_cols = [
            col for col in concentration_cols if col in combined_smps_data.columns
        ]

        if existing_cols:
            na_rows = combined_smps_data[existing_cols].isna().all(axis=1).sum()
            if na_rows > 0:
                combined_smps_data = combined_smps_data.dropna(
                    subset=existing_cols, how="all"
                )

    return combined_smps_data


# Function to process QuantAQ data
def process_quantaq_data(file_path, instrument="QuantAQB"):
    """Process QuantAQ data with standardized output format"""
    quantaq_data = pd.read_csv(file_path)
    quantaq_data = quantaq_data.iloc[::-1].reset_index(drop=True)
    quantaq_data.columns = quantaq_data.columns.str.strip()

    # Rename columns
    quantaq_data.rename(
        columns={"pm1": "PM1 (µg/m³)", "pm25": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)"},
        inplace=True,
    )

    # Convert timestamp to datetime without timezone information
    quantaq_data["timestamp_local"] = pd.to_datetime(
        quantaq_data["timestamp_local"].str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    ).dt.tz_localize(None)

    # Get burn range from instrument configuration
    burn_range = INSTRUMENT_CONFIG[instrument].get("burn_range", range(4, 11))

    # Filter by burn dates
    filtered_data = filter_by_burn_dates(quantaq_data, burn_range, "timestamp_local")

    # Apply time shift for each burn
    burn_ids = [f"burn{i}" for i in burn_range]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    return filtered_data


def determine_instrument_location(instrument, burn_id):
    """Determine if instrument is in bedroom or kitchen for the given burn"""
    # Extract burn number
    burn_num = int(burn_id.replace("burn", ""))

    if instrument.endswith("B"):  # Bedroom suffix
        return "bedroom"
    elif instrument.endswith("K"):  # Kitchen suffix
        return "kitchen"
    elif instrument == "DustTrak":
        # In bedroom for burns 1-6, in kitchen for burns 7-10
        return "bedroom" if burn_num <= 6 else "kitchen"
    elif instrument == "SMPS":
        return "bedroom"  # Always in bedroom
    elif instrument == "MiniAMS":
        return "bedroom"  # Assume bedroom (not clearly specified in original)
    else:
        return "unknown"


# Legend names in the manuscript's 1/2 unit numbering (1 = Bedroom 2,
# 2 = Morning Room). The K/B suffixes are internal config keys only.
LEGEND_NAMES = {
    "AeroTrakB": "AeroTrak1",
    "AeroTrakK": "AeroTrak2",
    "QuantAQB": "MODULAIR-PM1",
    "QuantAQK": "MODULAIR-PM2",
    "PurpleAirK": "PurpleAir",
    "DustTrak": "DustTrak",
    "SMPS": "SMPS",
}

# Drawing and legend order: roughly highest to lowest peak, so the legend
# reads top-down like the traces.
COMPARISON_ORDER = [
    "DustTrak",
    "PurpleAirK",
    "AeroTrakK",
    "QuantAQK",
    "AeroTrakB",
    "QuantAQB",
    "SMPS",
]

_BOKEH_DASH = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}


def get_pm25_equivalent_pollutant(instrument):
    """Get the appropriate PM2.5 equivalent pollutant for each instrument"""
    if instrument.startswith("AeroTrak"):
        return "PM3 (µg/m³)"  # Use PM3.0 for AeroTrak
    elif instrument == "SMPS":
        return "Total Concentration (µg/m³)"  # Use PM total for SMPS
    else:
        return "PM2.5 (µg/m³)"  # Use PM2.5 for others


def _collect_burn_traces(burn_to_plot, rooms):
    """Load the PM2.5-equivalent trace of each instrument for one burn.

    Args:
        burn_to_plot: Burn ID (e.g., "burn9").
        rooms: Locations to keep, as returned by determine_instrument_location
            ("kitchen" is the Morning Room, "bedroom" is Bedroom 2).

    Returns:
        Tuple (traces, events). traces is a list of dicts with keys
        "instrument", "label", "location", "pollutant", "t_h", "y", in
        COMPARISON_ORDER. events maps "Garage closed" and "CR Box on" to
        hours since garage closure (CR Box on omitted if not logged). Returns
        (None, None) if the burn is not in the burn log.
    """
    burn_row = burn_log[burn_log["Burn ID"] == burn_to_plot]
    if burn_row.empty:
        print(f"Burn {burn_to_plot} not found in burn log")
        return None, None

    burn_date = pd.to_datetime(burn_row["Date"].iloc[0])
    garage_closed_time_str = burn_row["garage closed"].iloc[0]
    garage_closed_time = create_naive_datetime(burn_date.date(), garage_closed_time_str)

    events = {"Garage closed": 0.0}
    cr_box_on_time_str = burn_row["CR Box on"].iloc[0]
    if pd.notna(cr_box_on_time_str):
        cr_box_on_time = create_naive_datetime(burn_date.date(), cr_box_on_time_str)
        if pd.notna(cr_box_on_time) and pd.notna(garage_closed_time):
            events["CR Box on"] = (cr_box_on_time - garage_closed_time).total_seconds() / 3600

    traces = []
    for instrument in COMPARISON_ORDER:
        config = INSTRUMENT_CONFIG[instrument]
        location = determine_instrument_location(instrument, burn_to_plot)
        if location not in rooms:
            continue

        try:
            # Check if this burn is in the instrument's range
            burn_range = config.get("burn_range", range(1, 11))
            burn_num = int(burn_to_plot.replace("burn", ""))
            if burn_num not in burn_range:
                print(f"Skipping {instrument} - {burn_to_plot} not in range")
                continue

            # Get the appropriate PM2.5 equivalent pollutant
            pollutant = get_pm25_equivalent_pollutant(instrument)
            if pollutant not in config["process_pollutants"]:
                print(f"Skipping {instrument} - {pollutant} not available")
                continue

            # Process the data
            process_func_name = config["process_function"]
            file_path = config["file_path"]

            if process_func_name == "process_aerotrak_data":
                processed_data = process_aerotrak_data(file_path, instrument)
            elif process_func_name == "process_dusttrak_data":
                processed_data = process_dusttrak_data(file_path, instrument)
            elif process_func_name == "process_miniams_data":
                processed_data = process_miniams_data(file_path, instrument)
            elif process_func_name == "process_purpleairk_data":
                processed_data = process_purpleairk_data(file_path)
            elif process_func_name == "process_quantaq_data":
                processed_data = process_quantaq_data(file_path, instrument)
            elif process_func_name == "process_smps_data":
                processed_data = process_smps_data(file_path, instrument)
            else:
                continue

            if processed_data.empty:
                continue

            # Filter data for the specific burn
            burn_data = processed_data[
                processed_data["Date"] == burn_date.date()
            ].copy()
            if burn_data.empty:
                continue

            # Calculate time since garage closed
            datetime_column = config["datetime_column"]
            burn_data[datetime_column] = pd.to_datetime(burn_data[datetime_column])

            # Make sure both datetime objects are timezone-naive
            burn_datetime = burn_data[datetime_column]
            if (
                hasattr(burn_datetime.dtype, "tz")
                and burn_datetime.dtype.tz is not None
            ):
                burn_datetime = burn_datetime.dt.tz_localize(None)

            burn_data["Time Since Garage Closed (hours)"] = (
                burn_datetime - garage_closed_time
            ).dt.total_seconds() / 3600

            # Filter data to time range (-1 to +4 hours)
            time_filtered_data = burn_data[
                (burn_data["Time Since Garage Closed (hours)"] >= -1)
                & (burn_data["Time Since Garage Closed (hours)"] <= 4)
            ].copy()

            if time_filtered_data.empty or pollutant not in time_filtered_data.columns:
                continue

            traces.append(
                {
                    "instrument": instrument,
                    "label": LEGEND_NAMES[instrument],
                    "location": location,
                    "pollutant": pollutant,
                    "t_h": time_filtered_data["Time Since Garage Closed (hours)"].to_numpy(),
                    "y": pd.to_numeric(time_filtered_data[pollutant], errors="coerce").to_numpy(),
                }
            )
            print(f"Loaded {instrument} ({location}) - {pollutant}")

        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            print(f"Error loading {instrument}: {str(e)}")
            continue

    return traces, events


def _trace_style(label, index):
    """Color and line style for a comparison trace (project colors first)."""
    if label in INSTRUMENT_COLORS:
        return INSTRUMENT_COLORS[label], INSTRUMENT_LINESTYLES[label]
    fallback = series_style(index + 1)  # skip black, reserved for event lines
    return fallback["color"], fallback["linestyle"]


def plot_burn_comparison(
    burn_to_plot=BURN_TO_PLOT,
    script_metadata=None,
    rooms=("kitchen",),
    out_dir=None,
    overwrite=False,
):
    """Plot the PM2.5-equivalent trace of each instrument for a single burn.

    Writes an interactive Bokeh HTML and a static journal-style figure (PDF
    and 600 dpi PNG), all ISO-dated. By default only the Morning Room
    ("kitchen") instruments are drawn, labeled with the 1/2 unit numbering
    used in the manuscript (AeroTrak2, MODULAIR-PM2).

    Args:
        burn_to_plot: Burn ID to plot (e.g., "burn9").
        script_metadata: Optional metadata string to include in the HTML.
        rooms: Locations to include ("kitchen" = Morning Room, "bedroom" =
            Bedroom 2).
        out_dir: Output folder; defaults to output_figures in data_config.json.
        overwrite: Replace same-day files instead of raising.

    Returns:
        List of Path objects written, or [] if the burn has no traces.
    """
    print(f"Creating comparison plot for {burn_to_plot} ({', '.join(rooms)})...")
    traces, events = _collect_burn_traces(burn_to_plot, rooms)
    if not traces:
        print(f"No traces for {burn_to_plot}; nothing plotted.")
        return []

    out_dir = Path(out_dir or get_common_file("output_figures"))
    room_tag = "morning_room" if tuple(rooms) == ("kitchen",) else "_".join(rooms)
    stem = dated_stem(out_dir, f"{burn_to_plot}_PM25_comparison_{room_tag}")
    written = []

    # --- Bokeh (interactive) ---
    p = figure(
        x_axis_label=AXIS_LABELS["time_since_garage_h"],
        y_axis_label=AXIS_LABELS["pm"],
        x_axis_type="linear",
        y_axis_type="log",
        width=800,
        height=500,
        x_range=Range1d(-1, 4),
        y_range=Range1d(10**-1, 10**5),
    )
    for i, tr in enumerate(traces):
        color, ls = _trace_style(tr["label"], i)
        p.line(
            tr["t_h"], tr["y"], legend_label=tr["label"], line_width=2,
            line_dash=_BOKEH_DASH[ls], color=color,
        )
    for (name, t_ev), dash in zip(events.items(), ("solid", "dashed")):
        p.line(x=[t_ev, t_ev], y=[10**-2, 10**6], line_color="black", line_width=1,
               line_dash=dash, legend_label=name)
    p.legend.click_policy = "hide"
    p.legend.location = "top_right"
    apply_bokeh_text(p)
    layout = column(p, Div(text=f"<small>{script_metadata}</small>", width=800)) \
        if script_metadata else p

    html_path = stem.with_suffix(".html")
    if html_path.exists() and not overwrite:
        raise FileExistsError(f"{html_path} exists. Pass overwrite=True to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file(str(html_path), title=f"{burn_to_plot} PM comparison")
    save(layout)
    written.append(html_path)
    print(f"[html] {html_path.name}")

    # --- matplotlib (static, journal style) ---
    with journal_style():
        fig, ax = plt.subplots(figsize=journal_figsize("double", aspect=0.5))
        for i, tr in enumerate(traces):
            color, ls = _trace_style(tr["label"], i)
            y = np.where(tr["y"] > 0, tr["y"], np.nan)
            ax.semilogy(tr["t_h"], y, color=color, ls=ls, lw=1.0, label=tr["label"])
        for (name, t_ev), ls in zip(events.items(), ("-", "--")):
            ax.axvline(t_ev, color=REF_LINE, lw=0.5, ls=ls, label=name)
        ax.set_xlim(-1, 4)
        ax.set_ylim(1e-1, 1e5)
        ax.set_xlabel(AXIS_LABELS["time_since_garage_h"])
        ax.set_ylabel(AXIS_LABELS["pm"])
        ax.legend(loc="upper right", frameon=False)
        written += save_pdf_png(fig, stem, overwrite=overwrite)

    return written


def find_peak_concentrations():
    """Process all instruments and find peak concentrations for each burn and pollutant"""
    print("Processing all instruments to find peak concentrations...")

    # Initialize results dictionary
    results = {}

    # Process each instrument
    for instrument, config in INSTRUMENT_CONFIG.items():
        print(f"\nProcessing {instrument}...")

        try:
            # Get processing function and file path
            process_func_name = config["process_function"]
            file_path = config["file_path"]

            # Call the appropriate processing function
            if process_func_name == "process_aerotrak_data":
                processed_data = process_aerotrak_data(file_path, instrument)
            elif process_func_name == "process_dusttrak_data":
                processed_data = process_dusttrak_data(file_path, instrument)
            elif process_func_name == "process_miniams_data":
                processed_data = process_miniams_data(file_path, instrument)
            elif process_func_name == "process_purpleairk_data":
                processed_data = process_purpleairk_data(file_path)
            elif process_func_name == "process_quantaq_data":
                processed_data = process_quantaq_data(file_path, instrument)
            elif process_func_name == "process_smps_data":
                processed_data = process_smps_data(file_path, instrument)
            else:
                print(f"Unknown processing function: {process_func_name}")
                continue

            if processed_data.empty:
                print(f"No data processed for {instrument}")
                continue

            # Get process pollutants for this instrument
            process_pollutants = config["process_pollutants"]

            # Find peak concentrations for each burn and pollutant
            unique_dates = processed_data["Date"].unique()

            for burn_date in unique_dates:
                # Get burn_id for this date
                burn_row = burn_log[burn_log["Date"] == pd.to_datetime(burn_date)]
                if burn_row.empty:
                    continue

                burn_id = burn_row["Burn ID"].iloc[0]

                # Filter data for this burn
                burn_data = processed_data[processed_data["Date"] == burn_date]

                # Find peak concentration for each pollutant
                for pollutant in process_pollutants:
                    if pollutant in burn_data.columns:
                        # Get maximum concentration, handling NaN values
                        max_conc = burn_data[pollutant].max()

                        # Store result with instrument-pollutant combination as column name
                        column_name = f"{instrument}_{pollutant}"

                        if burn_id not in results:
                            results[burn_id] = {}

                        results[burn_id][column_name] = max_conc

                        if pd.notna(max_conc):
                            print(f"  {burn_id} {pollutant}: {max_conc:.3f}")

        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            print(f"Error processing {instrument}: {str(e)}")
            continue

    # Convert results to DataFrame
    if results:
        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.index.name = "Burn_ID"

        # Sort by burn number
        burn_numbers = [
            int(burn_id.replace("burn", "")) for burn_id in df_results.index
        ]
        df_results = df_results.iloc[np.argsort(burn_numbers)]

        # Save to Excel in burn_data directory
        output_filename = "./burn_data/peak_concentrations_all_instruments.xlsx"
        df_results.to_excel(output_filename)

        print(f"\n*** Peak concentrations saved to {output_filename} ***")
        print(
            f"Results shape: {df_results.shape} (instrument-pollutant combinations × burns)"
        )
        print("\nFirst few rows:")
        print(df_results.iloc[:5, :].to_string())

        return df_results
    else:
        print("No results to save!")
        return None


def main():
    """Main function to process all instruments and create peak concentration matrix"""
    try:
        # Get script metadata
        metadata = get_script_metadata()

        # Find peak concentrations for all instruments
        results_df = find_peak_concentrations()

        if results_df is not None:
            print("\nProcessing completed successfully!")
            print(
                f"Created matrix with {len(results_df)} rows "
                f"(instrument-pollutant combinations) and "
                f"{len(results_df.columns)} columns (burns)"
            )
        else:
            print("No data was processed successfully")

        # Create comparison plot for the specified burn
        print(f"\nCreating comparison plot for {BURN_TO_PLOT}...")
        plot_burn_comparison(BURN_TO_PLOT, script_metadata=metadata)

    except (ValueError, TypeError, KeyError, AttributeError, FileNotFoundError, OSError) as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()


# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()
