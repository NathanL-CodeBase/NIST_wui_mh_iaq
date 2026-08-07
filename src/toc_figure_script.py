"""
WUI TOC Figure: Single Burn PM2.5 Equivalent Comparison

This script generates a high-quality figure for the Table of Contents (TOC) of a paper,
showing PM2.5 equivalent data for all instruments for a single burn.

Figure Specifications:
    - PNG format with transparent background
    - Resolution: 550px wide × 1050px tall
    - High DPI for publication quality
    - X-axis: -1 to 3 hours (time since garage closed)
    - Y-axis: 10^-2 to 10^5 µg/m³ (log scale)

Instrument Label Mapping:
    - AeroTrak → OPC1
    - DustTrak → Nef+OPC1
    - PurpleAir → Nef1
    - QuantAQ → Nef+OPC2
    - SMPS → SMPS

Author: Nathan Lima
Date: 2024-2025
"""

# %% IMPORT MODULES
import os
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import (  # noqa: E402
    get_common_file,
    get_data_root,
    get_instrument_path,
)

# Set the absolute path for the dataset
data_root = get_data_root()  # Portable path - auto-configured
os.chdir(str(data_root))

# Variable to set which burn to plot
BURN_TO_PLOT = "burn9"

# Variable to set which instruments to include in the plot
# Options: "AeroTrakB", "AeroTrakK", "DustTrak", "MiniAMS", "PurpleAirK", "QuantAQB", "QuantAQK", "SMPS"
# Set to None to include all instruments, or specify a list/set of instrument names
INSTRUMENTS_TO_PLOT = {
    "AeroTrakK"
}  # Use None for all, or set like: {"AeroTrakB", "DustTrak", "SMPS"}

# Variable to set text sizes for the plot
# Adjust these values to control the appearance of text in the figure
TEXT_SIZES = {
    "xlabel": 32,  # X-axis label font size
    "ylabel": 32,  # Y-axis label font size
    "legend": 18,  # Legend font size
    "title": 18,  # Title font size (if added)
    "xticks": 18,  # X-axis tick label font size
    "yticks": 18,  # Y-axis tick label font size
}

# Load burn log once
BURN_LOG_PATH = str(get_common_file("burn_log"))
burn_log = pd.read_excel(BURN_LOG_PATH, sheet_name="Sheet2")

# Define instrument configurations
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "file_path": str(get_instrument_path("aerotrak_bedroom") / "all_data.xlsx"),
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
        "display_name": "OPC1B",
    },
    "AeroTrakK": {
        "file_path": str(get_instrument_path("aerotrak_kitchen") / "all_data.xlsx"),
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
        "display_name": "OPC1K",
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
        "special_cases": {"burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}},
        "display_name": "Nef+OPC1",
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
        "display_name": "MiniAMS",
    },
    "PurpleAirK": {
        "file_path": "./burn_data/purpleair/garage-kitchen.xlsx",
        "process_function": "process_purpleairk_data",
        "time_shift": 0,
        "process_pollutants": ["PM2.5 (µg/m³)"],
        "datetime_column": "DateTime",
        "burn_range": range(6, 11),
        "special_cases": {},
        "display_name": "Nef1",
    },
    "QuantAQB": {
        "file_path": str(
            get_instrument_path("quantaq_bedroom")
            / "MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv"
        ),
        "process_function": "process_quantaq_data",
        "time_shift": -2.97,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "special_cases": {"burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}},
        "display_name": "Nef+OPC2B",
    },
    "QuantAQK": {
        "file_path": str(
            get_instrument_path("quantaq_kitchen")
            / "MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv"
        ),
        "process_function": "process_quantaq_data",
        "time_shift": 0,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "special_cases": {},
        "display_name": "Nef+OPC2K",
    },
    "SMPS": {
        "file_path": "./burn_data/smps",
        "process_function": "process_smps_data",
        "time_shift": 0,
        "process_pollutants": ["Total Concentration (µg/m³)"],
        "datetime_column": "datetime",
        "special_cases": {"burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}},
        "display_name": "SMPS",
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
def apply_time_shift(df, instrument, burn_date):
    """Apply time shift based on instrument configuration"""
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
    datetime_column = INSTRUMENT_CONFIG[instrument].get("datetime_column", "Date and Time")

    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
    burn_date = pd.to_datetime(burn_date).date()

    if time_shift != 0:
        mask = df[datetime_column].dt.date == burn_date
        if mask.any():
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


# Helper function to filter data by burn dates
def filter_by_burn_dates(data, burn_range, datetime_column):
    """Filter data by burn dates from burn log"""
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
                aerotrak_data[new_diff_col_µg_m3] = (particle_counts / (volume_cm * 1e-6)) * (
                    particle_mass
                )
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
    filtered_aerotrak_data = aerotrak_data[aerotrak_data["Date"].isin(burn_dates.dt.date)]
    filtered_aerotrak_data = filtered_aerotrak_data.copy()

    # Apply time shift for each burn ID in the filtered data
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_aerotrak_data = apply_time_shift(filtered_aerotrak_data, instrument, burn_date)

    # Check if there's a special case for burn3 rolling average
    special_cases = INSTRUMENT_CONFIG[instrument].get("special_cases", {})
    if "burn3" in special_cases and special_cases["burn3"].get("apply_rolling_average", False):
        filtered_aerotrak_data = calculate_rolling_average_burn3(filtered_aerotrak_data)

    return filtered_aerotrak_data


# Function to calculate 5-minute rolling average for burn3
def calculate_rolling_average_burn3(data):
    """Calculate 5-minute rolling average for burn3 data"""
    burn3_date = burn_log[burn_log["Burn ID"] == "burn3"]["Date"].values[0]
    burn3_date = pd.to_datetime(burn3_date).date()

    burn3_data = data[data["Date"] == burn3_date]
    if burn3_data.empty:
        return data

    burn3_data = burn3_data.set_index("Date and Time")
    rolling_avg_data = {}

    numeric_columns = burn3_data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        rolling_avg_data[column] = (
            burn3_data[column]
            .rolling(pd.Timedelta(minutes=5))
            .mean()
            .astype(burn3_data[column].dtype)
        )

    status_columns = ["Flow Status", "Instrument Status", "Laser Status"]
    for column in status_columns:
        if column in burn3_data.columns:
            rolling_avg_data[column] = burn3_data[column].iloc[0]

    rolling_avg_df = pd.DataFrame(rolling_avg_data, index=burn3_data.index)
    rolling_avg_df.reset_index(inplace=True)

    data.loc[
        data["Date"] == burn3_date,
        rolling_avg_df.columns.difference(["Date", "Date and Time"]),
    ] = rolling_avg_df[rolling_avg_df.columns.difference(["Date", "Date and Time"])].values
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
            filtered_data = apply_time_shift(filtered_data, instrument, burn_date)

    return filtered_data


# Function to process MiniAMS data
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
            filtered_data = apply_time_shift(filtered_data, instrument, burn_date)

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
            if "Date" not in smps_data.columns and "Start Time" not in smps_data.columns:
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

            missing_columns = [col for col in required_columns if col not in smps_data.columns]
            if missing_columns:
                continue

            # Rename 'Total Concentration(µg/m³)' to 'Total Concentration (µg/m³)' to add space
            if "Total Concentration(µg/m³)" in smps_data.columns:
                smps_data = smps_data.rename(
                    columns={"Total Concentration(µg/m³)": "Total Concentration (µg/m³)"}
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
                bin_columns = [f"Æ©{int(start)}-{int(end)}nm (µg/m³)" for start, end in bin_ranges]
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
                    date_series.dt.strftime("%Y-%m-%d") + " " + time_series.dt.strftime("%H:%M:%S"),
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
                            row["datetime"] + (row["next_datetime"] - row["datetime"]) / 2
                            if pd.notna(row["next_datetime"])
                            else row["datetime"]
                        ),
                        axis=1,
                    )
                )
                smps_data["datetime"] = smps_data["mid_datetime"]
            except (ValueError, TypeError, KeyError, AttributeError):
                pass

            smps_data = smps_data.drop(["next_datetime", "mid_datetime"], axis=1, errors="ignore")

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
            except (ValueError, TypeError, KeyError, AttributeError):
                pass

            # Create a new DataFrame to avoid fragmentation warning
            new_data = {}

            # Sum columns within each bin range
            for i, (start, end) in enumerate(bin_ranges):
                try:
                    bin_cols = [col for col in numeric_cols if start <= float(col) <= end]
                except (ValueError, TypeError):
                    bin_cols = []

                if bin_cols:
                    bin_name = bin_columns[i]

                    try:
                        numeric_data = {}
                        for col in bin_cols:
                            numeric_data[col] = pd.to_numeric(smps_data[col], errors="coerce")
                        new_data[bin_name] = pd.DataFrame(numeric_data).sum(axis=1)
                    except (ValueError, TypeError, KeyError, AttributeError):
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
                        "Total Concentration (µg/m³)": smps_data["Total Concentration (µg/m³)"],
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
                result_df["datetime"] = pd.to_datetime(result_df["datetime"], errors="coerce")
                result_df["Total Concentration (µg/m³)"] = pd.to_numeric(
                    result_df["Total Concentration (µg/m³)"], errors="coerce"
                )

                for bin_name in bin_columns:
                    if bin_name in result_df:
                        result_df[bin_name] = pd.to_numeric(result_df[bin_name], errors="coerce")

                combined_smps_data = pd.concat([combined_smps_data, result_df], ignore_index=True)

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error processing SMPS file for date {burn_date}: {str(e)}")

    # Final check of data quality
    if not combined_smps_data.empty:
        combined_smps_data = combined_smps_data.replace([np.inf, -np.inf], np.nan)

        concentration_cols = ["Total Concentration (µg/m³)"] + bin_columns
        existing_cols = [col for col in concentration_cols if col in combined_smps_data.columns]

        if existing_cols:
            na_rows = combined_smps_data[existing_cols].isna().all(axis=1).sum()
            if na_rows > 0:
                combined_smps_data = combined_smps_data.dropna(subset=existing_cols, how="all")

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
            filtered_data = apply_time_shift(filtered_data, instrument, burn_date)

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


def get_pm25_equivalent_pollutant(instrument):
    """Get the appropriate PM2.5 equivalent pollutant for each instrument"""
    if instrument.startswith("AeroTrak"):
        return "PM3 (µg/m³)"  # Use PM3.0 for AeroTrak
    elif instrument == "SMPS":
        return "Total Concentration (µg/m³)"  # Use PM total for SMPS
    else:
        return "PM2.5 (µg/m³)"  # Use PM2.5 for others


def create_toc_figure(burn_to_plot=BURN_TO_PLOT, instruments_to_plot=None, text_sizes=None):
    """Create high-quality TOC figure for a single burn
    Parameters:
    -----------
    burn_to_plot : str
        The burn ID to plot (e.g., "burn9")
    instruments_to_plot : set, list, or None
        Instruments to include in the plot. If None, includes all except MiniAMS.
        Examples: {"AeroTrakB", "DustTrak", "SMPS"} or None for all available
    text_sizes : dict or None
        Dictionary with text size settings. If None, uses global TEXT_SIZES.
        Keys: 'xlabel', 'ylabel', 'legend', 'title'
    """
    if instruments_to_plot is None:
        instruments_to_plot = INSTRUMENTS_TO_PLOT
    if text_sizes is None:
        text_sizes = TEXT_SIZES
    print(f"Creating TOC figure for {burn_to_plot}...")
    print(
        f"Instruments to plot: {instruments_to_plot if instruments_to_plot else 'All available (except MiniAMS)'}"
    )

    # Get burn date
    burn_row = burn_log[burn_log["Burn ID"] == burn_to_plot]
    if burn_row.empty:
        print(f"Burn {burn_to_plot} not found in burn log")
        return

    burn_date = pd.to_datetime(burn_row["Date"].iloc[0])
    garage_closed_time_str = burn_row["garage closed"].iloc[0]
    garage_closed_time = create_naive_datetime(burn_date.date(), garage_closed_time_str)

    # Set figure size and DPI for exact pixel dimensions
    # 650px wide × 1050px tall at 100 DPI
    fig_width_inches = 650 / 100
    fig_height_inches = 1050 / 100
    dpi = 100

    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Define colors for instruments
    colors = [
        "#ef5675",
        "#d45087",
        "#b35093",
        "#8d5196",
        "#665191",
        "#404e84",
        "#1d4772",
        "#003f5c",
    ]
    color_idx = 0

    # Process each instrument
    for instrument, config in INSTRUMENT_CONFIG.items():
        # Skip instruments not in the selection
        if instruments_to_plot is None:
            # Default behavior: skip MiniAMS if not explicitly selected
            if instrument == "MiniAMS":
                continue
        else:
            # Only plot instruments in the specified set/list
            if instrument not in instruments_to_plot:
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
            burn_data = processed_data[processed_data["Date"] == burn_date.date()].copy()
            if burn_data.empty:
                continue

            # Calculate time since garage closed
            datetime_column = config["datetime_column"]
            burn_data[datetime_column] = pd.to_datetime(burn_data[datetime_column])

            # Make sure both datetime objects are timezone-naive
            burn_datetime = burn_data[datetime_column]
            if hasattr(burn_datetime.dtype, "tz") and burn_datetime.dtype.tz is not None:
                burn_datetime = burn_datetime.dt.tz_localize(None)

            burn_data["Time Since Garage Closed (hours)"] = (
                burn_datetime - garage_closed_time
            ).dt.total_seconds() / 3600

            # Filter data to time range (-1 to +3 hours)
            time_filtered_data = burn_data[
                (burn_data["Time Since Garage Closed (hours)"] >= -1)
                & (burn_data["Time Since Garage Closed (hours)"] <= 3)
            ].copy()

            if time_filtered_data.empty or pollutant not in time_filtered_data.columns:
                continue

            # Determine line style based on location
            location = determine_instrument_location(instrument, burn_to_plot)
            linestyle = "--" if location == "bedroom" else "-"

            # Get display name for legend
            display_name = config.get("display_name", instrument)

            # Plot the data
            ax.plot(
                time_filtered_data["Time Since Garage Closed (hours)"],
                time_filtered_data[pollutant],
                label=display_name,
                linewidth=4,
                linestyle=linestyle,
                color=colors[color_idx % len(colors)],
            )

            color_idx += 1
            print(f"Plotted {instrument} ({location}) - {pollutant} as {display_name}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error plotting {instrument}: {str(e)}")
            continue

    # Set axis properties
    ax.set_xlabel("Time", fontsize=text_sizes["xlabel"])
    ax.set_ylabel("Concentration", fontsize=text_sizes["ylabel"])
    ax.set_yscale("log")
    ax.set_xlim(-1, 2)
    ax.set_ylim(10**-2, 10**4)

    # Remove x and y axis tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot as PNG with transparent background
    os.makedirs(str(get_common_file("output_figures")), exist_ok=True)
    png_filename = f"./Paper_figures/{burn_to_plot}_TOC_figure.png"
    plt.savefig(png_filename, format="png", dpi=dpi, transparent=True, bbox_inches="tight")

    print(f"TOC figure saved to {png_filename}")
    print(f"Figure dimensions: {fig_width_inches * dpi:.0f}px × {fig_height_inches * dpi:.0f}px")

    plt.close()


def main():
    """Main function to create the TOC figure"""
    try:
        # Create TOC figure for the specified burn
        print(f"Creating TOC figure for {BURN_TO_PLOT}...")
        create_toc_figure(BURN_TO_PLOT, INSTRUMENTS_TO_PLOT, TEXT_SIZES)
        print("\nTOC figure created successfully!")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error in main function: {e}")
        traceback.print_exc()


# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()
