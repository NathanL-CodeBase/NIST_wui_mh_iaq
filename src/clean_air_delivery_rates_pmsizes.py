#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Air Delivery Rate Analysis by Particle Size
==================================================

This script calculates clean air delivery rates (CADR) and exponential decay
parameters from particulate matter (PM) concentration data collected during
wildfire smoke infiltration experiments in a manufactured home. It processes
data from multiple air quality instruments to characterize how quickly indoor
PM concentrations decrease when CR Box (Corsi-Rosenthal Box) air filtration
devices are activated.

The analysis is part of the NIST Wildland-Urban Interface (WUI) Mobile Home
Indoor Air Quality (IAQ) study, which evaluates low-cost air filtration
strategies for protecting occupants during wildfire smoke events. By analyzing
decay rates across different particle size fractions, the script quantifies
filtration effectiveness for particles of varying diameters.

Key Metrics Calculated:
    - Exponential decay rate (h^-1): Rate constant from fitted decay curves
    - Decay uncertainty: 95% confidence interval for decay rate estimates
    - Relative standard deviation (RSD): Quality metric for fit reliability
    - Normalized concentration profiles: For cross-burn comparisons
    - Time since garage closed: Standardized time reference for all experiments

Analysis Features:
    - Multi-instrument support: AeroTrak, DustTrak, QuantAQ, SMPS, MiniAMS, PurpleAir
    - Size-resolved analysis: PM0.5, PM1, PM2.5, PM3, PM5, PM10, PM15, PM25
    - SMPS size bins: 9-100nm, 100-200nm, 200-300nm, 300-437nm
    - Chemical species (MiniAMS): Organic, Nitrate, Sulfate, Ammonium, Chloride
    - Automatic decay period detection based on concentration derivatives
    - Special case handling for burn3 (rolling average) and burn6 (CR Box timing)
    - Quality control filtering based on instrument status columns
    - Interactive Bokeh plots with fitted decay curves

Methodology:
    1. Load and preprocess instrument data (time shifts, unit conversions)
    2. Filter data by burn dates using burn log reference
    3. Apply instrument-specific data quality filters
    4. Calculate time since garage closed for temporal alignment
    5. Identify decay start time (maximum concentration or CR Box activation)
    6. Find decay end time (5% of maximum threshold or fixed offset)
    7. Fit exponential decay model: C(t) = A * exp(-k * t)
    8. Calculate decay rate (k), uncertainty, and RSD
    9. Exclude fits with RSD > 10% as unreliable
    10. Generate interactive plots with data and fitted curves

Output Files:
    - HTML plots: {instrument}_{burn_id}_PM-dependent-size_mass-concentration.html
    - Decay parameters: Stored in global decay_parameters dictionary
    - Burn calculations: Compiled in burn_calc list for export

Applications:
    - Quantifying CR Box filtration effectiveness by particle size
    - Comparing decay rates across different burn conditions
    - Evaluating spatial variability between bedroom and kitchen locations
    - Characterizing smoke infiltration dynamics in manufactured housing
    - Supporting development of protective action recommendations

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2024-2026
"""

# %%
import os

# Import data path resolver and metadata utilities
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import output_file, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div
from bokeh.plotting import figure, show

sys.path.insert(0, str(Path(__file__).parent))
from data_paths import get_common_file, get_instrument_path

# Import metadata_utils from scripts folder
scripts_path = Path(__file__).parent.parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))
from metadata_utils import get_script_metadata

# Set output to display plots in the notebook
output_notebook()

# Variable for the dataset to be processed
dataset = "MiniAMS"  # Change this variable as needed

# Load burn log once
burn_log_path = get_common_file("burn_log")
burn_log = pd.read_excel(burn_log_path, sheet_name="Sheet2")

# Declare global variable burn_calc
burn_calc = []

# Define global color mapping for pollutants (consistent across all instruments)
POLLUTANT_COLORS = {
    "PM0.5 (µg/m³)": "#ef5675",
    "PM1 (µg/m³)": "#d45087",
    "PM2.5 (µg/m³)": "#b35093",
    "PM3 (µg/m³)": "#b35093",
    "PM4 (µg/m³)": "#8d5196",
    "PM5 (µg/m³)": "#665191",
    "PM10 (µg/m³)": "#404e84",
    "PM15 (µg/m³)": "#1d4772",
    "PM25 (µg/m³)": "#003f5c",
}
SMPS_BIN_COLORS = {
    "Ʃ9-100nm (µg/m³)": "#ffa600",
    "Ʃ100-200nm (µg/m³)": "#ff8d2f",
    "Ʃ200-300nm (µg/m³)": "#ff764a",
    "Ʃ300-437nm (µg/m³)": "#ff6361",
    "Total Concentration (µg/m³)": "#ef5675",
}
MINIAMS_SPECIES_COLORS = {
    "Organic (µg/m³)": "#2E8B57",  # Sea Green
    "Nitrate (µg/m³)": "#FF6347",  # Tomato
    "Sulfate (µg/m³)": "#4169E1",  # Royal Blue
    "Ammonium (µg/m³)": "#FF8C00",  # Dark Orange
    "Chloride (µg/m³)": "#9932CC",  # Dark Orchid
}

# Update the POLLUTANT_COLORS dictionary with SMPS bin colors
POLLUTANT_COLORS.update(SMPS_BIN_COLORS)
# Update the POLLUTANT_COLORS dictionary with Mini-AMS species colors
POLLUTANT_COLORS.update(MINIAMS_SPECIES_COLORS)


# Helper function to build file paths from data_paths resolver
def get_instrument_file_path(instrument_key, filename=None):
    """
    Get the file path for an instrument using the data_paths resolver.

    Parameters
    ----------
    instrument_key : str
        Key from data_config.json instruments section
    filename : str, optional
        Specific filename to append to instrument path

    Returns
    -------
    Path or str
        Full path to the instrument data file or folder
    """
    instrument_path = get_instrument_path(instrument_key)
    if filename:
        return instrument_path / filename
    return instrument_path


# Define instrument configurations
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "instrument_key": "aerotrak_bedroom",
        "filename": "all_data.xlsx",
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
        "plot_pollutants": ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM3 (µg/m³)",
        "special_cases": {
            "burn3": {"apply_rolling_average": True},
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25},
        },
        "datetime_column": "Date and Time",
        "baseline_values": {
            "PM0.5 (µg/m³)": (0.5121, 0.0079),
            "PM1 (µg/m³)": (0.5492, 0.0116),
            "PM3 (µg/m³)": (1.0855, 0.0511),
            "PM5 (µg/m³)": (2.0051, 0.0831),
            "PM10 (µg/m³)": (2.7994, 0.1160),
            "PM25 (µg/m³)": (3.3799, 0.1397),
        },
    },
    "AeroTrakK": {
        "instrument_key": "aerotrak_kitchen",
        "filename": "all_data.xlsx",
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
        "plot_pollutants": ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM3 (µg/m³)",
        "special_cases": {},
        "datetime_column": "Date and Time",
        "baseline_values": None,  # Will be calculated during processing
        "baseline_method": "weighted_average",
        "baseline_burns": ["burn5", "burn6"],
    },
    "DustTrak": {
        "instrument_key": "dusttrak",
        "filename": "all_data.xlsx",
        "process_function": "process_dusttrak_data",
        "time_shift": 7,
        "process_pollutants": [
            "PM1 (µg/m³)",
            "PM2.5 (µg/m³)",
            "PM4 (µg/m³)",
            "PM10 (µg/m³)",
            "PM15 (µg/m³)",
        ],
        "plot_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM2.5 (µg/m³)",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
        "datetime_column": "datetime",
        "location_notes": "In bedroom for burns 1-6, in kitchen for burns 7-10",
        "baseline_values": None,  # Will be calculated during processing
        "baseline_source": "burn1",  # Use burn1 as baseline for all burns except burn6
    },
    "MiniAMS": {
        "instrument_key": "miniams",
        "filename": "WUI_AMS_Species.xlsx",
        "process_function": "process_miniams_data",
        "time_shift": 0,
        "process_pollutants": [
            "Organic (µg/m³)",
            "Nitrate (µg/m³)",
            "Sulfate (µg/m³)",
            "Ammonium (µg/m³)",
            "Chloride (µg/m³)",
        ],
        "plot_pollutants": [
            "Organic (µg/m³)",
            "Nitrate (µg/m³)",
            "Sulfate (µg/m³)",
            "Ammonium (µg/m³)",
            "Chloride (µg/m³)",
        ],
        "normalize_pollutant": "Organic (µg/m³)",
        "special_cases": {},
        "datetime_column": "DateTime",
        "burn_range": range(1, 11),  # Burns 1-10
        "location_notes": "Chemical composition measurements (70-700nm aerodynamic diameter)",
        "baseline_values": None,  # Will be calculated during processing
        "baseline_source": "burn1",  # Use burn1 as baseline
    },
    "PurpleAirK": {
        "instrument_key": "purpleair",
        "filename": "garage-kitchen.xlsx",
        "process_function": "process_purpleairk_data",
        "time_shift": 0,
        "process_pollutants": ["PM2.5 (µg/m³)"],
        "plot_pollutants": ["PM2.5 (µg/m³)"],
        "normalize_pollutant": "PM2.5 (µg/m³)",
        "special_cases": {},
        "datetime_column": "DateTime",
        "burn_range": range(6, 11),
        "baseline_values": None,  # Will be calculated during processing
        "baseline_source": "burn6",  # Indicates to use burn1 as the source for baseline values
    },
    "QuantAQB": {
        "instrument_key": "quantaq_bedroom",
        "filename": "MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv",
        "process_function": "process_quantaq_data",
        "time_shift": -2.97,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "plot_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM2.5 (µg/m³)",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "baseline_values": {
            "PM1 (µg/m³)": (0.5970, 0.0073),
            "PM2.5 (µg/m³)": (0.6343, 0.0049),
            "PM10 (µg/m³)": (0.8289, 0.0318),
        },
    },
    "QuantAQK": {
        "instrument_key": "quantaq_kitchen",
        "filename": "MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv",
        "process_function": "process_quantaq_data",
        "time_shift": 0,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "plot_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM2.5 (µg/m³)",
        "special_cases": {},
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "baseline_values": None,  # Will be calculated during processing
        "baseline_method": "weighted_average",
        "baseline_burns": ["burn5", "burn6"],
    },
    "SMPS": {
        "instrument_key": "smps",
        "filename": None,  # SMPS uses folder path, not a single file
        "process_function": "process_smps_data",
        "time_shift": 0,
        # Using generic names for columns - will be populated with actual size boundaries during processing
        "process_pollutants": [
            "Total Concentration (µg/m³)"
        ],  # Will be dynamically populated with correct spacing
        "plot_pollutants": [
            "Total Concentration (µg/m³)"
        ],  # Will be dynamically populated
        "normalize_pollutant": "Total Concentration (µg/m³)",  # Updated with space
        "special_cases": {
            "burn6": {
                "custom_decay_time": True,
                "decay_end_offset": 0.25,
            }  # 15 minutes after max
        },
        "datetime_column": "datetime",
        "location_notes": "Always in bedroom",
        "baseline_values": None,  # Will be calculated during processing
        "baseline_source": "burn1",  # Use burn1 as baseline for all burns except burn6
    },
}

# Define burn groups for normalized plots
BURN_GROUPS = {
    "figure1": [
        "burn1",
        "burn2",
        "burn3",
        "burn4",
        "burn5",
        "burn6",
        "burn7",
        "burn8",
        "burn9",
        "burn10",
    ],
    "figure2": ["burn2", "burn4", "burn9"],
    "figure3": ["burn3", "burn4", "burn7", "burn8", "burn9", "burn10"],
    "figure4": ["burn1", "burn5", "burn6"],
}

# Define burn styles (colors and line styles)
BURN_STYLES = {
    "burn1": {"color": "#ef5675", "line_dash": "solid"},
    "burn2": {"color": "#d45087", "line_dash": "solid"},
    "burn3": {"color": "#b35093", "line_dash": "dashed"},
    "burn4": {"color": "#b35093", "line_dash": "solid"},
    "burn5": {"color": "#8d5196", "line_dash": "solid"},
    "burn6": {"color": "#665191", "line_dash": "solid"},
    "burn7": {"color": "#404e84", "line_dash": "solid"},
    "burn8": {"color": "#404e84", "line_dash": "dashed"},
    "burn9": {"color": "#003f5c", "line_dash": "solid"},
    "burn10": {"color": "#003f5c", "line_dash": "dashed"},
}

# Define burn legend labels
BURN_LABELS = {
    "burn1": "01-House",
    "burn2": "02-House-4-N",
    "burn3": "03-House-1-U",
    "burn4": "04-House-1-N",
    "burn5": "05-Room",
    "burn6": "06-Room-1-N",
    "burn7": "07-House-2A-N",
    "burn8": "08-House-2A-U",
    "burn9": "09-House-2-N",
    "burn10": "10-House-2-U",
}


# Modified apply_time_shift function to use simplified config
def apply_time_shift(df, instrument, burn_id, burn_date):
    """Apply time shift based on instrument configuration"""
    # Get time shift from configuration
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)

    # Get datetime column name from configuration
    datetime_column = INSTRUMENT_CONFIG[instrument].get(
        "datetime_column", "Date and Time"
    )

    # Ensure datetime column is in datetime format - Using .loc to avoid SettingWithCopyWarning
    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])

    # Convert burn_date to datetime
    burn_date = pd.to_datetime(burn_date).date()

    # Apply the shift only if shift value is non-zero
    if time_shift != 0:
        # Filter rows for the burn date
        mask = df[datetime_column].dt.date == burn_date

        # Check if there are any rows to shift
        if mask.any():
            # Use .loc to safely modify the original DataFrame
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


# Utility function to create timezone-naive datetime
def create_naive_datetime(date_str, time_str):
    """Create a timezone-naive datetime object from date and time strings"""
    dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
    if hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


# Helper function to filter data by burn dates
def filter_by_burn_dates(data, burn_range, datetime_column):
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
    """Process AeroTrak data with instrument-specific settings"""
    # Load the AeroTrak data from the Excel file
    aerotrak_data = pd.read_excel(file_path)

    # Strip whitespace from column names to avoid issues
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
    if volume_column in aerotrak_data.columns:
        aerotrak_data["Volume (cm³)"] = (
            aerotrak_data[volume_column] * 1000
        )  # Convert to cm³
        volume_cm = aerotrak_data["Volume (cm³)"]

    def g_mean(x):
        a = np.log(x)
        return np.exp(a.mean())

    # Initialize new columns for mass concentration and calculate values
    pm_columns = []  # List to store the names of new PM concentration columns
    for i, channel in enumerate(size_channels):
        if channel in size_values:
            next_channel = size_channels[i + 1] if i < len(size_channels) - 1 else None
            next_size_value = (
                size_values[next_channel] if next_channel else 25
            )  # Default to 25 if no next channel

            # Calculate the geometric mean of the size range
            particle_size = g_mean([size_values[channel], next_size_value])
            particle_size_m = particle_size * 1e-6  # Convert size from µm to m

            diff_col = f"{channel} Diff (#)"
            if diff_col in aerotrak_data.columns:
                particle_counts = aerotrak_data[diff_col]

                # Calculate the volume of a single particle
                radius_m = particle_size_m / 2
                volume_per_particle = (4 / 3) * np.pi * (radius_m**3)  # Volume in m³

                # Calclate partical mass density (1 g/cm³)
                particle_mass = volume_per_particle * 1e6 * 1e6  # Convert to µg

                # Create new column for mass concentration in µg/m³
                new_diff_col_µg_m3 = (
                    f"PM{size_values[channel]}-{next_size_value} Diff (µg/m³)"
                )

                aerotrak_data[new_diff_col_µg_m3] = (
                    particle_counts / (volume_cm * 1e-6)
                ) * (particle_mass)

                # Add the new PM column name to the list
                pm_columns.append(new_diff_col_µg_m3)

                # Create new column for Diff (#/cm³)
                new_diff_col_cm3 = (
                    f"PM{size_values[channel]}-{next_size_value} Diff (#/cm³)"
                )
                aerotrak_data[new_diff_col_cm3] = (
                    particle_counts / volume_cm
                )  # Convert to #/cm³

            # Handle cumulative counts for PM concentrations
            cumul_col = f"{channel} Cumul (#)"
            if cumul_col in aerotrak_data.columns:
                new_cumul_col = f"{channel} Cumul (#/cm³)"
                aerotrak_data[new_cumul_col] = (
                    aerotrak_data[cumul_col] / volume_cm
                )  # Convert to #/cm³

                # Create new PM concentration column from the Diff column
                pm_column_name = f"PM{next_size_value} (µg/m³)"
                aerotrak_data[pm_column_name] = aerotrak_data[
                    new_diff_col_µg_m3
                ]  # Copy the concentration values

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
    for i in range(len(cumulative_columns)):
        if i == 0:
            aerotrak_data[cumulative_columns[i]] = aerotrak_data[
                pm_columns[i]
            ]  # First column is just copied
        else:
            # Sum current PM column with the previous cumulative column
            aerotrak_data[cumulative_columns[i]] = aerotrak_data[pm_columns[i]].add(
                aerotrak_data[cumulative_columns[i - 1]], fill_value=0
            )

    # Replace invalid entries with NaN for numeric columns only
    status_columns = ["Flow Status", "Laser Status"]
    valid_status = (aerotrak_data[status_columns] == "OK").all(
        axis=1
    )  # Valid rows where all statuses are OK
    for col in aerotrak_data.columns:
        if pd.api.types.is_numeric_dtype(aerotrak_data[col]) and col not in [
            "Date and Time",
            "Sample Time",
            "Volume (L)",
        ]:
            aerotrak_data.loc[~valid_status, col] = pd.NA  # Set invalid entries to NaN

    # Filter based on burn dates
    burn_ids = [f"burn{i}" for i in range(1, 11)]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)

    # Convert 'Date and Time' to date and filter AeroTrak data for the burn dates
    aerotrak_data["Date"] = pd.to_datetime(aerotrak_data["Date and Time"]).dt.date
    filtered_aerotrak_data = aerotrak_data[
        aerotrak_data["Date"].isin(burn_dates.dt.date)
    ]

    # Create a copy of the filtered DataFrame
    filtered_aerotrak_data = filtered_aerotrak_data.copy()

    # Apply time shift for each burn ID in the filtered data
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            # Get the burn date for the current burn ID
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            # Apply the time shift for that specific burn date
            filtered_aerotrak_data = apply_time_shift(
                filtered_aerotrak_data, instrument, burn_id, burn_date
            )

    # Check if there's a special case for burn3 rolling average
    special_cases = INSTRUMENT_CONFIG[instrument].get("special_cases", {})
    if "burn3" in special_cases and special_cases["burn3"].get(
        "apply_rolling_average", False
    ):
        filtered_aerotrak_data = calculate_rolling_average_burn3(
            filtered_aerotrak_data, burn_log
        )

    # Ensure 'Date and Time' is in datetime format before calculating time since closed
    filtered_aerotrak_data["Date and Time"] = pd.to_datetime(
        filtered_aerotrak_data["Date and Time"]
    )

    # Initialize the new column for time since the garage closed
    filtered_aerotrak_data["Time Since Garage Closed (hours)"] = np.nan

    # Calculate time since the garage closed for each burn date
    for index in burn_dates.index:
        burn_date = burn_dates[index]
        garage_closed_time = burn_log.loc[index, "garage closed"]

        # Ensure garage_closed_time is a datetime.time object
        if isinstance(garage_closed_time, str):
            garage_closed_time = pd.to_datetime(garage_closed_time).time()

        # Create a datetime for the closing time
        garage_closed_datetime = pd.Timestamp.combine(burn_date, garage_closed_time)

        # Filter the rows in filtered_aerotrak_data that match the burn date
        matching_rows = filtered_aerotrak_data[
            filtered_aerotrak_data["Date"] == burn_date.date()
        ]

        # Calculate the time difference for matching rows
        if not matching_rows.empty:
            filtered_aerotrak_data.loc[
                filtered_aerotrak_data["Date"] == burn_date.date(), "Date and Time"
            ] = pd.to_datetime(
                filtered_aerotrak_data.loc[
                    filtered_aerotrak_data["Date"] == burn_date.date(), "Date and Time"
                ]
            )

            # Get datetime values and ensure they're timezone-naive
            datetime_values = filtered_aerotrak_data.loc[
                filtered_aerotrak_data["Date"] == burn_date.date(), "Date and Time"
            ]
            if (
                hasattr(datetime_values.dtype, "tz")
                and datetime_values.dtype.tz is not None
            ):
                datetime_values = datetime_values.dt.tz_localize(None)

            # Calculate the time difference in hours
            time_since_closed = (
                datetime_values - garage_closed_datetime
            ).dt.total_seconds() / 3600  # Convert to hours

            # Update the new column for the matching rows
            filtered_aerotrak_data.loc[
                matching_rows.index, "Time Since Garage Closed (hours)"
            ] = time_since_closed
        else:
            print(f"No matching rows found for burn date {burn_date}")

    return filtered_aerotrak_data


# Function to calculate 5-minute rolling average for burn3
def calculate_rolling_average_burn3(data, burn_log):
    # Get the burn date for burn3
    burn3_date = get_burn_date(burn_log, "burn3")
    burn3_date = pd.to_datetime(burn3_date).date()

    # Filter the data for burn3
    burn3_data = data[data["Date"] == burn3_date]

    if burn3_data.empty:
        print("No data available for burn 3.")
        return data  # Return original data if no records

    # Set 'Date and Time' as the index for rolling average calculation
    burn3_data = burn3_data.set_index("Date and Time")

    # Initialize a dictionary to hold the results
    rolling_avg_data = {}

    # Columns to calculate rolling averages (numeric only)
    numeric_columns = burn3_data.select_dtypes(include=[np.number]).columns

    # Calculate rolling averages for numeric columns
    for col_name in numeric_columns:
        rolling_avg_data[col_name] = (
            burn3_data[col_name]
            .rolling(pd.Timedelta(minutes=5))
            .mean()
            .astype(burn3_data[col_name].dtype)
        )

    # For status columns, keep the first value
    status_columns = ["Flow Status", "Instrument Status", "Laser Status"]
    for col_name in status_columns:
        if col_name in burn3_data.columns:
            rolling_avg_data[col_name] = burn3_data[col_name].iloc[
                0
            ]  # Keep the first value

    # Create a new DataFrame with rolling averages and status values
    rolling_avg_df = pd.DataFrame(rolling_avg_data, index=burn3_data.index)

    # Reset index to bring 'Date and Time' back as a column
    rolling_avg_df.reset_index(inplace=True)

    # Replace existing burn3 data in data with rolling averages
    data.loc[
        data["Date"] == burn3_date,
        rolling_avg_df.columns.difference(["Date", "Date and Time"]),
    ] = rolling_avg_df[
        rolling_avg_df.columns.difference(["Date", "Date and Time"])
    ].values

    return data


# Function to get the date of a specific burn from the burn log
def get_burn_date(burn_log, burn_id):
    return burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]


# Function to process DustTrak data
def process_dusttrak_data(file_path, instrument="DustTrak"):
    """Process DustTrak data with standardized output format"""
    # Load the DustTrak data
    dusttrak_data = pd.read_excel(file_path)

    # Strip whitespace from column names
    dusttrak_data.columns = dusttrak_data.columns.str.strip()

    # Specify the columns that need unit conversion (from [mg/m³] to (µg/m³))
    pm_columns = [
        "PM1 [mg/m3]",
        "PM2.5 [mg/m3]",
        "PM4 [mg/m3]",
        "PM10 [mg/m3]",
        "TOTAL [mg/m3]",
    ]

    # Check if the relevant columns exist before proceeding
    for col in pm_columns:
        if col in dusttrak_data.columns:
            # Determine new column name
            if col == "TOTAL [mg/m3]":
                new_col_name = "PM15 (µg/m³)"
            else:
                new_col_name = col.replace("[mg/m3]", "(µg/m³)")

            # Convert and assign to new column name
            dusttrak_data[new_col_name] = (
                dusttrak_data[col] * 1000
            )  # Convert from mg/m³ to µg/m³

    # Filter by burn dates using 'datetime' column
    filtered_data = filter_by_burn_dates(dusttrak_data, range(1, 11), "datetime")

    # Apply time shift for each burn
    burn_ids = [f"burn{i}" for i in range(1, 11)]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            # Get the burn date for the current burn ID
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            # Apply the time shift for that specific burn date
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    return filtered_data


def process_miniams_data(file_path, instrument="MiniAMS"):
    """Process Mini-AMS data with standardized output format"""
    # Load the Mini-AMS data
    miniams_data = pd.read_excel(file_path)

    # Strip whitespace from column names
    miniams_data.columns = miniams_data.columns.str.strip()

    print(f"Original Mini-AMS columns: {miniams_data.columns.tolist()}")

    # Rename columns to standard format with units
    column_mapping = {
        "Org": "Organic (µg/m³)",
        "NO3": "Nitrate (µg/m³)",
        "SO4": "Sulfate (µg/m³)",
        "NH4": "Ammonium (µg/m³)",
        "Chl": "Chloride (µg/m³)",
    }

    miniams_data.rename(columns=column_mapping, inplace=True)

    print(f"Renamed Mini-AMS columns: {miniams_data.columns.tolist()}")

    # Convert DateTime to proper datetime format
    miniams_data["DateTime"] = pd.to_datetime(miniams_data["DateTime"], errors="coerce")

    # Get burn range from instrument configuration (burns 1-3)
    burn_range = INSTRUMENT_CONFIG[instrument].get("burn_range", range(1, 4))

    # Filter by burn dates
    filtered_data = filter_by_burn_dates(miniams_data, burn_range, "DateTime")

    print(f"Filtered Mini-AMS data shape: {filtered_data.shape}")
    print(
        f"Date range: {filtered_data['DateTime'].min()} to {filtered_data['DateTime'].max()}"
    )

    # Apply time shift for each burn (0 minutes for Mini-AMS)
    burn_ids = [f"burn{i}" for i in burn_range]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            # Get the burn date for the current burn ID
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            # Apply the time shift for that specific burn date (0 minutes)
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    # Calculate time since garage closed for all data points
    if not filtered_data.empty:
        filtered_data = filtered_data.copy()
        filtered_data["Time Since Garage Closed (hours)"] = np.nan

        for burn_id in burn_ids:
            burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
            if burn_date_row.empty:
                continue

            burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
            garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

            if pd.isna(garage_closed_time_str):
                continue

            try:
                garage_closed_time = create_naive_datetime(
                    burn_date.date(), garage_closed_time_str
                )

                # Filter data for this burn_id
                burn_mask = filtered_data["Date"] == burn_date.date()

                if not any(burn_mask):
                    continue

                # Ensure datetime is properly formatted
                burn_datetime = filtered_data.loc[burn_mask, "DateTime"]

                # Make sure datetimes are timezone-naive
                if (
                    hasattr(burn_datetime.dtype, "tz")
                    and burn_datetime.dtype.tz is not None
                ):
                    burn_datetime = burn_datetime.dt.tz_localize(None)

                # Calculate time since garage closed (in hours)
                filtered_data.loc[burn_mask, "Time Since Garage Closed (hours)"] = (
                    burn_datetime - garage_closed_time
                ).dt.total_seconds() / 3600
            except Exception as e:
                print(f"Error calculating garage closed time for {burn_id}: {str(e)}")
                continue

    print(f"Final Mini-AMS processed data shape: {filtered_data.shape}")
    if not filtered_data.empty:
        print("Sample of processed data:")
        print(
            filtered_data[
                [
                    "DateTime",
                    "Organic (µg/m³)",
                    "Nitrate (µg/m³)",
                    "Time Since Garage Closed (hours)",
                ]
            ].head()
        )

    return filtered_data


# Function to process PurpleAirK data
def process_purpleairk_data(file_path):
    """Process PurpleAirK data with standardized output format"""
    # Load the PurpleAirK data
    purpleair_data = pd.read_excel(file_path, sheet_name="(P2)kitchen")

    # Strip whitespace from column names
    purpleair_data.columns = purpleair_data.columns.str.strip()

    # Rename 'Average' column to 'PM2.5 (µg/m³)'
    purpleair_data.rename(columns={"Average": "PM2.5 (µg/m³)"}, inplace=True)

    # Filter by burn dates using 'DateTime' column (burns 6-10 for PurpleAir)
    return filter_by_burn_dates(purpleair_data, range(6, 11), "DateTime")


# Function to process SMPS data
def process_smps_data(file_path, instrument="SMPS"):
    """Process SMPS data with standardized output format"""
    print(f"Processing SMPS data from {file_path}")

    # Initialize an empty DataFrame to store combined SMPS data
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
            # Format the date for the filename
            date_str = burn_date.strftime("%m%d%Y")
            smps_filename = f"MH_apollo_bed_{date_str}_MassConc.xlsx"
            smps_file_path = os.path.join(file_path, smps_filename)

            # Check if file exists
            if not os.path.exists(smps_file_path):
                print(f"File not found: {smps_file_path}")
                continue

            print(f"Reading file: {smps_file_path}")

            # Read the first sheet of the Excel file
            smps_data = pd.read_excel(smps_file_path, sheet_name=0)

            # Transpose the data if it's not already in the right format
            if (
                "Date" not in smps_data.columns
                and "Start Time" not in smps_data.columns
            ):
                # Assume the first row should be headers
                smps_data = smps_data.transpose()

                # Use the first row as headers
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
                print(f"Required columns missing in {smps_filename}: {missing_columns}")
                continue

            # Rename 'Total Concentration(µg/m³)' to 'Total Concentration (µg/m³)' to add space
            if "Total Concentration(µg/m³)" in smps_data.columns:
                smps_data = smps_data.rename(
                    columns={
                        "Total Concentration(µg/m³)": "Total Concentration (µg/m³)"
                    }
                )

            # Get the minimum and maximum size boundaries and ensure they're numbers
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
                print(f"Error converting size boundaries to float in {smps_filename}")
                min_size = 9.47
                max_size = 414.2

            # Create bin ranges if not already defined
            if not bin_ranges:
                # Define bin boundaries based on min_size
                bin_ranges = [(min_size, 100), (100, 200), (200, 300), (300, max_size)]

                # Create column names for bin ranges
                bin_columns = [
                    f"Ʃ{int(start)}-{int(end)}nm (µg/m³)" for start, end in bin_ranges
                ]

                # Update INSTRUMENT_CONFIG with actual column names
                # Note: first we add the bins, then Total Concentration at the end
                INSTRUMENT_CONFIG["SMPS"]["process_pollutants"] = bin_columns + [
                    "Total Concentration (µg/m³)"
                ]
                INSTRUMENT_CONFIG["SMPS"]["plot_pollutants"] = bin_columns + [
                    "Total Concentration (µg/m³)"
                ]

            # Convert Date and Start Time to datetime with better error handling
            try:
                smps_data["Date"] = pd.to_datetime(smps_data["Date"], errors="coerce")
                smps_data["Start Time"] = pd.to_datetime(
                    smps_data["Start Time"], format="%H:%M:%S", errors="coerce"
                )
            except Exception as e:
                print(f"Error converting datetime in {smps_filename}: {str(e)}")
                continue

            # Check for NaN values after conversion
            if smps_data["Date"].isna().all() or smps_data["Start Time"].isna().all():
                print(f"All datetime values are NaN in {smps_filename}")
                continue

            # Create datetime column by combining Date and Start Time
            try:
                smps_data["datetime"] = pd.to_datetime(
                    smps_data["Date"].dt.strftime("%Y-%m-%d")
                    + " "
                    + smps_data["Start Time"].dt.strftime("%H:%M:%S"),
                    errors="coerce",
                )
            except Exception as e:
                print(f"Error creating datetime in {smps_filename}: {str(e)}")
                continue

            # Drop rows with invalid datetime
            invalid_rows = smps_data["datetime"].isna().sum()
            if invalid_rows > 0:
                print(
                    f"Dropping {invalid_rows} rows with invalid datetime in {smps_filename}"
                )
                smps_data = smps_data.dropna(subset=["datetime"])

            if smps_data.empty:
                print(f"No valid data after datetime filtering in {smps_filename}")
                continue

            # Calculate mid-time between current and next entries
            # First sort by datetime
            smps_data = smps_data.sort_values("datetime").reset_index(drop=True)

            # Shift datetime to get next datetime
            smps_data["next_datetime"] = smps_data["datetime"].shift(-1)

            # Calculate mid-time (average of current and next)
            # For the last row, keep the original time
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

                # Replace datetime with mid_datetime
                smps_data["datetime"] = smps_data["mid_datetime"]
            except Exception as e:
                print(
                    f"Error calculating mid-time in {smps_filename}, using original times: {str(e)}"
                )
                # Keep original datetime if calculation fails

            # Drop temporary columns
            smps_data = smps_data.drop(
                ["next_datetime", "mid_datetime"], axis=1, errors="ignore"
            )

            # Get numeric columns (size bins) by filtering out non-numeric ones
            known_non_numeric = [
                "Date",
                "Start Time",
                "datetime",
                "Lower Size(nm)",
                "Upper Size(nm)",
                "Total Concentration (µg/m³)",
            ]

            # Identify numeric columns by checking if they can be cast to float
            numeric_cols = []
            for col in smps_data.columns:
                if col not in known_non_numeric:
                    try:
                        float(col)
                        numeric_cols.append(col)
                    except (ValueError, TypeError):
                        pass

            if not numeric_cols:
                print(f"No size bin columns found in {smps_filename}")
                continue

            # Ensure Total Concentration is numeric
            try:
                smps_data["Total Concentration (µg/m³)"] = pd.to_numeric(
                    smps_data["Total Concentration (µg/m³)"], errors="coerce"
                )
            except Exception as e:
                print(
                    f"Error converting Total Concentration to numeric in {smps_filename}: {str(e)}"
                )

            # OPTIMIZATION: Create a new DataFrame to avoid fragmentation warning
            new_data = {}

            # Sum columns within each bin range
            for i, (start, end) in enumerate(bin_ranges):
                # Get columns in this range
                try:
                    bin_cols = [
                        col for col in numeric_cols if start <= float(col) <= end
                    ]
                except (ValueError, TypeError):
                    print(
                        f"Error filtering columns for bin {start}-{end} in {smps_filename}"
                    )
                    bin_cols = []

                if bin_cols:
                    # Use the correct bin name
                    bin_name = bin_columns[i]

                    # Sum the columns in this bin with error handling - OPTIMIZED
                    try:
                        # Create a dictionary of numeric columns
                        numeric_data = {}
                        for col in bin_cols:
                            numeric_data[col] = pd.to_numeric(
                                smps_data[col], errors="coerce"
                            )

                        # Calculate sum and store in new_data
                        new_data[bin_name] = pd.DataFrame(numeric_data).sum(axis=1)
                    except Exception as e:
                        print(
                            f"Error summing bin {bin_name} in {smps_filename}: {str(e)}"
                        )
                        # Create empty column
                        new_data[bin_name] = pd.Series(np.nan, index=smps_data.index)

            # Get the burn_id for this date
            burn_id_row = burn_log[burn_log["Date"] == burn_date]
            if not burn_id_row.empty:
                burn_id = burn_id_row["Burn ID"].iloc[0]

                # Create a new DataFrame with all needed columns to avoid fragmentation
                result_df = pd.DataFrame(
                    {
                        "datetime": smps_data["datetime"],
                        "Date": smps_data["datetime"].dt.date,
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

                # Concatenate with combined data
                combined_smps_data = pd.concat(
                    [combined_smps_data, result_df], ignore_index=True
                )

        except Exception as e:
            print(f"Error processing SMPS file for date {burn_date}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Calculate time since garage closed for all data points
    if not combined_smps_data.empty:
        for burn_id in burn_ids:
            burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
            if burn_date_row.empty:
                continue

            burn_date = burn_date_row["Date"].iloc[0]
            garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

            if pd.isna(garage_closed_time_str):
                continue

            try:
                garage_closed_time = create_naive_datetime(
                    burn_date, garage_closed_time_str
                )

                # Filter data for this burn_id
                burn_mask = combined_smps_data["burn_id"] == burn_id

                if not any(burn_mask):
                    continue

                # Ensure datetime is properly formatted
                burn_datetime = combined_smps_data.loc[burn_mask, "datetime"]

                # Make sure datetimes are timezone-naive
                if (
                    hasattr(burn_datetime.dtype, "tz")
                    and burn_datetime.dtype.tz is not None
                ):
                    burn_datetime = burn_datetime.dt.tz_localize(None)

                # Calculate time since garage closed (in hours)
                combined_smps_data.loc[
                    burn_mask, "Time Since Garage Closed (hours)"
                ] = (burn_datetime - garage_closed_time).dt.total_seconds() / 3600
            except Exception as e:
                print(f"Error calculating garage closed time for {burn_id}: {str(e)}")
                continue

    # Final check of data quality
    if not combined_smps_data.empty:
        # Replace any infinite values with NaN
        combined_smps_data = combined_smps_data.replace([np.inf, -np.inf], np.nan)

        # Drop rows where all concentration measurements are NaN
        concentration_cols = ["Total Concentration (µg/m³)"] + bin_columns
        existing_cols = [
            col for col in concentration_cols if col in combined_smps_data.columns
        ]

        if existing_cols:
            na_rows = combined_smps_data[existing_cols].isna().all(axis=1).sum()
            if na_rows > 0:
                print(f"Dropping {na_rows} rows with all NaN concentration values")
                combined_smps_data = combined_smps_data.dropna(
                    subset=existing_cols, how="all"
                )

    print(f"Processed SMPS data: {len(combined_smps_data)} records")
    if not combined_smps_data.empty:
        print(f"Columns available: {', '.join(combined_smps_data.columns)}")
    else:
        print("No valid SMPS data after processing")

    return combined_smps_data


# Function to process QuantAQ data
def process_quantaq_data(file_path, instrument="QuantAQB"):
    """Process QuantAQ data with standardized output format"""
    # Load the QuantAQ data
    quantaq_data = pd.read_csv(file_path)

    # Reverse the data order
    quantaq_data = quantaq_data.iloc[::-1].reset_index(drop=True)

    # Strip whitespace from column names
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
    ).dt.tz_localize(None)  # Remove timezone info

    # Get burn range from instrument configuration
    burn_range = INSTRUMENT_CONFIG[instrument].get("burn_range", range(4, 11))

    # Filter by burn dates
    filtered_data = filter_by_burn_dates(quantaq_data, burn_range, "timestamp_local")

    # Apply time shift for each burn
    burn_ids = [f"burn{i}" for i in burn_range]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            # Get the burn date for the current burn ID
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            # Apply the time shift for that specific burn date
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    return filtered_data


# Function to fit exponential curve
def fit_exponential_curve(x_data, y_data, initial_guess):
    import numpy as np
    from scipy.optimize import curve_fit

    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("Input data for fitting is empty.")

    # Exponential decay function
    def exp_decreasing(t, a, b):
        return a * np.exp(-b * t)

    # Perform the curve fit
    try:
        popt, pcov = curve_fit(
            exp_decreasing, x_data, y_data, p0=initial_guess, maxfev=10000
        )
    except Exception as e:
        print(f"Curve fitting error: {e}")
        raise

    # Calculate the fitted y values based on the optimal parameters
    y_fit = exp_decreasing(x_data, *popt)

    # Calculate standard error
    perr = np.sqrt(np.diag(pcov))

    return popt, y_fit, perr


# Generalized calculate_all_decay_parameters function
def calculate_all_decay_parameters(data, instrument):
    """Calculate decay parameters for all pollutants with instrument-specific settings"""
    global burn_log, decay_parameters

    # Get configuration for the instrument
    config = INSTRUMENT_CONFIG[instrument]
    pollutants = config["process_pollutants"]
    datetime_column = config["datetime_column"]
    special_cases = config.get("special_cases", {})
    normalize_pollutant = config.get("normalize_pollutant", "PM10 (µg/m³)")

    # Create directory for decay info files if it doesn't exist
    burn_calcs_path = get_common_file("burn_calcs")
    os.makedirs(burn_calcs_path, exist_ok=True)

    # Initialize list to store decay info for saving to file
    decay_info_lines = []
    decay_info_lines.append(f"Decay Parameters for {instrument}")
    decay_info_lines.append("-" * 50)

    decay_details_rows = []

    # Ensure datetime column is properly formatted
    data[datetime_column] = pd.to_datetime(data[datetime_column])

    # Get unique burn dates
    unique_burn_dates = data["Date"].unique()

    # Process each burn
    for burn_date in unique_burn_dates:
        burn_data = data[data["Date"] == burn_date].copy()
        burn_data = burn_data.reset_index(drop=True)

        # Skip if there's no data for this burn
        if burn_data.empty:
            continue

        # Get the burn ID
        burn_id_row = burn_log[burn_log["Date"] == pd.to_datetime(burn_date)]
        burn_id = burn_id_row["Burn ID"].values[0] if not burn_id_row.empty else None

        if burn_id is None:
            continue

        # Get garage closed time
        garage_closed_time_str = burn_id_row["garage closed"].values[0]
        garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)

        if pd.isna(garage_closed_time):
            continue

        # Ensure both datetime objects are timezone-naive for consistent subtraction
        burn_data_datetime = burn_data[datetime_column]
        if (
            hasattr(burn_data_datetime.dtype, "tz")
            and burn_data_datetime.dtype.tz is not None
        ):
            burn_data_datetime = burn_data_datetime.dt.tz_localize(None)

        # Calculate time since garage closed
        burn_data["Time Since Garage Closed (hours)"] = (
            burn_data_datetime - garage_closed_time
        ).dt.total_seconds() / 3600

        # Initialize entry for this burn in the decay parameters dictionary
        decay_parameters[burn_id] = {}

        # Add normalized data if the normalization pollutant exists
        if normalize_pollutant in burn_data.columns:
            # Calculate the maximum concentration for normalization
            max_concentration = burn_data[normalize_pollutant].max()
            if pd.notna(max_concentration) and max_concentration > 0:
                normalized_column = f"Normalized {normalize_pollutant.split()[0]}"
                burn_data[normalized_column] = burn_data[normalize_pollutant] / (
                    max_concentration
                )

                # Add normalized pollutant to the list for analysis
                all_pollutants = pollutants + [normalized_column]
            else:
                all_pollutants = pollutants
        else:
            all_pollutants = pollutants

        for pollutant in all_pollutants:
            # Skip if pollutant not in columns
            if pollutant not in burn_data.columns:
                continue

            # Check if this is a normalized pollutant
            is_normalized = pollutant.startswith("Normalized")

            # If it's a normalized pollutant, we need to get the base pollutant name
            if is_normalized:
                # Extract the base pollutant name (e.g., "PM10" from "Normalized PM10")
                base_pollutant_prefix = pollutant.split(" ")[
                    1
                ]  # Extract "PM10" from "Normalized PM10"
                base_pollutant = next(
                    (
                        p
                        for p in pollutants
                        if p.startswith(f"{base_pollutant_prefix} ")
                    ),
                    None,
                )

                # Skip if the base pollutant doesn't exist or hasn't been processed yet
                if base_pollutant is None or base_pollutant not in decay_parameters.get(
                    burn_id, {}
                ):
                    print(
                        f"Base pollutant {base_pollutant_prefix} not found or not processed for {burn_id}, skipping normalized calculation"
                    )
                    continue

                # Reuse the decay start and end times from the base pollutant
                base_decay_params = decay_parameters[burn_id][base_pollutant]
                decay_start_time = base_decay_params["decay_start_time"]
                decay_end_time = base_decay_params["decay_end_time"]

                # Find the maximum concentration for the current pollutant for other calculations
                max_concentration = burn_data[pollutant].max()
                if pd.isna(max_concentration):
                    print(
                        f"No valid data for {pollutant} on {burn_date}, skipping decay calculation."
                    )
                    continue
            else:
                # This is a regular pollutant, so proceed with normal decay parameter calculation
                # Find the maximum concentration for the current pollutant
                max_concentration = burn_data[pollutant].max()

                if pd.isna(max_concentration):
                    print(
                        f"No valid data for {pollutant} on {burn_date}, skipping decay calculation."
                    )
                    continue

                # Identify the timestamp of the maximum concentration
                max_index = burn_data[pollutant].idxmax()
                max_time = burn_data["Time Since Garage Closed (hours)"].iloc[max_index]

                # Filter out non-positive concentrations and calculate log values
                burn_data["filtered_concentration"] = pd.to_numeric(
                    burn_data[pollutant], errors="coerce"
                ).where(
                    pd.to_numeric(burn_data[pollutant], errors="coerce") > 0,
                    other=np.nan,
                )

                # Safely calculate log values
                try:
                    # Ensure filtered_concentration is numeric and doesn't have any problematic values
                    numeric_values = pd.to_numeric(
                        burn_data["filtered_concentration"], errors="coerce"
                    )
                    # Calculate log only for valid values
                    burn_data["log_concentration"] = np.log(numeric_values)
                except Exception as e:
                    print(
                        f"Error calculating logarithm for {burn_id} {pollutant}: {str(e)}"
                    )
                    print(
                        f"Data types: filtered_concentration={burn_data['filtered_concentration'].dtype}"
                    )
                    print(
                        f"Sample values: {burn_data['filtered_concentration'].head()}"
                    )
                    # Provide a fallback to avoid breaking the entire process
                    burn_data["log_concentration"] = np.nan
                    continue  # Skip this pollutant and continue with others

                # Calculate rolling derivative of log-transformed data
                burn_data["rolling_derivative"] = (
                    burn_data["log_concentration"].diff().rolling(window=5).mean()
                )

                # Initialize decay start time to max concentration time
                decay_start_time = max_time

                # Check if this burn has special case for decay time calculation
                burn_special_case = special_cases.get(burn_id, {})
                if burn_special_case.get("custom_decay_time", False):
                    # CUSTOM CODE FOR BURN6 IN SMPS
                    if burn_id == "burn6" and instrument == "SMPS":
                        # Get CR Box on time
                        cr_box_on_time_str = burn_id_row["CR Box on"].values[0]
                        cr_box_on_time = create_naive_datetime(
                            burn_date, cr_box_on_time_str
                        )

                        if pd.notna(cr_box_on_time):
                            # Calculate time since garage closed for CR box on
                            cr_box_on_time_since_garage_closed = (
                                cr_box_on_time - garage_closed_time
                            ).total_seconds() / 3600

                            # Find data within 3 minutes (0.05 hours) of CR box on time
                            window_data = burn_data[
                                (
                                    burn_data["Time Since Garage Closed (hours)"]
                                    >= cr_box_on_time_since_garage_closed - 0.05
                                )
                                & (
                                    burn_data["Time Since Garage Closed (hours)"]
                                    <= cr_box_on_time_since_garage_closed + 0.05
                                )
                            ]

                            if not window_data.empty:
                                # Find the maximum concentration within this window
                                max_window_idx = window_data[pollutant].idxmax()
                                max_time = window_data[
                                    "Time Since Garage Closed (hours)"
                                ].iloc[max_window_idx - window_data.index[0]]

                                # Set decay start time to this maximum and end time based on offset
                                decay_start_time = max_time
                                decay_end_time = (
                                    decay_start_time
                                    + burn_special_case.get("decay_end_offset", 0.25)
                                )

                                print(
                                    f"SMPS burn6: Using max concentration within 3 minutes of CR Box on time. Max at {max_time:.2f}h"
                                )
                            else:
                                print(
                                    f"No data found within 3 minutes of CR Box on time for SMPS {burn_id}, using default max"
                                )
                                # Set decay start time to max_concentration and decay end time based on offset
                                decay_start_time = max_time  # Default max
                                decay_end_time = (
                                    decay_start_time
                                    + burn_special_case.get("decay_end_offset", 0.25)
                                )
                    else:
                        # Original code for other instruments
                        # Set decay start time to max_concentration and decay end time based on offset
                        decay_start_time = max_time
                        decay_end_time = decay_start_time + burn_special_case.get(
                            "decay_end_offset", 0.25
                        )  # Default 15 minutes
                else:
                    # Initialize decay start time to max concentration time, but ensure it's not before garage closed
                    decay_start_time = max(
                        0, max_time
                    )  # 0 is the time when garage closed

                    # Get CR Box on time for other burns
                    if not burn_id_row.empty:
                        cr_box_on_time_str = burn_id_row["CR Box on"].values[0]
                        cr_box_on_time = create_naive_datetime(
                            burn_date, cr_box_on_time_str
                        )

                        if pd.notna(cr_box_on_time):
                            cr_box_on_time_since_garage_closed = (
                                cr_box_on_time - garage_closed_time
                            ).total_seconds() / 3600
                            # Use the later of max concentration time or CR Box on time
                            decay_start_time = max(
                                decay_start_time, cr_box_on_time_since_garage_closed
                            )

                    # Search for stable derivative after decay_start_time for other burns
                    valid_start_found = False
                    for idx in range(max_index + 1, len(burn_data)):
                        if (
                            idx >= max_index + 5
                        ):  # Ensure enough points for rolling mean
                            rolling_mean = (
                                burn_data["rolling_derivative"]
                                .iloc[idx - 3 : idx + 1]
                                .mean()
                            )
                            # Fixed stability threshold (no adjustment for normalized data here)
                            stability_threshold = 0.1
                            if (
                                abs(
                                    burn_data["rolling_derivative"].iloc[idx]
                                    - rolling_mean
                                )
                                < stability_threshold
                            ):
                                potential_start_time = burn_data[
                                    "Time Since Garage Closed (hours)"
                                ].iloc[idx]
                                if potential_start_time >= decay_start_time:
                                    decay_start_time = potential_start_time
                                    valid_start_found = True
                                    break

                    if not valid_start_found:
                        print(
                            f"Could not find stable decay start for {burn_id} {pollutant}"
                        )
                        continue

                    # Find decay end time (5% threshold)
                    threshold_value = 0.05 * max_concentration

                    below_threshold = burn_data[
                        (
                            burn_data["Time Since Garage Closed (hours)"]
                            > decay_start_time
                        )
                        & (burn_data[pollutant] < threshold_value)
                    ]

                    if below_threshold.empty:
                        print(f"No values below 5% threshold for {burn_id} {pollutant}")
                        # Use the last available time point instead
                        valid_data = burn_data[
                            (
                                burn_data["Time Since Garage Closed (hours)"]
                                > decay_start_time
                            )
                            & (burn_data[pollutant] > 0)
                        ]
                        if valid_data.empty:
                            print(
                                f"No valid data points after decay start time for {burn_id} {pollutant}"
                            )
                            continue
                        # Use the last valid data point as decay end time
                        decay_end_time = valid_data[
                            "Time Since Garage Closed (hours)"
                        ].iloc[-1]
                        print(
                            f"Using last available time point ({decay_end_time:.2f} h) as decay end time for {burn_id} {pollutant}"
                        )
                    else:
                        decay_end_time = below_threshold[
                            "Time Since Garage Closed (hours)"
                        ].iloc[0]

            # Extract final decay data
            decay_data = burn_data[
                (burn_data["Time Since Garage Closed (hours)"] >= decay_start_time)
                & (burn_data["Time Since Garage Closed (hours)"] <= decay_end_time)
                & (burn_data[pollutant] > 0)  # Ensure no zero/negative values
            ].copy()

            if decay_data.empty or len(decay_data) < 3:
                print(f"Insufficient decay data points for {burn_id} {pollutant}")
                continue

            # Print the decay range information
            # print(f"{burn_id} {pollutant}")
            # print(f"  Max Value: {max_concentration:.2f}")
            # print(f"  Decay Start: {decay_start_time:.2f} h (Value: {decay_data[pollutant].iloc[0]:.2f})")
            # print(f"  Decay End: {decay_end_time:.2f} h (Value: {decay_data[pollutant].iloc[-1]:.2f})")

            # Add to decay info for file output
            decay_info_lines.append(f"{burn_id} {pollutant}")
            decay_info_lines.append(f"  Max Value: {max_concentration:.4f}")
            decay_info_lines.append(
                f"  Decay Start: {decay_start_time:.4f} h (Value: {decay_data[pollutant].iloc[0]:.4f})"
            )
            decay_info_lines.append(
                f"  Decay End: {decay_end_time:.4f} h (Value: {decay_data[pollutant].iloc[-1]:.4f})"
            )

            # Prepare data for exponential fitting
            x_data = (
                decay_data["Time Since Garage Closed (hours)"].values - decay_start_time
            )
            y_data = decay_data[pollutant].values

            # Initial guess based on data
            initial_amplitude = y_data[0]
            initial_decay_rate = 0.1
            initial_guess = [initial_amplitude, initial_decay_rate]

            # Fit exponential curve
            try:
                popt, y_fit, perr = fit_exponential_curve(x_data, y_data, initial_guess)

                # Check if relative standard deviation is > 0.1 (10%)
                # Calculate relative standard deviation as uncertainty/decay_rate
                rsd = perr[1] / popt[1]

                if rsd > 0.1:
                    print(
                        f"Decay for {burn_id} {pollutant} has RSD of {rsd:.2f} (> 0.1), excluding from results"
                    )
                    decay_info_lines.append(
                        f"  Decay Rate: {popt[1]:.4f} ± {1.96 * perr[1]:.4f}  h⁻¹ (RSD: {rsd:.2f}) - EXCLUDED"
                    )
                    decay_info_lines.append("")  # Empty line for readability
                    continue

                # Store parameters in the global dictionary
                decay_parameters[burn_id][pollutant] = {
                    "decay_start_time": decay_start_time,
                    "decay_end_time": decay_end_time,
                    "amplitude": popt[0],
                    "decay_rate": popt[1],
                    "uncertainty": 1.96 * perr[1],
                    "rsd": rsd,
                    "x_data": x_data,
                    "y_data": y_data,
                }

                # Add to decay info for file output
                decay_info_lines.append(
                    f"  Decay Rate: {popt[1]:.4f} ± {1.96 * perr[1]:.4f}  h⁻¹ (RSD: {rsd:.2f})"
                )
                decay_info_lines.append("")  # Empty line for readability

                # Add row to the decay details DataFrame
                new_row = {
                    "burn": burn_id,
                    "pollutant": pollutant,
                    "max_concentration": max_concentration,
                    "decay_start_time": decay_start_time,
                    "start_value": decay_data[pollutant].iloc[0],
                    "decay_end_time": decay_end_time,
                    "end_value": decay_data[pollutant].iloc[-1],
                    "decay": popt[1],
                    "decay_uncertainty": 1.96 * perr[1],
                    "rsd": rsd,
                }
                decay_details_rows.append(new_row)

                # Add to burn_calc for PM data (not for normalized)
                if not pollutant.startswith("Normalized"):
                    new_row = {
                        "burn": burn_id,
                        "pollutant": pollutant,
                        "decay": f"{popt[1]:.4f}",
                        "decay_uncertainty": f"{1.96 * perr[1]:.4f}",
                        "rsd": f"{rsd:.4f}",
                    }
                    burn_calc.append(new_row)

            except Exception as e:
                print(f"Error fitting {burn_id} {pollutant}: {str(e)}")
                continue

    # Store the decay details DataFrame for later use
    global decay_details_dataframe
    decay_details_dataframe = pd.DataFrame(decay_details_rows)

    # Return the calculated parameters
    return decay_parameters


# Generalized plot_pm_data function
def plot_pm_data(data, instrument):
    """Plot PM data with instrument-specific settings"""
    global burn_log, decay_parameters

    # Get configuration for the instrument
    config = INSTRUMENT_CONFIG[instrument]
    pollutants = config["plot_pollutants"]
    datetime_column = config["datetime_column"]

    # Get script metadata
    metadata = get_script_metadata()

    # Ensure directory exists
    output_figures_path = get_common_file("output_figures")
    os.makedirs(output_figures_path, exist_ok=True)

    # Check if decay parameters have been calculated
    if not decay_parameters:
        print("Calculating decay parameters first...")
        calculate_all_decay_parameters(data, instrument)

    # Ensure datetime column is properly formatted
    data.loc[:, datetime_column] = pd.to_datetime(data[datetime_column])

    # Get unique burn dates
    unique_burn_dates = data["Date"].unique()

    # Create a separate plot for each burn
    for burn_date in unique_burn_dates:
        burn_data = data[data["Date"] == burn_date]
        burn_data = burn_data.reset_index(drop=True)

        # Skip if there's no data for this burn
        if burn_data.empty:
            continue

        # Get the burn ID for later checks
        burn_id_row = burn_log[burn_log["Date"] == pd.to_datetime(burn_date)]
        burn_id = burn_id_row["Burn ID"].values[0] if not burn_id_row.empty else None

        if burn_id not in decay_parameters:
            print(f"No decay parameters available for {burn_id}, skipping plot")
            continue

        # Get garage closed time
        garage_closed_time_str = burn_id_row["garage closed"].values[0]
        garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)

        if pd.notna(garage_closed_time):
            # Create a new column for time since garage closed
            burn_data["Time Since Garage Closed (hours)"] = (
                burn_data[datetime_column] - garage_closed_time
            ).dt.total_seconds() / 3600
        else:
            print(f"Missing garage closed time for {burn_id}")
            continue

        # Create a ColumnDataSource after adding required columns
        source = ColumnDataSource(burn_data)

        # Create a new plot with y-range limits
        p = figure(
            x_axis_label="Time Since Garage Closed (hours)",
            y_axis_label="PM Concentration (µg/m³)",
            x_axis_type="linear",
            y_axis_type="log",
            width=800,
            height=500,
            y_range=(10**-2, 10**4),  # Set y-axis limits
            title=f"{instrument} {burn_id} PM-dependent-size mass-concentration",
        )

        # Set x-axis range
        p.x_range.start = -1  # Start at -1 hour
        p.x_range.end = 4  # End at +4 hours

        # Get garage closed time
        garage_closed_time_str = burn_id_row["garage closed"].values[0]
        garage_closed_time = pd.to_datetime(
            f"{burn_date} {garage_closed_time_str}", errors="coerce"
        )

        if pd.notna(garage_closed_time):
            # Create a new column for time since garage closed
            burn_data["Time Since Garage Closed (hours)"] = (
                burn_data[datetime_column] - garage_closed_time
            ).dt.total_seconds() / 3600

            # For SMPS, reorder pollutants to put Total Concentration last
            if instrument == "SMPS":
                # Put Total Concentration last
                ordered_pollutants = [
                    p for p in pollutants if p != "Total Concentration (µg/m³)"
                ]
                if "Total Concentration (µg/m³)" in pollutants:
                    ordered_pollutants.append("Total Concentration (µg/m³)")
            else:
                ordered_pollutants = pollutants

            # Loop through each pollutant to plot
            for pollutant in ordered_pollutants:
                if pollutant in burn_data.columns:
                    # For SMPS data, handle special Ʃ labeling
                    if "Ʃ" in pollutant:
                        # For bins like 'Ʃ9-100nm (µg/m³)', use '9-100nm' as legend label
                        start_pos = pollutant.find("Ʃ") + 1
                        end_pos = pollutant.find(" (µg/m³)")
                        if end_pos > start_pos:
                            legend_label = pollutant[start_pos:end_pos]
                        else:
                            legend_label = pollutant.replace(" (µg/m³)", "")
                    else:
                        # Remove ' (µg/m³)' from the pollutant name for the legend
                        legend_label = pollutant.replace(" (µg/m³)", "")

                    color = POLLUTANT_COLORS.get(
                        pollutant, "gray"
                    )  # Use gray as fallback
                    p.line(
                        "Time Since Garage Closed (hours)",
                        pollutant,
                        source=source,
                        legend_label=legend_label,
                        line_width=1.5,
                        color=color,
                    )

            # Add vertical line for garage closed
            garage_closed_time_full = 0  # Set this to 0 since it's the reference point
            p.line(
                x=[garage_closed_time_full] * 2,
                y=[10**-2, 10**4],
                line_color="black",
                line_width=1,
                line_dash="solid",
                legend_label="Garage Closed",
            )

            # Add vertical line for CR Box On, adjusted for time since garage closed
            cr_box_on_time_str = burn_id_row["CR Box on"].values[0]
            cr_box_on_time = create_naive_datetime(burn_date, cr_box_on_time_str)

            if pd.notna(cr_box_on_time):
                cr_box_on_time_since_garage_closed = (
                    cr_box_on_time - garage_closed_time
                ).total_seconds() / 3600  # Convert to hours
                p.line(
                    x=[cr_box_on_time_since_garage_closed] * 2,
                    y=[10**-2, 10**4],
                    line_color="black",
                    line_width=1,
                    line_dash="dashed",
                    legend_label="CR Box on",
                )

        # --- Decay Plotting Section Using Pre-calculated Parameters ---
        decay_rates_info = []

        # Use the same ordered pollutants for decay plots
        if instrument == "SMPS":
            ordered_decay_pollutants = [
                p for p in pollutants if p != "Total Concentration (µg/m³)"
            ]
            if "Total Concentration (µg/m³)" in pollutants:
                ordered_decay_pollutants.append("Total Concentration (µg/m³)")
        else:
            ordered_decay_pollutants = pollutants

        for pollutant in ordered_decay_pollutants:
            if pollutant not in decay_parameters[burn_id]:
                print(f"No decay parameters for {pollutant} in {burn_id}")
                continue

            color = POLLUTANT_COLORS.get(
                pollutant, "gray"
            )  # Get the color for the current pollutant
            decay_params = decay_parameters[burn_id][pollutant]

            try:
                # Get parameters
                decay_start_time = decay_params["decay_start_time"]
                amplitude = decay_params["amplitude"]
                decay_rate = decay_params["decay_rate"]
                uncertainty = decay_params["uncertainty"]
                x_data = decay_params["x_data"]

                # Create points for the fitted curve
                x_fit = np.linspace(0, x_data[-1], 100)
                y_fit = amplitude * np.exp(-decay_rate * x_fit)

                # Plot the fitted curve
                p.line(
                    x_fit + decay_start_time,
                    y_fit,
                    line_color=color,
                    line_dash="dashdot",
                    line_cap="round",
                    line_width=2,
                )

                # Add decay rate info to display
                if "Ʃ" in pollutant:
                    # For bins like 'Ʃ9-100nm (µg/m³)', use '9-100nm' in the display text
                    start_pos = pollutant.find("Ʃ") + 1
                    end_pos = pollutant.find(" (µg/m³)")
                    if end_pos > start_pos:
                        display_name = pollutant[start_pos:end_pos]
                        decay_rate_text = f"{display_name} Decay Rate: {decay_rate:.1f} ± {uncertainty:.2f}  h⁻¹"
                    else:
                        decay_rate_text = f"{pollutant} Decay Rate: {decay_rate:.1f} ± {uncertainty:.2f}  h⁻¹"
                else:
                    decay_rate_text = f"{pollutant} Decay Rate: {decay_rate:.1f} ± {uncertainty:.2f}  h⁻¹"

                decay_rates_info.append(decay_rate_text)

            except Exception as e:
                print(
                    f"Error plotting decay curve for {pollutant} in {burn_id}: {str(e)}"
                )
                continue

        # Add decay rate information to plot and save files
        if decay_rates_info:
            text_div = Div(
                text="<br>".join(decay_rates_info) + f"<br><small>{metadata}</small>"
            )
            layout = column(p, text_div)

            # Define the output file paths
            html_filename = (
                output_figures_path
                / f"{instrument}_{burn_id}_PM-dependent-size_mass-concentration.html"
            )

            # Save figure to HTML file
            output_file(str(html_filename))
            show(layout)

        else:
            # Define the output file paths
            html_filename = (
                output_figures_path
                / f"{instrument}_{burn_id}_PM-dependent-size_mass-concentration.html"
            )

            # Save figure to HTML file
            output_file(str(html_filename))
            show(p)


# Generalized plot_normalized_data function
def plot_normalized_data(data, instrument):
    """Plot normalized PM data with instrument-specific settings"""
    global burn_log, decay_parameters

    # Get configuration for the instrument
    config = INSTRUMENT_CONFIG[instrument]
    datetime_column = config["datetime_column"]
    normalize_pollutant = config.get("normalize_pollutant", "PM10 (µg/m³)")

    # Handle renamed Total Concentration
    if normalize_pollutant == "Total Concentration(µg/m³)":
        normalize_pollutant = "Total Concentration (µg/m³)"

    normalized_column = f"Normalized {normalize_pollutant.split()[0]}"

    # Get script metadata
    metadata = get_script_metadata()

    # Ensure directory exists
    output_figures_path = get_common_file("output_figures")
    os.makedirs(output_figures_path, exist_ok=True)

    # Check if decay parameters have been calculated
    if not decay_parameters:
        print("Calculating decay parameters first...")
        calculate_all_decay_parameters(data, instrument)

    # Ensure datetime column is properly formatted
    data.loc[:, datetime_column] = pd.to_datetime(data[datetime_column])

    # Function to create a figure for a given set of burns
    def create_figure(burn_ids, group_name):
        # Update the y-axis label to reflect correct spacing
        y_label = f"{normalize_pollutant.split()[0]} ln(C)/ln(C₀)"

        p = figure(
            x_axis_label="Time Since Garage Closed (hours)",
            y_axis_label=y_label,
            x_axis_type="linear",
            y_axis_type="log",
            width=800,
            height=500,
            y_range=(10**-3, 10**0.1),
            title=f"{instrument} {group_name} normalized mass-concentration",
        )

        # Set x-axis range
        p.x_range.start = -1
        p.x_range.end = 4

        decay_rates_info = []  # Store decay rate information for this figure

        # Process each burn in the provided list
        for burn_id in burn_ids:
            if burn_id not in decay_parameters:
                print(f"No decay parameters for {burn_id}")
                continue

            try:
                # Get burn date from burn log
                burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
                if burn_date_row.empty:
                    continue

                burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0]).date()

                # Filter data for this burn
                burn_data = data[data["Date"] == burn_date].copy()

                if burn_data.empty:
                    continue

                # Get garage closed time
                garage_closed_time_str = burn_date_row["garage closed"].iloc[0]
                if pd.isna(garage_closed_time_str):
                    print(f"Missing garage closed time for {burn_id}")
                    continue

                garage_closed_time = create_naive_datetime(
                    burn_date, garage_closed_time_str
                )

                # Ensure both datetime objects are timezone-naive for consistent subtraction
                burn_data_datetime = burn_data[datetime_column]
                if (
                    hasattr(burn_data_datetime.dtype, "tz")
                    and burn_data_datetime.dtype.tz is not None
                ):
                    burn_data_datetime = burn_data_datetime.dt.tz_localize(None)

                # Calculate time since garage closed
                burn_data["Time Since Garage Closed (hours)"] = (
                    burn_data_datetime - garage_closed_time
                ).dt.total_seconds() / 3600

                # Check if we have the pollutant data to normalize
                if normalize_pollutant in burn_data.columns:
                    # Normalize pollutant data
                    max_value = burn_data[normalize_pollutant].max()
                    if pd.isna(max_value) or max_value <= 0:
                        print(
                            f"Invalid maximum value for {normalize_pollutant} in {burn_id}"
                        )
                        continue

                    burn_data[normalized_column] = (
                        burn_data[normalize_pollutant] / max_value
                    )

                    # Plot the normalized data
                    source = ColumnDataSource(burn_data)
                    p.line(
                        "Time Since Garage Closed (hours)",
                        normalized_column,
                        source=source,
                        line_color=BURN_STYLES[burn_id]["color"],
                        line_dash=BURN_STYLES[burn_id]["line_dash"],
                        legend_label=BURN_LABELS.get(burn_id, burn_id),
                        line_width=2,
                    )

                    # Check if we have decay parameters for normalized data and it's not figure1
                    if (
                        normalized_column in decay_parameters[burn_id]
                        and group_name != "figure1"
                    ):
                        # Only plot fit curves if NOT figure1
                        decay_params = decay_parameters[burn_id][normalized_column]

                        # Get parameters
                        decay_start_time = decay_params["decay_start_time"]
                        decay_end_time = decay_params["decay_end_time"]
                        # amplitude = decay_params['amplitude']
                        decay_rate = decay_params["decay_rate"]
                        uncertainty = decay_params["uncertainty"]

                        # Get normalized data values at the decay start and end points
                        decay_start_idx = burn_data[
                            "Time Since Garage Closed (hours)"
                        ].searchsorted(decay_start_time)
                        if decay_start_idx >= len(burn_data):
                            decay_start_idx = len(burn_data) - 1
                        start_value = burn_data.iloc[decay_start_idx][normalized_column]

                        # Create points for the fitted curve using the actual decay curve
                        # First create x values from decay start to decay end
                        x_range = np.linspace(decay_start_time, decay_end_time, 100)

                        # Calculate y values using the start value and decay rate
                        # Use the formula: y = start_value * exp(-decay_rate * (t - decay_start_time))
                        y_fit = start_value * np.exp(
                            -decay_rate * (x_range - decay_start_time)
                        )

                        # Plot the fitted curve
                        p.line(
                            x_range,
                            y_fit,
                            line_color=BURN_STYLES[burn_id]["color"],
                            line_dash="dotted",
                            line_width=2,
                        )

                        # Add decay rate info to display
                        decay_rate_text = f"{burn_id} Decay Rate: {decay_rate:.1f} ± {uncertainty:.2f}  h⁻¹"
                        decay_rates_info.append(decay_rate_text)
                    else:
                        # For figure1, still collect the decay rate info for display
                        if normalized_column in decay_parameters[burn_id]:
                            decay_params = decay_parameters[burn_id][normalized_column]
                            decay_rate = decay_params["decay_rate"]
                            uncertainty = decay_params["uncertainty"]
                            decay_rate_text = f"{burn_id} Decay Rate: {decay_rate:.1f} ± {uncertainty:.2f}  h⁻¹"
                            decay_rates_info.append(decay_rate_text)
                        else:
                            print(
                                f"No decay parameters for {normalized_column} in {burn_id}"
                            )
                else:
                    print(
                        f"No {normalize_pollutant} data for normalization in {burn_id}"
                    )

            except Exception as e:
                print(f"Error processing {burn_id}: {str(e)}")
                continue

        # Customize the legend
        p.legend.click_policy = "hide"
        p.legend.location = "bottom_right"

        # Add decay rate information and save files
        if decay_rates_info:
            text_div = Div(
                text="<br>".join(decay_rates_info) + f"<br><small>{metadata}</small>"
            )
            layout = column(p, text_div)

            # Define the output file paths
            html_filename = (
                output_figures_path
                / f"{instrument}_{group_name}_normalized_mass-concentration.html"
            )

            # Save figure to HTML file
            output_file(str(html_filename))
            show(layout)

        else:
            # Define the output file paths
            html_filename = (
                output_figures_path
                / f"{instrument}_{group_name}_normalized_mass-concentration.html"
            )

            # Save figure to HTML file
            output_file(str(html_filename))
            show(p)

    # Create figures for each group
    for group_name, burn_ids in BURN_GROUPS.items():
        print(f"\nCreating {group_name} with burns: {', '.join(burn_ids)}")
        create_figure(burn_ids, group_name)


# Function to calculate baseline decay values
def calculate_baseline_values(burn_calc_df, instrument, config):
    """
    Calculate baseline decay values based on configuration.
    Handles fallback to single burn when some weighted_average burns are excluded.

    Parameters:
    -----------
    burn_calc_df : DataFrame
        DataFrame containing burn information and decay parameters
    instrument : str
        Name of the instrument being processed
    config : dict
        Instrument configuration dictionary

    Returns:
    --------
    tuple
        (baseline_decay_dict, baseline_decay_uncertainty_dict)
    """
    baseline_decay = {}
    baseline_decay_uncertainty = {}

    # Get baseline method and parameters from configuration
    baseline_method = config.get("baseline_method", "single_burn")
    baseline_burns = config.get("baseline_burns", ["burn1"])
    pollutants = config.get("process_pollutants", [])

    # Helper function for weighted average calculation
    def calculate_weighted_average(x1, x2, u1, u2):
        """Calculate the weighted average and uncertainty."""
        weight1 = 1 / (u1**2)
        weight2 = 1 / (u2**2)
        weighted_avg = (x1 * weight1 + x2 * weight2) / (weight1 + weight2)
        combined_uncertainty = np.sqrt(1 / (weight1 + weight2))
        return weighted_avg, combined_uncertainty

    # Process each pollutant
    for pollutant in pollutants:
        # Get data for this pollutant that matches baseline burns
        valid_measurements = []
        valid_uncertainties = []
        valid_burn_ids = []

        # Collect valid measurements from specified burns
        for burn_id in baseline_burns:
            burn_data = burn_calc_df[
                (burn_calc_df["burn"] == burn_id)
                & (burn_calc_df["pollutant"] == pollutant)
            ]

            if (
                not burn_data.empty
                and pd.notna(burn_data["decay"].iloc[0])
                and pd.notna(burn_data["decay_uncertainty"].iloc[0])
            ):
                valid_measurements.append(burn_data["decay"].iloc[0])
                valid_uncertainties.append(burn_data["decay_uncertainty"].iloc[0])
                valid_burn_ids.append(burn_id)

        # Process based on selected method and available data
        if baseline_method == "weighted_average" and len(valid_measurements) >= 2:
            # Use weighted average with available measurements
            if len(valid_measurements) == 2:
                baseline_decay[pollutant], baseline_decay_uncertainty[pollutant] = (
                    calculate_weighted_average(
                        valid_measurements[0],
                        valid_measurements[1],
                        valid_uncertainties[0],
                        valid_uncertainties[1],
                    )
                )
                print(
                    f"Calculated Baseline Decay for {pollutant} using weighted average of {valid_burn_ids[0]} and {valid_burn_ids[1]}: "
                    f"{baseline_decay[pollutant]:.4f} ± {baseline_decay_uncertainty[pollutant]:.4f}"
                )
            else:
                # More than 2 measurements - use weights for all
                weights = [1 / (u**2) for u in valid_uncertainties]
                total_weight = sum(weights)
                baseline_decay[pollutant] = (
                    sum(m * w for m, w in zip(valid_measurements, weights))
                    / total_weight
                )
                baseline_decay_uncertainty[pollutant] = np.sqrt(1 / total_weight)
                burn_list = ", ".join(valid_burn_ids)
                print(
                    f"Calculated Baseline Decay for {pollutant} using weighted average of burns {burn_list}: "
                    f"{baseline_decay[pollutant]:.4f} ± {baseline_decay_uncertainty[pollutant]:.4f}"
                )

        elif len(valid_measurements) == 1:
            # Fall back to single burn if only one measurement is valid
            # (This happens either when baseline_method is 'single_burn' or when
            # 'weighted_average' was selected but some burns were excluded)
            baseline_decay[pollutant] = valid_measurements[0]
            baseline_decay_uncertainty[pollutant] = valid_uncertainties[0]
            print(
                f"Using single value from {valid_burn_ids[0]} for Baseline Decay for {pollutant}: "
                f"{baseline_decay[pollutant]:.4f} ± {baseline_decay_uncertainty[pollutant]:.4f}"
            )
        else:
            # No valid measurements for this pollutant
            print(
                f"No valid baseline decay measurements found for {pollutant} in {instrument}"
            )

    return baseline_decay, baseline_decay_uncertainty


# Clean Air Delivery Rate calculation funtion
def calculate_cadr(burn_calc_df, instrument):
    """
    Calculate Clean Air Delivery Rate (CADR) from decay parameters.

    Parameters:
    -----------
    burn_calc_df : DataFrame
        DataFrame containing burn information and decay parameters
    instrument : str
        Name of the instrument being processed

    Returns:
    --------
    DataFrame
        Updated DataFrame with CADR calculations
    """
    global decay_details_dataframe

    # Define room volumes
    ROOM_VOLUMES = {
        "house": 324,  # Volume for house/garage environment in cubic meters
        "room": 33,  # Volume for room/bedroom environment in cubic meters
        "house_minus_room": 291,  # House volume minus room volume (324 - 33)
    }

    # Make a copy to avoid modifying the original
    burn_calc = burn_calc_df.copy()

    # Convert 'decay' and 'decay_uncertainty' columns to numeric
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values to the number of CR boxes
    crbox_mapping = {
        "burn1": 0,
        "burn2": 4,
        "burn3": 1,
        "burn4": 1,
        "burn5": 0,
        "burn6": 1,
        "burn7": 2,
        "burn8": 2,
        "burn9": 2,
        "burn10": 2,
    }

    # Add CR boxes to DataFrame
    burn_calc["CRboxes"] = burn_calc["burn"].map(crbox_mapping)

    # Get configuration for the instrument
    config = INSTRUMENT_CONFIG[instrument]

    # Initialize baseline decay dictionaries
    baseline_decay = {}
    baseline_decay_uncertainty = {}

    # Get or calculate baseline values
    baseline_values = config.get("baseline_values")

    if baseline_values is None:
        # Calculate baseline values dynamically based on configuration
        baseline_decay, baseline_decay_uncertainty = calculate_baseline_values(
            burn_calc, instrument, config
        )
    else:
        # Use pre-defined baseline values
        baseline_decay = {}
        baseline_decay_uncertainty = {}
        for pollutant, (decay, uncertainty) in baseline_values.items():
            baseline_decay[pollutant] = decay
            baseline_decay_uncertainty[pollutant] = uncertainty
            print(
                f"Using predefined Baseline Decay for {pollutant}: {decay:.4f} ± {uncertainty:.4f}"
            )

    # Function to determine the appropriate volume based on instrument and burn ID
    def get_volume_for_calculation(instrument, burn_id):
        """
        Determine the appropriate volume based on instrument and burn ID.

        Special handling:
        - DustTrak: In bedroom for burns 1-6, in kitchen for burns 7-10
        - SMPS: Always in bedroom
        - Instruments with suffix B: Always in bedroom
        - Instruments with suffix K: Always in kitchen

        For burns 5 and 6 (when room was isolated):
            - Bedroom instruments use room volume
            - Kitchen instruments use house_minus_room volume
        For all other burns:
            - All instruments use house volume
        """
        # Check if this is burn 5 or 6 (special cases with isolated room)
        if burn_id in ["burn5", "burn6"]:
            # Handle DustTrak based on burn number (in bedroom for burns 1-6)
            if instrument == "DustTrak":
                return ROOM_VOLUMES["room"]

            # Handle SMPS (always in bedroom)
            elif instrument == "SMPS":
                return ROOM_VOLUMES["room"]

            # Handle instruments by location suffix
            elif instrument.endswith("B"):  # Bedroom instruments
                return ROOM_VOLUMES["room"]
            elif instrument.endswith("K"):  # Kitchen instruments
                return ROOM_VOLUMES["house_minus_room"]

        # For all other burn scenarios, use house volume
        return ROOM_VOLUMES["house"]

    # Function to calculate CADR and uncertainty for each row
    def calculate_cadr_and_uncertainty(row):
        pollutant = row["pollutant"]
        burn_id = row["burn"]
        baseline_d = baseline_decay.get(pollutant, np.nan)
        baseline_u = baseline_decay_uncertainty.get(pollutant, np.nan)

        # Determine if instrument is in bedroom based on name and burn number
        in_bedroom = False
        if instrument.endswith("B") or instrument == "SMPS":
            in_bedroom = True
        elif instrument == "DustTrak" and int(burn_id.replace("burn", "")) <= 6:
            in_bedroom = True

        # Get appropriate volume for this instrument and burn
        volume = get_volume_for_calculation(instrument, burn_id)

        # Special case handling
        if instrument == "DustTrak" and burn_id == "burn1":
            # Skip CADR calculation for burn1 for DustTrak since it's used as baseline
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        elif not in_bedroom and burn_id in ["burn5", "burn6"]:
            # For kitchen instruments during burn 5 and 6
            # Skip CADR calculation as room was isolated
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        elif in_bedroom and burn_id == "burn5":
            # For bedroom instruments during burn 5 (no CR boxes)
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        elif in_bedroom and burn_id == "burn6":
            # For bedroom instruments during burn 6, use burn5 as baseline
            burn5_data = burn_calc[
                (burn_calc["burn"] == "burn5") & (burn_calc["pollutant"] == pollutant)
            ]
            if not burn5_data.empty:
                decay_burn5 = burn5_data["decay"].iloc[0]
                decay_uncertainty_burn5 = burn5_data["decay_uncertainty"].iloc[0]
                print(
                    f"Using Burn 5 values for {pollutant}: Decay = {decay_burn5} ± {decay_uncertainty_burn5}"
                )

                # Calculate CADR using room volume and burn5 decay values
                cadr = volume * (row["decay"] - decay_burn5)

                # Calculate uncertainty using the original formula from paste.txt for consistency
                cadr_uncertainty = volume * (
                    (row["decay"] + row["decay_uncertainty"])
                    - (decay_burn5 - decay_uncertainty_burn5)
                ) - volume * (
                    (row["decay"] - row["decay_uncertainty"])
                    - (decay_burn5 + decay_uncertainty_burn5)
                )
            else:
                cadr = pd.NA
                cadr_uncertainty = pd.NA
        else:
            # Standard calculations for other burns
            cadr = volume * (row["decay"] - baseline_d)

            # Calculate uncertainty using the original formula from paste.txt for consistency
            cadr_uncertainty = volume * (
                (row["decay"] + row["decay_uncertainty"]) - (baseline_d - baseline_u)
            ) - volume * (
                (row["decay"] - row["decay_uncertainty"]) - (baseline_d + baseline_u)
            )

        return pd.Series([cadr, cadr_uncertainty])

    # Calculate CADR and uncertainty
    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CR box
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    # Filter out normalized pollutants from the decay details DataFrame
    filtered_decay_details = decay_details_dataframe[
        ~decay_details_dataframe["pollutant"].str.startswith("Normalized")
    ]

    # Merge the CADR calculations with the filtered decay information
    merged_data = pd.merge(
        filtered_decay_details,
        burn_calc[
            [
                "burn",
                "pollutant",
                "CRboxes",
                "CADR",
                "CADR_uncertainty",
                "CADR_per_CRbox",
                "CADR_per_CRbox_uncertainty",
            ]
        ],
        on=["burn", "pollutant"],
        how="left",
    )

    # Reorder columns to place CRboxes between decay_uncertainty and CADR
    cols = list(merged_data.columns)
    # Find the position of decay_uncertainty and CADR
    decay_uncertainty_pos = cols.index("decay_uncertainty")

    # Remove CRboxes from its current position
    cols.remove("CRboxes")

    # Insert CRboxes between decay_uncertainty and CADR
    cols.insert(decay_uncertainty_pos + 1, "CRboxes")

    # Reorder the DataFrame columns according to new order
    merged_data = merged_data[cols]

    # Replace negative values with <NA> in numeric columns
    # First, handle columns without dependencies
    independent_numeric_columns = [
        "max_concentration",
        "decay_start_time",
        "start_value",
        "decay_end_time",
        "end_value",
    ]

    # Process independent columns
    for col in independent_numeric_columns:
        if col in merged_data.columns:
            # Replace negative values with pd.NA
            merged_data[col] = merged_data[col].mask(merged_data[col] < 0, pd.NA)

    # Define column pairs (value and associated uncertainty)
    column_pairs = [
        ("decay", "decay_uncertainty"),
        ("CADR", "CADR_uncertainty"),
        ("CADR_per_CRbox", "CADR_per_CRbox_uncertainty"),
    ]

    # Process column pairs to handle dependencies
    for value_col, uncertainty_col in column_pairs:
        if value_col in merged_data.columns and uncertainty_col in merged_data.columns:
            # Create mask for negative values in value column
            negative_mask = merged_data[value_col] < 0

            # Replace negative values with pd.NA in both columns
            merged_data[value_col] = merged_data[value_col].mask(negative_mask, pd.NA)
            merged_data[uncertainty_col] = merged_data[uncertainty_col].mask(
                negative_mask, pd.NA
            )

    # Save the comprehensive results to Excel
    burn_calcs_path = get_common_file("burn_calcs")
    os.makedirs(burn_calcs_path, exist_ok=True)
    output_file_path = burn_calcs_path / f"{instrument}_decay_and_CADR.xlsx"
    merged_data.to_excel(output_file_path, index=False)
    print(f"Comprehensive decay and CADR data saved to {output_file_path}")

    # Print the Excel contents to the terminal
    print("\nDecay and CADR Results for", instrument)
    print("-" * 80)
    print(merged_data.to_string())
    print("-" * 80)

    return burn_calc


# Modified main function to initialize the decay_details_dataframe
def main():
    global decay_parameters, burn_calc, decay_details_dataframe

    try:
        # Reset decay parameters, burn_calc, and decay_details_dataframe for each run
        decay_parameters = {}
        burn_calc = []
        decay_details_dataframe = pd.DataFrame()

        # Process the selected dataset
        if dataset in INSTRUMENT_CONFIG:
            config = INSTRUMENT_CONFIG[dataset]
            # Build file path using data_paths resolver
            file_path = get_instrument_file_path(
                config["instrument_key"], config.get("filename")
            )
            process_func_name = config["process_function"]

            # Call the appropriate processing function
            if process_func_name == "process_aerotrak_data":
                processed_data = process_aerotrak_data(file_path, dataset)
            elif process_func_name == "process_dusttrak_data":
                processed_data = process_dusttrak_data(file_path)
            elif process_func_name == "process_miniams_data":
                processed_data = process_miniams_data(file_path, dataset)
            elif process_func_name == "process_purpleairk_data":
                processed_data = process_purpleairk_data(file_path)
            elif process_func_name == "process_quantaq_data":
                processed_data = process_quantaq_data(file_path, dataset)
            elif process_func_name == "process_smps_data":
                processed_data = process_smps_data(file_path)

                # Print overview of the data for debugging
                print("SMPS data overview:")
                print(f"Shape: {processed_data.shape}")
                print(f"Columns: {processed_data.columns.tolist()}")
                print(f"Sample data:\n{processed_data.head()}")

                # Verify that the required columns are present
                required_columns = [
                    "datetime",
                    "Date",
                    "Time Since Garage Closed (hours)",
                    "Total Concentration(µg/m³)",
                ]
                missing_columns = [
                    col for col in required_columns if col not in processed_data.columns
                ]
                if missing_columns:
                    print(
                        f"Warning: Missing required columns in SMPS data: {missing_columns}"
                    )
            else:
                print(f"Unknown processing function: {process_func_name}")
                return

            # Calculate decay parameters
            print(f"\nCalculating decay parameters for {dataset}...")
            decay_parameters = calculate_all_decay_parameters(processed_data, dataset)

            # Plot the data
            print(f"\nPlotting PM data for {dataset}...")
            plot_pm_data(processed_data, dataset)

            # Check if normalizing pollutant is available for normalized plots
            normalize_pollutant = config.get("normalize_pollutant")
            if normalize_pollutant and normalize_pollutant in processed_data.columns:
                print(
                    f"\nPlotting normalized data for {dataset} using {normalize_pollutant}..."
                )
                plot_normalized_data(processed_data, dataset)
            else:
                print(f"No {normalize_pollutant} data available for normalized plots")

            # Convert burn_calc to DataFrame and calculate CADR if data is available
            if burn_calc:
                print("\nCalculating CADR values...")
                burn_calc_df = pd.DataFrame(burn_calc)
                # Calculate CADR with the updated function
                calculate_cadr(burn_calc_df, dataset)
                print(f"CADR calculations completed for {dataset}")
            else:
                print("No decay parameters available for CADR calculations")
        else:
            print(f"Unknown dataset: {dataset}")

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback

        traceback.print_exc()


# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()
