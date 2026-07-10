#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WUI Spatial Variation Analysis Script
=====================================

This script quantifies spatial variability of particulate matter (PM) concentrations
between two locations (bedroom2 vs. morning room) within a manufactured home during
wildfire smoke infiltration experiments. The analysis characterizes smoke distribution
uniformity and mitigation effectiveness across the indoor environment.

Note: Processes burns with 1 and 2 CR Box configurations only. Burns with 4 CR Boxes
excluded due to data quality issues.

Key Metrics Calculated:
    - Peak Ratio Index (PRI): Ratio of peak PM concentrations between locations
    - CR Box Activation Ratio: PM concentration ratio at air cleaner activation time
    - Average Ratio: Time-averaged concentration ratio during 2-hour decay period
    - Relative Standard Deviation (RSD): Temporal variability of spatial ratios

Analysis Features:
    - Multi-instrument processing (AeroTrak optical particle counters, QuantAQ sensors)
    - Size-resolved PM analysis (PM0.5, PM1, PM2.5, PM3, PM5, PM10, PM25)
    - Time-synchronized data alignment with burn timeline
    - Instrument-specific time shifts and baseline corrections
    - Statistical outlier removal (ratios outside 0.1-10 range)

Methodology:
    1. Load peak concentration data and time-series data from instruments
    2. Apply time shifts and baseline corrections
    3. Calculate peak ratios from maximum concentrations during each burn
    4. Calculate CR Box activation ratios at air cleaner turn-on moment
    5. Calculate average ratios over 2-hour decay windows
    6. Compute RSD to quantify temporal variability
    7. Export results to Excel with instrument-specific sheets

Outputs:
    Files:
        - spatial_variation_analysis.xlsx: Results with AeroTrak and QuantAQ sheets
          containing burn-by-burn ratios for each PM size

    Terminal:
        - Data loading confirmations and file paths
        - Processing progress for each burn and instrument
        - Ratio metrics (R_I, R_CR, R_ave, RSD) for each PM size
        - Summary statistics (mean ratios, number of burns analyzed)

Applications:
    - Assess smoke mixing and stratification within test house
    - Evaluate spatial uniformity assumptions for CADR calculations
    - Identify room-to-room transport characteristics
    - Validate single-point measurement representativeness

Utility Functions Used:
    From scripts/datetime_utils:
        - create_naive_datetime: Timezone-naive datetime creation from date/time strings
        - TIME_SHIFTS: Centralized instrument time shift values

    From scripts/data_filters:
        - g_mean: Geometric mean calculation for particle sizing distributions

    From scripts/data_loaders:
        - load_burn_log: Load and parse burn log Excel file

    From scripts/instrument_config:
        - get_instrument_datetime_column: Get datetime column name per instrument
        - get_process_pollutants: Get pollutant list per instrument
        - get_special_cases: Get burn-specific special case configurations
        - get_baseline_values: Get baseline correction values
        - get_burn_range_for_instrument: Get valid burn range per instrument
        - get_instrument_location: Get physical location of instrument

    From scripts/spatial_analysis_utils:
        - calculate_peak_ratio: Peak concentration ratio calculation between locations

Refactoring Status:
    -INSTRUMENT_CONFIG now uses centralized utilities for all configuration values
    -Significantly reduced code duplication (~120 lines saved)
    -Ensures consistency with other analysis scripts
    -Easier maintenance - updates to constants propagate automatically

    Custom implementations retained (with justification):
    - apply_time_shift: Custom signature using INSTRUMENT_CONFIG lookup
    - calculate_rolling_average_burn3: Depends on global burn_log variable
    - calculate_crbox_activation_ratio: Depends on global burn_log variable
    - calculate_average_ratio_and_rsd: Different signature pattern for this workflow

    File paths remain script-specific as they point to processed data files.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025
Last Modified: 2025-12-23 (Migrated to use centralized utility modules)
Update log:
    2026-07-10 (N. Lima): Added Section 3.3 count-independent spatial metrics.
        Joins the AeroTrak suppressed-regime flag from
        coincidence_analysis/aerotrak_coincidence_per_burn.csv (Edit 1); adds a
        reliable-regime restricted ratio R_sub over steps where both units read
        PM below 500 ug/m^3 (Edit 2); adds a signed inter-room smoke arrival lag
        at 100 and 50 ug/m^3 computed in a separate pass over every burn where
        both AeroTraks ran, independent of CR Box use (Edit 3); and cross-checks
        R_sub on the MODULAIR PM2.5 series (Edit 4). The MODULAIR R_sub carries a
        QA/QC-bound caveat: it is only meaningful where both MODULAIR units
        operated below their Section 3.2.3 bound.
"""

import os
import sys
from pathlib import Path
import warnings
import traceback
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add src to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# pylint: disable=import-error,wrong-import-position
from src.data_paths import get_data_root, get_common_file, get_instrument_path

# Import utility functions from centralized modules
from scripts.datetime_utils import create_naive_datetime, TIME_SHIFTS
from scripts.data_filters import g_mean
from scripts.spatial_analysis_utils import (
    calculate_peak_ratio,
    calculate_subthreshold_ratio,
    calculate_arrival_lag,
)
from scripts.data_loaders import load_burn_log
from scripts.instrument_config import (
    get_instrument_datetime_column,
    get_process_pollutants,
    get_special_cases,
    get_baseline_values,
    get_burn_range_for_instrument,
    get_instrument_location,
)
# pylint: enable=import-error,wrong-import-position

# Set up portable paths
data_root = get_data_root()
os.chdir(str(data_root))  # Keep this for backward compatibility with relative paths

# Output directory for results (now using portable path)
OUTPUT_PATH = str(data_root / "burn_data")

# Load burn log (using portable path)
burn_log_path = get_common_file('burn_log')
burn_log = load_burn_log(burn_log_path)

# Section 3.3 thresholds (see Section 3.1 / 3.2.2 of the WUI smoke paper):
#   RELIABLE_REGIME_UG_M3 - co-located instruments agree within a factor of two
#     below this level, so R_sub is trustworthy only over both-below steps.
#   SUPPRESSED_PM3_UG_M3   - peak PM3 mass above which AeroTrak Ch1/Ch2 counts
#     roll over (unreliable peaks).
#   CEILING_FRAC_FLAG      - a unit within 10 % of the Poisson rollover ceiling.
RELIABLE_REGIME_UG_M3 = 500.0
SUPPRESSED_PM3_UG_M3 = 901.0
CEILING_FRAC_FLAG = 0.9
# Arrival-lag crossing thresholds (primary, sensitivity)
ARRIVAL_THRESHOLDS_UG_M3 = (100.0, 50.0)

print(f"[OK] Data root: {data_root}")
print(f"[OK] Output path: {OUTPUT_PATH}")
print(f"[OK] Burn log loaded from: {burn_log_path}")

# Define instrument configurations (using portable paths + centralized utilities)
# Note: This builds on centralized constants from scripts/instrument_config.py
# File paths are still script-specific as they point to processed data files
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "file_path": str(get_instrument_path('aerotrak_bedroom') / "all_data.xlsx"),
        "time_shift": TIME_SHIFTS["AeroTrakB"],
        "process_pollutants": get_process_pollutants("AeroTrakB"),
        "datetime_column": get_instrument_datetime_column("AeroTrakB"),
        "location": get_instrument_location("AeroTrakB"),
        "special_cases": get_special_cases("AeroTrakB"),
        "baseline_values": get_baseline_values("AeroTrakB"),
    },
    "AeroTrakK": {
        "file_path": str(get_instrument_path('aerotrak_kitchen') / "all_data.xlsx"),
        "time_shift": TIME_SHIFTS["AeroTrakK"],
        "process_pollutants": get_process_pollutants("AeroTrakK"),
        "datetime_column": get_instrument_datetime_column("AeroTrakK"),
        "location": get_instrument_location("AeroTrakK"),
        "special_cases": get_special_cases("AeroTrakK"),
        "baseline_values": get_baseline_values("AeroTrakK"),
    },
    "QuantAQB": {
        "file_path": str(get_instrument_path('quantaq_bedroom') / "MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv"),
        "time_shift": TIME_SHIFTS["QuantAQB"],
        "process_pollutants": get_process_pollutants("QuantAQB"),
        "datetime_column": get_instrument_datetime_column("QuantAQB"),
        "burn_range": get_burn_range_for_instrument("QuantAQB"),
        "location": get_instrument_location("QuantAQB"),
        "special_cases": get_special_cases("QuantAQB"),
    },
    "QuantAQK": {
        "file_path": str(get_instrument_path('quantaq_kitchen') / "MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv"),
        "time_shift": TIME_SHIFTS["QuantAQK"],
        "process_pollutants": get_process_pollutants("QuantAQK"),
        "datetime_column": get_instrument_datetime_column("QuantAQK"),
        "burn_range": get_burn_range_for_instrument("QuantAQK"),
        "location": get_instrument_location("QuantAQK"),
        "special_cases": get_special_cases("QuantAQK"),
    },
}


# ============================================================================
# UTILITY FUNCTIONS (from wui_clean_air_delivery_rates_pmsizes_v5.py)
# ============================================================================
def apply_time_shift(df, instrument, burn_date):
    """Apply time shift based on instrument configuration

    Note: burn_id parameter removed as it was not used in the function.
    """
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
    datetime_column = INSTRUMENT_CONFIG[instrument].get(
        "datetime_column", "Date and Time"
    )

    # Ensure datetime column is in datetime format
    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])

    # Convert burn_date to datetime
    burn_date = pd.to_datetime(burn_date).date()

    # Apply the shift only if shift value is non-zero
    if time_shift != 0:
        mask = df[datetime_column].dt.date == burn_date
        if mask.any():
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


def filter_by_burn_dates(data, burn_range, datetime_column):
    """Helper function to filter data by burn dates

    Parameters:
    -----------
    burn_range : range or list
        Either a range object (e.g., range(4, 11)) or list of burn IDs (e.g., ['burn4', 'burn5'])
    """
    # Handle both range objects and lists of burn IDs
    if isinstance(burn_range, range):
        burn_ids = [f"burn{i}" for i in burn_range]
    elif isinstance(burn_range, list) and len(burn_range) > 0:
        # Check if already formatted as burn IDs
        if isinstance(burn_range[0], str) and burn_range[0].startswith('burn'):
            burn_ids = burn_range
        else:
            burn_ids = [f"burn{i}" for i in burn_range]
    else:
        burn_ids = [f"burn{i}" for i in burn_range]

    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)
    if datetime_column in data.columns:
        data["Date"] = pd.to_datetime(data[datetime_column]).dt.date
        return data[data["Date"].isin(burn_dates.dt.date)]
    else:
        raise KeyError(f"Column '{datetime_column}' not found in the dataset.")


def calculate_rolling_average_burn3(data):
    """Calculate 5-minute rolling average for burn3"""
    burn3_date = burn_log[burn_log["Burn ID"] == "burn3"]["Date"].values[0]
    burn3_date = pd.to_datetime(burn3_date).date()

    burn3_data = data[data["Date"] == burn3_date].copy()
    pollutant_columns = [col for col in burn3_data.columns if "(µg/m³)" in col]
    for col in pollutant_columns:
        # Ensure column is numeric before applying rolling average
        burn3_data.loc[:, col] = pd.to_numeric(burn3_data[col], errors="coerce")
        burn3_data.loc[:, col] = burn3_data[col].rolling(window=5, center=True).mean()
    # Replace original data with rolling average data for burn3
    data.loc[data["Date"] == burn3_date, pollutant_columns] = burn3_data[
        pollutant_columns
    ]
    return data


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def process_aerotrak_data(file_path, instrument="AeroTrakB"):
    """Process AeroTrak data with complete conversion from particle counts to mass concentration"""
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
    else:
        raise ValueError(f"Required column '{volume_column}' not found in AeroTrak data")

    # Initialize new columns for mass concentration and calculate values
    pm_columns = []  # List to store the names of new PM concentration columns
    for i, channel in enumerate(size_channels):
        if channel in size_values:
            next_channel = size_channels[i + 1] if i < len(size_channels) - 1 else None
            next_size_value = size_values[next_channel] if next_channel else 25
            # Calculate the geometric mean of the size range
            particle_size = g_mean([size_values[channel], next_size_value])
            particle_size_m = particle_size * 1e-6  # Convert size from µm to m
            # Initialize column name variable
            new_diff_col_µg_m3 = (
                f"PM{size_values[channel]}-{next_size_value} Diff (µg/m³)"
            )
            diff_col = f"{channel} Diff (#)"
            if diff_col in aerotrak_data.columns:
                particle_counts = aerotrak_data[diff_col]
                # Calculate the volume of a single particle
                radius_m = particle_size_m / 2
                volume_per_particle = (4 / 3) * np.pi * (radius_m**3)  # Volume in m³
                # Calculate particle mass density (1 g/cm³)
                particle_mass = volume_per_particle * 1e6 * 1e6  # Convert to µg
                # Ensure particle counts are numeric
                particle_counts = pd.to_numeric(particle_counts, errors="coerce")
                volume_cm = pd.to_numeric(volume_cm, errors="coerce")
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
                # Ensure numeric conversion
                cumul_data = pd.to_numeric(aerotrak_data[cumul_col], errors="coerce")
                aerotrak_data[new_cumul_col] = (
                    cumul_data / volume_cm
                )  # Convert to #/cm³
                # Create new PM concentration column from the Diff column if it was created
                pm_column_name = f"PM{next_size_value} (µg/m³)"
                if (
                    diff_col in aerotrak_data.columns
                    and new_diff_col_µg_m3 in aerotrak_data.columns
                ):
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
    for i, cumul_col in enumerate(cumulative_columns):
        if i == 0 and len(pm_columns) > 0:
            aerotrak_data[cumul_col] = pd.to_numeric(
                aerotrak_data[pm_columns[i]], errors="coerce"
            )
        elif i < len(pm_columns):
            # Sum current PM column with the previous cumulative column
            prev_cumul_col = cumulative_columns[i - 1]
            aerotrak_data[cumul_col] = pd.to_numeric(
                aerotrak_data[pm_columns[i]], errors="coerce"
            ).add(
                pd.to_numeric(aerotrak_data[prev_cumul_col], errors="coerce"),
                fill_value=0,
            )
    # Replace invalid entries with NaN for numeric columns only
    status_columns = ["Flow Status", "Laser Status"]
    if all(col in aerotrak_data.columns for col in status_columns):
        valid_status = (aerotrak_data[status_columns] == "OK").all(axis=1)
        exclude_cols = ["Date and Time", "Sample Time", "Volume (L)"]
        for col in list(aerotrak_data.columns):
            if pd.api.types.is_numeric_dtype(aerotrak_data[col]) and col not in exclude_cols:
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
                filtered_aerotrak_data, instrument, burn_date
            )

    # Check if there's a special case for burn3 rolling average
    special_cases = INSTRUMENT_CONFIG[instrument].get("special_cases", {})
    if "burn3" in special_cases and special_cases["burn3"].get(
        "apply_rolling_average", False
    ):
        filtered_aerotrak_data = calculate_rolling_average_burn3(filtered_aerotrak_data)

    # Ensure all PM columns are numeric
    for col in cumulative_columns:
        if col in filtered_aerotrak_data.columns:
            filtered_aerotrak_data[col] = pd.to_numeric(
                filtered_aerotrak_data[col], errors="coerce"
            )

    # Apply baseline correction - DISABLED for spatial variation analysis
    # Baseline correction removed to avoid near-zero artifacts in concentration ratios
    # For spatial variation analysis, we compare raw concentrations between locations
    # since baseline correction can introduce artifacts if baselines differ between instruments
    # config = INSTRUMENT_CONFIG[instrument]
    # baseline_values = config.get("baseline_values")

    # if baseline_values:
    #     # Check if this is a special case (like AeroTrakK with weighted average method)
    #     if "baseline_method" in baseline_values:
    #         # AeroTrakK uses weighted average - baseline calculated during processing
    #         # For now, skip baseline correction for this instrument
    #         # Future: implement weighted average baseline calculation here
    #         pass
    #     else:
    #         # Apply baseline correction to each pollutant (standard case like AeroTrakB)
    #         for pollutant, (baseline_val, _) in baseline_values.items():
    #             # Note: baseline_uncertainty (_) is stored but not currently used in calculations
    #             if pollutant in filtered_aerotrak_data.columns:
    #                 # Ensure column is numeric first
    #                 filtered_aerotrak_data[pollutant] = pd.to_numeric(
    #                     filtered_aerotrak_data[pollutant], errors="coerce"
    #                 )
    #                 # Subtract baseline from measurements
    #                 filtered_aerotrak_data[pollutant] = (
    #                     filtered_aerotrak_data[pollutant] - baseline_val
    #                 )
    #                 # Set negative values to 0
    #                 filtered_aerotrak_data[pollutant] = filtered_aerotrak_data[
    #                     pollutant
    #                 ].clip(lower=0)

    # Calculate Time Since Garage Closed for each burn
    for burn_id in burn_ids:
        if burn_id not in burn_log["Burn ID"].values:
            continue

        burn_date_value = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
        burn_date_as_date = pd.to_datetime(burn_date_value).date()
        burn_mask = filtered_aerotrak_data["Date"] == burn_date_as_date

        if burn_mask.any():
            burn_info = burn_log[burn_log["Burn ID"] == burn_id]
            garage_closed_time_str = burn_info["garage closed"].iloc[0]
            burn_date = burn_info["Date"].iloc[0]

            if pd.notna(garage_closed_time_str) and garage_closed_time_str != "n/a":
                garage_closed_time = create_naive_datetime(
                    burn_date, garage_closed_time_str
                )

                if pd.notna(garage_closed_time):
                    # Calculate time since garage closed in hours
                    # Get indices where mask is True
                    mask_indices = filtered_aerotrak_data.index[burn_mask]

                    # Iterate through mask and calculate time differences
                    garage_closed_timestamp = pd.Timestamp(garage_closed_time)
                    for idx in mask_indices:
                        datetime_val = filtered_aerotrak_data.at[idx, "Date and Time"]
                        if pd.notna(datetime_val) and isinstance(datetime_val, (pd.Timestamp, np.datetime64)):
                            time_diff = pd.Timestamp(datetime_val) - garage_closed_timestamp
                            filtered_aerotrak_data.at[idx, "Time Since Garage Closed (hours)"] = (
                                time_diff.total_seconds() / 3600
                            )

    return filtered_aerotrak_data


def process_quantaq_data(file_path, instrument="QuantAQB"):
    """Process QuantAQ data with standardized output format"""
    quantaq_data = pd.read_csv(file_path)
    quantaq_data = quantaq_data.iloc[::-1].reset_index(drop=True)
    quantaq_data.columns = quantaq_data.columns.str.strip()

    # Rename columns and ensure they're numeric
    rename_dict = {
        "pm1": "PM1 (µg/m³)",
        "pm25": "PM2.5 (µg/m³)",
        "pm10": "PM10 (µg/m³)",
    }
    quantaq_data.rename(columns=rename_dict, inplace=True)

    # Ensure PM columns are numeric
    for col in ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]:
        if col in quantaq_data.columns:
            quantaq_data[col] = pd.to_numeric(quantaq_data[col], errors="coerce")

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
    # Handle both range objects and lists of burn IDs
    if isinstance(burn_range, range):
        burn_ids = [f"burn{i}" for i in burn_range]
    elif isinstance(burn_range, list) and len(burn_range) > 0 and isinstance(burn_range[0], str):
        burn_ids = burn_range  # Already formatted as burn IDs
    else:
        burn_ids = [f"burn{i}" for i in burn_range]

    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(filtered_data, instrument, burn_date)

    # Calculate Time Since Garage Closed for each burn
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_info = burn_log[burn_log["Burn ID"] == burn_id]
            burn_date = pd.to_datetime(burn_info["Date"].iloc[0])

            burn_mask = filtered_data["Date"] == burn_date.date()

            if burn_mask.any():
                garage_closed_time_str = burn_info["garage closed"].iloc[0]

                if pd.notna(garage_closed_time_str) and garage_closed_time_str != "n/a":
                    garage_closed_time = create_naive_datetime(
                        burn_date, garage_closed_time_str
                    )

                    if pd.notna(garage_closed_time):
                        # Calculate time since garage closed in hours
                        # Get indices where mask is True
                        mask_indices = filtered_data.index[burn_mask]

                        # Iterate through mask and calculate time differences
                        garage_closed_timestamp = pd.Timestamp(garage_closed_time)
                        for idx in mask_indices:
                            datetime_val = filtered_data.at[idx, "timestamp_local"]
                            if pd.notna(datetime_val) and isinstance(datetime_val, (pd.Timestamp, np.datetime64)):
                                time_diff = pd.Timestamp(datetime_val) - garage_closed_timestamp
                                filtered_data.at[idx, "Time Since Garage Closed (hours)"] = (
                                    time_diff.total_seconds() / 3600
                                )

    return filtered_data


def load_peak_concentrations():
    """Load peak concentration data from Excel file"""
    peak_file = "./burn_data/peak_concentrations_all_instruments_edited.xlsx"

    try:
        # Read the data sheet
        peak_data = pd.read_excel(peak_file, sheet_name="data")

        print(f"Peak concentration data loaded: {peak_data.shape}")

        # Show actual AeroTrak and QuantAQ columns
        aerotrak_cols = [col for col in peak_data.columns if "AeroTrak" in col]
        quantaq_cols = [col for col in peak_data.columns if "QuantAQ" in col]

        print("\nActual AeroTrak columns in peak data:")
        for col in aerotrak_cols[:6]:  # Show first 6
            print(f"  '{col}'")

        print("\nActual QuantAQ columns in peak data:")
        for col in quantaq_cols[:6]:  # Show first 6
            print(f"  '{col}'")

        return peak_data
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading peak concentrations: {e}")
        return None


def load_coincidence_flags():
    """Load the AeroTrak suppressed-regime flags for the Section 3.3 join.

    Reads coincidence_analysis/aerotrak_coincidence_per_burn.csv (written by
    src/aerotrak_coincidence.py) and returns a per-burn table with the ceiling
    fraction and peak PM3 mass for each unit, plus the derived boolean flag.

    Returns
    -------
    pd.DataFrame or None
        Columns: Burn_ID, peak_frac_ceiling_bed2, peak_frac_ceiling_morningroom,
        peak_total_pm3_bed2, peak_total_pm3_morningroom, bedroom_door_closed,
        peak_in_suppressed_regime. Returns None if the CSV is absent so the main
        run continues with NaN flag columns rather than failing.
    """
    csv_path = get_common_file("coincidence_analysis") / "aerotrak_coincidence_per_burn.csv"

    if not os.path.exists(csv_path):
        print(f"[WARNING] Coincidence flag CSV not found: {csv_path}")
        print("          Suppressed-regime columns will be NaN.")
        return None

    coinc = pd.read_csv(csv_path)

    # AeroTrak1 sits in Bedroom 2, AeroTrak2 in the Morning Room.
    bed = coinc[coinc["location"] == "bedroom2"].set_index("burn")
    morn = coinc[coinc["location"] == "morning_room"].set_index("burn")

    burns = sorted(set(bed.index) | set(morn.index))
    rows = []
    for burn in burns:
        frac_bed = bed["n_peak_frac_of_ceiling"].get(burn, np.nan)
        frac_morn = morn["n_peak_frac_of_ceiling"].get(burn, np.nan)
        pm3_bed = bed["peak_total_PM3_mass_ug_m3"].get(burn, np.nan)
        pm3_morn = morn["peak_total_PM3_mass_ug_m3"].get(burn, np.nan)

        # bedroom_sealed marks burns with the bedroom door closed (burns 5, 6).
        door_closed = bool(
            bed["bedroom_sealed"].get(burn, False)
            or morn["bedroom_sealed"].get(burn, False)
        )

        # Suppressed if either unit peaks within 10 % of the rollover ceiling or
        # above the 901 ug/m^3 PM3 mass bound.
        suppressed = False
        for frac, pm3 in ((frac_bed, pm3_bed), (frac_morn, pm3_morn)):
            if pd.notna(frac) and frac >= CEILING_FRAC_FLAG:
                suppressed = True
            if pd.notna(pm3) and pm3 > SUPPRESSED_PM3_UG_M3:
                suppressed = True

        rows.append(
            {
                "Burn_ID": burn,
                "peak_frac_ceiling_bed2": frac_bed,
                "peak_frac_ceiling_morningroom": frac_morn,
                "peak_total_pm3_bed2": pm3_bed,
                "peak_total_pm3_morningroom": pm3_morn,
                "bedroom_door_closed": door_closed,
                "peak_in_suppressed_regime": suppressed,
            }
        )

    flags = pd.DataFrame(rows)
    print(f"Coincidence flags loaded: {flags.shape[0]} burns")
    return flags


# ============================================================================
# RATIO CALCULATION FUNCTIONS
# ============================================================================
# Note: calculate_peak_ratio is imported from scripts.spatial_analysis_utils

def calculate_crbox_activation_ratio(
    bedroom_data,
    morning_data,
    burn_id,
    pm_size,
    datetime_col_b,
    datetime_col_m,
):
    """
    Calculate concentration ratio at CR Box activation time between bedroom2 and morning room

    This ratio represents the spatial variation at the moment the portable air cleaner
    was turned on, providing insight into smoke distribution at the start of decay.

    Parameters
    ----------
    bedroom_data : pd.DataFrame
        Bedroom2 concentration data
    morning_data : pd.DataFrame
        Morning room concentration data
    burn_id : str
        Burn identifier (e.g., 'burn4')
    pm_size : str
        PM size column name (e.g., 'PM2.5 (µg/m³)')
    datetime_col_b : str
        Datetime column name for bedroom data
    datetime_col_m : str
        Datetime column name for morning room data

    Returns
    -------
    float or None
        Ratio of bedroom2/morning room concentration at CR Box activation time,
        or None if data is unavailable
    """
    # Get burn information
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]

    if burn_info.empty:
        return None

    burn_date = burn_info["Date"].iloc[0]
    cr_box_time_str = burn_info["CR Box on"].iloc[0]

    # Check if CR Box was used
    if pd.isna(cr_box_time_str) or cr_box_time_str == "n/a":
        return None

    # Create datetime for CR Box activation
    cr_box_time = create_naive_datetime(burn_date, cr_box_time_str)

    if pd.isna(cr_box_time):
        return None

    # Filter data for the current burn date
    burn_date_only = pd.to_datetime(burn_date).date()

    bedroom_burn_data = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn_data = morning_data[morning_data["Date"] == burn_date_only].copy()

    if bedroom_burn_data.empty or morning_burn_data.empty:
        return None

    # Find the closest measurement to CR Box activation time (within ±5 minutes)
    time_window = pd.Timedelta(minutes=5)

    bedroom_window = bedroom_burn_data[
        (bedroom_burn_data[datetime_col_b] >= cr_box_time - time_window)
        & (bedroom_burn_data[datetime_col_b] <= cr_box_time + time_window)
    ].copy()

    morning_window = morning_burn_data[
        (morning_burn_data[datetime_col_m] >= cr_box_time - time_window)
        & (morning_burn_data[datetime_col_m] <= cr_box_time + time_window)
    ].copy()

    if bedroom_window.empty or morning_window.empty:
        return None

    # Check if PM size column exists
    if pm_size not in bedroom_window.columns or pm_size not in morning_window.columns:
        return None

    # Ensure PM columns are numeric
    bedroom_window[pm_size] = pd.to_numeric(bedroom_window[pm_size], errors="coerce")
    morning_window[pm_size] = pd.to_numeric(morning_window[pm_size], errors="coerce")

    # Get the measurement closest to CR Box activation time
    bedroom_window["time_diff"] = abs(
        (bedroom_window[datetime_col_b] - cr_box_time).dt.total_seconds()
    )
    morning_window["time_diff"] = abs(
        (morning_window[datetime_col_m] - cr_box_time).dt.total_seconds()
    )

    bedroom_closest = bedroom_window.loc[bedroom_window["time_diff"].idxmin()]
    morning_closest = morning_window.loc[morning_window["time_diff"].idxmin()]

    bedroom_conc = bedroom_closest[pm_size]
    morning_conc = morning_closest[pm_size]

    # Check for valid data
    if pd.isna(bedroom_conc) or pd.isna(morning_conc) or morning_conc <= 0:
        return None

    # Calculate ratio (bedroom2/morning room)
    ratio = bedroom_conc / morning_conc

    return ratio


def calculate_average_ratio_and_rsd(
    bedroom_data,
    morning_data,
    burn_id,
    pm_size,
    datetime_col_b,
    datetime_col_m,
    analysis_duration_hours=2,
):
    """
    Calculate average concentration ratio and RSD for the analysis window
    Starting from CR Box on time for specified duration
    """

    # Get burn information
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]

    if burn_info.empty:
        return None, None

    burn_date = burn_info["Date"].iloc[0]
    cr_box_time_str = burn_info["CR Box on"].iloc[0]

    # Check if CR Box was used
    if pd.isna(cr_box_time_str) or cr_box_time_str == "n/a":
        return None, None

    # Create datetime for CR Box activation
    cr_box_time = create_naive_datetime(burn_date, cr_box_time_str)

    if pd.isna(cr_box_time):
        return None, None

    # Define analysis window
    start_time = cr_box_time
    end_time = cr_box_time + pd.Timedelta(hours=analysis_duration_hours)

    # Filter data for the current burn date first
    burn_date_only = pd.to_datetime(burn_date).date()

    # Filter bedroom data
    bedroom_burn_data = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn_data = morning_data[morning_data["Date"] == burn_date_only].copy()

    if bedroom_burn_data.empty or morning_burn_data.empty:
        return None, None

    # Now filter for the analysis window
    bedroom_window = bedroom_burn_data[
        (bedroom_burn_data[datetime_col_b] >= start_time)
        & (bedroom_burn_data[datetime_col_b] <= end_time)
    ].copy()

    morning_window = morning_burn_data[
        (morning_burn_data[datetime_col_m] >= start_time)
        & (morning_burn_data[datetime_col_m] <= end_time)
    ].copy()

    if bedroom_window.empty or morning_window.empty:
        return None, None

    # Check if PM size column exists and is numeric
    if pm_size not in bedroom_window.columns or pm_size not in morning_window.columns:
        return None, None

    # Ensure PM columns are numeric
    bedroom_window[pm_size] = pd.to_numeric(bedroom_window[pm_size], errors="coerce")
    morning_window[pm_size] = pd.to_numeric(morning_window[pm_size], errors="coerce")

    # Remove NaN values
    bedroom_window = bedroom_window.dropna(subset=[pm_size])
    morning_window = morning_window.dropna(subset=[pm_size])

    if bedroom_window.empty or morning_window.empty:
        return None, None

    # Align data by time (resample to 1-minute intervals)
    # Select only the datetime column and PM size column for resampling
    bedroom_resample = bedroom_window[[datetime_col_b, pm_size]].copy()
    morning_resample = morning_window[[datetime_col_m, pm_size]].copy()

    # Set index and resample
    bedroom_resample = (
        bedroom_resample.set_index(datetime_col_b)[pm_size].resample("1T").mean()
    )
    morning_resample = (
        morning_resample.set_index(datetime_col_m)[pm_size].resample("1T").mean()
    )

    # Convert back to DataFrame for merging
    bedroom_resample = pd.DataFrame({f"{pm_size}_bedroom": bedroom_resample})
    morning_resample = pd.DataFrame({f"{pm_size}_morning": morning_resample})

    # Merge on time index
    merged = pd.merge(
        bedroom_resample,
        morning_resample,
        left_index=True,
        right_index=True,
        how="inner",
    )

    if merged.empty:
        return None, None

    # Remove rows with NaN or zero/negative values
    merged = merged.dropna()
    merged = merged[
        (merged[f"{pm_size}_morning"] > 0) & (merged[f"{pm_size}_bedroom"] > 0)
    ]

    if merged.empty or len(merged) < 5:  # Need at least 5 data points
        return None, None

    # Calculate ratios for each time point
    ratios = merged[f"{pm_size}_bedroom"] / merged[f"{pm_size}_morning"]

    # Remove outliers (ratios > 10 or < 0.1)
    ratios = ratios[(ratios > 0.1) & (ratios < 10)]

    if ratios.empty:
        return None, None

    # Calculate average ratio and RSD
    avg_ratio = ratios.mean()
    std_ratio = ratios.std()
    rsd = (std_ratio / avg_ratio * 100) if avg_ratio > 0 else None

    return avg_ratio, rsd


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def analyze_spatial_variation():
    """Main function to analyze spatial variation for all burns and instruments"""

    print("=" * 80)
    print("WUI SPATIAL VARIATION ANALYSIS")
    print("=" * 80)

    # Load peak concentration data
    peak_data = load_peak_concentrations()
    if peak_data is None:
        print("ERROR: Could not load peak concentration data")
        return None, None

    # Initialize results storage
    results = {"AeroTrak": [], "QuantAQ": []}

    # Process AeroTrak instruments
    print("\n" + "=" * 40)
    print("PROCESSING AEROTRAK DATA")
    print("=" * 40)

    try:
        aerotrakb_data = process_aerotrak_data(
            INSTRUMENT_CONFIG["AeroTrakB"]["file_path"], "AeroTrakB"
        )
        aerotrakk_data = process_aerotrak_data(
            INSTRUMENT_CONFIG["AeroTrakK"]["file_path"], "AeroTrakK"
        )
        print(f"AeroTrakB data loaded: {aerotrakb_data.shape}")
        print(f"AeroTrakK data loaded: {aerotrakk_data.shape}")

        # Check for required columns
        aerotrakb_pm_cols = [
            col for col in aerotrakb_data.columns if "PM" in col and "(µg/m³)" in col
        ]
        aerotrakk_pm_cols = [
            col for col in aerotrakk_data.columns if "PM" in col and "(µg/m³)" in col
        ]
        print(f"AeroTrakB PM columns: {aerotrakb_pm_cols}")
        print(f"AeroTrakK PM columns: {aerotrakk_pm_cols}")

    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"Error loading AeroTrak data: {e}")
        traceback.print_exc()
        aerotrakb_data = None
        aerotrakk_data = None

    # Process QuantAQ instruments
    print("\n" + "=" * 40)
    print("PROCESSING QUANTAQ DATA")
    print("=" * 40)

    try:
        quantaqb_data = process_quantaq_data(
            INSTRUMENT_CONFIG["QuantAQB"]["file_path"], "QuantAQB"
        )
        quantaqk_data = process_quantaq_data(
            INSTRUMENT_CONFIG["QuantAQK"]["file_path"], "QuantAQK"
        )
        print(f"QuantAQB data loaded: {quantaqb_data.shape}")
        print(f"QuantAQK data loaded: {quantaqk_data.shape}")
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"Error loading QuantAQ data: {e}")
        traceback.print_exc()
        quantaqb_data = None
        quantaqk_data = None

    # Define PM sizes for each instrument type
    aerotrak_pm_sizes = [
        "PM0.5 (µg/m³)",
        "PM1 (µg/m³)",
        "PM3 (µg/m³)",
        "PM5 (µg/m³)",
        "PM10 (µg/m³)",
        "PM25 (µg/m³)",
    ]
    quantaq_pm_sizes = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

    # Analyze each burn
    for burn_id in burn_log["Burn ID"]:

        print(f"\n{'='*60}")
        print(f"Analyzing {burn_id}")
        print(f"{'='*60}")

        # Check if CR Box was used
        burn_info = burn_log[burn_log["Burn ID"] == burn_id]
        cr_box_time = burn_info["CR Box on"].iloc[0]

        if pd.isna(cr_box_time) or cr_box_time == "n/a":
            print(f"  CR Box not used for {burn_id}, skipping")
            continue

        # Process AeroTrak data for this burn
        if aerotrakb_data is not None and aerotrakk_data is not None:
            print("\n  Processing AeroTrak...")

            for pm_size in aerotrak_pm_sizes:
                try:
                    # Calculate peak ratio
                    peak_ratio = calculate_peak_ratio(
                        peak_data, burn_id, "AeroTrak", pm_size
                    )

                    # Calculate CR Box activation ratio
                    crbox_ratio = calculate_crbox_activation_ratio(
                        aerotrakb_data,
                        aerotrakk_data,
                        burn_id,
                        pm_size,
                        "Date and Time",
                        "Date and Time",
                    )

                    # Calculate average ratio and RSD
                    avg_ratio, rsd = calculate_average_ratio_and_rsd(
                        aerotrakb_data,
                        aerotrakk_data,
                        burn_id,
                        pm_size,
                        "Date and Time",
                        "Date and Time",
                    )

                    # Reliable-regime restricted ratio (Edit 2): median Bed2/MR
                    # over steps where both units read below 500 ug/m^3.
                    burn_date_ai = burn_info["Date"].iloc[0]
                    (
                        r_sub_med,
                        r_sub_lo,
                        r_sub_hi,
                        n_sub,
                    ) = calculate_subthreshold_ratio(
                        aerotrakb_data,
                        aerotrakk_data,
                        burn_date_ai,
                        pm_size,
                        "Date and Time",
                        "Date and Time",
                        threshold=RELIABLE_REGIME_UG_M3,
                    )

                    # Store results if we have at least one metric
                    if (
                        peak_ratio is not None
                        or crbox_ratio is not None
                        or avg_ratio is not None
                    ):
                        results["AeroTrak"].append(
                            {
                                "Burn_ID": burn_id,
                                "PM_Size": pm_size,
                                "Peak_Ratio_Index": peak_ratio,
                                "CRBox_Activation_Ratio": crbox_ratio,
                                "Average_Ratio": avg_ratio,
                                "RSD_%": rsd,
                                "R_sub_median": r_sub_med,
                                "R_sub_iqr_low": r_sub_lo,
                                "R_sub_iqr_high": r_sub_hi,
                                "n_sub_steps": n_sub,
                            }
                        )

                        # Fixed f-string formatting
                        peak_str = (
                            f"{peak_ratio:.3f}" if peak_ratio is not None else "N/A"
                        )
                        crbox_str = (
                            f"{crbox_ratio:.3f}" if crbox_ratio is not None else "N/A"
                        )
                        avg_str = f"{avg_ratio:.3f}" if avg_ratio is not None else "N/A"
                        rsd_str = f"{rsd:.1f}" if rsd is not None else "N/A"

                        print(
                            f"    {pm_size}: R_I={peak_str}, R_CR={crbox_str}, "
                            f"R_ave={avg_str}, RSD={rsd_str}%"
                        )
                    else:
                        print(f"    {pm_size}: No valid data for ratios")

                except (KeyError, ValueError, TypeError, IndexError) as e:
                    print(f"    Error processing {pm_size}: {str(e)[:100]}")

        # Process QuantAQ data for this burn (only for burns 4-10)
        burn_num = int(burn_id.replace("burn", ""))
        if burn_num >= 4 and quantaqb_data is not None and quantaqk_data is not None:
            print("\n  Processing QuantAQ...")

            for pm_size in quantaq_pm_sizes:
                try:
                    # Calculate peak ratio
                    peak_ratio = calculate_peak_ratio(
                        peak_data, burn_id, "QuantAQ", pm_size
                    )

                    # Calculate CR Box activation ratio
                    crbox_ratio = calculate_crbox_activation_ratio(
                        quantaqb_data,
                        quantaqk_data,
                        burn_id,
                        pm_size,
                        "timestamp_local",
                        "timestamp_local",
                    )

                    # Calculate average ratio and RSD
                    avg_ratio, rsd = calculate_average_ratio_and_rsd(
                        quantaqb_data,
                        quantaqk_data,
                        burn_id,
                        pm_size,
                        "timestamp_local",
                        "timestamp_local",
                    )

                    # MODULAIR reliable-regime ratio (Edit 4). Cross-check of the
                    # AeroTrak R_sub using MODULAIR PM2.5. QA/QC caveat: only
                    # meaningful where both MODULAIR units operated below their
                    # Section 3.2.3 bound; interpret with that in mind.
                    burn_date_qa = burn_info["Date"].iloc[0]
                    (
                        r_sub_med,
                        r_sub_lo,
                        r_sub_hi,
                        n_sub,
                    ) = calculate_subthreshold_ratio(
                        quantaqb_data,
                        quantaqk_data,
                        burn_date_qa,
                        pm_size,
                        "timestamp_local",
                        "timestamp_local",
                        threshold=RELIABLE_REGIME_UG_M3,
                    )

                    # Store results if we have at least one metric
                    if (
                        peak_ratio is not None
                        or crbox_ratio is not None
                        or avg_ratio is not None
                    ):
                        results["QuantAQ"].append(
                            {
                                "Burn_ID": burn_id,
                                "PM_Size": pm_size,
                                "Peak_Ratio_Index": peak_ratio,
                                "CRBox_Activation_Ratio": crbox_ratio,
                                "Average_Ratio": avg_ratio,
                                "RSD_%": rsd,
                                "R_sub_median": r_sub_med,
                                "R_sub_iqr_low": r_sub_lo,
                                "R_sub_iqr_high": r_sub_hi,
                                "n_sub_steps": n_sub,
                            }
                        )

                        # Fixed f-string formatting
                        peak_str = (
                            f"{peak_ratio:.3f}" if peak_ratio is not None else "N/A"
                        )
                        crbox_str = (
                            f"{crbox_ratio:.3f}" if crbox_ratio is not None else "N/A"
                        )
                        avg_str = f"{avg_ratio:.3f}" if avg_ratio is not None else "N/A"
                        rsd_str = f"{rsd:.1f}" if rsd is not None else "N/A"

                        print(
                            f"    {pm_size}: R_I={peak_str}, R_CR={crbox_str}, "
                            f"R_ave={avg_str}, RSD={rsd_str}%"
                        )
                    else:
                        print(f"    {pm_size}: No valid data for ratios")

                except (KeyError, ValueError, TypeError, IndexError) as e:
                    print(f"    Error processing {pm_size}: {str(e)[:100]}")

    # Create DataFrames from results
    aerotrak_df = pd.DataFrame(results["AeroTrak"])
    quantaq_df = pd.DataFrame(results["QuantAQ"])

    # ------------------------------------------------------------------
    # Edit 3: inter-room smoke arrival lag (separate pass over ALL burns)
    # ------------------------------------------------------------------
    # Arrival lag is count-independent and meaningful for burns the main loop
    # skips (no CR Box). Compute it once per burn on the AeroTrak PM3 series for
    # every burn where both units have data, independent of CR Box use.
    print(f"\n{'='*60}")
    print("ARRIVAL LAG (AeroTrak PM3, all burns with both units)")
    print(f"{'='*60}")

    lag_rows = []
    if aerotrakb_data is not None and aerotrakk_data is not None:
        for burn_id in burn_log["Burn ID"]:
            burn_info = burn_log[burn_log["Burn ID"] == burn_id]
            burn_date = burn_info["Date"].iloc[0]

            lag_100 = calculate_arrival_lag(
                aerotrakb_data,
                aerotrakk_data,
                burn_date,
                "PM3 (µg/m³)",
                "Date and Time",
                "Date and Time",
                threshold=ARRIVAL_THRESHOLDS_UG_M3[0],
            )
            lag_50 = calculate_arrival_lag(
                aerotrakb_data,
                aerotrakk_data,
                burn_date,
                "PM3 (µg/m³)",
                "Date and Time",
                "Date and Time",
                threshold=ARRIVAL_THRESHOLDS_UG_M3[1],
            )

            lag_rows.append(
                {
                    "Burn_ID": burn_id,
                    "arrival_lag_min_100": lag_100,
                    "arrival_lag_min_50": lag_50,
                }
            )
            lag100_str = f"{lag_100:+.1f}" if pd.notna(lag_100) else "N/A"
            lag50_str = f"{lag_50:+.1f}" if pd.notna(lag_50) else "N/A"
            print(
                f"  {burn_id}: lag@100={lag100_str} min, lag@50={lag50_str} min "
                f"(negative = Morning Room first)"
            )

    lag_df = pd.DataFrame(lag_rows)

    # ------------------------------------------------------------------
    # Edit 1: join the suppressed-regime flag (per burn)
    # ------------------------------------------------------------------
    coinc_flags = load_coincidence_flags()

    # Merge lag + flags into the AeroTrak sheet. Arrival lag and the suppressed
    # flag are per-burn, so broadcast across that burn's PM-size rows. Burns that
    # the main loop skipped (no CR Box) get one PM3 row carrying lag + flags with
    # the ratio columns left NaN.
    ratio_cols = [
        "Peak_Ratio_Index",
        "CRBox_Activation_Ratio",
        "Average_Ratio",
        "RSD_%",
        "R_sub_median",
        "R_sub_iqr_low",
        "R_sub_iqr_high",
        "n_sub_steps",
    ]
    if not lag_df.empty:
        existing_burns = (
            set(aerotrak_df["Burn_ID"]) if not aerotrak_df.empty else set()
        )
        missing_rows = []
        for _, lr in lag_df.iterrows():
            if lr["Burn_ID"] not in existing_burns:
                row = {"Burn_ID": lr["Burn_ID"], "PM_Size": "PM3 (µg/m³)"}
                for col in ratio_cols:
                    row[col] = np.nan
                missing_rows.append(row)
        if missing_rows:
            aerotrak_df = pd.concat(
                [aerotrak_df, pd.DataFrame(missing_rows)], ignore_index=True
            )

        aerotrak_df = aerotrak_df.merge(lag_df, on="Burn_ID", how="left")

    if coinc_flags is not None and not aerotrak_df.empty:
        aerotrak_df = aerotrak_df.merge(coinc_flags, on="Burn_ID", how="left")
    elif not aerotrak_df.empty:
        for col in [
            "peak_frac_ceiling_bed2",
            "peak_frac_ceiling_morningroom",
            "peak_total_pm3_bed2",
            "peak_total_pm3_morningroom",
            "bedroom_door_closed",
            "peak_in_suppressed_regime",
        ]:
            aerotrak_df[col] = np.nan

    # Save to Excel file
    output_file = os.path.join(OUTPUT_PATH, "spatial_variation_analysis.xlsx")

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    # Check if there's any data to save
    if aerotrak_df.empty and quantaq_df.empty:
        print(
            "WARNING: No results to save - both AeroTrak and QuantAQ DataFrames are empty"
        )
        print("Check the error messages above for issues with data processing")
    else:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            if not aerotrak_df.empty:
                aerotrak_df.to_excel(writer, sheet_name="AeroTrak", index=False)
                print(f"AeroTrak results: {len(aerotrak_df)} rows")

            if not quantaq_df.empty:
                quantaq_df.to_excel(writer, sheet_name="QuantAQ", index=False)
                print(f"QuantAQ results: {len(quantaq_df)} rows")

        print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    if not aerotrak_df.empty:
        print("\nAeroTrak Summary:")
        print(f"  Burns analyzed: {aerotrak_df['Burn_ID'].nunique()}")
        print(f"  Total measurements: {len(aerotrak_df)}")

        # Calculate means only for non-null values
        peak_mean = aerotrak_df["Peak_Ratio_Index"].dropna().mean()
        crbox_mean = aerotrak_df["CRBox_Activation_Ratio"].dropna().mean()
        avg_mean = aerotrak_df["Average_Ratio"].dropna().mean()
        rsd_mean = aerotrak_df["RSD_%"].dropna().mean()

        # Fixed f-string formatting
        print(
            f"  Mean Peak Ratio: {peak_mean:.3f}"
            if not pd.isna(peak_mean)
            else "  Mean Peak Ratio: N/A"
        )
        print(
            f"  Mean CR Box Activation Ratio: {crbox_mean:.3f}"
            if not pd.isna(crbox_mean)
            else "  Mean CR Box Activation Ratio: N/A"
        )
        print(
            f"  Mean Average Ratio: {avg_mean:.3f}"
            if not pd.isna(avg_mean)
            else "  Mean Average Ratio: N/A"
        )
        print(
            f"  Mean RSD: {rsd_mean:.1f}%"
            if not pd.isna(rsd_mean)
            else "  Mean RSD: N/A"
        )

    if not quantaq_df.empty:
        print("\nQuantAQ Summary:")
        print(f"  Burns analyzed: {quantaq_df['Burn_ID'].nunique()}")
        print(f"  Total measurements: {len(quantaq_df)}")

        # Calculate means only for non-null values
        peak_mean = quantaq_df["Peak_Ratio_Index"].dropna().mean()
        crbox_mean = quantaq_df["CRBox_Activation_Ratio"].dropna().mean()
        avg_mean = quantaq_df["Average_Ratio"].dropna().mean()
        rsd_mean = quantaq_df["RSD_%"].dropna().mean()

        # Fixed f-string formatting
        print(
            f"  Mean Peak Ratio: {peak_mean:.3f}"
            if not pd.isna(peak_mean)
            else "  Mean Peak Ratio: N/A"
        )
        print(
            f"  Mean CR Box Activation Ratio: {crbox_mean:.3f}"
            if not pd.isna(crbox_mean)
            else "  Mean CR Box Activation Ratio: N/A"
        )
        print(
            f"  Mean Average Ratio: {avg_mean:.3f}"
            if not pd.isna(avg_mean)
            else "  Mean Average Ratio: N/A"
        )
        print(
            f"  Mean RSD: {rsd_mean:.1f}%"
            if not pd.isna(rsd_mean)
            else "  Mean RSD: N/A"
        )

    # ------------------------------------------------------------------
    # Section 3.3 text inputs: arrival lag, R_sub, Burn 03 suppressed flag
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SECTION 3.3 TEXT INPUTS")
    print(f"{'='*60}")

    if not aerotrak_df.empty and "arrival_lag_min_100" in aerotrak_df.columns:
        # One value per burn; collapse to per-burn before summarizing.
        per_burn = aerotrak_df.drop_duplicates(subset="Burn_ID")

        lag100 = per_burn["arrival_lag_min_100"].dropna()
        if not lag100.empty:
            print(
                f"\nArrival lag @100 ug/m^3 (t_MorningRoom - t_Bed2, "
                f"negative = Morning Room first):"
            )
            print(
                f"  median={lag100.median():+.1f} min, "
                f"range {lag100.min():+.1f} to {lag100.max():+.1f} min, "
                f"N={len(lag100)} burns"
            )
            if "bedroom_door_closed" in per_burn.columns:
                closed = per_burn[per_burn["bedroom_door_closed"] == True]
                for _, row in closed.iterrows():
                    l100 = row["arrival_lag_min_100"]
                    l50 = row["arrival_lag_min_50"]
                    l100_s = f"{l100:+.1f} min" if pd.notna(l100) else "no crossing"
                    l50_s = f"{l50:+.1f} min" if pd.notna(l50) else "no crossing"
                    print(
                        f"  [door closed] {row['Burn_ID']}: "
                        f"lag@100={l100_s}, lag@50={l50_s} "
                        f"(no crossing = Bedroom 2 PM3 stayed below threshold)"
                    )

        # PM3 R_sub across burns
        pm3 = aerotrak_df[aerotrak_df["PM_Size"] == "PM3 (µg/m³)"]
        r_sub = pm3["R_sub_median"].dropna() if "R_sub_median" in pm3.columns else pd.Series(dtype=float)
        if not r_sub.empty:
            print(f"\nR_sub_median (Bed2/MR, PM3, both < {RELIABLE_REGIME_UG_M3:.0f} ug/m^3):")
            print(
                f"  median across burns={r_sub.median():.3f}, "
                f"range {r_sub.min():.3f} to {r_sub.max():.3f}, N={len(r_sub)} burns"
            )

        # Burn 03 suppressed-regime status
        b3 = per_burn[per_burn["Burn_ID"] == "burn3"]
        if not b3.empty and "peak_in_suppressed_regime" in b3.columns:
            r = b3.iloc[0]
            print("\nBurn 03 suppressed-regime status:")
            print(f"  peak_in_suppressed_regime = {r.get('peak_in_suppressed_regime')}")
            print(
                f"  n_peak_frac_of_ceiling: Bed2="
                f"{r.get('peak_frac_ceiling_bed2')}, "
                f"MorningRoom={r.get('peak_frac_ceiling_morningroom')}"
            )

    return aerotrak_df, quantaq_df


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run the analysis
    aerotrak_results, quantaq_results = analyze_spatial_variation()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
