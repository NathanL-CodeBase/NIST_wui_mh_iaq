#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Script for Missing Hourly Average Ratios

This script diagnoses why some hourly average ratios are missing for AeroTrak data
but present for QuantAQ data in the spatial variation timeseries analysis.

Author: Nathan Lima
Date: 2026-01-12
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts.data_loaders import load_burn_log
from scripts.datetime_utils import create_naive_datetime
from src.data_paths import get_common_file, get_data_root
from src.spatial_variation_analysis import (
    INSTRUMENT_CONFIG,
    process_aerotrak_data,
    process_quantaq_data,
)

# Load burn log
burn_log_path = get_common_file("burn_log")
burn_log = load_burn_log(burn_log_path)

# Hourly bins to check
HOURLY_BINS = [
    (-1, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

# Burns to check (those with missing data)
BURNS_TO_CHECK = ["burn3", "burn4", "burn7", "burn8", "burn9", "burn10"]

# PM size to check
PM_SIZE_AEROTRAK = "PM3 (µg/m³)"  # AeroTrak PM3 is proxy for PM2.5
PM_SIZE_QUANTAQ = "PM2.5 (µg/m³)"


def diagnose_hourly_data(
    bedroom_data, morning_data, burn_id, pm_size, datetime_col, instrument_name
):
    """
    Diagnose why hourly ratios might be missing
    """
    print(f"\n{'=' * 80}")
    print(f"DIAGNOSING {instrument_name} - {burn_id} - {pm_size}")
    print(f"{'=' * 80}")

    burn_info = burn_log[burn_log["Burn ID"] == burn_id]
    if burn_info.empty:
        print(f"Burn {burn_id} not found in burn log")
        return

    burn_date = burn_info["Date"].iloc[0]
    garage_closed_str = burn_info["garage closed"].iloc[0]

    if pd.isna(garage_closed_str) or garage_closed_str == "n/a":
        print(f"No garage closed time for {burn_id}")
        return

    garage_closed_time = create_naive_datetime(burn_date, garage_closed_str)
    if pd.isna(garage_closed_time):
        print("Could not parse garage closed time")
        return

    print(f"Garage closed: {garage_closed_time}")

    # Filter for burn date
    burn_date_only = pd.to_datetime(burn_date).date()
    bedroom_burn = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn = morning_data[morning_data["Date"] == burn_date_only].copy()

    print(f"Bedroom data points: {len(bedroom_burn)}")
    print(f"Morning room data points: {len(morning_burn)}")

    if bedroom_burn.empty or morning_burn.empty:
        print("No data for this burn date")
        return

    # Check if PM size exists
    if pm_size not in bedroom_burn.columns or pm_size not in morning_burn.columns:
        print(f"PM size {pm_size} not found in data")
        return

    # Ensure numeric
    bedroom_burn[pm_size] = pd.to_numeric(bedroom_burn[pm_size], errors="coerce")
    morning_burn[pm_size] = pd.to_numeric(morning_burn[pm_size], errors="coerce")

    # Check for NaN values
    bedroom_valid = bedroom_burn[pm_size].notna().sum()
    morning_valid = morning_burn[pm_size].notna().sum()
    print(f"Bedroom valid (non-NaN) values: {bedroom_valid}/{len(bedroom_burn)}")
    print(f"Morning room valid (non-NaN) values: {morning_valid}/{len(morning_burn)}")

    # Check each hourly bin
    print("\n  Hourly Bin Analysis:")
    print(f"  {'Bin':<12} {'Bedroom':<12} {'Morning':<12} {'Merged':<12} {'Status'}")
    print(f"  {'-' * 70}")

    for start_hour, end_hour in HOURLY_BINS:
        # Define time window
        start_time = garage_closed_time + pd.Timedelta(hours=start_hour)
        end_time = garage_closed_time + pd.Timedelta(hours=end_hour)

        # Filter data for this window
        bedroom_window = bedroom_burn[
            (bedroom_burn[datetime_col] >= start_time)
            & (bedroom_burn[datetime_col] < end_time)
        ].copy()

        morning_window = morning_burn[
            (morning_burn[datetime_col] >= start_time)
            & (morning_burn[datetime_col] < end_time)
        ].copy()

        bin_label = f"{start_hour} to {end_hour}"
        bedroom_count = len(bedroom_window)
        morning_count = len(morning_window)

        if bedroom_window.empty or morning_window.empty:
            status = "EMPTY"
            merged_count = 0
        else:
            # Remove NaN values
            bedroom_window = bedroom_window.dropna(subset=[pm_size])
            morning_window = morning_window.dropna(subset=[pm_size])

            bedroom_count_valid = len(bedroom_window)
            morning_count_valid = len(morning_window)

            if bedroom_window.empty or morning_window.empty:
                status = "ALL NaN"
                merged_count = 0
            else:
                # Resample to 1-minute intervals and align
                bedroom_resample = bedroom_window[[datetime_col, pm_size]].copy()
                morning_resample = morning_window[[datetime_col, pm_size]].copy()

                bedroom_resample = (
                    bedroom_resample.set_index(datetime_col)[pm_size]
                    .resample("1T")
                    .mean()
                )
                morning_resample = (
                    morning_resample.set_index(datetime_col)[pm_size]
                    .resample("1T")
                    .mean()
                )

                # Convert to DataFrames for merging
                bedroom_resample = pd.DataFrame(
                    {f"{pm_size}_bedroom": bedroom_resample}
                )
                morning_resample = pd.DataFrame(
                    {f"{pm_size}_morning": morning_resample}
                )

                # Merge on time index
                merged = pd.merge(
                    bedroom_resample,
                    morning_resample,
                    left_index=True,
                    right_index=True,
                    how="inner",
                )

                if merged.empty:
                    status = "NO OVERLAP"
                    merged_count = 0
                else:
                    # Remove NaN or zero/negative values
                    merged = merged.dropna()
                    merged = merged[
                        (merged[f"{pm_size}_morning"] > 0)
                        & (merged[f"{pm_size}_bedroom"] > 0)
                    ]

                    merged_count = len(merged)

                    if merged.empty or len(merged) < 3:
                        status = f"< 3 points ({merged_count})"
                    else:
                        # Calculate ratios
                        ratios = (
                            merged[f"{pm_size}_bedroom"] / merged[f"{pm_size}_morning"]
                        )

                        # Remove outliers
                        ratios_filtered = ratios[(ratios > 0.1) & (ratios < 10)]

                        if ratios_filtered.empty:
                            status = f"OUTLIERS ({len(ratios)} ratios)"
                        else:
                            avg_ratio = ratios_filtered.mean()
                            status = f"OK (ratio={avg_ratio:.3f})"

        print(
            f"  {bin_label:<12} {bedroom_count:<12} {morning_count:<12} {merged_count:<12} {status}"
        )


def main():
    """Main diagnostic function"""
    print("=" * 80)
    print("HOURLY RATIO DIAGNOSTIC TOOL")
    print("=" * 80)

    # Load instrument data
    print("\nLoading AeroTrak data...")
    aerotrak_bedroom = process_aerotrak_data(
        INSTRUMENT_CONFIG["AeroTrakB"]["file_path"], "AeroTrakB"
    )
    aerotrak_kitchen = process_aerotrak_data(
        INSTRUMENT_CONFIG["AeroTrakK"]["file_path"], "AeroTrakK"
    )
    print(f"AeroTrak Bedroom: {len(aerotrak_bedroom)} rows")
    print(f"AeroTrak Kitchen: {len(aerotrak_kitchen)} rows")

    print("\nLoading QuantAQ data...")
    quantaq_bedroom = process_quantaq_data(
        INSTRUMENT_CONFIG["QuantAQB"]["file_path"], "QuantAQB"
    )
    quantaq_kitchen = process_quantaq_data(
        INSTRUMENT_CONFIG["QuantAQK"]["file_path"], "QuantAQK"
    )
    print(f"QuantAQ Bedroom: {len(quantaq_bedroom)} rows")
    print(f"QuantAQ Kitchen: {len(quantaq_kitchen)} rows")

    # Diagnose each burn
    for burn_id in BURNS_TO_CHECK:
        # Check AeroTrak
        diagnose_hourly_data(
            aerotrak_bedroom,
            aerotrak_kitchen,
            burn_id,
            PM_SIZE_AEROTRAK,
            "Date and Time",
            "AeroTrak",
        )

        # Check QuantAQ (only for burns 4-10)
        burn_num = int(burn_id.replace("burn", ""))
        if burn_num >= 4:
            diagnose_hourly_data(
                quantaq_bedroom,
                quantaq_kitchen,
                burn_id,
                PM_SIZE_QUANTAQ,
                "timestamp_local",
                "QuantAQ",
            )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
