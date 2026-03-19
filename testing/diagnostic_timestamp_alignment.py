#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Script for Timestamp Alignment Issues

This script examines the raw timestamps to understand why AeroTrak data
doesn't merge properly in later hourly bins.

Author: Nathan Lima
Date: 2026-01-12
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts.data_loaders import load_burn_log
from scripts.datetime_utils import create_naive_datetime
from src.data_paths import get_common_file
from src.spatial_variation_analysis import (
    INSTRUMENT_CONFIG,
    process_aerotrak_data,
)

# Load burn log
burn_log_path = get_common_file("burn_log")
burn_log = load_burn_log(burn_log_path)

# Test case: burn7, 2-3 hour bin (which shows 0 merged points)
TEST_BURN = "burn7"
TEST_BIN_START = 2
TEST_BIN_END = 3


def analyze_timestamps():
    """Analyze timestamp patterns for problematic hourly bins"""

    print("=" * 80)
    print(f"TIMESTAMP ALIGNMENT DIAGNOSTIC: {TEST_BURN}")
    print("=" * 80)

    # Load AeroTrak data
    print("\nLoading AeroTrak data...")
    aerotrak_bedroom = process_aerotrak_data(
        INSTRUMENT_CONFIG["AeroTrakB"]["file_path"], "AeroTrakB"
    )
    aerotrak_kitchen = process_aerotrak_data(
        INSTRUMENT_CONFIG["AeroTrakK"]["file_path"], "AeroTrakK"
    )

    # Get burn info
    burn_info = burn_log[burn_log["Burn ID"] == TEST_BURN]
    burn_date = burn_info["Date"].iloc[0]
    garage_closed_str = burn_info["garage closed"].iloc[0]
    garage_closed_time = create_naive_datetime(burn_date, garage_closed_str)

    print(f"Burn date: {burn_date}")
    print(f"Garage closed: {garage_closed_time}")

    # Define time window
    start_time = garage_closed_time + pd.Timedelta(hours=TEST_BIN_START)
    end_time = garage_closed_time + pd.Timedelta(hours=TEST_BIN_END)

    print(f"\nAnalyzing time window: {start_time} to {end_time}")

    # Filter for burn date
    burn_date_only = pd.to_datetime(burn_date).date()
    bedroom_burn = aerotrak_bedroom[
        aerotrak_bedroom["Date"] == burn_date_only
    ].copy()
    morning_burn = aerotrak_kitchen[
        aerotrak_kitchen["Date"] == burn_date_only
    ].copy()

    # Filter for time window
    bedroom_window = bedroom_burn[
        (bedroom_burn["Date and Time"] >= start_time)
        & (bedroom_burn["Date and Time"] < end_time)
    ].copy()

    morning_window = morning_burn[
        (morning_burn["Date and Time"] >= start_time)
        & (morning_burn["Date and Time"] < end_time)
    ].copy()

    print(f"\nBedroom data points: {len(bedroom_window)}")
    print(f"Morning data points: {len(morning_window)}")

    if bedroom_window.empty or morning_window.empty:
        print("\nERROR: One or both windows are empty!")
        return

    # Check for NaN values in PM3
    pm_size = "PM3 (µg/m³)"
    bedroom_window = bedroom_window.dropna(subset=[pm_size])
    morning_window = morning_window.dropna(subset=[pm_size])

    print(f"\nAfter dropping NaN:")
    print(f"Bedroom data points: {len(bedroom_window)}")
    print(f"Morning data points: {len(morning_window)}")

    if bedroom_window.empty or morning_window.empty:
        print("\nERROR: All values are NaN!")
        return

    # Show sample timestamps
    print(f"\n{'='*80}")
    print("SAMPLE RAW TIMESTAMPS")
    print(f"{'='*80}")
    print("\nBedroom first 10 timestamps:")
    for ts in bedroom_window["Date and Time"].head(10):
        print(f"  {ts} (seconds: {ts.second})")

    print("\nMorning room first 10 timestamps:")
    for ts in morning_window["Date and Time"].head(10):
        print(f"  {ts} (seconds: {ts.second})")

    # Analyze timestamp patterns
    print(f"\n{'='*80}")
    print("TIMESTAMP PATTERN ANALYSIS")
    print(f"{'='*80}")

    # Calculate seconds within each minute
    bedroom_seconds = bedroom_window["Date and Time"].apply(lambda x: x.second)
    morning_seconds = morning_window["Date and Time"].apply(lambda x: x.second)

    print(f"\nBedroom timestamp seconds distribution:")
    print(f"  Mean: {bedroom_seconds.mean():.1f}s")
    print(f"  Std: {bedroom_seconds.std():.1f}s")
    print(f"  Range: {bedroom_seconds.min()}-{bedroom_seconds.max()}s")

    print(f"\nMorning room timestamp seconds distribution:")
    print(f"  Mean: {morning_seconds.mean():.1f}s")
    print(f"  Std: {morning_seconds.std():.1f}s")
    print(f"  Range: {morning_seconds.min()}-{morning_seconds.max()}s")

    # Now try the resampling process
    print(f"\n{'='*80}")
    print("RESAMPLING SIMULATION")
    print(f"{'='*80}")

    # Resample to 1-minute intervals
    bedroom_resample = bedroom_window[["Date and Time", pm_size]].copy()
    morning_resample = morning_window[["Date and Time", pm_size]].copy()

    bedroom_resample = (
        bedroom_resample.set_index("Date and Time")[pm_size].resample("1T").mean()
    )
    morning_resample = (
        morning_resample.set_index("Date and Time")[pm_size].resample("1T").mean()
    )

    print(f"\nAfter resampling to 1-minute intervals:")
    print(f"  Bedroom bins: {len(bedroom_resample)}")
    print(f"  Morning bins: {len(morning_resample)}")

    # Convert to DataFrames
    bedroom_resample = pd.DataFrame({f"{pm_size}_bedroom": bedroom_resample})
    morning_resample = pd.DataFrame({f"{pm_size}_morning": morning_resample})

    # Show the resampled timestamps
    print(f"\nBedroom resampled timestamps (first 10):")
    for idx in bedroom_resample.head(10).index:
        val = bedroom_resample.loc[idx, f"{pm_size}_bedroom"]
        print(f"  {idx}: {val:.2f}" if pd.notna(val) else f"  {idx}: NaN")

    print(f"\nMorning resampled timestamps (first 10):")
    for idx in morning_resample.head(10).index:
        val = morning_resample.loc[idx, f"{pm_size}_morning"]
        print(f"  {idx}: {val:.2f}" if pd.notna(val) else f"  {idx}: NaN")

    # Try the merge
    merged = pd.merge(
        bedroom_resample,
        morning_resample,
        left_index=True,
        right_index=True,
        how="inner",
    )

    print(f"\n{'='*80}")
    print("MERGE RESULT")
    print(f"{'='*80}")
    print(f"\nMerged data points: {len(merged)}")

    if not merged.empty:
        print(f"\nMerge successful! First 5 rows:")
        print(merged.head())
    else:
        print(f"\nERROR: Merge failed! No overlapping timestamps.")

        # Show the actual timestamp ranges
        print(f"\nBedroom timestamp range:")
        print(f"  First: {bedroom_resample.index.min()}")
        print(f"  Last:  {bedroom_resample.index.max()}")

        print(f"\nMorning timestamp range:")
        print(f"  First: {morning_resample.index.min()}")
        print(f"  Last:  {morning_resample.index.max()}")

        # Check if ranges overlap at all
        if (bedroom_resample.index.max() < morning_resample.index.min() or
            morning_resample.index.max() < bedroom_resample.index.min()):
            print(f"\nERROR: NO TEMPORAL OVERLAP between the two instruments!")
        else:
            print(f"\nWARNING: Ranges overlap but no matching minute bins!")

            # Check for NaN after resampling
            bedroom_valid = bedroom_resample.notna().sum().iloc[0]
            morning_valid = morning_resample.notna().sum().iloc[0]
            print(f"\nValid (non-NaN) bins after resampling:")
            print(f"  Bedroom: {bedroom_valid}/{len(bedroom_resample)}")
            print(f"  Morning: {morning_valid}/{len(morning_resample)}")


if __name__ == "__main__":
    analyze_timestamps()
