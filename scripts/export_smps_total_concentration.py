"""
SMPS Data Export Utility

This script reads raw SMPS (Scanning Mobility Particle Sizer) data files and
exports a CSV containing datetime, size range metadata, all diameter-midpoint
size bin columns, summary statistics, and total concentration.
The output file is intended for sharing with collaborators.

The script handles the transposed SMPS data format where:
- First column contains all data labels
- Each subsequent column represents a time point
- Diameter midpoint rows are identified by numeric (decimal) labels

Output column order:
    datetime | Lower Size(nm) | Upper Size(nm) | [sorted midpoint bins] |
    D50(nm) | Median(nm) | Mean(nm) | Geo. Mean(nm) | Mode(nm) |
    Geo. Std. Dev. | Total Concentration

Midpoint column names include units, e.g. "14.13 nm (µg/m³)" or "14.13 nm (#/cm³)".
Bins may differ across files; missing bins for a given file are filled with NaN.

Usage:
    1. Set CONCENTRATION_TYPE to either 'MassConc' or 'NumConc'
    2. Set OUTPUT_PATH to desired output directory
    3. Run: conda run -n wui python scripts/export_smps_total_concentration.py

Author: Nathan Lima
Date: 2025-01-08
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ====== CONFIGURATION VARIABLES ======
# Set this to 'MassConc' or 'NumConc'
CONCENTRATION_TYPE = "MassConc"  # Options: 'MassConc' or 'NumConc'

# ======================================


def load_config():
    """Load data_config.json to get SMPS data path"""
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "data_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def read_transposed_smps_file(file_path, conc_type="MassConc"):
    """
    Read a raw SMPS file in transposed format and extract all relevant columns.

    Parameters:
    -----------
    file_path : Path or str
        Path to the SMPS Excel file
    conc_type : str
        'MassConc' or 'NumConc' - determines units appended to midpoint column names

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: datetime, Lower Size(nm), Upper Size(nm),
        [sorted diameter midpoints], D50(nm), Median(nm), Mean(nm),
        Geo. Mean(nm), Mode(nm), Geo. Std. Dev., Total Concentration, _units
    """
    size_units = "nm (µg/m³)" if conc_type == "MassConc" else "nm (#/cm³)"

    # Read the Excel file without headers
    df = pd.read_excel(file_path, header=None)

    # First column contains all row labels
    labels = df.iloc[:, 0].astype(str)

    # Row index trackers
    date_row_idx = None
    time_row_idx = None
    total_conc_row_idx = None
    lower_size_row_idx = None
    upper_size_row_idx = None
    d50_row_idx = None
    median_row_idx = None
    mean_row_idx = None
    geo_mean_row_idx = None
    mode_row_idx = None
    geo_std_row_idx = None
    diameter_rows = []  # list of (float_value, original_label_string, row_idx)
    units = "unknown"

    for idx, label in enumerate(labels):
        if label == "Date":
            date_row_idx = idx
        elif label == "Start Time":
            time_row_idx = idx
        elif "Total Concentration" in label:
            total_conc_row_idx = idx
            if "(" in label and ")" in label:
                units = label[label.find("(") + 1 : label.find(")")]
        elif label == "Lower Size(nm)":
            lower_size_row_idx = idx
        elif label == "Upper Size(nm)":
            upper_size_row_idx = idx
        elif label == "D50(nm)":
            d50_row_idx = idx
        elif label == "Median(nm)":
            median_row_idx = idx
        elif label == "Mean(nm)":
            mean_row_idx = idx
        elif label == "Geo. Mean(nm)":
            geo_mean_row_idx = idx
        elif label == "Mode(nm)":
            mode_row_idx = idx
        elif label in ("Geo. Std. Dev.", "Geo. Std. Dev"):
            geo_std_row_idx = idx
        else:
            # Numeric labels are diameter midpoints
            try:
                val = float(label)
                diameter_rows.append((val, label, idx))
            except ValueError:
                pass

    if date_row_idx is None or time_row_idx is None or total_conc_row_idx is None:
        raise ValueError(f"Could not find required rows in {file_path}")

    # Sort diameter midpoints numerically (ascending)
    diameter_rows.sort(key=lambda x: x[0])

    # Build datetime list
    dates = df.iloc[date_row_idx, 1:].values
    times = df.iloc[time_row_idx, 1:].values
    datetimes = []
    for date, time in zip(dates, times):
        try:
            if pd.notna(date) and pd.notna(time):
                datetimes.append(pd.to_datetime(f"{date} {time}"))
            else:
                datetimes.append(pd.NaT)
        except Exception:
            datetimes.append(pd.NaT)

    # Build result column-by-column in desired output order
    result = {"datetime": datetimes}

    if lower_size_row_idx is not None:
        result["Lower Size(nm)"] = df.iloc[lower_size_row_idx, 1:].tolist()

    if upper_size_row_idx is not None:
        result["Upper Size(nm)"] = df.iloc[upper_size_row_idx, 1:].tolist()

    # Diameter midpoint bins (sorted)
    for _, label, row_idx in diameter_rows:
        result[f"{label} {size_units}"] = pd.to_numeric(df.iloc[row_idx, 1:], errors="coerce").tolist()

    # Summary statistics
    for row_idx, col_name in [
        (d50_row_idx, "D50(nm)"),
        (median_row_idx, "Median(nm)"),
        (mean_row_idx, "Mean(nm)"),
        (geo_mean_row_idx, "Geo. Mean(nm)"),
        (mode_row_idx, "Mode(nm)"),
        (geo_std_row_idx, "Geo. Std. Dev."),
    ]:
        if row_idx is not None:
            result[col_name] = df.iloc[row_idx, 1:].tolist()

    result["Total Concentration"] = df.iloc[total_conc_row_idx, 1:].tolist()
    result["_units"] = units

    result_df = pd.DataFrame(result)
    result_df = result_df.dropna(subset=["datetime"])

    return result_df


def process_all_smps_files(conc_type="MassConc"):
    """
    Process all SMPS files of the specified concentration type and export to CSV.

    Parameters:
    -----------
    conc_type : str
        'MassConc' or 'NumConc'
    """
    size_units = "nm (µg/m³)" if conc_type == "MassConc" else "nm (#/cm³)"

    print(f"\n{'=' * 60}")
    print("SMPS Data Export")
    print(f"{'=' * 60}")
    print(f"Concentration Type: {conc_type}")

    # Load configuration
    config = load_config()
    smps_path = Path(config["instruments"]["smps"]["path"])
    output_dir = Path(config["common_folders"]["smps_export"])

    print(f"SMPS Data Directory: {smps_path}")

    # Find all files matching the concentration type
    pattern = f"MH_apollo_bed_*_{conc_type}.xlsx"
    smps_files = sorted(smps_path.glob(pattern))

    print(f"\nFound {len(smps_files)} {conc_type} files:")
    for f in smps_files:
        print(f"  - {f.name}")

    if len(smps_files) == 0:
        print(f"\nERROR: No {conc_type} files found matching pattern: {pattern}")
        print(f"       in directory: {smps_path}")
        return

    # Process each file
    all_data = []
    units = None

    for file_path in smps_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            df = read_transposed_smps_file(file_path, conc_type)
            all_data.append(df)

            # Capture units from first successful file
            if units is None and len(df) > 0:
                units = df["_units"].iloc[0]

            n_midpoints = sum(1 for c in df.columns if c.endswith(size_units))
            print(f"  Extracted {len(df)} data points, {n_midpoints} diameter bins")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        except Exception as e:
            print(f"  Error processing {file_path.name}: {str(e)}")
            continue

    if len(all_data) == 0:
        print("\nERROR: No data was successfully processed")
        return

    # Combine all data (missing bins across files become NaN)
    print(f"\n{'=' * 60}")
    print("Combining data from all files...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by datetime, remove duplicates
    combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(subset="datetime", keep="first")

    # Drop internal units column
    combined_df = combined_df.drop(columns=["_units"])

    # Re-sort diameter midpoint columns numerically (bins may differ across files)
    midpoint_cols = []
    for col in combined_df.columns:
        if col.endswith(" " + size_units):
            try:
                midpoint_cols.append((float(col.replace(" " + size_units, "")), col))
            except ValueError:
                pass
    midpoint_cols_sorted = [col for _, col in sorted(midpoint_cols)]

    # Fixed columns in desired order
    meta_cols = [c for c in ["Lower Size(nm)", "Upper Size(nm)"] if c in combined_df.columns]
    stat_cols = [
        c
        for c in [
            "D50(nm)",
            "Median(nm)",
            "Mean(nm)",
            "Geo. Mean(nm)",
            "Mode(nm)",
            "Geo. Std. Dev.",
        ]
        if c in combined_df.columns
    ]

    # Rename Total Concentration to include units
    total_conc_col = "Total Concentration"
    if units:
        combined_df = combined_df.rename(columns={total_conc_col: f"Total Concentration ({units})"})
        total_conc_col = f"Total Concentration ({units})"

    final_col_order = ["datetime"] + meta_cols + midpoint_cols_sorted + stat_cols + [total_conc_col]
    combined_df = combined_df[final_col_order]

    print(f"Total data points: {len(combined_df)}")
    print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")

    # Export to CSV
    if output_dir is None:
        output_dir = Path.cwd() / "output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"SMPS_{conc_type}_full.csv"
    output_path = output_dir / output_filename

    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print("Export complete!")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"{'=' * 60}\n")

    # Display first few rows
    print("Preview of exported data:")
    print(combined_df.head(10).to_string(index=False))
    print("...")
    print(combined_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    # Validate configuration
    if CONCENTRATION_TYPE not in ["MassConc", "NumConc"]:
        print(
            f"ERROR: CONCENTRATION_TYPE must be 'MassConc' or 'NumConc', got '{CONCENTRATION_TYPE}'"
        )
        sys.exit(1)

    # Process files
    process_all_smps_files(conc_type=CONCENTRATION_TYPE)
