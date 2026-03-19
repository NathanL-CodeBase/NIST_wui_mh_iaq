#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WUI Spatial Variation Analysis Plotting Script
===============================================

This script generates interactive Bokeh visualizations of spatial variation analysis
results from wildfire smoke infiltration experiments. It creates plots comparing
particulate matter (PM) concentration ratios between two locations (bedroom2 vs
morning room) under different CR Box (portable air cleaner) operating conditions.

Key Features:
    - Visualizes three ratio metrics: Peak Ratio Index, CR Box Activation Ratio,
      and Average Ratio
    - Compares data from two instrument types: AeroTrak and QuantAQ
    - Analyzes two burns with different CR Box configurations (1 and 2 units)
    - Generates size-resolved plots for PM1, PM2.5, and PM10
    - Fits smooth curves to data for trend analysis
    - Embeds script metadata into HTML output for reproducibility
    - Automatically detects desktop vs laptop system for file paths

Note: Burns with 4 CR Boxes have been excluded due to data quality issues.

Input:
    - spatial_variation_analysis.xlsx: Excel file with AeroTrak and QuantAQ sheets
      containing calculated ratios for each burn and PM size

Output:
    - Individual HTML files for each PM size with interactive Bokeh plots
    - Plots saved to Paper_figures directory

Methodology:
    - X-axis represents number of CR Boxes operating (1 or 2 units)
    - Y-axis shows concentration ratios (bedroom2/morning room)
    - Scatter points show measured ratios for each burn
    - Colors distinguish different ratio types (Peak, CR Box Activation, Average)
    - Marker shapes distinguish instruments (AeroTrak circles vs QuantAQ squares)
    - Fitted curves use linear interpolation (for 2 data points)
    - Ratio = 1.0 indicates perfect spatial uniformity

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025
"""

import os
import sys

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div, Span
from bokeh.plotting import figure, output_file, save
from scipy import stats
from scipy.interpolate import interp1d, make_interp_spline

# ============================================================================
# SYSTEM DETECTION AND PATH SETUP
# ============================================================================

# Use portable data paths - no longer need system detection!
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts import get_script_metadata  # noqa: E402
from scripts.plotting_utils import apply_text_formatting  # noqa: E402
from src.data_paths import get_common_file, get_data_root  # noqa: E402

# Get portable paths
BASE_DIR = str(get_data_root())
print(f"[OK] Using data directory: {BASE_DIR}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input file path - Excel file with spatial variation analysis results (using portable path)
EXCEL_FILE_PATH = str(get_common_file("spatial_variation"))

# Output directory for figures (using portable path)
OUTPUT_DIR = str(get_common_file("output_figures"))

# Burns to analyze and their corresponding CR Box counts
# Note: burn2 (4 CR Boxes) excluded due to data quality issues
# burn4: 1 CR Box, burn9: 2 CR Boxes
BURN_IDS = ["burn4", "burn9"]
BURN_TO_CRBOX_COUNT = {"burn4": 1, "burn9": 2}

# Color mapping for ratio types (each ratio gets a unique color)
RATIO_COLORS = {
    "Peak_Ratio_Index": "#d62728",  # Red
    "CRBox_Activation_Ratio": "#1f77b4",  # Blue
    "Average_Ratio": "#2ca02c",  # Green
}

# Display names for ratios in legend
RATIO_DISPLAY_NAMES = {
    "Peak_Ratio_Index": "Peak Ratio",
    "CRBox_Activation_Ratio": "CR Box Activation",
    "Average_Ratio": "Average Ratio",
}

# Marker mapping for instruments (each instrument gets a unique marker)
# Note: device_name values are "OPC" (AeroTrak) and "Nef+OPC" (QuantAQ)
INSTRUMENT_MARKERS = {
    "OPC": "circle",
    "Nef+OPC": "square",
}

# PM sizes to create plots for (QuantAQ sizes)
PM_SIZES = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

# AeroTrak PM3 is used as a proxy for PM2.5 comparison
# Note: AeroTrak measures PM3 which is the closest size bin to PM2.5
AEROTRAK_PM_SIZE_FOR_PM25 = "PM3 (µg/m³)"

# Ratio metrics to plot
RATIO_METRICS = ["Peak_Ratio_Index", "CRBox_Activation_Ratio", "Average_Ratio"]

# Plot configuration
PLOT_Y_RANGE = (0, 1.5)  # Y-axis range for ratio plots
X_AXIS_TICKS = [1, 2]  # X-axis tick positions (CR Box counts)
X_AXIS_LABELS = {1.0: "1", 2.0: "2"}  # Custom labels for x-axis (keys must be float)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_spatial_variation_data(file_path):
    """
    Load spatial variation analysis results from Excel file

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing spatial variation analysis results

    Returns
    -------
    tuple of pd.DataFrame
        (aerotrak_df, quantaq_df) - DataFrames for AeroTrak and QuantAQ data

    Raises
    ------
    FileNotFoundError
        If the Excel file is not found
    ValueError
        If required sheets are missing from the Excel file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    try:
        aerotrak_df = pd.read_excel(file_path, sheet_name="AeroTrak")
        quantaq_df = pd.read_excel(file_path, sheet_name="QuantAQ")
    except ValueError as e:
        raise ValueError(
            f"Error reading Excel file. Ensure it has 'AeroTrak' and 'QuantAQ' sheets: {e}"
        ) from e

    return aerotrak_df, quantaq_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def create_fitted_curve(x_data, y_data, num_points=100):
    """
    Create a smooth fitted curve through data points

    Uses cubic spline interpolation if possible, falls back to quadratic
    or linear interpolation for fewer data points.

    Parameters
    ----------
    x_data : array-like
        X coordinates of data points
    y_data : array-like
        Y coordinates of data points
    num_points : int, optional
        Number of points in the fitted curve (default: 100)

    Returns
    -------
    tuple of np.ndarray
        (x_fit, y_fit) - X and Y coordinates of fitted curve

    Notes
    -----
    - Cubic spline (k=3) requires at least 4 data points
    - Quadratic interpolation requires at least 3 data points
    - Linear interpolation works with 2 data points
    - Returns None, None if fitting fails
    """
    try:
        n_points = len(x_data)

        # Need at least 2 points for any interpolation
        if n_points < 2:
            return None, None

        x_fit = np.linspace(x_data.min(), x_data.max(), num_points)

        # For 2 points, use linear interpolation
        if n_points == 2:
            try:
                f = interp1d(x_data, y_data, kind="linear")
                y_fit = f(x_fit)
                return x_fit, y_fit
            except (ValueError, TypeError):
                return None, None

        # For 3+ points, try spline/quadratic/linear in order
        try:
            # Try cubic spline for smooth curve (needs 4+ points)
            if n_points >= 4:
                f = make_interp_spline(x_data, y_data, k=3)
            else:
                # For 3 points, use quadratic spline
                f = make_interp_spline(x_data, y_data, k=2)
            y_fit = f(x_fit)
        except (ValueError, TypeError):
            # Fallback to quadratic fit if spline fails
            try:
                f = interp1d(x_data, y_data, kind="quadratic")
                y_fit = f(x_fit)
            except (ValueError, TypeError):
                # If quadratic also fails, try linear
                try:
                    f = interp1d(x_data, y_data, kind="linear")
                    y_fit = f(x_fit)
                except (ValueError, TypeError):
                    return None, None

        return x_fit, y_fit

    except (ValueError, TypeError) as e:
        print(f"  Warning: Could not create fitted curve: {e}")
        return None, None


def perform_linear_fit(x_data, y_data):
    """Perform linear regression with full statistics

    Note: For 2-point data, the fit is perfect (R²=1.0, residuals=0).
    In this case, AIC is set to -inf to indicate the best possible fit,
    and standard errors are set to 0.
    """
    try:
        if len(x_data) < 2:
            return None
        x, y = np.array(x_data), np.array(y_data)
        n = len(x)
        # Unpack linregress result - returns LinregressResult with named attributes
        linreg_result = stats.linregress(x, y)
        slope = float(linreg_result.slope)  # type: ignore
        intercept = float(linreg_result.intercept)  # type: ignore
        r_value = float(linreg_result.rvalue)  # type: ignore
        slope_stderr = float(linreg_result.stderr)  # type: ignore
        r_squared = float(r_value) ** 2
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        # Calculate x_mean first, then ss_x (ss_x depends on x_mean)
        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean) ** 2)

        # Handle 2-point case specially (perfect fit, no degrees of freedom for error)
        if n == 2:
            mse = 0
            intercept_stderr = 0  # Cannot estimate with only 2 points
            adj_r_squared = r_squared  # Already 1.0 for 2 points
            aic = float("-inf")  # Perfect fit has best (lowest) AIC
        else:
            mse = ss_res / (n - 2)
            intercept_stderr = (
                np.sqrt(mse * (1 / n + x_mean**2 / ss_x)) if ss_x > 0 else 0
            )
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
            # AIC calculation - handle case where ss_res is very small
            if ss_res > 1e-15:
                aic = 2 * 2 + n * np.log(ss_res / n)
            else:
                aic = float("-inf")  # Near-perfect fit

        return {
            "type": "linear",
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "slope": slope,
            "intercept": intercept,
            "slope_stderr": slope_stderr,
            "intercept_stderr": intercept_stderr,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "aic": aic,
            "n_points": n,
        }
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def perform_polynomial_fit(x_data, y_data, degree=2):
    """Perform polynomial regression with full statistics"""
    try:
        if len(x_data) < degree + 1:
            return None
        x, y = np.array(x_data), np.array(y_data)
        n = len(x)
        coeffs, cov_matrix = np.polyfit(x, y, degree, cov=True)
        coeff_stderr = np.sqrt(np.diag(cov_matrix))
        y_pred = np.polyval(coeffs, x)
        ss_res, ss_tot = np.sum((y - y_pred) ** 2), np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        k = degree + 1
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else r_squared
        aic = 2 * k + n * np.log(ss_res / n) if ss_res > 0 else np.inf
        terms = []
        for i, coeff in enumerate(coeffs):
            power = degree - i
            if abs(coeff) > 1e-10:
                if power == 0:
                    terms.append(f"{coeff:.4f}")
                elif power == 1:
                    terms.append(f"{coeff:.4f}x")
                else:
                    terms.append(f"{coeff:.4f}x^{power}")
        equation = "y = " + " + ".join(terms)
        equation = equation.replace("+ -", "- ")
        degree_names = {2: "quadratic", 3: "cubic"}
        fit_type = degree_names.get(degree, f"poly_deg{degree}")
        result = {
            "type": fit_type,
            "degree": degree,
            "equation": equation,
            "coefficients": coeffs.tolist(),
            "coeff_stderr": coeff_stderr.tolist(),
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "aic": aic,
            "n_points": n,
        }
        if degree == 2:
            result.update(
                {
                    "a": coeffs[0],
                    "b": coeffs[1],
                    "c": coeffs[2],
                    "a_stderr": coeff_stderr[0],
                    "b_stderr": coeff_stderr[1],
                    "c_stderr": coeff_stderr[2],
                }
            )
        return result
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def select_best_fit(x_data, y_data):
    """Try multiple fits and select best using AIC"""
    n = len(x_data)
    if n < 2:
        return None
    fits = []
    linear_fit = perform_linear_fit(x_data, y_data)
    if linear_fit:
        fits.append(linear_fit)
    if n >= 3:
        quad_fit = perform_polynomial_fit(x_data, y_data, degree=2)
        if quad_fit:
            fits.append(quad_fit)
    if n >= 5:
        cubic_fit = perform_polynomial_fit(x_data, y_data, degree=3)
        if cubic_fit:
            fits.append(cubic_fit)
    if not fits:
        return None
    best_fit = min(fits, key=lambda f: f["aic"])
    best_fit["comparison"] = {
        "types_tested": [f["type"] for f in fits],
        "aic_values": {f["type"]: f["aic"] for f in fits},
    }
    return best_fit


def create_spatial_variation_plot(
    aerotrak_data, quantaq_data, pm_size, burn_to_crbox_map
):
    """
    Create Bokeh plot for spatial variation ratios

    Parameters
    ----------
    aerotrak_data : pd.DataFrame
        Filtered AeroTrak data for the specific PM size
    quantaq_data : pd.DataFrame
        Filtered QuantAQ data for the specific PM size
    pm_size : str
        PM size label (e.g., 'PM2.5 (µg/m³)')
    burn_to_crbox_map : dict
        Mapping from burn ID to CR Box count

    Returns
    -------
    tuple
        (bokeh.plotting.figure.Figure, dict) - Bokeh figure and fit information
    """
    # Create figure
    p = figure(
        #title=f"Spatial Variation Analysis - {pm_size}",
        x_axis_label="Number of CR Boxes",
        y_axis_label="Concentration Ratio (Bedroom2 / Morning Room)",
        y_range=(float(PLOT_Y_RANGE[0]), float(PLOT_Y_RANGE[1])),  # type: ignore
        width=800,
        height=600,
    )

    # Apply standard text formatting configuration
    apply_text_formatting(p)

    # Track fit info
    fit_info = {}

    # Plot data for each ratio type (colored by ratio)
    for ratio in RATIO_METRICS:
        # Get color for this ratio type
        ratio_color = RATIO_COLORS.get(ratio, "black")
        ratio_display = RATIO_DISPLAY_NAMES.get(ratio, ratio)

        # Process each instrument separately (different marker shapes)
        for device_data, device_name in [
            (aerotrak_data, "OPC"),
            (quantaq_data, "Nef+OPC"),
        ]:
            if device_data.empty:
                continue

            # Skip if ratio column doesn't exist or has no valid data
            if ratio not in device_data.columns:
                print(f"  Warning: {ratio} not found in {device_name} data")
                continue

            # Get marker shape for this instrument
            marker = INSTRUMENT_MARKERS.get(device_name, "circle")

            # Collect ALL data points for this instrument and ratio FIRST
            all_x = []
            all_y = []

            for burn_id in BURN_IDS:
                burn_data = device_data[device_data["Burn_ID"] == burn_id]

                if burn_data.empty:
                    continue

                # Map burn ID to CR Box count for x-axis
                x_val = burn_to_crbox_map.get(burn_id)
                y_val = burn_data[ratio].values

                if x_val is None or len(y_val) == 0:
                    continue

                y_val = y_val[0]  # Get first value

                # Skip NaN values
                if pd.isna(x_val) or pd.isna(y_val):
                    continue

                all_x.append(x_val)
                all_y.append(y_val)

            # Skip if no valid data points
            if len(all_x) == 0:
                continue

            # Create a SINGLE scatter plot with ALL points for this ratio-instrument combo
            # This ensures legend toggle works for all points together
            source = ColumnDataSource(data={"x": all_x, "y": all_y})

            p.scatter(
                x="x",
                y="y",
                source=source,
                color=ratio_color,
                marker=marker,
                size=10,
                alpha=1.0,
                legend_label=f"{ratio_display} - {device_name}",
            )

            # Add fitted curve for this instrument and ratio (if we have enough points)
            if len(all_x) >= 2:
                # Convert to numpy arrays and sort by x
                x_array = np.array(all_x)
                y_array = np.array(all_y)
                sort_idx = np.argsort(x_array)
                x_sorted = x_array[sort_idx]
                y_sorted = y_array[sort_idx]

                # Select best statistical fit (works for 2+ points)
                best_fit = select_best_fit(x_sorted, y_sorted)

                # Store fit info ALWAYS if we have a valid statistical fit
                # This is separate from curve visualization
                fit_key = f"{ratio_display} - {device_name}"

                if best_fit:
                    fit_info[fit_key] = {"n_points": len(all_x), "best_fit": best_fit}
                    eq = best_fit["equation"]
                    r2 = best_fit["r_squared"]
                    adj_r2 = best_fit["adj_r_squared"]
                    ftype = best_fit["type"]
                    print(
                        f"    OK {fit_key}: {eq}, R²={r2:.4f} (adj. R²={adj_r2:.4f}) [{ftype}]"
                    )

                # Create smooth curve for visualization (now works for 2+ points)
                x_fit, y_fit = create_fitted_curve(x_sorted, y_sorted)
                if x_fit is not None and y_fit is not None:
                    # Determine line dash style based on device (AeroTrak: dashed, QuantAQ: dashdot)
                    line_dash_style = "dashed" if device_name == "OPC" else "dotdash"

                    # Plot fit line WITHOUT adding to legend
                    p.line(
                        x_fit,
                        y_fit,
                        color=ratio_color,
                        line_width=2,
                        alpha=1.0,
                        line_dash=line_dash_style,
                    )
                else:
                    # Even if curve visualization fails, we still have fit_info stored above
                    print(
                        f"    WARNING: {ratio_display} - {device_name}: Curve visualization failed (fit info still recorded)"
                    )
            elif len(all_x) == 1:
                print(
                    f"    SKIP: {ratio_display} - {device_name}: Insufficient data ({len(all_x)} pts)"
                )

    # Customize x-axis ticks and labels
    p.xaxis.ticker = X_AXIS_TICKS
    p.xaxis.major_label_overrides = X_AXIS_LABELS  # type: ignore

    # Customize legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # Allow clicking legend to hide/show all points

    # Add horizontal line at y=1.0 (perfect spatial uniformity)
    uniform_line = Span(
        location=1.0,
        dimension="width",
        line_color="black",
        line_dash="dotted",
        line_width=1,
        line_alpha=1.0,
    )
    p.add_layout(uniform_line)

    return p, fit_info


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function to generate spatial variation plots

    Loads data, filters by selected burns, creates plots for each PM size,
    and saves HTML files with embedded metadata.
    """
    print("=" * 80)
    print("WUI SPATIAL VARIATION ANALYSIS - PLOTTING")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")

    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"\nWARNING: Base directory not found: {BASE_DIR}")
        print("Please check the path configuration.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Get script metadata
    try:
        metadata = get_script_metadata()
        print("Metadata loaded successfully")
    except (ImportError, AttributeError, OSError, RuntimeError) as e:
        # ImportError: module not found, AttributeError: function not found
        # OSError: file system issues, RuntimeError: execution errors
        metadata = "Metadata unavailable"
        print(f"Metadata not available: {e}")

    # Load data
    print(f"\nLoading data from: {EXCEL_FILE_PATH}")
    try:
        aerotrak_df, quantaq_df = load_spatial_variation_data(EXCEL_FILE_PATH)
        print(f"  AeroTrak data: {aerotrak_df.shape}")
        print(f"  QuantAQ data: {quantaq_df.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    # Filter data for selected burn IDs
    print(f"\nFiltering data for burns: {BURN_IDS}")
    aerotrak_df = aerotrak_df[aerotrak_df["Burn_ID"].isin(BURN_IDS)]
    quantaq_df = quantaq_df[quantaq_df["Burn_ID"].isin(BURN_IDS)]
    print(f"  Filtered AeroTrak data: {aerotrak_df.shape}")
    print(f"  Filtered QuantAQ data: {quantaq_df.shape}")

    # Generate plots for each PM size
    print(f"\nGenerating plots for PM sizes: {PM_SIZES}")
    for pm_size in PM_SIZES:
        print(f"\n{'=' * 60}")
        print(f"Processing {pm_size}")
        print(f"{'=' * 60}")

        # Filter data for current PM size
        # Special case: Use PM3 from AeroTrak as proxy for PM2.5
        if pm_size == "PM2.5 (µg/m³)":
            aerotrak_data = aerotrak_df[
                aerotrak_df["PM_Size"] == AEROTRAK_PM_SIZE_FOR_PM25
            ]
            print(f"  Using {AEROTRAK_PM_SIZE_FOR_PM25} from AeroTrak as PM2.5 proxy")
        else:
            aerotrak_data = aerotrak_df[aerotrak_df["PM_Size"] == pm_size]

        quantaq_data = quantaq_df[quantaq_df["PM_Size"] == pm_size]

        print(f"  AeroTrak data points: {len(aerotrak_data)}")
        print(f"  QuantAQ data points: {len(quantaq_data)}")

        # Create plot
        p, fit_info = create_spatial_variation_plot(
            aerotrak_data, quantaq_data, pm_size, BURN_TO_CRBOX_COUNT
        )

        # Format fit information for metadata
        fit_metadata_lines = ["<strong>Fitted Curves Information:</strong><br><br>"]
        if fit_info:
            for fit_key, info in sorted(fit_info.items()):
                n_pts = info["n_points"]
                best_fit = info.get("best_fit")

                if best_fit:
                    ftype = best_fit["type"]
                    eq = best_fit["equation"]
                    r2 = best_fit["r_squared"]
                    adj_r2 = best_fit["adj_r_squared"]

                    fit_metadata_lines.append(
                        f"<strong>{fit_key}:</strong> {n_pts} data points<br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;<strong>{ftype.capitalize()} fit:</strong> {eq}<br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}<br>"
                    )

                    # Add parameters with uncertainties
                    if ftype == "linear":
                        slope, intercept = best_fit["slope"], best_fit["intercept"]
                        slope_err, int_err = (
                            best_fit["slope_stderr"],
                            best_fit["intercept_stderr"],
                        )
                        fit_metadata_lines.append(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;Parameters: slope = {slope:.4f} ± {slope_err:.4f}, "
                            f"intercept = {intercept:.4f} ± {int_err:.4f}<br>"
                        )
                    elif ftype == "quadratic":
                        a, b, c = best_fit["a"], best_fit["b"], best_fit["c"]
                        a_err, b_err, c_err = (
                            best_fit["a_stderr"],
                            best_fit["b_stderr"],
                            best_fit["c_stderr"],
                        )
                        fit_metadata_lines.append(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;Parameters: a = {a:.4f} ± {a_err:.4f}, "
                            f"b = {b:.4f} ± {b_err:.4f}, c = {c:.4f} ± {c_err:.4f}<br>"
                        )

                    # Add comparison info
                    if "comparison" in best_fit:
                        comp = best_fit["comparison"]
                        tested = ", ".join(comp["types_tested"])
                        aic_vals = comp["aic_values"]
                        aic_str = ", ".join(
                            [f"{k}: {v:.1f}" for k, v in aic_vals.items()]
                        )
                        fit_metadata_lines.append(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;<em>Fits compared: {tested} (AIC: {aic_str})</em><br>"
                        )

                    fit_metadata_lines.append("<br>")
                else:
                    fit_metadata_lines.append(
                        f"<strong>{fit_key}:</strong> {n_pts} pts (fit unavailable)<br><br>"
                    )
        else:
            fit_metadata_lines.append("No fits created for this PM size.<br>")

        fit_metadata = "".join(fit_metadata_lines)

        # Create metadata div with FITS FIRST, then script info
        div_text = (
            f'<div style="font-size: 10pt; font-weight: normal;">'
            f"{fit_metadata}<br><hr><br>{metadata}</div>"
        )
        metadata_div = Div(text=div_text, width=800)

        # Combine plot and metadata
        layout = column(p, metadata_div)

        # Save figure
        output_filename = f"{pm_size.replace(' ', '_').replace('/', '_')}.html"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        output_file(output_path)
        save(layout)
        print(f"  Saved plot to: {output_path}")

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
