"""
Statistical analysis and visualization module for WUI smoke decay rate data.

This script processes decay rate data from various instruments, performs statistical
analyses (ANOVA, t-tests, and z-tests), and generates bar charts for different
particulate matter (PM) sizes and experimental conditions.

The module contains functions for:
1. Data reading and preprocessing
2. Baseline correction
3. Statistical analyses (filter count, new vs used, MERV comparison)
4. Chart creation
5. Statistical summary generation

The main processing section executes the statistical analyses and chart generation
for PM0.4, PM1, PM2.5, and PM10 data.

Requires:
    - pandas
    - numpy
    - bokeh
    - scipy
    - metadata_utils (custom utility module)

Output:
    - Statistical analysis results (decay_statistical_analyses.txt)
    - Statistical summary (decay_statistical_summary.txt)
    - Bar charts for different experimental conditions (HTML files)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div, Range1d
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.transform import dodge
from scipy.stats import f_oneway, norm, ttest_ind

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from scripts import (
    get_script_metadata,  # pylint: disable=import-error,wrong-import-position
)
from src.data_paths import get_common_file, get_data_root

# ============================================================================
# STATISTICAL ANALYSIS CONFIGURATION
# ============================================================================
STATISTICAL_CONFIG = {
    "alpha": 0.05,  # Significance level for hypothesis tests
    "confidence_level": 0.95,  # Confidence level (1 - alpha)
}

# Define needed paths - using portable data paths
data_root = get_data_root()
ABSOLUTE_PATH = str(data_root / "burn_data" / "burn_calcs")
BASE_PATH = str(get_common_file("output_figures"))
STATS_OUTPUT_PATH = str(data_root / "burn_data")

# Define the instruments
instruments = [
    "AeroTrakB",
    "AeroTrakK",
    "DustTrak",
    "PurpleAirK",
    "QuantAQB",
    "QuantAQK",
    "SMPS",
]

# Define pollutant types
pollutant_types_pm04 = {"SMPS": "Total Concentration (µg/m³)"}

pollutant_types_pm1 = {
    "AeroTrakB": "PM1 (µg/m³)",
    "AeroTrakK": "PM1 (µg/m³)",
    "DustTrak": "PM1 (µg/m³)",
    "QuantAQB": "PM1 (µg/m³)",
    "QuantAQK": "PM1 (µg/m³)",
}

pollutant_types_pm25 = {
    "AeroTrakB": "PM3 (µg/m³)",
    "AeroTrakK": "PM3 (µg/m³)",
    "DustTrak": "PM2.5 (µg/m³)",
    "PurpleAirK": "PM2.5 (µg/m³)",
    "QuantAQB": "PM2.5 (µg/m³)",
    "QuantAQK": "PM2.5 (µg/m³)",
}

pollutant_types_pm10 = {
    "AeroTrakB": "PM10 (µg/m³)",
    "AeroTrakK": "PM10 (µg/m³)",
    "DustTrak": "PM10 (µg/m³)",
    "QuantAQB": "PM10 (µg/m³)",
    "QuantAQK": "PM10 (µg/m³)",
}

# Define colors for instruments
colors = {
    "AeroTrakB": "#003f5c",
    "AeroTrakK": "#1d4772",
    "DustTrak": "#404e84",
    "PurpleAirK": "#665191",
    "QuantAQB": "#8d5196",
    "QuantAQK": "#b35093",
    "SMPS": "#d45087",
    "Mean": "#2ca02c",
    "Mean_PM0.4": "#FF6B35",
    "Mean_PM1": "#90EE90",
    "Mean_PM2.5": "#228B22",
}

# Define display names for the plot legend
display_names = {
    "AeroTrakB": "AeroTrak1",
    "AeroTrakK": "AeroTrak2",
    "DustTrak": "DustTrak",
    "PurpleAirK": "PurpleAir",
    "QuantAQB": "QuantAQ1",
    "QuantAQK": "QuantAQ2",
    "SMPS": "SMPS",
    "Mean": "Mean",
}

# Define burn labels
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

# Text formatting configuration - centralized for easy modification
TEXT_CONFIG = {
    "font_size": "12pt",
    "title_font_size": "12pt",
    "axis_label_font_size": "12pt",
    "axis_tick_font_size": "12pt",
    "legend_font_size": "12pt",
    "label_font_size": "12pt",
    "font_style": "normal",
    "plot_font_style": "bold",  # For Bokeh plot elements
    "html_font_weight": "normal",  # For HTML div elements
}

# PM size ordering - ensures consistent display order
PM_SIZE_ORDER = ["PM0.4", "PM1", "PM2.5", "PM10"]


# Function to read data from Excel files
def read_instrument_data(instrument, pollutant_types_dict):
    """Read decay data for a specific instrument from Excel files."""
    file_path = os.path.join(ABSOLUTE_PATH, f"{instrument}_decay_and_CADR.xlsx")
    try:
        df = pd.read_excel(file_path)

        # Get the correct pollutant for this instrument
        pollutant = pollutant_types_dict[instrument]

        # Filter data for the specified pollutant
        df_filtered = df[df["pollutant"] == pollutant].copy()

        # Some rows might not have decay values - filter those out
        df_filtered = df_filtered[df_filtered["decay"].notna()]

        # Extract burn, decay, and uncertainty information
        result = {}
        for _, row in df_filtered.iterrows():
            burn = row["burn"]
            decay = row["decay"]
            uncertainty = row["decay_uncertainty"]
            crboxes = row["CRboxes"] if "CRboxes" in row and pd.notna(row["CRboxes"]) else 1

            result[burn] = {
                "decay": decay,
                "uncertainty": uncertainty,
                "CRboxes": crboxes,
            }

        return result
    except FileNotFoundError:
        print(f"Warning: File not found {file_path} - skipping {instrument}")
        return {}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}


class BaselineCalculator:
    """Centralized baseline calculation for all instruments and burns."""

    def __init__(self, pm04_data, pm1_data, pm25_data, pm10_data):
        """Initialize with all PM size data."""
        self.baseline_burns = ["burn1", "burn5"]

        # Store baseline data for each PM size
        self.baselines = {
            "PM0.4": self._calculate_baselines(pm04_data),
            "PM1": self._calculate_baselines(pm1_data),
            "PM2.5": self._calculate_baselines(pm25_data),
            "PM10": self._calculate_baselines(pm10_data),
        }

    def _calculate_baselines(self, data_by_instrument):
        """Calculate baseline for each instrument."""
        baselines = {}

        for instrument in data_by_instrument:
            baseline_values = []
            for burn in self.baseline_burns:
                if burn in data_by_instrument[instrument]:
                    baseline_values.append(data_by_instrument[instrument][burn]["decay"])

            if baseline_values:
                baselines[instrument] = np.mean(baseline_values)

        return baselines

    def get_baseline(self, instrument, pm_size):
        """Get baseline for a specific instrument and PM size."""
        if pm_size in self.baselines and instrument in self.baselines[pm_size]:
            return self.baselines[pm_size][instrument]
        return 0.0


def get_matching_instruments_for_burns(data_by_instrument, burn_list):
    """Find instruments that have data for ALL specified burns."""
    if not burn_list:
        return []

    # Start with instruments that have data for the first burn
    first_burn = burn_list[0]
    common_instruments = set()

    for instrument in data_by_instrument:
        if instrument == "Mean":  # Skip Mean
            continue
        if first_burn in data_by_instrument[instrument]:
            common_instruments.add(instrument)

    # Keep only instruments that have data for ALL burns
    for burn in burn_list[1:]:
        burn_instruments = set()
        for instrument in data_by_instrument:
            if instrument == "Mean":  # Skip Mean
                continue
            if burn in data_by_instrument[instrument]:
                burn_instruments.add(instrument)

        # Intersection with previous common instruments
        common_instruments = common_instruments.intersection(burn_instruments)

    return list(common_instruments)


def create_baseline_corrected_data(
    data_by_instrument,
    baseline_calculator,
    pollutant_type,
    burns_to_process,
    exclude_instruments=None,
    normalize_by_crboxes=True,
):
    """Create baseline-corrected data for all instruments with centralized baseline calculator.

    Parameters:
    -----------
    normalize_by_crboxes : bool, optional (default=True)
        If True, normalizes decay rate by dividing by the number of PACs (per-PAC rate).
        If False, returns total decay rate without normalization.
    """
    if exclude_instruments is None:
        exclude_instruments = []

    corrected_data = {}

    for instrument in data_by_instrument:
        if instrument in exclude_instruments:
            continue

        corrected_data[instrument] = {}
        baseline = baseline_calculator.get_baseline(instrument, pollutant_type)

        for burn in burns_to_process:
            if burn in data_by_instrument[instrument]:
                original_decay = data_by_instrument[instrument][burn]["decay"]
                original_uncertainty = data_by_instrument[instrument][burn]["uncertainty"]
                crboxes = data_by_instrument[instrument][burn]["CRboxes"]

                # Baseline correction: subtract baseline
                if normalize_by_crboxes:
                    # Normalize by number of PACs (per-PAC rate)
                    corrected_decay = (original_decay - baseline) / crboxes
                    corrected_uncertainty = original_uncertainty / crboxes
                    normalized_crboxes = 1  # Already normalized
                else:
                    # Total decay rate (not normalized by PACs)
                    corrected_decay = original_decay - baseline
                    corrected_uncertainty = original_uncertainty
                    normalized_crboxes = crboxes  # Keep original count

                corrected_data[instrument][burn] = {
                    "decay": corrected_decay,
                    "uncertainty": corrected_uncertainty,
                    "CRboxes": normalized_crboxes,
                }

    return corrected_data


def calculate_mean_data(corrected_data, burns_to_process):
    """Calculate mean across instruments for each burn."""
    mean_data = {}

    for burn in burns_to_process:
        # Collect baseline-corrected values for this burn
        baseline_corrected_values = []
        baseline_corrected_uncertainties = []
        contributing_instruments = []

        for instrument in corrected_data:
            if burn in corrected_data[instrument]:
                baseline_corrected_values.append(corrected_data[instrument][burn]["decay"])
                baseline_corrected_uncertainties.append(
                    corrected_data[instrument][burn]["uncertainty"]
                )
                contributing_instruments.append(instrument)

        # Calculate mean if we have values
        if baseline_corrected_values:
            mean_decay = np.mean(baseline_corrected_values)

            # Calculate uncertainty
            sem = np.std(baseline_corrected_values, ddof=1) / np.sqrt(
                len(baseline_corrected_values)
            )
            instrument_uncertainty_contribution = np.sqrt(
                np.sum(np.array(baseline_corrected_uncertainties) ** 2)
            ) / len(baseline_corrected_values)
            mean_uncertainty = np.sqrt(sem**2 + instrument_uncertainty_contribution**2)

            mean_data[burn] = {
                "decay": mean_decay,
                "uncertainty": mean_uncertainty,
                "CRboxes": 1,  # Already converted to per-PAC
                "contributing_instruments": contributing_instruments,
            }

    return mean_data


# ============================================================================
# NEW PM0.4 STATISTICAL ANALYSIS FUNCTIONS (Z-TEST APPROACH)
# ============================================================================


def perform_z_test_comparison(mu_a, sd_a, mu_b, sd_b, label_a, label_b, output_file):
    """
    Perform z-test comparison between two groups using Type B uncertainty approach.

    Based on John Lu's email from September 17, 2025:
    - var(mu_a - mu_b) = var(mu_a) + var(mu_b) = sd(mu_a)^2 + sd(mu_b)^2
    - z = (mu_a - mu_b) / sqrt(var(mu_a - mu_b))
    - p-value = 2 * (1 - Pnorm(|z|))

    Parameters:
    -----------
    mu_a : float
        Mean value for group A
    sd_a : float
        Standard deviation (uncertainty) for group A
    mu_b : float
        Mean value for group B
    sd_b : float
        Standard deviation (uncertainty) for group B
    label_a : str
        Label for group A
    label_b : str
        Label for group B
    output_file : file object
        File to write output to

    Returns:
    --------
    dict : Dictionary with test results
    """
    # Calculate combined variance
    var_diff = sd_a**2 + sd_b**2

    # Calculate z-statistic
    z_stat = (mu_a - mu_b) / np.sqrt(var_diff)

    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    # Determine significance
    alpha = STATISTICAL_CONFIG["alpha"]
    is_significant = p_value < alpha

    # Create significance indicator
    if p_value < 0.001:
        sig_indicator = "***"
    elif p_value < 0.01:
        sig_indicator = "**"
    elif p_value < 0.05:
        sig_indicator = "*"
    else:
        sig_indicator = "ns"

    # Write results
    output_file.write(f"\n  Comparison: {label_a} vs {label_b}\n")
    output_file.write(f"    {label_a}: mean = {mu_a:.4f} ± {sd_a:.4f} h⁻¹\n")
    output_file.write(f"    {label_b}: mean = {mu_b:.4f} ± {sd_b:.4f} h⁻¹\n")
    output_file.write(f"    Difference: {mu_a - mu_b:.4f} h⁻¹\n")
    output_file.write(f"    Combined variance: {var_diff:.6f}\n")
    output_file.write(f"    Z-statistic: {z_stat:.4f}\n")
    output_file.write(f"    P-value: {p_value:.6f}\n")
    output_file.write(f"    Significant at α={alpha}: {is_significant} ({sig_indicator})\n")

    return {
        "label_a": label_a,
        "label_b": label_b,
        "mu_a": mu_a,
        "sd_a": sd_a,
        "mu_b": mu_b,
        "sd_b": sd_b,
        "z_stat": z_stat,
        "p_value": p_value,
        "significant": is_significant,
        "sig_indicator": sig_indicator,
    }


def perform_pm04_filter_count_ztest(data_by_instrument, baseline_calculator, output_file):
    """
    Perform PM0.4 filter count analysis using z-test approach.
    Compares 1 PAC, 2 PACs, and 4 PACs (all pairwise comparisons).
    """
    output_file.write("\n" + "=" * 60 + "\n")
    output_file.write("PM0.4 (SMPS) FILTER COUNT ANALYSIS (Z-TEST)\n")
    output_file.write("=" * 60 + "\n")

    if "SMPS" not in data_by_instrument:
        output_file.write("No SMPS data available for PM0.4 analysis\n")
        return {}

    smps_data = data_by_instrument["SMPS"]

    # Define burns and labels
    filter_count_burns = ["burn4", "burn9", "burn2"]  # 1, 2, 4 PACs
    filter_count_labels = ["1 PAC", "2 PACs", "4 PACs"]

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        {"SMPS": smps_data},
        baseline_calculator,
        "PM0.4",
        filter_count_burns,
        exclude_instruments=[],
    )

    # Get baseline-corrected values
    values = {}
    for i, burn in enumerate(filter_count_burns):
        if burn in corrected_data["SMPS"]:
            values[filter_count_labels[i]] = {
                "mean": corrected_data["SMPS"][burn]["decay"],
                "sd": corrected_data["SMPS"][burn]["uncertainty"],
            }

    output_file.write("\n1. Filter Count Analysis (1 vs 2 vs 4 PACs)\n")
    output_file.write("-" * 60 + "\n")

    # Perform all pairwise comparisons
    results = []
    comparisons = [("1 PAC", "2 PACs"), ("1 PAC", "4 PACs"), ("2 PACs", "4 PACs")]

    for label_a, label_b in comparisons:
        if label_a in values and label_b in values:
            result = perform_z_test_comparison(
                values[label_a]["mean"],
                values[label_a]["sd"],
                values[label_b]["mean"],
                values[label_b]["sd"],
                label_a,
                label_b,
                output_file,
            )
            results.append(result)

    return {"filter_count": results}


def perform_pm04_new_vs_used_ztest(data_by_instrument, baseline_calculator, output_file):
    """
    Perform PM0.4 new vs used filter analysis using z-test approach.
    """
    output_file.write("\n2. New vs Used Filter Analysis\n")
    output_file.write("-" * 60 + "\n")

    if "SMPS" not in data_by_instrument:
        output_file.write("No SMPS data available for PM0.4 analysis\n")
        return {}

    smps_data = data_by_instrument["SMPS"]

    # Define burns for new vs used
    new_used_burns = ["burn4", "burn3", "burn9", "burn10"]  # 1N, 1U, 2N, 2U

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        {"SMPS": smps_data},
        baseline_calculator,
        "PM0.4",
        new_used_burns,
        exclude_instruments=[],
    )

    # Get baseline-corrected values
    values = {}
    burn_labels = {
        "burn4": "1 New",
        "burn3": "1 Used",
        "burn9": "2 New",
        "burn10": "2 Used",
    }

    for burn, label in burn_labels.items():
        if burn in corrected_data["SMPS"]:
            values[label] = {
                "mean": corrected_data["SMPS"][burn]["decay"],
                "sd": corrected_data["SMPS"][burn]["uncertainty"],
            }

    # Perform comparisons
    results = []
    comparisons = [("1 New", "1 Used"), ("2 New", "2 Used")]

    for label_a, label_b in comparisons:
        if label_a in values and label_b in values:
            result = perform_z_test_comparison(
                values[label_a]["mean"],
                values[label_a]["sd"],
                values[label_b]["mean"],
                values[label_b]["sd"],
                label_a,
                label_b,
                output_file,
            )
            results.append(result)

    return {"new_vs_used": results}


def perform_pm04_merv_comparison_ztest(data_by_instrument, baseline_calculator, output_file):
    """
    Perform PM0.4 MERV filter comparison using z-test approach.
    Compares MERV12A (burn7, burn8) vs MERV13 (burn9, burn10).
    """
    output_file.write("\n3. MERV Filter Comparison (MERV12A vs MERV13)\n")
    output_file.write("-" * 60 + "\n")

    if "SMPS" not in data_by_instrument:
        output_file.write("No SMPS data available for PM0.4 analysis\n")
        return {}

    smps_data = data_by_instrument["SMPS"]

    # Define burns for MERV comparison
    merv_burns = ["burn7", "burn8", "burn9", "burn10"]

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        {"SMPS": smps_data},
        baseline_calculator,
        "PM0.4",
        merv_burns,
        exclude_instruments=[],
    )

    # Get baseline-corrected values - separate new and used for each MERV type
    values = {}
    burn_labels = {
        "burn7": "MERV12A_new",
        "burn8": "MERV12A_used",
        "burn9": "MERV13_new",
        "burn10": "MERV13_used",
    }

    for burn, label in burn_labels.items():
        if burn in corrected_data["SMPS"]:
            values[label] = {
                "mean": corrected_data["SMPS"][burn]["decay"],
                "sd": corrected_data["SMPS"][burn]["uncertainty"],
            }

    # Perform comparisons
    results = []

    # Compare MERV12A new vs MERV13 new
    if "MERV12A_new" in values and "MERV13_new" in values:
        result = perform_z_test_comparison(
            values["MERV12A_new"]["mean"],
            values["MERV12A_new"]["sd"],
            values["MERV13_new"]["mean"],
            values["MERV13_new"]["sd"],
            "MERV12A New",
            "MERV13 New",
            output_file,
        )
        results.append(result)

    # Compare MERV12A used vs MERV13 used
    if "MERV12A_used" in values and "MERV13_used" in values:
        result = perform_z_test_comparison(
            values["MERV12A_used"]["mean"],
            values["MERV12A_used"]["sd"],
            values["MERV13_used"]["mean"],
            values["MERV13_used"]["sd"],
            "MERV12A Used",
            "MERV13 Used",
            output_file,
        )
        results.append(result)

    return {"merv_comparison": results}


# ============================================================================
# T-TEST AND ANOVA FUNCTIONS FOR PM1, PM2.5, PM10
# ============================================================================


def perform_filter_count_analysis(
    data_by_instrument, baseline_calculator, pollutant_type, output_file
):
    """Perform filter count analysis using both ANOVA and pairwise t-tests."""
    output_file.write("\n" + "=" * 60 + "\n")
    output_file.write(f"{pollutant_type} FILTER COUNT ANALYSIS\n")
    output_file.write("=" * 60 + "\n")

    filter_count_burns = ["burn4", "burn9", "burn2"]
    filter_count_labels = ["1 PAC", "2 PACs", "4 PACs"]

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        data_by_instrument,
        baseline_calculator,
        pollutant_type,
        filter_count_burns,
        exclude_instruments=[],
    )

    # Calculate mean data
    mean_data = calculate_mean_data(corrected_data, filter_count_burns)

    # Collect values for ANOVA
    groups = []
    for burn in filter_count_burns:
        if burn in mean_data:
            # For ANOVA, we need the individual instrument values
            burn_values = []
            for instrument in corrected_data:
                if burn in corrected_data[instrument]:
                    burn_values.append(corrected_data[instrument][burn]["decay"])
            groups.append(burn_values)

    if len(groups) >= 2:
        # perform ANOVA
        f_stat, p_value = f_oneway(*groups)

        output_file.write("\n1. Filter Count ANOVA\n")
        output_file.write("-" * 60 + "\n")

        for i, burn in enumerate(filter_count_burns):
            if burn in mean_data:
                output_file.write(
                    f"  {filter_count_labels[i]}: {mean_data[burn]['decay']:.4f} ± {mean_data[burn]['uncertainty']:.4f} h⁻¹\n"
                )

        output_file.write(f"\n  F-statistic: {f_stat:.4f}\n")
        output_file.write(f"  P-value: {p_value:.6f}\n")

        alpha = STATISTICAL_CONFIG["alpha"]
        is_significant = p_value < alpha

        if p_value < 0.001:
            sig_indicator = "***"
        elif p_value < 0.01:
            sig_indicator = "**"
        elif p_value < 0.05:
            sig_indicator = "*"
        else:
            sig_indicator = "ns"

        output_file.write(f"  Significant at α={alpha}: {is_significant} ({sig_indicator})\n")

        # Perform pairwise t-tests
        output_file.write("\n  Pairwise t-tests:\n")
        pairwise_results = []
        comparisons = [(0, 1), (0, 2), (1, 2)]  # 1 vs 2, 1 vs 4, 2 vs 4

        for idx_a, idx_b in comparisons:
            if idx_a < len(groups) and idx_b < len(groups):
                group_a = groups[idx_a]
                group_b = groups[idx_b]
                label_a = filter_count_labels[idx_a]
                label_b = filter_count_labels[idx_b]

                # Perform two-sample t-test
                t_stat, t_p_value = ttest_ind(group_a, group_b)

                t_sig = t_p_value < alpha
                if t_p_value < 0.001:
                    t_sig_indicator = "***"
                elif t_p_value < 0.01:
                    t_sig_indicator = "**"
                elif t_p_value < 0.05:
                    t_sig_indicator = "*"
                else:
                    t_sig_indicator = "ns"

                output_file.write(
                    f"    {label_a} vs {label_b}: t={t_stat:.4f}, p={t_p_value:.6f} {t_sig_indicator}\n"
                )

                pairwise_results.append(
                    {
                        "comparison": f"{label_a} vs {label_b}",
                        "t_stat": t_stat,
                        "p_value": t_p_value,
                        "significant": t_sig,
                        "sig_indicator": t_sig_indicator,
                    }
                )

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "significant": is_significant,
            "sig_indicator": sig_indicator,
            "groups": {
                filter_count_labels[i]: mean_data[burn]
                for i, burn in enumerate(filter_count_burns)
                if burn in mean_data
            },
            "pairwise_ttests": pairwise_results,
        }

    return {}


def perform_new_vs_used_analysis(
    data_by_instrument, baseline_calculator, pollutant_type, output_file
):
    """Perform new vs used filter analysis using both ANOVA and t-test."""
    output_file.write("\n2. New vs Used Filter Analysis\n")
    output_file.write("-" * 60 + "\n")

    new_used_burns = ["burn4", "burn3", "burn9", "burn10"]

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        data_by_instrument,
        baseline_calculator,
        pollutant_type,
        new_used_burns,
        exclude_instruments=[],
    )

    # Separate new vs used
    new_values = []
    used_values = []

    for instrument in corrected_data:
        if "burn4" in corrected_data[instrument]:
            new_values.append(corrected_data[instrument]["burn4"]["decay"])
        if "burn9" in corrected_data[instrument]:
            new_values.append(corrected_data[instrument]["burn9"]["decay"])
        if "burn3" in corrected_data[instrument]:
            used_values.append(corrected_data[instrument]["burn3"]["decay"])
        if "burn10" in corrected_data[instrument]:
            used_values.append(corrected_data[instrument]["burn10"]["decay"])

    if new_values and used_values:
        # ANOVA
        f_stat, p_value = f_oneway(new_values, used_values)

        # T-test
        t_stat, t_p_value = ttest_ind(new_values, used_values)

        output_file.write(
            f"  New filters: n={len(new_values)}, mean={np.mean(new_values):.4f} h⁻¹\n"
        )
        output_file.write(
            f"  Used filters: n={len(used_values)}, mean={np.mean(used_values):.4f} h⁻¹\n"
        )

        output_file.write(f"\n  ANOVA F-statistic: {f_stat:.4f}\n")
        output_file.write(f"  ANOVA P-value: {p_value:.6f}\n")

        output_file.write(f"\n  T-test statistic: {t_stat:.4f}\n")
        output_file.write(f"  T-test P-value: {t_p_value:.6f}\n")

        alpha = STATISTICAL_CONFIG["alpha"]
        is_significant = t_p_value < alpha

        if t_p_value < 0.001:
            sig_indicator = "***"
        elif t_p_value < 0.01:
            sig_indicator = "**"
        elif t_p_value < 0.05:
            sig_indicator = "*"
        else:
            sig_indicator = "ns"

        output_file.write(f"  Significant at α={alpha}: {is_significant} ({sig_indicator})\n")

        return {
            "f_stat": f_stat,
            "anova_p_value": p_value,
            "t_stat": t_stat,
            "p_value": t_p_value,
            "significant": is_significant,
            "sig_indicator": sig_indicator,
            "new_mean": np.mean(new_values),
            "used_mean": np.mean(used_values),
        }

    return {}


def perform_merv_comparison_analysis(
    data_by_instrument, baseline_calculator, pollutant_type, output_file
):
    """Perform MERV filter comparison using both ANOVA and t-test."""
    output_file.write("\n3. MERV Filter Comparison\n")
    output_file.write("-" * 60 + "\n")

    merv_burns = ["burn7", "burn8", "burn9", "burn10"]

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        data_by_instrument,
        baseline_calculator,
        pollutant_type,
        merv_burns,
        exclude_instruments=[],
    )

    # Separate MERV12A vs MERV13 (combining new and used)
    merv12a_values = []
    merv13_values = []

    for instrument in corrected_data:
        if "burn7" in corrected_data[instrument]:
            merv12a_values.append(corrected_data[instrument]["burn7"]["decay"])
        if "burn8" in corrected_data[instrument]:
            merv12a_values.append(corrected_data[instrument]["burn8"]["decay"])
        if "burn9" in corrected_data[instrument]:
            merv13_values.append(corrected_data[instrument]["burn9"]["decay"])
        if "burn10" in corrected_data[instrument]:
            merv13_values.append(corrected_data[instrument]["burn10"]["decay"])

    if merv12a_values and merv13_values:
        # ANOVA
        f_stat, p_value = f_oneway(merv12a_values, merv13_values)

        # T-test
        t_stat, t_p_value = ttest_ind(merv12a_values, merv13_values)

        output_file.write(
            f"  MERV12A: n={len(merv12a_values)}, mean={np.mean(merv12a_values):.4f} h⁻¹\n"
        )
        output_file.write(
            f"  MERV13: n={len(merv13_values)}, mean={np.mean(merv13_values):.4f} h⁻¹\n"
        )

        output_file.write(f"\n  ANOVA F-statistic: {f_stat:.4f}\n")
        output_file.write(f"  ANOVA P-value: {p_value:.6f}\n")

        output_file.write(f"\n  T-test statistic: {t_stat:.4f}\n")
        output_file.write(f"  T-test P-value: {t_p_value:.6f}\n")

        alpha = STATISTICAL_CONFIG["alpha"]
        is_significant = t_p_value < alpha

        if t_p_value < 0.001:
            sig_indicator = "***"
        elif t_p_value < 0.01:
            sig_indicator = "**"
        elif t_p_value < 0.05:
            sig_indicator = "*"
        else:
            sig_indicator = "ns"

        output_file.write(f"  Significant at α={alpha}: {is_significant} ({sig_indicator})\n")

        return {
            "f_stat": f_stat,
            "anova_p_value": p_value,
            "t_stat": t_stat,
            "p_value": t_p_value,
            "significant": is_significant,
            "sig_indicator": sig_indicator,
        }

    return {}


def perform_two_way_anova_filter_analysis(
    data_by_instrument, baseline_calculator, pollutant_type, output_file
):
    """
    Perform two-way ANOVA for filter analysis.
    Analyzes Filter Type (MERV12A vs MERV13) × Condition (New vs Used).
    """
    output_file.write("\n4. Two-Way ANOVA (Filter Type × Condition)\n")
    output_file.write("-" * 60 + "\n")
    # Define burns with their characteristics
    burn_characteristics = {
        "burn7": {"filter_type": "MERV12A", "condition": "New"},
        "burn8": {"filter_type": "MERV12A", "condition": "Used"},
        "burn9": {"filter_type": "MERV13", "condition": "New"},
        "burn10": {"filter_type": "MERV13", "condition": "Used"},
    }

    merv_burns = list(burn_characteristics.keys())

    # Create baseline-corrected data
    corrected_data = create_baseline_corrected_data(
        data_by_instrument,
        baseline_calculator,
        pollutant_type,
        merv_burns,
        exclude_instruments=[],
    )

    # Organize data for two-way ANOVA
    anova_data = []
    for burn, characteristics in burn_characteristics.items():
        for instrument in corrected_data:
            if burn in corrected_data[instrument]:
                anova_data.append(
                    {
                        "filter_type": characteristics["filter_type"],
                        "condition": characteristics["condition"],
                        "decay": corrected_data[instrument][burn]["decay"],
                        "instrument": instrument,
                    }
                )

    if not anova_data:
        output_file.write("  No data available for two-way ANOVA\n")
        return {}

    # Convert to DataFrame
    df_anova = pd.DataFrame(anova_data)

    # Calculate group means
    output_file.write("\n  Group Means:\n")
    group_means = (
        df_anova.groupby(["filter_type", "condition"])["decay"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for _, row in group_means.iterrows():
        output_file.write(
            f"    {row['filter_type']} {row['condition']}: "
            f"mean={row['mean']:.4f}, std={row['std']:.4f}, n={int(row['count'])}\n"
        )

    # Manual two-way ANOVA calculation
    # Grand mean
    grand_mean = df_anova["decay"].mean()
    n_total = len(df_anova)

    # Factor A (filter_type) sums of squares
    filter_means = df_anova.groupby("filter_type")["decay"].mean()
    filter_counts = df_anova.groupby("filter_type").size()
    ss_filter = sum(filter_counts * (filter_means - grand_mean) ** 2)
    df_filter = len(filter_means) - 1

    # Factor B (condition) sums of squares
    condition_means = df_anova.groupby("condition")["decay"].mean()
    condition_counts = df_anova.groupby("condition").size()
    ss_condition = sum(condition_counts * (condition_means - grand_mean) ** 2)
    df_condition = len(condition_means) - 1

    # Interaction sums of squares
    interaction_means = df_anova.groupby(["filter_type", "condition"])["decay"].mean()
    interaction_counts = df_anova.groupby(["filter_type", "condition"]).size()

    ss_interaction = 0
    for (ft, cond), mean in interaction_means.items():
        filter_mean = filter_means[ft]
        cond_mean = condition_means[cond]
        count = interaction_counts[(ft, cond)]
        ss_interaction += count * (mean - filter_mean - cond_mean + grand_mean) ** 2

    df_interaction = df_filter * df_condition

    # Error (residual) sums of squares
    ss_total = sum((df_anova["decay"] - grand_mean) ** 2)
    ss_error = ss_total - ss_filter - ss_condition - ss_interaction
    df_error = n_total - len(interaction_means)

    # Mean squares
    ms_filter = ss_filter / df_filter if df_filter > 0 else 0
    ms_condition = ss_condition / df_condition if df_condition > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    # F-statistics
    f_filter = ms_filter / ms_error if ms_error > 0 else 0
    f_condition = ms_condition / ms_error if ms_error > 0 else 0
    f_interaction = ms_interaction / ms_error if ms_error > 0 else 0

    # P-values
    from scipy.stats import f as f_dist

    p_filter = 1 - f_dist.cdf(f_filter, df_filter, df_error) if f_filter > 0 else 1
    p_condition = 1 - f_dist.cdf(f_condition, df_condition, df_error) if f_condition > 0 else 1
    p_interaction = (
        1 - f_dist.cdf(f_interaction, df_interaction, df_error) if f_interaction > 0 else 1
    )

    # Write results
    output_file.write("\n  Two-Way ANOVA Results:\n")
    output_file.write(
        f"    Filter Type:    F({df_filter},{df_error}) = {f_filter:.4f}, p = {p_filter:.6f}\n"
    )
    output_file.write(
        f"    Condition:      F({df_condition},{df_error}) = {f_condition:.4f}, p = {p_condition:.6f}\n"
    )
    output_file.write(
        f"    Interaction:    F({df_interaction},{df_error}) = {f_interaction:.4f}, p = {p_interaction:.6f}\n"
    )

    alpha = STATISTICAL_CONFIG["alpha"]

    def get_sig_indicator(p_val):
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        else:
            return "ns"

    output_file.write(f"\n  Significance at α={alpha}:\n")
    output_file.write(f"    Filter Type:    {p_filter < alpha} ({get_sig_indicator(p_filter)})\n")
    output_file.write(
        f"    Condition:      {p_condition < alpha} ({get_sig_indicator(p_condition)})\n"
    )
    output_file.write(
        f"    Interaction:    {p_interaction < alpha} ({get_sig_indicator(p_interaction)})\n"
    )

    return {
        "filter_type": {
            "f_stat": f_filter,
            "p_value": p_filter,
            "significant": p_filter < alpha,
            "sig_indicator": get_sig_indicator(p_filter),
        },
        "condition": {
            "f_stat": f_condition,
            "p_value": p_condition,
            "significant": p_condition < alpha,
            "sig_indicator": get_sig_indicator(p_condition),
        },
        "interaction": {
            "f_stat": f_interaction,
            "p_value": p_interaction,
            "significant": p_interaction < alpha,
            "sig_indicator": get_sig_indicator(p_interaction),
        },
    }


# ============================================================================
# STATISTICAL SUMMARY GENERATION
# ============================================================================


def generate_statistical_summary(summary_results, summary_file_path):
    """
    Generate a concise statistical summary file with key results.
    """
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY: WUI DECAY RATE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Significance level: α = {STATISTICAL_CONFIG['alpha']}\n")
        f.write("Significance indicators: *** p<0.001, ** p<0.01, * p<0.05, ns not significant\n")
        f.write("=" * 80 + "\n\n")

        # PM0.4 Results (Z-test)
        f.write("PM0.4 RESULTS (Z-Test)\n")
        f.write("-" * 80 + "\n")

        if "PM0.4" in summary_results:
            pm04_results = summary_results["PM0.4"]

            # Filter count
            if "filter_count" in pm04_results:
                f.write("\n1. Filter Count Comparison:\n")
                for result in pm04_results["filter_count"]:
                    f.write(f"   {result['label_a']} vs {result['label_b']}:\n")
                    f.write(
                        f"     Mean: {result['mu_a']:.4f}±{result['sd_a']:.4f} vs {result['mu_b']:.4f}±{result['sd_b']:.4f} h⁻¹\n"
                    )
                    f.write(
                        f"     Z={result['z_stat']:.4f}, p={result['p_value']:.6f} {result['sig_indicator']}\n"
                    )

            # New vs used
            if "new_vs_used" in pm04_results:
                f.write("\n2. New vs Used Filters:\n")
                for result in pm04_results["new_vs_used"]:
                    f.write(f"   {result['label_a']} vs {result['label_b']}:\n")
                    f.write(
                        f"     Mean: {result['mu_a']:.4f}±{result['sd_a']:.4f} vs {result['mu_b']:.4f}±{result['sd_b']:.4f} h⁻¹\n"
                    )
                    f.write(
                        f"     Z={result['z_stat']:.4f}, p={result['p_value']:.6f} {result['sig_indicator']}\n"
                    )

            # MERV comparison
            if "merv_comparison" in pm04_results:
                f.write("\n3. MERV Filter Comparison:\n")
                for result in pm04_results["merv_comparison"]:
                    f.write(f"   {result['label_a']} vs {result['label_b']}:\n")
                    f.write(
                        f"     Mean: {result['mu_a']:.4f}±{result['sd_a']:.4f} vs {result['mu_b']:.4f}±{result['sd_b']:.4f} h⁻¹\n"
                    )
                    f.write(
                        f"     Z={result['z_stat']:.4f}, p={result['p_value']:.6f} {result['sig_indicator']}\n"
                    )

        # PM1, PM2.5, PM10 Results (ANOVA and t-tests)
        for pm_size in ["PM1", "PM2.5", "PM10"]:
            f.write(f"\n\n{pm_size} RESULTS (ANOVA and t-tests)\n")
            f.write("-" * 80 + "\n")

            if pm_size in summary_results:
                pm_results = summary_results[pm_size]

                # Filter count
                if "filter_count" in pm_results:
                    result = pm_results["filter_count"]
                    f.write("\n1. Filter Count Comparison:\n")
                    if "groups" in result:
                        for label, data in result["groups"].items():
                            f.write(
                                f"   {label}: {data['decay']:.4f}±{data['uncertainty']:.4f} h⁻¹\n"
                            )
                    f.write(
                        f"   ANOVA: F={result.get('f_stat', 0):.4f}, p={result.get('p_value', 1):.6f} "
                        f"{result.get('sig_indicator', 'ns')}\n"
                    )

                    # Pairwise t-tests
                    if "pairwise_ttests" in result:
                        f.write("   Pairwise t-tests:\n")
                        for ttest in result["pairwise_ttests"]:
                            f.write(
                                f"     {ttest['comparison']}: t={ttest['t_stat']:.4f}, "
                                f"p={ttest['p_value']:.6f} {ttest['sig_indicator']}\n"
                            )

                # New vs used
                if "new_vs_used" in pm_results:
                    result = pm_results["new_vs_used"]
                    f.write("\n2. New vs Used Filters:\n")
                    f.write(f"   New mean: {result.get('new_mean', 0):.4f} h⁻¹\n")
                    f.write(f"   Used mean: {result.get('used_mean', 0):.4f} h⁻¹\n")
                    f.write(
                        f"   ANOVA: F={result.get('f_stat', 0):.4f}, p={result.get('anova_p_value', 1):.6f}\n"
                    )
                    f.write(
                        f"   T-test: t={result.get('t_stat', 0):.4f}, p={result.get('p_value', 1):.6f} "
                        f"{result.get('sig_indicator', 'ns')}\n"
                    )

                # MERV comparison
                if "merv_comparison" in pm_results:
                    result = pm_results["merv_comparison"]
                    f.write("\n3. MERV Filter Comparison:\n")
                    f.write(
                        f"   ANOVA: F={result.get('f_stat', 0):.4f}, p={result.get('anova_p_value', 1):.6f}\n"
                    )
                    f.write(
                        f"   T-test: t={result.get('t_stat', 0):.4f}, p={result.get('p_value', 1):.6f} "
                        f"{result.get('sig_indicator', 'ns')}\n"
                    )

                # Two-way ANOVA
                if "two_way_anova" in pm_results:
                    result = pm_results["two_way_anova"]
                    f.write("\n4. Two-Way ANOVA (Filter Type × Condition):\n")
                    if "filter_type" in result:
                        ft = result["filter_type"]
                        f.write(
                            f"   Filter Type: F={ft.get('f_stat', 0):.4f}, "
                            f"p={ft.get('p_value', 1):.6f} {ft.get('sig_indicator', 'ns')}\n"
                        )
                    if "condition" in result:
                        cond = result["condition"]
                        f.write(
                            f"   Condition: F={cond.get('f_stat', 0):.4f}, "
                            f"p={cond.get('p_value', 1):.6f} {cond.get('sig_indicator', 'ns')}\n"
                        )
                    if "interaction" in result:
                        inter = result["interaction"]
                        f.write(
                            f"   Interaction: F={inter.get('f_stat', 0):.4f}, "
                            f"p={inter.get('p_value', 1):.6f} {inter.get('sig_indicator', 'ns')}\n"
                        )

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================


def create_transposed_bar_chart(
    pm04_data,
    pm1_data,
    pm25_data,
    pm10_data,
    config,
    script_metadata,
    y_axis_label=None,
):
    """Create transposed bar chart with PM types on x-axis and conditions as legend.

    Parameters:
    -----------
    y_axis_label : str, optional
        Custom y-axis label. If None, uses default label.
    """

    # Prepare data based on selected PM types
    pm_data_dict = {
        "PM0.4": pm04_data,
        "PM1": pm1_data,
        "PM2.5": pm25_data,
        "PM10": pm10_data,
    }

    # Filter to only include selected PM types and ensure correct order
    selected_pm_types = []
    for pm_type in PM_SIZE_ORDER:  # Use the global ordering
        if pm_type in config.get("pm_types", []):
            selected_pm_types.append(pm_type)

    # Create figure
    p = figure(
        x_range=selected_pm_types,
        height=500,
        width=900,
        # title=config["title"], # commented out to as to not show for paper figures
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        background_fill_color="white",
        border_fill_color="white",
    )

    # Prepare data for plotting
    source_data = {
        "pm_type": [],
        "condition": [],
        "decay": [],
        "upper": [],
        "lower": [],
        "is_used": [],
    }

    detailed_info = []

    # Collect all data
    for pm_type in selected_pm_types:
        pm_data = pm_data_dict.get(pm_type, {})

        for burn in config["burns"]:
            if burn in pm_data:
                label = config["labels"][burn]
                decay = pm_data[burn]["decay"]
                uncertainty = pm_data[burn]["uncertainty"]

                # Get contributing instruments for this burn/pm_type
                instruments_list = []
                if pm_type == "PM0.4":
                    instruments_list = ["SMPS"]
                elif "contributing_instruments" in pm_data[burn]:
                    instruments_list = pm_data[burn]["contributing_instruments"]

                instruments_str = ", ".join(instruments_list) if instruments_list else ""

                # Determine if used filter
                is_used = burn in ["burn3", "burn8", "burn10"]

                source_data["pm_type"].append(pm_type)
                source_data["condition"].append(label)
                source_data["decay"].append(decay)
                source_data["upper"].append(decay + uncertainty)
                source_data["lower"].append(decay - uncertainty)
                source_data["is_used"].append(is_used)

                detailed_info.append(
                    f"{label} ({pm_type}): {decay:.3f} ± {uncertainty:.3f} h⁻¹ ({instruments_str})"
                )

    # Get unique conditions
    unique_conditions = []
    seen = set()
    for cond in source_data["condition"]:
        if cond not in seen:
            unique_conditions.append(cond)
            seen.add(cond)

    # Calculate bar positioning
    n_conditions = len(unique_conditions)
    bar_width = 0.8 / n_conditions
    offset = -0.4 + bar_width / 2

    # Define colors to use from the predefined colors dictionary
    chart_colors = [
        colors["AeroTrakB"],
        colors["PurpleAirK"],
        colors["SMPS"],
    ]  # #003f5c, #665191, #d45087

    # Check if conditions already include New/Used in labels (for MERV chart)
    conditions_include_new_used = all("New" in c or "Used" in c for c in unique_conditions)

    # If conditions include filter types (MERV12A, MERV13), set up colors by filter type
    filter_type_colors = {}
    if conditions_include_new_used:
        # Extract unique filter types
        filter_types = []
        for condition in unique_conditions:
            filter_type = condition.replace(" New", "").replace(" Used", "")
            if filter_type not in filter_types:
                filter_types.append(filter_type)

        # Assign colors to filter types
        for i, filter_type in enumerate(filter_types):
            filter_type_colors[filter_type] = chart_colors[i % len(chart_colors)]

    # Add bars for each condition
    for i, condition in enumerate(unique_conditions):
        # Filter data for this condition
        condition_data = {
            "pm_type": [],
            "decay": [],
            "upper": [],
            "lower": [],
            "is_used": [],
        }

        for j in range(len(source_data["condition"])):
            if source_data["condition"][j] == condition:
                condition_data["pm_type"].append(source_data["pm_type"][j])
                condition_data["decay"].append(source_data["decay"][j])
                condition_data["upper"].append(source_data["upper"][j])
                condition_data["lower"].append(source_data["lower"][j])
                condition_data["is_used"].append(source_data["is_used"][j])

        if condition_data["pm_type"]:
            pos = offset + i * bar_width

            # If conditions include New/Used info, color by filter type
            if conditions_include_new_used:
                # Extract filter type from condition
                filter_type = condition.replace(" New", "").replace(" Used", "")
                color = filter_type_colors[filter_type]

                # Check if this is a "Used" condition
                is_used_condition = "Used" in condition

                # Create data source
                source = ColumnDataSource(data=condition_data)

                # Create vbar arguments
                vbar_args = {
                    "x": dodge("pm_type", pos, range=p.x_range),
                    "top": "decay",
                    "width": bar_width * 0.8,
                    "source": source,
                    "color": color,
                    "legend_label": condition,
                }

                # Add hatching for used filters
                if is_used_condition:
                    vbar_args["hatch_pattern"] = "right_diagonal_line"
                    vbar_args["hatch_color"] = "black"

                p.vbar(**vbar_args)

                # Add error bars
                p.segment(
                    x0=dodge("pm_type", pos, range=p.x_range),
                    y0="lower",
                    x1=dodge("pm_type", pos, range=p.x_range),
                    y1="upper",
                    source=source,
                    line_color="black",
                    line_width=1.5,
                )
            else:
                # Original logic for non-MERV charts
                color = chart_colors[i % len(chart_colors)]

                # Separate new and used data
                new_data = {"pm_type": [], "decay": [], "upper": [], "lower": []}
                used_data = {"pm_type": [], "decay": [], "upper": [], "lower": []}

                for k in range(len(condition_data["pm_type"])):
                    if condition_data["is_used"][k]:
                        used_data["pm_type"].append(condition_data["pm_type"][k])
                        used_data["decay"].append(condition_data["decay"][k])
                        used_data["upper"].append(condition_data["upper"][k])
                        used_data["lower"].append(condition_data["lower"][k])
                    else:
                        new_data["pm_type"].append(condition_data["pm_type"][k])
                        new_data["decay"].append(condition_data["decay"][k])
                        new_data["upper"].append(condition_data["upper"][k])
                        new_data["lower"].append(condition_data["lower"][k])

                # Add bars for new filters (solid color, no hatching)
                if new_data["pm_type"]:
                    new_source = ColumnDataSource(data=new_data)
                    p.vbar(
                        x=dodge("pm_type", pos, range=p.x_range),
                        top="decay",
                        width=bar_width * 0.8,
                        source=new_source,
                        color=color,
                        legend_label=condition,
                    )

                    # Add error bars
                    p.segment(
                        x0=dodge("pm_type", pos, range=p.x_range),
                        y0="lower",
                        x1=dodge("pm_type", pos, range=p.x_range),
                        y1="upper",
                        source=new_source,
                        line_color="black",
                        line_width=1.5,
                    )

                # Add bars for used filters (same color with hatching)
                if used_data["pm_type"]:
                    used_source = ColumnDataSource(data=used_data)

                    vbar_args = {
                        "x": dodge("pm_type", pos, range=p.x_range),
                        "top": "decay",
                        "width": bar_width * 0.8,
                        "source": used_source,
                        "color": color,
                        "hatch_pattern": "right_diagonal_line",
                        "hatch_color": "black",
                    }

                    # Only add legend label if this is the first occurrence of this condition
                    if not new_data["pm_type"]:
                        vbar_args["legend_label"] = condition

                    p.vbar(**vbar_args)

                    # Add error bars
                    p.segment(
                        x0=dodge("pm_type", pos, range=p.x_range),
                        y0="lower",
                        x1=dodge("pm_type", pos, range=p.x_range),
                        y1="upper",
                        source=used_source,
                        line_color="black",
                        line_width=1.5,
                    )

    # Apply text formatting to the plot
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "lightgray"
    p.ygrid.grid_line_alpha = 0.6

    # Set y-axis range based on PM10, chart type, and normalization
    # Check if this is a total decay rate chart (not normalized by CRboxes)
    is_total_decay_rate = not config.get("normalize_by_crboxes", True)

    if is_total_decay_rate:
        # Total decay rate charts need larger ranges
        if "PM10" in selected_pm_types:
            p.y_range = Range1d(0, 12)
        else:
            p.y_range = Range1d(0, 8)
    elif "PM10" in selected_pm_types:
        # Filter count chart needs 0-4 range when PM10 is included
        if "filter_count" in config.get("filename", ""):
            p.y_range = Range1d(0, 4.5)
        elif "merv" in config.get("filename", ""):
            p.y_range = Range1d(0, 4)
        else:
            p.y_range = Range1d(0, 3.5)
    elif "filter_condition_combined" in config.get("filename", ""):
        p.y_range = Range1d(0, 2.7)
    else:
        p.y_range = Range1d(0, 2.5)

    # Set y-axis label (use custom label if provided, otherwise use default)
    if y_axis_label is None:
        p.yaxis.axis_label = "Baseline-Corrected Decay Rate per Portable Air Cleaner (h⁻¹)"
    else:
        p.yaxis.axis_label = y_axis_label
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    p.xaxis.major_label_orientation = "horizontal"

    # Apply text formatting to plot elements
    p.title.text_font_size = TEXT_CONFIG["title_font_size"]
    p.title.text_font_style = TEXT_CONFIG["plot_font_style"]
    p.xaxis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
    p.xaxis.axis_label_text_font_style = TEXT_CONFIG["plot_font_style"]
    p.yaxis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
    p.yaxis.axis_label_text_font_style = TEXT_CONFIG["plot_font_style"]
    p.xaxis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
    p.xaxis.major_label_text_font_style = TEXT_CONFIG["font_style"]
    p.yaxis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
    p.yaxis.major_label_text_font_style = TEXT_CONFIG["font_style"]

    # Position the legend with text formatting
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.legend.background_fill_alpha = 0.6
    p.legend.label_text_font_size = TEXT_CONFIG["legend_font_size"]
    p.legend.label_text_font_style = TEXT_CONFIG["font_style"]
    p.legend.spacing = 1
    p.legend.padding = 5
    p.legend.margin = 0

    # Create enhanced metadata div with detailed values and text formatting
    detailed_values_text = "<br>".join(detailed_info)
    div_text = f"""<div style="font-size: {TEXT_CONFIG["font_size"]}; font-weight: {TEXT_CONFIG["html_font_weight"]}; font-style: {TEXT_CONFIG["font_style"]};">
    <b>Detailed Values:</b><br>{detailed_values_text}<br><br>
    Baseline-corrected decay rates (baseline subtracted from all measurements)<br>
    Solid bars: New filters, Hatched bars: Used filters<br><br>
    {script_metadata}
    </div>"""
    text_div = Div(text=div_text, width=900)

    return column(p, text_div)


# Main processing function
def process_data_for_charts(pollutant_types_dict, pollutant_suffix):
    """Process data for charts."""
    print(f"\nProcessing {pollutant_suffix} data...")

    # Read data for all instruments
    data_by_instrument = {}
    for instrument in instruments:
        if instrument in pollutant_types_dict:
            instrument_data = read_instrument_data(instrument, pollutant_types_dict)
            if instrument_data:
                data_by_instrument[instrument] = instrument_data

    return data_by_instrument


# ============================================================================
# MAIN PROCESSING SECTION
# ============================================================================

print("Processing PM0.4, PM1, PM2.5, and PM10 data...")

# Process PM0.4 data (SMPS only)
pm04_data_raw = process_data_for_charts(pollutant_types_pm04, "PM0.4")

# Process PM1 data (excluding SMPS)
pm1_data_raw = process_data_for_charts(pollutant_types_pm1, "PM1")

# Process PM2.5 data
pm25_data_raw = process_data_for_charts(pollutant_types_pm25, "PM2.5")

# Process PM10 data
pm10_data_raw = process_data_for_charts(pollutant_types_pm10, "PM10")

# Create centralized baseline calculator
baseline_calculator = BaselineCalculator(pm04_data_raw, pm1_data_raw, pm25_data_raw, pm10_data_raw)

# Dictionary to store key statistical results for summary
summary_results = {"PM0.4": {}, "PM1": {}, "PM2.5": {}, "PM10": {}}

# ============================================================================
# PERFORM STATISTICAL ANALYSES AND WRITE TO FILE
# ============================================================================

print("#" * 80)
print("STATISTICAL ANALYSES")
print("#" * 80)
print("Writing statistical analyses to file...")

# Create output file paths
stats_file_path = os.path.join(STATS_OUTPUT_PATH, "decay_statistical_analyses.txt")
summary_file_path = os.path.join(STATS_OUTPUT_PATH, "decay_statistical_summary.txt")

# Open the detailed statistical analyses file
with open(stats_file_path, "w", encoding="utf-8") as stats_file:
    stats_file.write("=" * 80 + "\n")
    stats_file.write("STATISTICAL ANALYSES: WUI DECAY RATE\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Configuration: α = {STATISTICAL_CONFIG['alpha']}\n")
    stats_file.write("=" * 80 + "\n")

    # PM0.4 Statistical Analyses (Z-test)
    stats_file.write("\n\nPM0.4 Statistical Analyses (Z-Test Method):\n")
    pm04_filter_count = perform_pm04_filter_count_ztest(
        pm04_data_raw, baseline_calculator, stats_file
    )
    pm04_new_vs_used = perform_pm04_new_vs_used_ztest(
        pm04_data_raw, baseline_calculator, stats_file
    )
    pm04_merv = perform_pm04_merv_comparison_ztest(pm04_data_raw, baseline_calculator, stats_file)

    summary_results["PM0.4"] = {**pm04_filter_count, **pm04_new_vs_used, **pm04_merv}

    # PM1 Statistical Analyses (ANOVA + T-tests)
    stats_file.write("\n\nPM1 Statistical Analyses:\n")
    pm1_filter_count = perform_filter_count_analysis(
        pm1_data_raw, baseline_calculator, "PM1", stats_file
    )
    pm1_new_vs_used = perform_new_vs_used_analysis(
        pm1_data_raw, baseline_calculator, "PM1", stats_file
    )
    pm1_merv = perform_merv_comparison_analysis(
        pm1_data_raw, baseline_calculator, "PM1", stats_file
    )
    pm1_two_way = perform_two_way_anova_filter_analysis(
        pm1_data_raw, baseline_calculator, "PM1", stats_file
    )

    summary_results["PM1"] = {
        "filter_count": pm1_filter_count,
        "new_vs_used": pm1_new_vs_used,
        "merv_comparison": pm1_merv,
        "two_way_anova": pm1_two_way,
    }

    # PM2.5 Statistical Analyses (ANOVA + T-tests)
    stats_file.write("\n\nPM2.5 Statistical Analyses:\n")
    pm25_filter_count = perform_filter_count_analysis(
        pm25_data_raw, baseline_calculator, "PM2.5", stats_file
    )
    pm25_new_vs_used = perform_new_vs_used_analysis(
        pm25_data_raw, baseline_calculator, "PM2.5", stats_file
    )
    pm25_merv = perform_merv_comparison_analysis(
        pm25_data_raw, baseline_calculator, "PM2.5", stats_file
    )
    pm25_two_way = perform_two_way_anova_filter_analysis(
        pm25_data_raw, baseline_calculator, "PM2.5", stats_file
    )

    summary_results["PM2.5"] = {
        "filter_count": pm25_filter_count,
        "new_vs_used": pm25_new_vs_used,
        "merv_comparison": pm25_merv,
        "two_way_anova": pm25_two_way,
    }

    # PM10 Statistical Analyses (ANOVA + T-tests)
    stats_file.write("\n\nPM10 Statistical Analyses:\n")
    pm10_filter_count = perform_filter_count_analysis(
        pm10_data_raw, baseline_calculator, "PM10", stats_file
    )
    pm10_new_vs_used = perform_new_vs_used_analysis(
        pm10_data_raw, baseline_calculator, "PM10", stats_file
    )
    pm10_merv = perform_merv_comparison_analysis(
        pm10_data_raw, baseline_calculator, "PM10", stats_file
    )
    pm10_two_way = perform_two_way_anova_filter_analysis(
        pm10_data_raw, baseline_calculator, "PM10", stats_file
    )

    summary_results["PM10"] = {
        "filter_count": pm10_filter_count,
        "new_vs_used": pm10_new_vs_used,
        "merv_comparison": pm10_merv,
        "two_way_anova": pm10_two_way,
    }

    stats_file.write("\n\n" + "=" * 80 + "\n")
    stats_file.write("END OF STATISTICAL ANALYSES\n")
    stats_file.write("=" * 80 + "\n")

print(f"Statistical analyses written to: {stats_file_path}")

# Generate statistical summary
generate_statistical_summary(summary_results, summary_file_path)
print(f"Statistical summary written to: {summary_file_path}")

# Get script metadata
metadata = get_script_metadata()

# Define the chart configurations with PM type selection
chart_configs = [
    {
        "burns": ["burn4", "burn9", "burn2"],
        "labels": {
            "burn4": "1 Air Cleaner",
            "burn9": "2 Air Cleaners",
            "burn2": "4 Air Cleaners",
        },
        "title": "Baseline-Corrected Decay Rate by Number of Portable Air Cleaners",
        "filename": "decay_bar_chart_filter_count_combined.html",
        "exclude_instruments": ["PurpleAirK"],
        "exclude_baseline_instruments": ["AeroTrakB", "QuantAQB"],
        "pm_types": [
            "PM0.4",
            "PM2.5",
        ],  # all available PM types 'PM0.4', 'PM1', 'PM2.5', 'PM10'
    },
    {
        "burns": ["burn4", "burn3", "burn9", "burn10"],
        "labels": {
            "burn4": "1 New Air Cleaner",
            "burn3": "1 Used Air Cleaner",
            "burn9": "2 New Air Cleaners",
            "burn10": "2 Used Air Cleaners",
        },
        "title": "Baseline-Corrected Decay Rate: New versus Used Portable Air Cleaners",
        "filename": "decay_bar_chart_filter_condition_combined.html",
        "exclude_baseline_instruments": ["AeroTrakB", "QuantAQB"],
        "use_matching_instruments": True,
        "pm_types": [
            "PM0.4",
            "PM2.5",
        ],  # all available PM types 'PM0.4', 'PM1', 'PM2.5', 'PM10'
    },
    {
        "burns": ["burn7", "burn8", "burn9", "burn10"],
        "labels": {
            "burn7": "MERV 12A New",
            "burn8": "MERV 12A Used",
            "burn9": "MERV 13 New",
            "burn10": "MERV 13 Used",
        },
        "title": "Baseline-Corrected Decay Rate: MERV 12A versus MERV 13 Filters",
        "filename": "decay_bar_chart_merv_comparison_combined.html",
        "exclude_baseline_instruments": ["AeroTrakB", "QuantAQB"],
        "use_matching_instruments": True,
        "pm_types": [
            "PM0.4",
            "PM2.5",
        ],  # all available PM types 'PM0.4', 'PM1', 'PM2.5', 'PM10'
    },
    {
        "burns": ["burn4", "burn9", "burn2"],
        "labels": {
            "burn4": "1 Air Cleaner",
            "burn9": "2 Air Cleaners",
            "burn2": "4 Air Cleaners",
        },
        "title": "Baseline-Corrected Total Decay Rate by Number of Portable Air Cleaners",
        "filename": "decay_bar_chart_filter_count_total.html",
        "exclude_instruments": ["PurpleAirK"],
        "exclude_baseline_instruments": ["AeroTrakB", "QuantAQB"],
        "pm_types": [
            "PM0.4",
            "PM2.5",
        ],  # all available PM types 'PM0.4', 'PM1', 'PM2.5', 'PM10'
        "normalize_by_crboxes": False,  # Show total decay rate, not per-PAC
        "y_axis_label": "Baseline-Corrected (Total) Decay Rate (h⁻¹)",
    },
]

# Generate charts
print("\nGenerating charts...")
for config in chart_configs:
    print(f"  Creating {config['filename']}...")

    # Process each PM size
    pm04_processed = {}
    pm1_processed = {}
    pm25_processed = {}
    pm10_processed = {}

    # Get normalize_by_crboxes flag (default True for existing charts)
    normalize_by_crboxes = config.get("normalize_by_crboxes", True)

    # Get list of instruments to use
    if config.get("use_matching_instruments", False):
        matching_instruments = get_matching_instruments_for_burns(pm1_data_raw, config["burns"])
        exclude_instruments = config.get("exclude_instruments", [])
        exclude_instruments = [
            i for i in instruments if i not in matching_instruments and i not in exclude_instruments
        ]
        config["exclude_instruments"] = exclude_instruments

    # Process PM0.4 if selected (always use SMPS data regardless of matching instruments)
    if "PM0.4" in config.get("pm_types", []):
        # For PM0.4, don't exclude SMPS even if use_matching_instruments is True
        pm04_exclude = [i for i in config.get("exclude_instruments", []) if i != "SMPS"]
        pm04_corrected = create_baseline_corrected_data(
            pm04_data_raw,
            baseline_calculator,
            "PM0.4",
            config["burns"],
            exclude_instruments=pm04_exclude,
            normalize_by_crboxes=normalize_by_crboxes,
        )
        if "SMPS" in pm04_corrected:
            for burn in config["burns"]:
                if burn in pm04_corrected["SMPS"]:
                    pm04_processed[burn] = pm04_corrected["SMPS"][burn]

    # Process PM1 if selected
    if "PM1" in config.get("pm_types", []):
        pm1_corrected = create_baseline_corrected_data(
            pm1_data_raw,
            baseline_calculator,
            "PM1",
            config["burns"],
            exclude_instruments=config.get("exclude_instruments", []),
            normalize_by_crboxes=normalize_by_crboxes,
        )
        pm1_processed = calculate_mean_data(pm1_corrected, config["burns"])

    # Process PM2.5 if selected
    if "PM2.5" in config.get("pm_types", []):
        pm25_corrected = create_baseline_corrected_data(
            pm25_data_raw,
            baseline_calculator,
            "PM2.5",
            config["burns"],
            exclude_instruments=config.get("exclude_instruments", []),
            normalize_by_crboxes=normalize_by_crboxes,
        )
        pm25_processed = calculate_mean_data(pm25_corrected, config["burns"])

    # Process PM10 if selected
    if "PM10" in config.get("pm_types", []):
        pm10_corrected = create_baseline_corrected_data(
            pm10_data_raw,
            baseline_calculator,
            "PM10",
            config["burns"],
            exclude_instruments=config.get("exclude_instruments", []),
            normalize_by_crboxes=normalize_by_crboxes,
        )
        pm10_processed = calculate_mean_data(pm10_corrected, config["burns"])

    # Create chart with custom y-axis label if specified
    y_axis_label = config.get("y_axis_label", None)
    chart = create_transposed_bar_chart(
        pm04_processed,
        pm1_processed,
        pm25_processed,
        pm10_processed,
        config,
        metadata,
        y_axis_label=y_axis_label,
    )

    # Save chart
    output_path = os.path.join(BASE_PATH, config["filename"])
    output_file(output_path)
    save(chart)
    print(f"    Saved to: {output_path}")

print("\nProcessing complete!")
print(f"Statistical analyses: {stats_file_path}")
print(f"Statistical summary: {summary_file_path}")
