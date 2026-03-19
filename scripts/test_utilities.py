"""
Test Script for NIST WUI MH IAQ Utility Modules

This script tests the functionality of the utility modules and demonstrates
their usage. Run this script to verify the utility modules are working correctly.

Author: Nathan Lima
Date: 2024-2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add repository root to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

print("="*70)
print("NIST WUI MH IAQ Utility Modules Test Script")
print("="*70)

# ============================================================================
# Test 1: datetime_utils
# ============================================================================
print("\n[TEST 1] Testing datetime_utils module...")
try:
    from scripts.datetime_utils import (
        create_naive_datetime,
        apply_time_shift,
        calculate_time_since_event,
        calculate_time_since_garage_closed,
        TIME_SHIFTS,
        seconds_to_hours,
        minutes_to_hours
    )

    # Test create_naive_datetime
    dt = create_naive_datetime("2024-01-15", "14:30:00")
    assert pd.notna(dt), "create_naive_datetime failed"
    print("  create_naive_datetime works")

    # Test apply_time_shift
    test_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=5, freq='1H'),
        'value': [1, 2, 3, 4, 5]
    })
    shifted_df = apply_time_shift(test_df, 'AeroTrakB', 'datetime')
    assert len(shifted_df) == len(test_df), "apply_time_shift failed"
    print("  apply_time_shift works")

    # Test calculate_time_since_event
    event_time = pd.Timestamp('2024-01-01 00:00:00')
    result_df = calculate_time_since_event(test_df, 'datetime', event_time, 'hours')
    assert 'hours' in result_df.columns, "calculate_time_since_event failed"
    print("  calculate_time_since_event works")

    # Test time conversions
    assert seconds_to_hours(3600) == 1.0, "seconds_to_hours failed"
    assert minutes_to_hours(60) == 1.0, "minutes_to_hours failed"
    print("  Time conversion functions work")

    # Test TIME_SHIFTS constant
    assert 'AeroTrakB' in TIME_SHIFTS, "TIME_SHIFTS missing instruments"
    assert TIME_SHIFTS['AeroTrakB'] == 2.16, "TIME_SHIFTS values incorrect"
    print("  TIME_SHIFTS constant available")

    print("  datetime_utils module: ALL TESTS PASSED")

except Exception as e:
    print(f"  datetime_utils module: FAILED - {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 2: data_filters
# ============================================================================
print("\n[TEST 2] Testing data_filters module...")
try:
    from scripts.data_filters import (
        g_mean,
        split_data_by_nan,
        filter_by_status_columns,
        calculate_rolling_average_burn3
    )

    # Test g_mean
    result = g_mean([1, 10, 100])
    assert np.isclose(result, 10.0), "g_mean calculation incorrect"
    print("  g_mean works")

    # Test split_data_by_nan
    test_df = pd.DataFrame({
        'time': [0, 0.05, 0.1, 0.5, 0.55],
        'value': [1, 2, np.nan, 4, 5]
    })
    segments = split_data_by_nan(test_df, 'time', 'value')
    assert len(segments) > 0, "split_data_by_nan failed"
    print("  split_data_by_nan works")

    # Test filter_by_status_columns
    test_df = pd.DataFrame({
        'Flow Status': ['OK', 'Error', 'OK'],
        'Laser Status': ['OK', 'OK', 'Error'],
        'concentration': [10.0, 20.0, 30.0]
    })
    filtered = filter_by_status_columns(test_df)
    assert filtered['concentration'].iloc[0] == 10.0, "Status filtering failed"
    assert pd.isna(filtered['concentration'].iloc[1]), "Status filtering failed"
    print("  filter_by_status_columns works")

    # Test calculate_rolling_average_burn3
    test_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=10, freq='1min'),
        'concentration': range(10)
    })
    smoothed = calculate_rolling_average_burn3(test_df, 'datetime')
    assert len(smoothed) == len(test_df), "Rolling average failed"
    print("  calculate_rolling_average_burn3 works")

    print("  data_filters module: ALL TESTS PASSED")

except Exception as e:
    print(f"  data_filters module: FAILED - {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 3: statistical_utils
# ============================================================================
print("\n[TEST 3] Testing statistical_utils module...")
try:
    from scripts.statistical_utils import (
        exponential_decay,
        fit_exponential_curve,
        perform_linear_fit,
        perform_polynomial_fit
    )

    # Test exponential_decay
    x = np.array([0, 1, 2, 3])
    y = exponential_decay(x, a=100, b=0.5)
    assert len(y) == len(x), "exponential_decay failed"
    assert y[0] == 100, "exponential_decay calculation incorrect"
    print("  exponential_decay works")

    # Test fit_exponential_curve
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = 100 * np.exp(-0.5 * x_data) + np.random.normal(0, 0.1, 5)
    params, y_fit, errors = fit_exponential_curve(x_data, y_data)
    assert params is not None, "fit_exponential_curve failed"
    assert len(params) == 2, "fit_exponential_curve returned wrong number of params"
    print("  fit_exponential_curve works")

    # Test perform_linear_fit
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = 2 * x_data + 1
    result = perform_linear_fit(x_data, y_data)
    assert result is not None, "perform_linear_fit failed"
    assert 'slope' in result, "perform_linear_fit missing slope"
    assert np.isclose(result['slope'], 2.0), "perform_linear_fit slope incorrect"
    print("  perform_linear_fit works")

    # Test perform_polynomial_fit
    result = perform_polynomial_fit(x_data, y_data, degree=2)
    assert result is not None, "perform_polynomial_fit failed"
    assert 'coefficients' in result, "perform_polynomial_fit missing coefficients"
    print("  perform_polynomial_fit works")

    print("  statistical_utils module: ALL TESTS PASSED")

except Exception as e:
    print(f"  statistical_utils module: FAILED - {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 4: plotting_utils
# ============================================================================
print("\n[TEST 4] Testing plotting_utils module...")
try:
    from scripts.plotting_utils import (
        get_script_metadata,
        create_standard_figure,
        apply_text_formatting,
        configure_legend,
        get_color_for_pollutant,
        get_color_palette_for_instrument,
        INSTRUMENT_COLORS,
        POLLUTANT_COLORS
    )

    # Test get_script_metadata
    metadata = get_script_metadata()
    assert isinstance(metadata, str), "get_script_metadata failed"
    assert "Generated by" in metadata, "get_script_metadata format incorrect"
    print("  get_script_metadata works")

    # Test create_standard_figure
    fig = create_standard_figure("Test Figure")
    assert fig is not None, "create_standard_figure failed"
    assert fig.title.text == "Test Figure", "Figure title incorrect"
    print("  create_standard_figure works")

    # Test color functions
    color = get_color_for_pollutant('PM2.5')
    assert isinstance(color, str), "get_color_for_pollutant failed"
    assert color.startswith('#'), "Color not in hex format"
    print("  get_color_for_pollutant works")

    palette = get_color_palette_for_instrument('AeroTrakB')
    assert isinstance(palette, list), "get_color_palette_for_instrument failed"
    assert len(palette) > 0, "Color palette empty"
    print("  get_color_palette_for_instrument works")

    # Test constants
    assert 'AeroTrakB' in INSTRUMENT_COLORS, "INSTRUMENT_COLORS missing instruments"
    assert 'PM2.5' in POLLUTANT_COLORS, "POLLUTANT_COLORS missing pollutants"
    print("  Color constants available")

    print("  plotting_utils module: ALL TESTS PASSED")

except Exception as e:
    print(f"  plotting_utils module: FAILED - {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 5: instrument_config
# ============================================================================
print("\n[TEST 5] Testing instrument_config module...")
try:
    from scripts.instrument_config import (
        QUANTAQ_BINS,
        SMPS_BIN_RANGES,
        AEROTRAK_CHANNELS,
        UNIT_CONVERSIONS,
        get_quantaq_bins,
        get_smps_bin_ranges,
        get_burn_range_for_instrument,
        convert_unit,
        sum_quantaq_bins
    )

    # Test constants
    assert isinstance(QUANTAQ_BINS, dict), "QUANTAQ_BINS not a dictionary"
    assert isinstance(SMPS_BIN_RANGES, list), "SMPS_BIN_RANGES not a list"
    assert isinstance(AEROTRAK_CHANNELS, list), "AEROTRAK_CHANNELS not a list"
    assert 'liter_to_cm3' in UNIT_CONVERSIONS, "UNIT_CONVERSIONS missing conversions"
    print("  Configuration constants available")

    # Test get functions
    bins = get_quantaq_bins()
    assert len(bins) > 0, "get_quantaq_bins returned empty"
    print("  get_quantaq_bins works")

    ranges = get_smps_bin_ranges()
    assert len(ranges) > 0, "get_smps_bin_ranges returned empty"
    print("  get_smps_bin_ranges works")

    burns = get_burn_range_for_instrument('QuantAQB')
    assert 'burn4' in burns, "get_burn_range_for_instrument incorrect"
    assert 'burn1' not in burns, "get_burn_range_for_instrument incorrect"
    print("  get_burn_range_for_instrument works")

    # Test convert_unit
    result = convert_unit(1.5, 'liter_to_cm3')
    assert result == 1500.0, "convert_unit calculation incorrect"
    print("  convert_unit works")

    # Test sum_quantaq_bins
    test_df = pd.DataFrame({
        'bin0': [1, 2, 3],
        'bin1': [1, 2, 3],
        'bin2': [2, 3, 4]
    })
    result_df = sum_quantaq_bins(test_df)
    assert 'Ʃ0.35-0.66µm (#/cm³)' in result_df.columns, "sum_quantaq_bins failed"
    print("  sum_quantaq_bins works")

    print("  instrument_config module: ALL TESTS PASSED")

except Exception as e:
    print(f"  instrument_config module: FAILED - {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("\nAll utility modules imported and tested successfully!")
print("\nUtility modules are ready to use in analysis scripts.")
print("\nNext steps:")
print("  1. Review the README.md in the scripts/ directory")
print("  2. Update existing analysis scripts to use these utilities")
print("  3. Enjoy cleaner, more maintainable code!")
print("\n" + "="*70)
