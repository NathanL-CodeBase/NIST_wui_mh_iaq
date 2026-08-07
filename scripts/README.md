# NIST WUI MH IAQ Utility Scripts

This directory contains reusable utility modules for the NIST Wildland-Urban Interface (WUI) Manufactured Housing Indoor Air Quality (IAQ) analysis project.

## Overview

These utility modules extract commonly-used functions from analysis scripts to:
- **Reduce code duplication** across the repository
- **Improve maintainability** by centralizing common functionality
- **Ensure consistency** in data processing and visualization
- **Simplify script development** with well-documented, tested functions

## Utility Modules

### 1. `datetime_utils.py`
DateTime handling and time synchronization functions.

**Key Functions:**
- `create_naive_datetime(date_str, time_str)` - Create timezone-naive datetime from strings
- `apply_time_shift(df, instrument, datetime_column)` - Apply instrument-specific time corrections
- `calculate_time_since_event(df, datetime_column, event_time)` - Calculate elapsed time since reference event
- `calculate_time_since_garage_closed(df, datetime_column, garage_closed_time)` - Convenience wrapper for garage closed event
- `fix_smps_datetime(smps_data, debug=False)` - Fix SMPS datetime formatting issues
- `seconds_to_hours(seconds)`, `minutes_to_hours(minutes)` - Time unit conversions

**Constants:**
- `TIME_SHIFTS` - Dictionary of instrument-specific time corrections (in minutes)

**Example Usage:**
```python
from scripts.datetime_utils import apply_time_shift, TIME_SHIFTS

# Apply time correction to AeroTrakB data
df = apply_time_shift(df, 'AeroTrakB', 'Date and Time')

# Calculate time since garage closed
df = calculate_time_since_garage_closed(df, 'datetime', garage_closed_time)
```

---

### 2. `data_filters.py`
Data filtering, transformation, and quality control functions.

**Key Functions:**
- `g_mean(x)` - Calculate geometric mean (for log-normal distributions)
- `filter_by_burn_dates(data, burn_range, datetime_column, burn_log)` - Filter to specific burn dates
- `calculate_rolling_average_burn3(data, datetime_column)` - Apply 5-minute rolling average for burn3
- `split_data_by_nan(df, x_col, y_col, gap_threshold_hours=0.1)` - Split data at NaN gaps for plotting
- `filter_by_status_columns(data, status_columns, valid_status='OK')` - Filter by instrument status
- `remove_outliers(data, column, method='iqr', threshold=1.5)` - Outlier detection and removal
- `resample_to_common_timebase(data, datetime_column, freq='1min')` - Resample data to common frequency

**Example Usage:**
```python
from scripts.data_filters import filter_by_status_columns, split_data_by_nan

# Filter AeroTrak data by status
df = filter_by_status_columns(df, status_columns=['Flow Status', 'Laser Status'])

# Split data for continuous line plotting
segments = split_data_by_nan(df, 'time', 'concentration')
```

---

### 3. `statistical_utils.py`
Statistical analysis and curve fitting functions.

**Key Functions:**
- `exponential_decay(x, a, b)` - Exponential decay model: y = a * exp(-b * x)
- `fit_exponential_curve(x_data, y_data, initial_guess=None)` - Fit exponential decay to data
- `perform_linear_fit(x_data, y_data)` - Linear regression with R², p-value, AIC
- `perform_polynomial_fit(x_data, y_data, degree=2)` - Polynomial regression
- `select_best_fit(x_data, y_data, models=None)` - Automatic best fit selection using AIC
- `perform_z_test_comparison(mu_a, sd_a, n_a, mu_b, sd_b, n_b)` - Z-test for comparing means
- `create_fitted_curve(x_data, y_data, num_points=100, kind='cubic')` - Smooth interpolation

**Example Usage:**
```python
from scripts.statistical_utils import fit_exponential_curve, exponential_decay

# Fit exponential decay to concentration data
params, y_fit, errors = fit_exponential_curve(time, concentration)
decay_rate = params[1]

# Generate smooth fitted curve
y_predicted = exponential_decay(time, *params)
```

---

### 4. `plotting_utils.py`
Bokeh plotting utilities and standardized visualization functions.

**Key Functions:**
- `get_script_metadata()` - Get script name and timestamp for annotations
- `create_standard_figure(title, ...)` - Create standardized Bokeh figure with common settings
- `apply_text_formatting(plot_object, config=None)` - Apply consistent text formatting
- `configure_legend(plot_object, location='top_right', ...)` - Configure legend settings
- `add_event_markers(plot_object, events, y_range=None)` - Add vertical lines for events
- `create_metadata_div(content, width=800)` - Create metadata annotation div
- `get_color_for_pollutant(pollutant_name)` - Get standard color for pollutant
- `get_color_palette_for_instrument(instrument_name)` - Get color palette for instrument

**Constants:**
- `INSTRUMENT_COLORS` - Color palettes for each instrument
- `POLLUTANT_COLORS` - Standard colors for PM species
- `TEXT_CONFIG` - Standard text formatting configuration

**Example Usage:**
```python
from scripts.plotting_utils import create_standard_figure, add_event_markers

# Create standardized plot
p = create_standard_figure(
    "PM2.5 Concentrations",
    y_axis_type='log',
    x_range=(-1, 4),
    y_range=(1e-4, 1e5)
)

# Add event markers
events = {'Garage Closed': 0, 'CR Boxes On': 1.5}
add_event_markers(p, events)
```

---

### 5. `instrument_config.py`
Instrument-specific configurations, bin definitions, and constants.

**Key Constants:**
- `QUANTAQ_BINS` - Bin definitions for QuantAQ particle count ranges
- `SMPS_BIN_RANGES` - Size ranges for SMPS data binning
- `AEROTRAK_CHANNELS` - Channel definitions for AeroTrak
- `UNIT_CONVERSIONS` - Common unit conversion factors
- `BURN_RANGES` - Valid burn ranges for each instrument
- `ALL_BURNS` - List of all burn IDs

**Key Functions:**
- `get_quantaq_bins()` - Get QuantAQ bin definitions
- `get_smps_bin_ranges()` - Get SMPS bin ranges
- `get_burn_range_for_instrument(instrument)` - Get valid burns for instrument
- `convert_unit(value, conversion_type)` - Convert units using standard factors
- `sum_quantaq_bins(data, bins=None)` - Sum QuantAQ bins to create size ranges
- `sum_smps_ranges(data, ranges=None)` - Sum SMPS columns by size ranges
- `get_instrument_datetime_column(instrument)` - Get datetime column name for instrument

**Example Usage:**
```python
from scripts.instrument_config import QUANTAQ_BINS, sum_quantaq_bins, convert_unit

# Sum QuantAQ bins
df = sum_quantaq_bins(df)

# Convert volume from liters to cm³
volume_cm3 = convert_unit(volume_liters, 'liter_to_cm3')

# Get valid burns for QuantAQ
from scripts.instrument_config import get_burn_range_for_instrument
valid_burns = get_burn_range_for_instrument('QuantAQB')
```

---

### 6. `data_loaders.py`
Instrument-specific data loading and processing functions.

**Key Functions:**
- `load_burn_log(burn_log_path, sheet_name="Sheet2")` - Load burn log Excel file
- `process_aerotrak_data(file_path, instrument, burn_date, ...)` - Process AeroTrak data
- `process_quantaq_data(file_path, instrument, burn_date, ...)` - Process QuantAQ data
- `process_smps_data(file_path, garage_closed_time, ...)` - Process SMPS data
- `process_dusttrak_data(file_path, burn_date, ...)` - Process DustTrak data
- `process_purpleair_data(file_path, burn_date, ...)` - Process PurpleAir data
- `process_miniams_data(file_path, burn_date, ...)` - Process MiniAMS data
- `get_garage_closed_times(burn_log, burn_numbers)` - Extract garage closed times
- `get_cr_box_times(burn_log, burn_numbers, relative_to_garage)` - Extract CR Box times

**Example Usage:**
```python
from scripts.data_loaders import (
    load_burn_log,
    get_garage_closed_times,
    process_aerotrak_data
)
from src.data_paths import get_common_file, get_instrument_path

# Load burn log and get times
burn_log = load_burn_log(get_common_file('burn_log'))
garage_times = get_garage_closed_times(burn_log, ['burn1', 'burn2'])

# Process AeroTrak data
aerotrak_path = get_instrument_path('aerotrak_bedroom') / 'all_data.xlsx'
df = process_aerotrak_data(
    aerotrak_path,
    instrument='AeroTrakB',
    burn_date='2024-01-15',
    garage_closed_time=garage_times['burn1'],
    burn_number='burn1'
)
```

---

### 7. `spatial_analysis_utils.py`
Spatial variability analysis functions for multi-location measurements.

**Key Functions:**
- `calculate_peak_ratio(peak_data, burn_id, instrument_pair, pm_size)` - Calculate Peak Ratio Index between locations
- `calculate_event_time_ratio(data1, data2, event_time, pm_size, ...)` - Concentration ratio at specific event time
- `calculate_average_ratio_and_rsd(data1, data2, start_time, pm_size, ...)` - Time-averaged ratio and RSD
- `calculate_crbox_activation_ratio(bedroom_data, morning_data, burn_id, ...)` - CR Box activation ratio (convenience wrapper)

**Example Usage:**
```python
from scripts.spatial_analysis_utils import (
    calculate_peak_ratio,
    calculate_event_time_ratio
)

# Calculate peak ratio from peak concentration data
peak_df = pd.read_excel('peak_concentrations.xlsx')
peak_ratio = calculate_peak_ratio(
    peak_df,
    burn_id='burn4',
    instrument_pair='AeroTrak',
    pm_size='PM2.5 (µg/m³)'
)
print(f"Peak Ratio Index: {peak_ratio:.2f}")

# Calculate ratio at specific event time
event_time = pd.Timestamp('2024-01-15 12:30:00')
event_ratio = calculate_event_time_ratio(
    bedroom_data,
    morning_data,
    event_time,
    pm_size='PM2.5 (µg/m³)',
    datetime_col_1='Date and Time',
    datetime_col_2='Date and Time',
    burn_date=pd.to_datetime('2024-01-15').date()
)
print(f"Event Time Ratio: {event_ratio:.2f}")
```

---

### 8. `metadata_utils.py`
Utility functions for generating metadata strings for script identification.

**Key Functions:**
- `get_script_metadata()` - Generate metadata string with calling script name and execution timestamp

**Example Usage:**
```python
from scripts.metadata_utils import get_script_metadata

# Get metadata for plot annotations or file headers
metadata = get_script_metadata()
print(metadata)  # Output: "Generated by: my_script.py | Date: 2024-01-15 14:30:00"
```

**Notes:**
- Automatically detects the calling script name using the inspect module
- Falls back to "unknown_script_name" if detection fails
- Useful for adding provenance information to plots and output files

---

### 9. `export_smps_total_concentration.py`
Standalone script to export SMPS total concentration data to CSV format.

**Purpose:**
- Reads raw SMPS (Scanning Mobility Particle Sizer) data files
- Exports simplified CSV containing datetime and total concentration
- Designed for sharing data with collaborators

**Configuration Variables:**
- `CONCENTRATION_TYPE`: Set to 'MassConc' or 'NumConc'
- `OUTPUT_PATH`: Directory for output CSV files

**Usage:**
```bash
conda run -n wui python scripts/export_smps_total_concentration.py
```

**Output:**
- CSV file with columns: `datetime`, `Total Concentration (µg/m³)` or `(#/cm³)`

---

### 10. `test_utilities.py`
Test script for verifying utility module functionality.

**Purpose:**
- Tests all utility modules to ensure correct installation and functionality
- Demonstrates usage patterns for each module
- Useful for validating changes to utility functions

**Modules Tested:**
1. `datetime_utils` - Time shifts, datetime creation, event time calculations
2. `data_filters` - Geometric mean, data splitting, status filtering, rolling averages
3. `statistical_utils` - Exponential decay fitting, linear/polynomial regression
4. `plotting_utils` - Figure creation, metadata, color functions
5. `instrument_config` - Bin definitions, unit conversions, burn ranges

**Usage:**
```bash
python scripts/test_utilities.py
```

---

## Installation / Setup

These modules are designed to be imported from the `scripts/` directory. Ensure your analysis scripts include the repository root in the Python path:

```python
import sys
from pathlib import Path

# Add repository root to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# Now you can import utility modules
from scripts.datetime_utils import apply_time_shift
from scripts.plotting_utils import create_standard_figure
```

## Dependencies

The utility modules require the following packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Statistical functions and curve fitting
- `bokeh` - Interactive plotting

These are already dependencies of the main analysis scripts.

## Usage Examples

### Complete Analysis Workflow Example

```python
import pandas as pd
from pathlib import Path
import sys

# Setup path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Import utilities
from scripts.datetime_utils import apply_time_shift, calculate_time_since_garage_closed
from scripts.data_filters import filter_by_status_columns, calculate_rolling_average_burn3
from scripts.plotting_utils import create_standard_figure, add_event_markers
from scripts.instrument_config import get_burn_range_for_instrument
from src.data_paths import get_instrument_path, get_common_file

# Load data
aerotrak_path = get_instrument_path('aerotrak_bedroom') / 'all_data.xlsx'
df = pd.read_excel(aerotrak_path)

# Process data
df = apply_time_shift(df, 'AeroTrakB', 'Date and Time')
df = filter_by_status_columns(df)
df = calculate_time_since_garage_closed(df, 'Date and Time', garage_closed_time)

# For burn3, apply rolling average
if burn_number == 'burn3':
    df = calculate_rolling_average_burn3(df, 'Date and Time')

# Create plot
p = create_standard_figure(
    title="AeroTrak Particle Concentrations",
    x_range=(-1, 4),
    y_range=(1e-4, 1e5)
)

# Add data
p.line(df['Time Since Garage Closed (hours)'], df['concentration'],
       legend_label='AeroTrak', line_width=2)

# Add event markers
events = {'Garage Closed': 0, 'CR Boxes On': 1.5}
add_event_markers(p, events)
```

### Curve Fitting Example

```python
from scripts.statistical_utils import fit_exponential_curve, exponential_decay
import numpy as np

# Prepare data (remove NaN, filter to decay period)
mask = (time > decay_start) & (time < decay_end)
x_data = time[mask]
y_data = concentration[mask]

# Fit exponential decay
params, y_fit, errors = fit_exponential_curve(x_data, y_data)

if params is not None:
    initial_conc = params[0]
    decay_rate = params[1]
    print(f"Decay rate: {decay_rate:.4f} hr⁻¹")

    # Calculate CADR
    volume_m3 = 300
    cadr = decay_rate * volume_m3
    print(f"CADR: {cadr:.1f} m³/hr")
```

## Migration Guide

When updating existing scripts to use these utilities:

### 1. Replace local function definitions
**Before:**
```python
def apply_time_shift(df, instrument, datetime_column):
    time_shifts = {"AeroTrakB": 2.16, ...}
    # ... implementation
```

**After:**
```python
from scripts.datetime_utils import apply_time_shift
# Function is now available, no local definition needed
```

### 2. Replace duplicated constants
**Before:**
```python
time_shifts = {"AeroTrakB": 2.16, "AeroTrakK": 5, ...}
```

**After:**
```python
from scripts.datetime_utils import TIME_SHIFTS
```

### 3. Simplify plotting code
**Before:**
```python
p = figure(title=title, x_axis_label="...", y_axis_label="...",
           y_axis_type="log", width=800, height=600, ...)
p.y_range = Range1d(1e-4, 1e5)
p.xgrid.grid_line_color = "lightgray"
# ... many more configuration lines
```

**After:**
```python
from scripts.plotting_utils import create_standard_figure
p = create_standard_figure(title, x_range=(-1, 4), y_range=(1e-4, 1e5))
```

## Benefits

**Reduced Code Duplication**: Functions like `apply_time_shift()` and `calculate_rolling_average_burn3()` appeared in 6+ scripts - now centralized
**Consistency**: All scripts use the same time shifts, color palettes, and processing logic
**Maintainability**: Bug fixes and improvements in one place benefit all scripts
**Documentation**: Comprehensive docstrings with examples for all functions
**Testability**: Isolated functions are easier to test and validate
**Cleaner Scripts**: Analysis scripts focus on analysis logic, not boilerplate

## Estimated Impact

Based on the codebase analysis:
- **~3000+ lines** of duplicated code across repository
- Functions extracted appear in **6-9 different scripts**
- **50-70% reduction** in script length for typical analysis scripts
- **Centralized configuration** eliminates inconsistencies

## Contributing

When adding new analysis scripts:
1. Check these utility modules first - the function you need may already exist
2. If writing a function that could be reused, consider adding it to the appropriate utility module
3. Update this README when adding new utility functions
4. Include docstrings with examples for all new functions

## Author

Nathan Lima
NIST WUI MH IAQ Project
2024-2026
