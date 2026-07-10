"""
Spatial Analysis Utilities for NIST WUI MH IAQ Analysis

This module provides functions for calculating spatial variability metrics
between different measurement locations in indoor air quality experiments.

Functions:
    - calculate_peak_ratio: Calculate ratio of peak concentrations between locations
    - calculate_event_time_ratio: Calculate concentration ratio at specific event time
    - calculate_average_ratio_and_rsd: Calculate time-averaged ratio and RSD
    - calculate_subthreshold_ratio: Median ratio over steps where both units are
      below a reliable-regime concentration threshold
    - calculate_arrival_lag: Signed inter-room lag between threshold crossings

Author: Nathan Lima
Date: 2024-2025
Update log:
    2026-07-10 (N. Lima): Added calculate_subthreshold_ratio and
        calculate_arrival_lag for the Section 3.3 count-independent spatial
        metrics. These support the AeroTrak inter-room comparison when peak
        counts sit in the Poisson-suppression regime documented in Section
        3.2.2 (rollover ceiling near 760 particles/cm^3, PM3 mass unreliable
        above ~901 ug/m^3).
"""

import pandas as pd
import numpy as np

try:
    from scripts.datetime_utils import create_naive_datetime
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.datetime_utils import create_naive_datetime


def calculate_peak_ratio(peak_data, burn_id, instrument_pair, pm_size):
    """
    Calculate ratio of peak concentrations between two locations.

    This function calculates the Peak Ratio Index (PRI), which quantifies
    spatial variability by comparing maximum concentrations between locations
    during a burn event.

    Parameters:
    -----------
    peak_data : pd.DataFrame
        DataFrame containing peak concentration data with columns formatted as
        '{InstrumentName}_{PMSize}' (e.g., 'AeroTrakB_PM2.5 (µg/m³)')
    burn_id : str
        Burn identifier (e.g., 'burn4')
    instrument_pair : str
        Instrument type: 'AeroTrak' or 'QuantAQ'
    pm_size : str
        PM size column name (e.g., 'PM2.5 (µg/m³)')

    Returns:
    --------
    float or None
        Ratio of location1/location2 peak concentrations, or None if data unavailable

    Examples:
    ---------
    >>> peak_df = pd.DataFrame({
    ...     'Burn_ID': ['burn4', 'burn5'],
    ...     'AeroTrakB_PM2.5 (µg/m³)': [150.0, 200.0],
    ...     'AeroTrakK_PM2.5 (µg/m³)': [100.0, 180.0]
    ... })
    >>> ratio = calculate_peak_ratio(peak_df, 'burn4', 'AeroTrak', 'PM2.5 (µg/m³)')
    >>> print(f"{ratio:.2f}")
    1.50

    Notes:
    ------
    - Returns None if either location has missing data
    - Returns None if denominator (location2) is zero
    - Column naming convention: '{Instrument}{Location}_{PMSize}'
      where Location is 'B' for bedroom2 or 'K' for morning room (kitchen)
    """
    # Build column names based on instrument pair
    if instrument_pair == "AeroTrak":
        location1_col = f"AeroTrakB_{pm_size}"
        location2_col = f"AeroTrakK_{pm_size}"
    else:  # QuantAQ
        location1_col = f"QuantAQB_{pm_size}"
        location2_col = f"QuantAQK_{pm_size}"

    # Check if columns exist
    available_cols = peak_data.columns.tolist()

    if location1_col not in available_cols or location2_col not in available_cols:
        return None

    # Get peak values for the specific burn
    burn_data = peak_data[peak_data["Burn_ID"] == burn_id]

    if burn_data.empty:
        return None

    location1_peak = burn_data[location1_col].values[0]
    location2_peak = burn_data[location2_col].values[0]

    # Check for valid data
    if pd.isna(location1_peak) or pd.isna(location2_peak) or location2_peak == 0:
        return None

    # Calculate ratio (location1/location2)
    ratio = location1_peak / location2_peak

    return ratio


def calculate_event_time_ratio(
    data_location1,
    data_location2,
    event_time,
    pm_size,
    datetime_col_1,
    datetime_col_2,
    time_window_minutes=5,
    burn_date=None,
):
    """
    Calculate concentration ratio at a specific event time between two locations.

    This function is commonly used to calculate the CR Box Activation Ratio,
    which represents spatial variation at the moment an air cleaner was turned on.
    More generally, it can calculate the ratio at any event time.

    Parameters:
    -----------
    data_location1 : pd.DataFrame
        Time-series concentration data for location 1
    data_location2 : pd.DataFrame
        Time-series concentration data for location 2
    event_time : pd.Timestamp or datetime
        Event datetime to calculate ratio at
    pm_size : str
        PM size column name (e.g., 'PM2.5 (µg/m³)')
    datetime_col_1 : str
        Datetime column name for location 1 data
    datetime_col_2 : str
        Datetime column name for location 2 data
    time_window_minutes : float, optional
        Time window (±) to search for closest measurement (default: 5 minutes)
    burn_date : datetime.date, optional
        Burn date to filter data (if None, uses all data)

    Returns:
    --------
    float or None
        Ratio of location1/location2 concentration at event time,
        or None if data is unavailable

    Examples:
    ---------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>>
    >>> # Create sample data
    >>> data1 = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01 12:00', periods=20, freq='1min'),
    ...     'PM2.5 (µg/m³)': [100 + i*5 for i in range(20)],
    ...     'Date': pd.to_datetime('2024-01-01').date()
    ... })
    >>> data2 = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01 12:00', periods=20, freq='1min'),
    ...     'PM2.5 (µg/m³)': [80 + i*4 for i in range(20)],
    ...     'Date': pd.to_datetime('2024-01-01').date()
    ... })
    >>> event = pd.Timestamp('2024-01-01 12:10')
    >>>
    >>> ratio = calculate_event_time_ratio(
    ...     data1, data2, event, 'PM2.5 (µg/m³)',
    ...     'datetime', 'datetime',
    ...     burn_date=pd.to_datetime('2024-01-01').date()
    ... )

    Notes:
    ------
    - Finds the measurement closest to event_time within the time window
    - Returns None if no data found within the time window
    - Returns None if either location has invalid/missing data
    - Both datasets must have a 'Date' column if burn_date is specified
    """
    if pd.isna(event_time):
        return None

    # Filter data for the burn date if specified
    if burn_date is not None:
        burn_date_only = pd.to_datetime(burn_date).date()
        data1_burn = data_location1[data_location1["Date"] == burn_date_only].copy()
        data2_burn = data_location2[data_location2["Date"] == burn_date_only].copy()
    else:
        data1_burn = data_location1.copy()
        data2_burn = data_location2.copy()

    if data1_burn.empty or data2_burn.empty:
        return None

    # Find measurements within time window
    time_window = pd.Timedelta(minutes=time_window_minutes)

    data1_window = data1_burn[
        (data1_burn[datetime_col_1] >= event_time - time_window)
        & (data1_burn[datetime_col_1] <= event_time + time_window)
    ].copy()

    data2_window = data2_burn[
        (data2_burn[datetime_col_2] >= event_time - time_window)
        & (data2_burn[datetime_col_2] <= event_time + time_window)
    ].copy()

    if data1_window.empty or data2_window.empty:
        return None

    # Check if PM size column exists
    if pm_size not in data1_window.columns or pm_size not in data2_window.columns:
        return None

    # Ensure PM columns are numeric
    data1_window[pm_size] = pd.to_numeric(data1_window[pm_size], errors="coerce")
    data2_window[pm_size] = pd.to_numeric(data2_window[pm_size], errors="coerce")

    # Get the measurement closest to event time
    data1_window["time_diff"] = abs(
        (data1_window[datetime_col_1] - event_time).dt.total_seconds()
    )
    data2_window["time_diff"] = abs(
        (data2_window[datetime_col_2] - event_time).dt.total_seconds()
    )

    location1_closest = data1_window.loc[data1_window["time_diff"].idxmin()]
    location2_closest = data2_window.loc[data2_window["time_diff"].idxmin()]

    location1_conc = location1_closest[pm_size]
    location2_conc = location2_closest[pm_size]

    # Check for valid data
    if pd.isna(location1_conc) or pd.isna(location2_conc) or location2_conc <= 0:
        return None

    # Calculate ratio (location1/location2)
    ratio = location1_conc / location2_conc

    return ratio


def calculate_average_ratio_and_rsd(
    data_location1,
    data_location2,
    start_time,
    pm_size,
    datetime_col_1,
    datetime_col_2,
    analysis_duration_hours=2,
    burn_date=None,
    outlier_threshold=(0.1, 10),
    min_data_points=5,
    resample_freq="1T",
):
    """
    Calculate time-averaged concentration ratio and relative standard deviation (RSD).

    This function computes the average concentration ratio between two locations
    over a specified time window and quantifies the temporal variability using RSD.

    Parameters:
    -----------
    data_location1 : pd.DataFrame
        Time-series concentration data for location 1
    data_location2 : pd.DataFrame
        Time-series concentration data for location 2
    start_time : pd.Timestamp or datetime
        Start time for analysis window
    pm_size : str
        PM size column name (e.g., 'PM2.5 (µg/m³)')
    datetime_col_1 : str
        Datetime column name for location 1 data
    datetime_col_2 : str
        Datetime column name for location 2 data
    analysis_duration_hours : float, optional
        Duration of analysis window in hours (default: 2)
    burn_date : datetime.date, optional
        Burn date to filter data (if None, uses all data)
    outlier_threshold : tuple of (float, float), optional
        (lower, upper) bounds for valid ratios (default: (0.1, 10))
    min_data_points : int, optional
        Minimum number of data points required (default: 5)
    resample_freq : str, optional
        Resampling frequency for alignment (default: '1T' for 1 minute)

    Returns:
    --------
    tuple of (float, float) or (None, None)
        (average_ratio, rsd_percent) or (None, None) if calculation fails

    Examples:
    ---------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>>
    >>> # Create sample data with 2 hours of measurements
    >>> data1 = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01 12:00', periods=120, freq='1min'),
    ...     'PM2.5 (µg/m³)': 100 * np.exp(-np.arange(120) / 60),  # Exponential decay
    ...     'Date': pd.to_datetime('2024-01-01').date()
    ... })
    >>> data2 = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01 12:00', periods=120, freq='1min'),
    ...     'PM2.5 (µg/m³)': 80 * np.exp(-np.arange(120) / 60),
    ...     'Date': pd.to_datetime('2024-01-01').date()
    ... })
    >>>
    >>> start = pd.Timestamp('2024-01-01 12:00')
    >>> avg_ratio, rsd = calculate_average_ratio_and_rsd(
    ...     data1, data2, start, 'PM2.5 (µg/m³)',
    ...     'datetime', 'datetime',
    ...     burn_date=pd.to_datetime('2024-01-01').date()
    ... )

    Notes:
    ------
    - Data from both locations is resampled to common timebase before ratio calculation
    - Outlier ratios outside the threshold range are excluded
    - Returns (None, None) if insufficient valid data points
    - RSD (Relative Standard Deviation) = (std/mean) * 100%
    """
    if pd.isna(start_time):
        return None, None

    # Define analysis window
    end_time = start_time + pd.Timedelta(hours=analysis_duration_hours)

    # Filter data for burn date if specified
    if burn_date is not None:
        burn_date_only = pd.to_datetime(burn_date).date()
        data1_burn = data_location1[data_location1["Date"] == burn_date_only].copy()
        data2_burn = data_location2[data_location2["Date"] == burn_date_only].copy()
    else:
        data1_burn = data_location1.copy()
        data2_burn = data_location2.copy()

    if data1_burn.empty or data2_burn.empty:
        return None, None

    # Filter for the analysis window
    data1_window = data1_burn[
        (data1_burn[datetime_col_1] >= start_time)
        & (data1_burn[datetime_col_1] <= end_time)
    ].copy()

    data2_window = data2_burn[
        (data2_burn[datetime_col_2] >= start_time)
        & (data2_burn[datetime_col_2] <= end_time)
    ].copy()

    if data1_window.empty or data2_window.empty:
        return None, None

    # Check if PM size column exists
    if pm_size not in data1_window.columns or pm_size not in data2_window.columns:
        return None, None

    # Ensure PM columns are numeric
    data1_window[pm_size] = pd.to_numeric(data1_window[pm_size], errors="coerce")
    data2_window[pm_size] = pd.to_numeric(data2_window[pm_size], errors="coerce")

    # Remove NaN values
    data1_window = data1_window.dropna(subset=[pm_size])
    data2_window = data2_window.dropna(subset=[pm_size])

    if data1_window.empty or data2_window.empty:
        return None, None

    # Align data by time (resample to common timebase)
    data1_resample = data1_window[[datetime_col_1, pm_size]].copy()
    data2_resample = data2_window[[datetime_col_2, pm_size]].copy()

    # Set index and resample
    data1_resample = (
        data1_resample.set_index(datetime_col_1)[pm_size]
        .resample(resample_freq)
        .mean()
    )
    data2_resample = (
        data2_resample.set_index(datetime_col_2)[pm_size]
        .resample(resample_freq)
        .mean()
    )

    # Convert back to DataFrame for merging
    data1_resample = pd.DataFrame({f"{pm_size}_location1": data1_resample})
    data2_resample = pd.DataFrame({f"{pm_size}_location2": data2_resample})

    # Merge on time index
    merged = pd.merge(
        data1_resample,
        data2_resample,
        left_index=True,
        right_index=True,
        how="inner",
    )

    if merged.empty:
        return None, None

    # Remove rows with NaN or zero/negative values
    merged = merged.dropna()
    merged = merged[
        (merged[f"{pm_size}_location2"] > 0) & (merged[f"{pm_size}_location1"] > 0)
    ]

    if merged.empty or len(merged) < min_data_points:
        return None, None

    # Calculate ratios for each time point
    ratios = merged[f"{pm_size}_location1"] / merged[f"{pm_size}_location2"]

    # Remove outliers
    lower_bound, upper_bound = outlier_threshold
    ratios = ratios[(ratios > lower_bound) & (ratios < upper_bound)]

    if ratios.empty:
        return None, None

    # Calculate average ratio and RSD
    avg_ratio = ratios.mean()
    std_ratio = ratios.std()
    rsd = (std_ratio / avg_ratio * 100) if avg_ratio > 0 else None

    return avg_ratio, rsd


def calculate_crbox_activation_ratio(
    bedroom_data,
    morning_data,
    burn_id,
    pm_size,
    datetime_col_b,
    datetime_col_m,
    burn_log,
):
    """
    Calculate concentration ratio at CR Box activation time (convenience wrapper).

    This is a convenience function that wraps calculate_event_time_ratio()
    specifically for CR Box activation events, extracting the event time
    from the burn log.

    Parameters:
    -----------
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
    burn_log : pd.DataFrame
        Burn log DataFrame with 'Burn ID', 'Date', and 'CR Box on' columns

    Returns:
    --------
    float or None
        Ratio of bedroom2/morning room concentration at CR Box activation,
        or None if data is unavailable

    Examples:
    ---------
    >>> burn_log = pd.DataFrame({
    ...     'Burn ID': ['burn4'],
    ...     'Date': ['2024-01-01'],
    ...     'CR Box on': ['12:30:00']
    ... })
    >>> ratio = calculate_crbox_activation_ratio(
    ...     bedroom_df, morning_df, 'burn4', 'PM2.5 (µg/m³)',
    ...     'Date and Time', 'Date and Time', burn_log
    ... )

    Notes:
    ------
    - This is specific to WUI experiments with CR Box interventions
    - Returns None if CR Box was not used for this burn
    - Uses a ±5 minute time window to find closest measurement
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

    # Use the general event_time_ratio function
    burn_date_only = pd.to_datetime(burn_date).date()

    return calculate_event_time_ratio(
        bedroom_data,
        morning_data,
        cr_box_time,
        pm_size,
        datetime_col_b,
        datetime_col_m,
        time_window_minutes=5,
        burn_date=burn_date_only,
    )


def calculate_subthreshold_ratio(
    data_bed,
    data_morning,
    burn_date,
    pm_size,
    datetime_col_b,
    datetime_col_m,
    threshold=500.0,
    outlier_threshold=(0.1, 10),
    resample_freq="1T",
):
    """
    Median Bedroom 2 / Morning Room ratio in the reliable concentration regime.

    Restricts the comparison to time steps where *both* units report a
    concentration below ``threshold``. Section 3.1 shows co-located instruments
    agree within a factor of two below 500 ug/m^3, so a ratio computed only over
    those steps is independent of the count-suppression regime that inflates the
    peak ratio above ~901 ug/m^3 (Section 3.2.2). A median near 1 means the large
    peak gap is mostly saturation; a median well above or below 1 means real
    inter-room heterogeneity persists in the reliable regime.

    Parameters
    ----------
    data_bed : pd.DataFrame
        Bedroom 2 time-series data (numerator location).
    data_morning : pd.DataFrame
        Morning Room time-series data (denominator location).
    burn_date : datetime-like
        Burn day used to select rows via the ``Date`` column present on both
        frames.
    pm_size : str
        PM size column name (e.g. ``'PM3 (ug/m3)'``).
    datetime_col_b : str
        Datetime column name for the bedroom frame.
    datetime_col_m : str
        Datetime column name for the morning room frame.
    threshold : float, optional
        Upper concentration bound (ug/m^3) both units must satisfy for a step to
        count (default 500.0, the Section 3.1 agreement level).
    outlier_threshold : tuple of (float, float), optional
        (lower, upper) bounds for valid ratios (default (0.1, 10)), matching the
        outlier rule used by ``calculate_average_ratio_and_rsd``.
    resample_freq : str, optional
        Resampling frequency for time alignment (default ``'1T'`` = 1 minute).

    Returns
    -------
    tuple of (float, float, float, int)
        ``(median, iqr_low, iqr_high, n_matched_steps)`` where the IQR bounds are
        the 25th and 75th percentiles of the retained ratios. Returns
        ``(nan, nan, nan, 0)`` if no step satisfies the mask or the PM column is
        absent.
    """
    burn_date_only = pd.to_datetime(burn_date).date()

    bed_burn = data_bed[data_bed["Date"] == burn_date_only].copy()
    morning_burn = data_morning[data_morning["Date"] == burn_date_only].copy()

    if bed_burn.empty or morning_burn.empty:
        return np.nan, np.nan, np.nan, 0

    if pm_size not in bed_burn.columns or pm_size not in morning_burn.columns:
        return np.nan, np.nan, np.nan, 0

    bed_burn[pm_size] = pd.to_numeric(bed_burn[pm_size], errors="coerce")
    morning_burn[pm_size] = pd.to_numeric(morning_burn[pm_size], errors="coerce")

    bed_burn = bed_burn.dropna(subset=[pm_size])
    morning_burn = morning_burn.dropna(subset=[pm_size])

    if bed_burn.empty or morning_burn.empty:
        return np.nan, np.nan, np.nan, 0

    # Align to a common 1-min timebase, same idiom as calculate_average_ratio_and_rsd
    bed_resample = (
        bed_burn[[datetime_col_b, pm_size]]
        .set_index(datetime_col_b)[pm_size]
        .resample(resample_freq)
        .mean()
    )
    morning_resample = (
        morning_burn[[datetime_col_m, pm_size]]
        .set_index(datetime_col_m)[pm_size]
        .resample(resample_freq)
        .mean()
    )

    merged = pd.merge(
        pd.DataFrame({"bed": bed_resample}),
        pd.DataFrame({"morning": morning_resample}),
        left_index=True,
        right_index=True,
        how="inner",
    ).dropna()

    # Reliable regime: both units below the agreement threshold and positive
    mask = (
        (merged["bed"] < threshold)
        & (merged["morning"] < threshold)
        & (merged["bed"] > 0)
        & (merged["morning"] > 0)
    )
    merged = merged[mask]

    if merged.empty:
        return np.nan, np.nan, np.nan, 0

    ratios = merged["bed"] / merged["morning"]

    lower_bound, upper_bound = outlier_threshold
    ratios = ratios[(ratios > lower_bound) & (ratios < upper_bound)]

    if ratios.empty:
        return np.nan, np.nan, np.nan, 0

    median = float(ratios.median())
    iqr_low = float(ratios.quantile(0.25))
    iqr_high = float(ratios.quantile(0.75))
    n_matched = int(len(ratios))

    return median, iqr_low, iqr_high, n_matched


def calculate_arrival_lag(
    data_bed,
    data_morning,
    burn_date,
    pm_size,
    datetime_col_b,
    datetime_col_m,
    threshold,
):
    """
    Signed inter-room lag between the two units crossing a fixed threshold.

    Finds the first time each unit's concentration rises to or above
    ``threshold`` on the burn day and returns the difference. Because it depends
    only on when each series first reaches an absolute level, not on the peak
    magnitude, this measure is valid even when the peak counts sit in the
    Poisson-suppression regime (Section 3.2.2). It is the primary
    count-independent evidence for room-to-room smoke transport.

    Parameters
    ----------
    data_bed : pd.DataFrame
        Bedroom 2 time-series data.
    data_morning : pd.DataFrame
        Morning Room time-series data.
    burn_date : datetime-like
        Burn day used to select rows via the ``Date`` column.
    pm_size : str
        PM size column name (e.g. ``'PM3 (ug/m3)'``).
    datetime_col_b : str
        Datetime column name for the bedroom frame.
    datetime_col_m : str
        Datetime column name for the morning room frame.
    threshold : float
        Absolute concentration (ug/m^3) whose first crossing defines arrival.

    Returns
    -------
    float
        ``t_cross(morning) - t_cross(bedroom)`` in minutes. Negative means the
        Morning Room crossed first (smoke arrived there earlier). NaN if either
        unit never crosses the threshold on the burn day.
    """
    burn_date_only = pd.to_datetime(burn_date).date()

    def _first_cross(data, dt_col):
        sub = data[data["Date"] == burn_date_only].copy()
        if sub.empty or pm_size not in sub.columns:
            return pd.NaT
        sub[pm_size] = pd.to_numeric(sub[pm_size], errors="coerce")
        sub = sub.dropna(subset=[pm_size, dt_col]).sort_values(dt_col)
        crossed = sub[sub[pm_size] >= threshold]
        if crossed.empty:
            return pd.NaT
        return crossed[dt_col].iloc[0]

    t_bed = _first_cross(data_bed, datetime_col_b)
    t_morning = _first_cross(data_morning, datetime_col_m)

    if pd.isna(t_bed) or pd.isna(t_morning):
        return np.nan

    return (pd.Timestamp(t_morning) - pd.Timestamp(t_bed)).total_seconds() / 60.0
