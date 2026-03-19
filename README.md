# Manufactured Housing Wildland-Urban Interface (WUI) Fire Smoke Study

Repository for NIST WUI manufactured home indoor air quality (IAQ) study Python tools. Includes scripts for particulate matter concentration analysis, Clean Air Delivery Rate (CADR) calculations, spatial variation quantification, instrument validation, and statistical modeling of wildfire smoke infiltration events in a manufactured home test structure.

## Project Overview

This project analyzes indoor air quality during simulated wildfire smoke infiltration events in the NIST [Indoor Air Quality (IAQ) Test House](https://www.nist.gov/laboratories/tools-instruments/indoor-air-quality-test-house). The research investigates the effectiveness of various mitigation strategies for protecting indoor environments from wildfire smoke:

- Particulate matter (PM) concentration dynamics during smoke infiltration events
- Clean Air Delivery Rate (CADR) calculations for air cleaning devices
- Spatial variation in PM concentrations across different rooms
- Particle size distribution analysis (0.3 μm to 25 μm)
- Decay rate characterization for different mitigation strategies
- Filter performance evaluation (MERV 12A and MERV 13)
- Compartmentalization strategy comparison

## Disclaimer

Certain commercial equipment, instruments, software, or materials are identified in this repository in order to specify the experimental and analytical procedures adequately. Such identification is not intended to imply recommendation or endorsement of any product or service by NIST, nor is it intended to imply that the materials or equipment identified are necessarily the best available for the purpose.

## Repository Structure

```
NIST_wui_mh_iaq/
├── .vscode/                          # VS Code configuration
│   └── settings.json
├── .gitignore
├── CODEMETA.yaml                     # NIST Software Portal metadata
├── CODEOWNERS                        # Repository ownership for PR reviews
├── LICENSE.md                        # NIST software licensing statement
├── README.md
├── data_config.json                  # Active configuration (gitignored)
├── data_config.template.json         # Configuration template
├── wui.yml                           # Conda environment specification
│
├── src/                              # Core analysis scripts
│   ├── __init__.py                   # Package initialization
│   ├── data_paths.py                 # Portable data access via data_config.json
│   │
│   │ # CADR (Clean Air Delivery Rate) Analysis
│   ├── clean_air_delivery_rates_update.py
│   ├── clean_air_delivery_rates_barchart.py
│   ├── clean_air_delivery_rates_pmsizes.py
│   ├── clean_air_delivery_rates_pmsizes_SIUniformaty.py
│   ├── clean_air_delivery_rates_vs_total_surface_area.py
│   ├── cadr_comparison_statistical_analysis.py
│   │
│   │ # Compartmentalization & Mitigation Strategy Analysis
│   ├── compartmentalization_strategy_comparison.py
│   ├── decay_rate_barchart.py
│   │
│   │ # Concentration Dynamics
│   ├── conc_increase_to_decrease.py
│   ├── peak_concentration_script.py
│   │
│   │ # Spatial Variation Analysis
│   ├── spatial_variation_analysis.py
│   ├── spatial_variation_analysis_plot.py
│   ├── spatial_variation_analysis_plot_timeseries.py
│   │
│   │ # Instrument Comparison & Validation
│   ├── dusttrak-rh_comparison.py
│   ├── purpleair_comparison.py
│   ├── quantaq_pm2.5_burn8.py
│   ├── general_particle_count_comparison.py
│   ├── aham_ac1_comparison.py
│   ├── temp-rh_comparison.py
│   │
│   │ # SMPS (Scanning Mobility Particle Sizer) Analysis
│   ├── smps_filterperformance.py
│   ├── smps_finepm_comparison.py
│   ├── smps_heatmap.py
│   ├── smps_mass_vs_conc.py
│   ├── smps_size_bin_barchart.py
│   │
│   │ # Data Processing & Utilities
│   ├── remove_aerotrak_dup_data.py
│   ├── mh_relay_control_log.py
│   └── toc_figure_script.py          # Publication TOC figure generator
│
├── scripts/                          # Reusable utility modules
│   ├── __init__.py
│   ├── datetime_utils.py             # DateTime handling and time synchronization
│   ├── data_filters.py               # Data filtering and quality control
│   ├── statistical_utils.py          # Curve fitting and statistical analysis
│   ├── plotting_utils.py             # Standardized Bokeh figure creation
│   ├── instrument_config.py          # Instrument configs and bin definitions
│   ├── data_loaders.py               # Instrument-specific data loading
│   ├── spatial_analysis_utils.py     # Spatial variability calculations
│   ├── metadata_utils.py             # Script metadata and provenance strings
│   ├── export_smps_total_concentration.py
│   ├── test_utilities.py             # Utility module tests
│   └── README.md                     # Detailed module documentation
│
└── testing/                          # Diagnostic and testing scripts
    ├── diagnostic_hourly_ratios.py
    ├── diagnostic_spatial_variation.py
    └── diagnostic_timestamp_alignment.py
```

## Experimental Design

The study conducted multiple controlled burn experiments (Burn 1–10) in a manufactured home test structure with the following instrumentation:

**Monitoring Instruments:**
- **AeroTrak 9510** — Optical particle counters in bedroom and kitchen/morning room
- **QuantAQ MODULAIR-PM** — Low-cost PM sensors in multiple locations
- **TSI DustTrak DRX 8533** — Real-time aerosol monitor
- **TSI SMPS 3938 (Scanning Mobility Particle Sizer)** — Ultrafine particle size distribution (9–437 nm)
- **PurpleAir PA-II-SD** — Community-grade PM sensors
- **Vaisala** — Temperature and relative humidity sensors

**Measured Pollutants:**
- PM₀.₅, PM₁, PM₂.₅, PM₃, PM₄, PM₅, PM₁₀, PM₁₅, PM₂₅ (μg/m³)
- Particle number concentrations (#/cm³)
- Particle size distributions

**Mitigation Strategies Tested:**
- Portable air cleaners with MERV filtration (various CADR ratings)
- Central HVAC systems with MERV 12A and MERV 13 filters
- Room compartmentalization (closed bedroom with/without filtration)
- Multiple filter configurations (1 and 2 filters analyzed; 4-filter configuration excluded due to data quality issues)

## Installation

### Prerequisites

- Python 3.13+
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/usnistgov/WUI_MH_IAQ.git
   cd WUI_MH_IAQ
   ```

2. Create the conda environment:
   ```bash
   conda env create -f wui.yml
   conda activate wui
   ```

3. Configure data paths:
   ```bash
   cp data_config.template.json data_config.json
   ```
   Edit `data_config.json` to set the correct `data_root` path for your machine:
   ```json
   {
     "machine_name": "YourMachineName",
     "data_root": "C:/path/to/your/WUI_smoke",
     "instruments": { ... }
   }
   ```
   See `data_config.template.json` for the complete structure with all instruments.

4. Verify configuration:
   ```bash
   python -c "from src.data_paths import resolver; resolver.list_instruments()"
   ```

**Note:** `data_config.json` is excluded from version control (`.gitignore`). Each user maintains their own local configuration.

## Usage

### Running Analysis Scripts

Scripts in `src/` can be run from the command line or interactively in Jupyter/VS Code:

```bash
# Run CADR analysis
python src/clean_air_delivery_rates_update.py

# Run spatial variation analysis
python src/spatial_variation_analysis.py
python src/spatial_variation_analysis_plot.py
```

### Data Structure

Ensure your data files are organized as follows:

```
WUI_smoke/
├── burn_log.xlsx
├── burn_data/
│   ├── aerotraks/
│   │   ├── bedroom2/
│   │   │   └── all_data.xlsx
│   │   └── kitchen/
│   │       └── all_data.xlsx
│   ├── quantaq/
│   │   ├── MOD-PM-00194-*.csv        # QuantAQ bedroom
│   │   └── MOD-PM-00197-*.csv        # QuantAQ kitchen
│   ├── dusttrak/
│   │   └── *.xlsx
│   ├── smps/
│   │   └── *.txt
│   ├── purpleair/
│   │   └── *.csv
│   ├── miniams/
│   │   └── *.csv
│   ├── vaisala_th/
│   │   └── *.xlsx
│   └── relaycontrol/
│       └── *.txt
├── burn_dates_decay_aerotraks_bedroom.xlsx
├── burn_dates_decay_aerotraks_kitchen.xlsx
├── burn_dates_decay_smps.xlsx
├── peak_concentrations_all_instruments.xlsx
├── spatial_variation_analysis.xlsx
└── Paper_figures/                    # Output directory for plots
```

### Analysis Scripts Overview

#### CADR (Clean Air Delivery Rate) Analysis
- **`clean_air_delivery_rates_update.py`** — Primary CADR calculation with exponential decay fitting for all instruments
- **`clean_air_delivery_rates_barchart.py`** — CADR visualization across burn experiments
- **`clean_air_delivery_rates_pmsizes.py`** — Size-resolved CADR and exponential decay fitting across all instruments (AeroTrak, DustTrak, QuantAQ, SMPS, MiniAMS, PurpleAir); exports decay rates and 95% confidence intervals to Excel for downstream barchart scripts
- **`clean_air_delivery_rates_pmsizes_SIUniformaty.py`** — CADR analysis with SI unit formatting and uniformity checks
- **`clean_air_delivery_rates_vs_total_surface_area.py`** — CADR vs. particle surface area correlation
- **`cadr_comparison_statistical_analysis.py`** — Statistical comparison of CADR values across experimental conditions

#### Compartmentalization and Mitigation Strategy Analysis
- **`compartmentalization_strategy_comparison.py`** — Comparison of compartmentalization approaches (open house vs. closed bedroom vs. closed bedroom with filtration)
- **`decay_rate_barchart.py`** — Decay rate visualization and comparison

#### Concentration Dynamics
- **`conc_increase_to_decrease.py`** — Analysis of concentration increase vs. decay phase
- **`peak_concentration_script.py`** — Peak PM concentration identification and characterization

#### Spatial Variation Analysis
- **`spatial_variation_analysis.py`** — Spatial variability quantification (Peak Ratio Index, Average Ratio, RSD)
- **`spatial_variation_analysis_plot.py`** — Interactive Bokeh visualizations comparing bedroom vs. morning room
- **`spatial_variation_analysis_plot_timeseries.py`** — Time-series spatial analysis plots

#### Instrument Comparison and Validation
- **`dusttrak-rh_comparison.py`** — DustTrak performance and relative humidity effects
- **`purpleair_comparison.py`** — PurpleAir sensor validation against reference instruments
- **`quantaq_pm2_5_burn8.py`** — QuantAQ sensor analysis for specific burn experiment
- **`general_particle_count_comparison.py`** — Cross-instrument particle count comparison
- **`aham_ac1_comparison.py`** — AHAM AC-1 smoke concentration standard comparison with WUI measurements
- **`temp-rh_comparison.py`** — Temperature and relative humidity correlation with PM measurements

#### SMPS (Scanning Mobility Particle Sizer) Analysis
- **`smps_filterperformance.py`** — Filter performance evaluation using ultrafine particle measurements
- **`smps_finepm_comparison.py`** — Fine PM behavior comparison across instruments
- **`smps_heatmap.py`** — Particle size distribution evolution heatmap
- **`smps_mass_vs_conc.py`** — Mass concentration vs. number concentration analysis
- **`smps_size_bin_barchart.py`** — CADR-per-CR-box barcharts grouped by SMPS size bins (9–100 nm, 100–200 nm, 200–300 nm, 300–437 nm) for filter count, new vs. used filter, and MERV grade comparisons

#### Data Processing Utilities
- **`data_paths.py`** — Portable path resolver; reads `data_config.json` to provide machine-independent access to instrument data folders and common files without hardcoded paths
- **`remove_aerotrak_dup_data.py`** — Duplicate AeroTrak timestamp removal
- **`mh_relay_control_log.py`** — HVAC relay control log processing
- **`toc_figure_script.py`** — Publication table of contents figure generator

#### Diagnostic and Testing Scripts (`testing/`)
- **`diagnostic_timestamp_alignment.py`** — Examines raw AeroTrak timestamps and resampling behavior to diagnose merge failures in hourly spatial variation bins
- **`diagnostic_hourly_ratios.py`** — Diagnostic analysis of hourly concentration ratios between instruments
- **`diagnostic_spatial_variation.py`** — Diagnostic checks for spatial variation calculation inputs and outputs

### Reusable Utility Modules (`scripts/`)

The `scripts/` directory contains shared utility modules used across analysis scripts. See [`scripts/README.md`](scripts/README.md) for full documentation.

| Module | Purpose |
|--------|---------|
| `datetime_utils.py` | DateTime handling, time shifts, synchronization |
| `data_filters.py` | Data filtering, quality control, transformations |
| `statistical_utils.py` | Curve fitting, exponential decay, regression |
| `plotting_utils.py` | Standardized Bokeh figure creation |
| `instrument_config.py` | Instrument configs, bin definitions, constants |
| `data_loaders.py` | Instrument-specific data loading functions |
| `spatial_analysis_utils.py` | Spatial variability calculations |
| `metadata_utils.py` | Script metadata and provenance strings |
| `export_smps_total_concentration.py` | Reads raw SMPS files and exports a combined datetime/total-concentration CSV for sharing; configurable for mass or number concentration; also runnable as a standalone script |
| `test_utilities.py` | Validates all shared utility modules (datetime_utils, data_filters, statistical_utils, plotting_utils, instrument_config) with assertions; run to verify environment setup after installation |

### Output

Most analysis scripts generate interactive Bokeh HTML plots saved to the configured `Paper_figures/` output directory. HTML files can be opened in any web browser for interactive data exploration.

## Data Processing Methods

### CADR Calculation

Clean Air Delivery Rate is calculated using first-order exponential decay analysis:

```
C(t) = C₀ × exp(-kt)

CADR = k × V

where:
  C(t) = PM concentration at time t
  C₀   = Initial (peak) concentration
  k    = Decay rate constant (min⁻¹)
  V    = Room volume (m³)
```

### Spatial Variation Metrics
- **Peak Ratio Index (PRI)**: Ratio of peak concentrations between locations
- **Average Ratio**: Time-averaged concentration ratio during the decay period
- **Relative Standard Deviation (RSD)**: Coefficient of variation across locations

### Uncertainty Quantification

Decay rate and CADR uncertainties are propagated through two stages.

**Stage 1 — Decay rate uncertainty**

Exponential decay curves are fitted with `scipy.optimize.curve_fit` (non-linear least squares). The 95% confidence interval on the decay rate *k* is derived from the covariance matrix returned by the solver:

```
σ_k = 1.96 × √pcov[1, 1]
```

Fits where the relative standard deviation exceeds 10% (σ_k / k > 0.10) are flagged and excluded from further analysis.

**Stage 2 — CADR uncertainty**

CADR is computed as V × (k − k_baseline). Uncertainty is propagated using the range method:

```
σ_CADR = V × [(k + σ_k) − (k_baseline − σ_k_baseline)]
       − V × [(k − σ_k) − (k_baseline + σ_k_baseline)]
       = 2V × (σ_k + σ_k_baseline)
```

where *V* is the effective room volume (m³). When two baseline burns are available, *k_baseline* and *σ_k_baseline* are calculated as an inverse-variance weighted average. CADR-per-CR-box uncertainty is obtained by dividing σ_CADR by the number of CR boxes.

### Statistical Analysis
- Exponential curve fitting with `scipy.optimize.curve_fit`
- Uncertainty propagation for CADR calculations
- Multi-instrument data synchronization
- Baseline subtraction

## Configured Instruments

| Instrument | Model | Purpose | Burns Available |
|-----------|-------|---------|----------------|
| AeroTrak Bedroom | TSI AeroTrak 9510 | Optical particle counting | 1–10 |
| AeroTrak Kitchen | TSI AeroTrak 9510 | Optical particle counting | 1–10 |
| DustTrak | TSI DRX 8533 | Real-time aerosol monitoring | 1–10 |
| MiniAMS | Mini Aerosol Mass Spectrometer | Chemical species analysis | 1–10 |
| PurpleAir | PA-II-SD | Low-cost PM monitoring | 6–10 |
| QuantAQ Bedroom | MODULAIR-PM (MOD-PM-00194) | PM monitoring | 4–10 |
| QuantAQ Kitchen | MODULAIR-PM (MOD-PM-00197) | PM monitoring | 4–10 |
| SMPS | TSI SMPS 3938 | Ultrafine size distribution (9–437 nm) | 1–10 |
| Vaisala T/RH | Vaisala sensors | Temperature and relative humidity | 1–10 |

## Data Availability

The experimental data associated with this project are not included in this repository. Data can be made available upon request by contacting the PI.

## Citation

If you use this software, please cite it as:

```bibtex
@software{nist_wui_mh_iaq,
  author       = {Lima, Nathan M. and Poppendieck, Dustin G.},
  title        = {Manufactured Housing Wildland-Urban Interface (WUI) Fire Smoke Study: Indoor Air Quality Analysis Tools},
  year         = {2025},
  publisher    = {National Institute of Standards and Technology},
  url          = {https://github.com/usnistgov/WUI_MH_IAQ}
}
```

**Related Publications:**
- Publication under review. Citation will be updated upon journal acceptance.

## Contact

- **PI:** Dustin G. Poppendieck
- **NIST Organizational Unit:** Engineering Laboratory
- **Division:** Building Energy & Environment Division
- **Group:** Indoor Air Quality and Ventilation Group
- **Email:** dustin.poppendieck@nist.gov

## Acknowledgments

This research was conducted at the National Institute of Standards and Technology (NIST) as part of the [Indoor Air Quality and Ventilation Group](https://www.nist.gov/el/beed/indoor-air-quality-ventilation) and [Wildland-Urban Interface Fire Group](https://www.nist.gov/el/fire/wildland-urban-interface-fire).

Data collection period: April – June 2024

## License

This software was developed by employees of the National Institute
of Standards and Technology (NIST), an agency of the Federal
Government and is being made available as a public service. Pursuant
to [Title 17 United States Code Section 105][usc-17], works of NIST
employees are not subject to copyright protection in the United
States. This software may be subject to foreign copyright. Permission
in the United States and in foreign countries, to the extent that
NIST may hold copyright, to use, copy, modify, create derivative
works, and distribute this software and its documentation without
fee is hereby granted on a non-exclusive basis, provided that this
notice and disclaimer of warranty appears in all copies.

See [LICENSE.md](LICENSE.md) for the full NIST licensing statement.


<!-- Link definitions -->

[18f-guide]: https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
[cornell-meta]: https://data.research.cornell.edu/content/readme
[gh-cdo]: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
[gh-mdn]: https://github.github.com/gfm/
[gh-nst]: https://github.com/usnistgov
[gh-odi]: https://odiwiki.nist.gov/ODI/GitHub.html
[gh-osr]: https://github.com/usnistgov/opensource-repo/
[gh-ost]: https://github.com/orgs/usnistgov/teams/opensource-team
[gh-rob]: https://odiwiki.nist.gov/pub/ODI/GitHub/GHROB.pdf
[li-bsd]: https://opensource.org/licenses/bsd-license
[li-gpl]: https://opensource.org/licenses/gpl-license
[li-mit]: https://opensource.org/licenses/mit-license
[nist-code]: https://code.nist.gov
[nist-disclaimer]: https://www.nist.gov/open/license
[nist-s-1801-02]: https://inet.nist.gov/adlp/directives/review-data-intended-publication
[nist-open]: https://www.nist.gov/open/license#software
[usc-17]: https://www.copyright.gov/title17/
[wk-rdm]: https://en.wikipedia.org/wiki/README
