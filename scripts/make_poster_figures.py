"""
Build the four updated poster and manuscript figures without rerunning the full
analysis scripts.

Purpose:
    Regenerate, in the shared journal style (src/plot_style.py), only:
        burn9    - Burn 09 Morning Room PM comparison (HTML, PDF, PNG)
        overlay  - AeroTrak2 0.3-0.5 um coincidence overlay with zones
        lag      - per-burn inter-room arrival lag (Bedroom 2 minus Morning Room)
        modulair - Burn 06 MODULAIR-PM2 QA/QC time series with portal PM1 panel
    The full scripts also rewrite their CSV, HTML, and other PNG outputs; this
    entry point loads only what each figure needs and writes nothing else.
    Outputs are ISO-dated; an existing same-day file raises unless --overwrite.
    A figure whose input data are missing on this machine is skipped and
    reported, not imputed.

Usage (from the repository root):
    python scripts/make_poster_figures.py
    python scripts/make_poster_figures.py --only overlay lag
    python scripts/make_poster_figures.py --overwrite

Author: Nathan Lima
Created: 2026-10-16

Update log:
    2026-10-16: Initial version.
"""

import argparse
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIGURES = ("burn9", "overlay", "lag", "modulair")


def make_burn9(overwrite):
    """Burn 09 Morning Room comparison (peak_concentration_script)."""
    # Imported here: the module changes the working directory to data_root
    # and reads the burn log at import time.
    from src import peak_concentration_script as pcs

    try:
        metadata = pcs.get_script_metadata()
    except Exception:  # metadata is provenance only; never block the figure
        metadata = None
    return pcs.plot_burn_comparison(
        "burn9", script_metadata=metadata, rooms=("kitchen",), overwrite=overwrite
    )


def make_overlay(overwrite):
    """AeroTrak2 coincidence overlay (aerotrak_coincidence)."""
    from src import aerotrak_coincidence as ac

    burn_log = ac._load_burn_log()
    df_at2 = ac._load_aerotrak_all("AeroTrak2")
    results = []
    for burn_id in ac.BURN_COVERAGE["AeroTrak2"]:
        rows = burn_log[burn_log["Burn ID"] == burn_id]
        if rows.empty:
            continue
        r = ac.analyze_burn_instrument(burn_id, "AeroTrak2", df_at2, rows.iloc[0], None)
        if r is not None:
            results.append(r)
    return ac._mpl_overlay(results, overwrite=overwrite)


def make_lag(overwrite):
    """Arrival-lag figure (spatial_variation_analysis_plot)."""
    from src import spatial_variation_analysis_plot as svp

    aerotrak_full, _ = svp.load_spatial_variation_data(svp.EXCEL_FILE_PATH)
    return svp.plot_arrival_lag(aerotrak_full, svp.OUTPUT_DIR, overwrite=overwrite)


def make_modulair(overwrite):
    """Burn 06 MODULAIR-PM2 QA/QC time series (modulair_5sec_peak_window)."""
    from src import modulair_5sec_peak_window as mp

    events = mp.load_event_times()
    at = mp._load_aerotrak()
    # The worked example saturates, so the fallback half-width is not needed.
    rec = mp.analyze_pair(mp.FIG2_BURN, mp.FIG2_UNIT, events, at, np.nan)
    return mp._mpl_qaqc_timeseries_fig2([rec], overwrite=overwrite)


BUILDERS = {
    "burn9": make_burn9,
    "overlay": make_overlay,
    "lag": make_lag,
    "modulair": make_modulair,
}


def main():
    """Build the requested figures and print a written/skipped summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--overwrite", action="store_true",
                        help="replace same-day output files")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    from src.plot_style import check_font, use_journal_style
    import matplotlib.pyplot as plt

    with plt.style.context("default"):
        use_journal_style()
        font_ok = check_font()
    if not font_ok:
        print("Helvetica/Arial not found; stopping before any export.")
        return 1

    summary = {}
    for name in args.only:
        print(f"\n=== {name} ===")
        try:
            written = BUILDERS[name](args.overwrite) or []
            summary[name] = ("written", [str(p) for p in written]) if written else (
                "skipped", ["no output (see messages above)"])
        except (FileNotFoundError, KeyError) as exc:
            summary[name] = ("skipped", [f"missing input: {exc}"])
        except FileExistsError as exc:
            summary[name] = ("not written", [str(exc)])
        except Exception as exc:  # report and continue with the other figures
            traceback.print_exc()
            summary[name] = ("error", [repr(exc)])

    print("\n=== summary ===")
    for name, (status, items) in summary.items():
        print(f"{name}: {status}")
        for item in items:
            print(f"    {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
