"""
Decay-window sensitivity: Morning Room decay rates on a common fit window.

Purpose:
    The published Morning Room decay rates (companion paper; burn_calcs
    *_decay_and_CADR_edited.xlsx) were fitted over a parameter-based window
    chosen per instrument: start at the latest of the concentration maximum,
    portable air cleaner activation, and the first stable log-slope; end at the
    first value below 5 % of the maximum. The windows therefore differ between
    co-located instruments. This script tests how much the decay rates change
    when every co-located instrument is fitted over one common window.

    For each burn with at least two co-located Morning Room instruments
    (Burns 06 to 10; the DustTrak is in the Morning Room only for Burns 07 to
    10), the script:
        1. reads each instrument's published window and decay rate;
        2. refits the published window with the same loaders and the same
           model, C(t) = A exp(-k t), to confirm the published value is
           reproduced;
        3. fits every instrument over the common window, defined as the
           intersection of the published windows (latest start to earliest
           end), so all instruments see the same time span of the same decay;
        4. reports the per-instrument change and the across-instrument spread
           (max k / min k) for both windows.

    The existing decay analysis and its output files are not modified. This is
    a new, separate sensitivity analysis.

Inputs (through data_config.json):
    burn log, instrument files read by src/clean_air_delivery_rates_pmsizes.py,
    burn_calcs/<instrument>_decay_and_CADR_edited.xlsx (published windows).

Outputs (under burn_calcs/decay_window_sensitivity/, ISO-dated):
    YYYY-MM-DD_decay_window_sensitivity.csv      per burn x instrument
    YYYY-MM-DD_decay_window_sensitivity.md       tables and summary
    YYYY-MM-DD_decay_window_sensitivity.pdf/.png per-burn decay curves with
                                                 the common window shaded

Usage (from the repository root):
    python src/decay_window_sensitivity.py
    python src/decay_window_sensitivity.py --overwrite

Author: Nathan Lima
Created: 2026-10-28
Update log:
    2026-10-28: Initial version (response to reviewer comment on decay-rate
        consistency across instruments, v4 manuscript comment 10).
"""

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
# The CADR module reads the burn log at import; its loaders and fit function are
# reused unchanged so the refit matches the published analysis.
from src import clean_air_delivery_rates_pmsizes as cadr  # noqa: E402
from src.data_paths import get_common_file  # noqa: E402
from src.plot_style import figsize as journal_figsize  # noqa: E402
from src.project_style import (  # noqa: E402
    INSTRUMENT_COLORS,
    INSTRUMENT_LINESTYLES,
    REF_LINE,
    journal_style,
    save_pdf_png,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Morning Room instruments: CADR-module dataset key -> (manuscript name,
# pollutant fitted in the published analysis). The DustTrak PM2.5 equals its
# total photometric mass in this campaign (all size fractions identical).
INSTRUMENTS = {
    "DustTrak": ("DustTrak", "PM2.5 (µg/m³)"),
    "AeroTrakK": ("AeroTrak2", "PM3 (µg/m³)"),
    "QuantAQK": ("MODULAIR-PM2", "PM2.5 (µg/m³)"),
    "PurpleAirK": ("PurpleAir", "PM2.5 (µg/m³)"),
}

# Burns with the instrument in the Morning Room. The DustTrak sat in Bedroom 2
# for Burns 01 to 06, so its Burn 06 decay is not a Morning Room value.
MORNING_ROOM_BURNS = {
    "DustTrak": ["burn7", "burn8", "burn9", "burn10"],
    "AeroTrakK": ["burn6", "burn7", "burn8", "burn9", "burn10"],
    "QuantAQK": ["burn6", "burn7", "burn8", "burn9", "burn10"],
    "PurpleAirK": ["burn6", "burn7", "burn8", "burn9", "burn10"],
}
BURNS = ["burn6", "burn7", "burn8", "burn9", "burn10"]

# Published decay files (the companion-paper values).
DECAY_FILE = "{inst}_decay_and_CADR_edited.xlsx"

# Fit-quality flag used by the published analysis (RSD = SE(k) / k).
RSD_MAX = 0.10
# Minimum points in a window for a fit to be reported.
MIN_POINTS = 5
# Published-versus-refit agreement flag (fractional difference in k).
REFIT_TOL = 0.05

TIME_COL = "Time Since Garage Closed (hours)"

# Plotting extent around the decay (hours since garage closure).
X_LO, X_HI = -0.25, 1.5

OUT_SUBDIR = "decay_window_sensitivity"


# ==============================================================================
# DATA
# ==============================================================================


def load_instrument(inst: str) -> pd.DataFrame:
    """
    Load one Morning Room instrument with the published CADR-module loader.

    Parameters
    ----------
    inst : str
        CADR-module dataset key (e.g. "DustTrak", "QuantAQK").

    Returns
    -------
    pd.DataFrame
        Loader output (time-shifted, unit-converted) with a 'Date' column and
        the instrument's datetime column.
    """
    config = cadr.INSTRUMENT_CONFIG[inst]
    path = cadr.get_instrument_file_path(config["instrument_key"], config.get("filename"))
    func = config["process_function"]
    if func == "process_aerotrak_data":
        return cadr.process_aerotrak_data(path, inst)
    if func == "process_dusttrak_data":
        return cadr.process_dusttrak_data(path)
    if func == "process_purpleairk_data":
        return cadr.process_purpleairk_data(path)
    if func == "process_quantaq_data":
        return cadr.process_quantaq_data(path, inst)
    raise ValueError(f"No loader mapped for {inst} ({func}).")


def burn_series(data: pd.DataFrame, inst: str, burn_id: str) -> pd.DataFrame | None:
    """
    Return one burn's record as hours since garage closure and concentration.

    Uses the burn-log garage-closure time, as the published analysis does.

    Returns
    -------
    pd.DataFrame or None
        Columns TIME_COL and 'conc' (ug/m3), sorted by time; None if the burn
        or the pollutant is absent.
    """
    pollutant = INSTRUMENTS[inst][1]
    dt_col = cadr.INSTRUMENT_CONFIG[inst]["datetime_column"]
    row = cadr.burn_log[cadr.burn_log["Burn ID"] == burn_id]
    if row.empty or pollutant not in data.columns:
        return None
    burn_date = pd.to_datetime(row["Date"].values[0])
    garage = cadr.create_naive_datetime(burn_date.date(), row["garage closed"].values[0])
    if pd.isna(garage):
        return None
    day = data[pd.to_datetime(data["Date"]) == burn_date].copy()
    if day.empty:
        return None
    t = pd.to_datetime(day[dt_col])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    out = pd.DataFrame({
        TIME_COL: (t - garage).dt.total_seconds() / 3600.0,
        "conc": pd.to_numeric(day[pollutant], errors="coerce"),
    }).dropna().sort_values(TIME_COL)
    return out.reset_index(drop=True)


def load_published(inst: str) -> pd.DataFrame:
    """Published decay rows for one instrument (first sheet of the edited file)."""
    path = get_common_file("burn_calcs") / DECAY_FILE.format(inst=inst)
    df = pd.read_excel(path, sheet_name=0)
    return df[df["pollutant"] == INSTRUMENTS[inst][1]].set_index("burn")


# ==============================================================================
# FIT
# ==============================================================================


def fit_window(series: pd.DataFrame, t0: float, t1: float) -> dict:
    """
    Fit C(t) = A exp(-k (t - t0)) over [t0, t1] with the published fit function.

    Returns
    -------
    dict
        k (1/h), ci95 (1/h, 1.96 x SE as published), rsd, n (points); NaN when
        fewer than MIN_POINTS positive values fall in the window or the fit
        fails.
    """
    out = dict(k=np.nan, ci95=np.nan, rsd=np.nan, n=0)
    if series is None or not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return out
    m = (series[TIME_COL] >= t0) & (series[TIME_COL] <= t1) & (series["conc"] > 0)
    win = series[m]
    out["n"] = int(len(win))
    if len(win) < MIN_POINTS:
        return out
    x = win[TIME_COL].to_numpy() - t0
    y = win["conc"].to_numpy()
    try:
        popt, _, perr = cadr.fit_exponential_curve(x, y, [y[0], 0.1])
    except Exception as exc:  # curve_fit raises RuntimeError on non-convergence
        print(f"    fit failed over [{t0:.3f}, {t1:.3f}] h: {exc}")
        return out
    out.update(k=float(popt[1]), ci95=float(1.96 * perr[1]),
               rsd=float(perr[1] / popt[1]) if popt[1] != 0 else np.nan)
    return out


# ==============================================================================
# ANALYSIS
# ==============================================================================


def analyze() -> tuple[pd.DataFrame, dict]:
    """
    Run the published-window refit and the common-window fit for every burn.

    Returns
    -------
    rows : pd.DataFrame
        One row per burn x instrument.
    series : dict
        (burn, inst) -> burn series, for the figure.
    """
    published = {inst: load_published(inst) for inst in INSTRUMENTS}
    data = {}
    for inst in INSTRUMENTS:
        print(f"Loading {inst} ...")
        data[inst] = load_instrument(inst)

    rows, series = [], {}
    for burn in BURNS:
        insts = [i for i in INSTRUMENTS if burn in MORNING_ROOM_BURNS[i]
                 and burn in published[i].index]
        windows = {}
        for inst in insts:
            s = burn_series(data[inst], inst, burn)
            series[(burn, inst)] = s
            p = published[inst].loc[burn]
            windows[inst] = (float(p["decay_start_time"]), float(p["decay_end_time"]))
            if s is not None and not s.empty:
                # Verification: print the range and units of each fitted series.
                print(f"  {burn} {INSTRUMENTS[inst][0]}: {INSTRUMENTS[inst][1]} "
                      f"min {s['conc'].min():.3g}, max {s['conc'].max():.3g} µg/m³")
        if len(windows) < 2:
            print(f"  {burn}: fewer than two Morning Room instruments, skipped.")
            continue
        c0 = max(w[0] for w in windows.values())
        c1 = min(w[1] for w in windows.values())
        if c1 <= c0:
            print(f"  {burn}: published windows do not overlap; no common window.")
        for inst in insts:
            p = published[inst].loc[burn]
            t0, t1 = windows[inst]
            ref = fit_window(series[(burn, inst)], t0, t1)
            com = fit_window(series[(burn, inst)], c0, c1) if c1 > c0 else fit_window(None, 0, 0)
            k_pub = float(p["decay"])
            rows.append({
                "burn": burn,
                "instrument": INSTRUMENTS[inst][0],
                "pollutant": INSTRUMENTS[inst][1],
                "pub_start_h": t0, "pub_end_h": t1,
                "k_published_per_h": k_pub,
                "ci95_published_per_h": float(p["decay_uncertainty"]),
                "k_refit_per_h": ref["k"], "n_refit": ref["n"],
                "refit_matches": bool(np.isfinite(ref["k"])
                                      and abs(ref["k"] - k_pub) / k_pub <= REFIT_TOL),
                "common_start_h": c0, "common_end_h": c1,
                "k_common_per_h": com["k"], "ci95_common_per_h": com["ci95"],
                "rsd_common": com["rsd"], "n_common": com["n"],
                "rsd_flag_common": bool(np.isfinite(com["rsd"]) and com["rsd"] > RSD_MAX),
                "change_frac": (com["k"] - k_pub) / k_pub if np.isfinite(com["k"]) else np.nan,
            })
    return pd.DataFrame(rows), series


def spread_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per burn: max/min decay rate across instruments for each window."""
    out = []
    for burn, g in df.groupby("burn", sort=False):
        kp, kc = g["k_published_per_h"], g["k_common_per_h"].dropna()
        out.append({
            "burn": burn,
            "n_instruments": len(g),
            "common_window_h": f"{g['common_start_h'].iloc[0]:.2f} to {g['common_end_h'].iloc[0]:.2f}",
            "k_range_published": f"{kp.min():.2g} to {kp.max():.2g}",
            "spread_published": kp.max() / kp.min(),
            "k_range_common": f"{kc.min():.2g} to {kc.max():.2g}" if len(kc) else "",
            "spread_common": kc.max() / kc.min() if len(kc) >= 2 else np.nan,
        })
    return pd.DataFrame(out)


# ==============================================================================
# OUTPUTS
# ==============================================================================


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_tables(df: pd.DataFrame, spread: pd.DataFrame, stem: Path, overwrite: bool) -> list:
    """Write the per-instrument CSV and a Markdown summary."""
    csv_path, md_path = stem.with_suffix(".csv"), stem.with_suffix(".md")
    for p in (csv_path, md_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists. Rerun with --overwrite or rename it.")
    df.to_csv(csv_path, index=False, float_format="%.4g", encoding="utf-8")

    show = df[["burn", "instrument", "pub_start_h", "pub_end_h", "k_published_per_h",
               "k_refit_per_h", "refit_matches", "k_common_per_h", "ci95_common_per_h",
               "n_common", "rsd_flag_common", "change_frac"]].copy()
    for c in ("pub_start_h", "pub_end_h"):
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    for c in ("k_published_per_h", "k_refit_per_h", "k_common_per_h", "ci95_common_per_h"):
        show[c] = show[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.2f}")
    show["change_frac"] = show["change_frac"].map(
        lambda v: "" if not np.isfinite(v) else f"{100 * v:+.0f} %")
    sp = spread.copy()
    for c in ("spread_published", "spread_common"):
        sp[c] = sp[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.2f}")

    n_mismatch = int((~df["refit_matches"]).sum())
    lines = [
        "# Morning Room decay rates: published windows versus a common window",
        "",
        f"Run {date.today().isoformat()}. Model C(t) = A exp(-k t), fitted with the "
        "published curve_fit routine. Published values from "
        "`<instrument>_decay_and_CADR_edited.xlsx`. Times are hours since "
        "garage-door closure. The common window is the intersection of the "
        "published windows for that burn. k in h⁻¹; CI is 1.96 × SE.",
        "",
        f"Refit check: {len(df) - n_mismatch} of {len(df)} published windows reproduce "
        f"the published k within {int(REFIT_TOL * 100)} %.",
        "",
        "## Across-instrument spread (max k / min k)",
        "",
        _md_table(sp),
        "",
        "## Per instrument",
        "",
        _md_table(show),
        "",
        f"rsd_flag_common is True where SE(k)/k exceeds {RSD_MAX:.2f}, the "
        "published exclusion threshold; those common-window values are reported "
        "but are not reliable.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    [CSV] {csv_path.name}")
    print(f"    [MD] {md_path.name}")
    return [csv_path, md_path]


def plot_curves(df: pd.DataFrame, series: dict, stem: Path, overwrite: bool) -> list:
    """
    Per-burn decay curves (log y) with each published fit and the common window.

    Each instrument keeps its Figure 2 color and line style. The thin black
    segment over each instrument's own window is the published-window refit;
    the grey band is the common window.
    """
    burns = list(dict.fromkeys(df["burn"]))
    n = len(burns)
    inv = {v[0]: k for k, v in INSTRUMENTS.items()}
    with journal_style():
        fig, axes = plt.subplots(n, 1, figsize=(journal_figsize("double")[0], 1.9 * n + 0.6),
                                 sharex=True, squeeze=False)
        for ax, burn in zip(axes[:, 0], burns):
            g = df[df["burn"] == burn]
            c0, c1 = g["common_start_h"].iloc[0], g["common_end_h"].iloc[0]
            if c1 > c0:
                ax.axvspan(c0, c1, color="#999999", alpha=0.25, lw=0)
            for r in g.itertuples():
                inst = inv[r.instrument]
                s = series.get((burn, inst))
                if s is None or s.empty:
                    continue
                m = (s[TIME_COL] >= X_LO) & (s[TIME_COL] <= X_HI) & (s["conc"] > 0)
                ax.semilogy(s.loc[m, TIME_COL], s.loc[m, "conc"],
                            color=INSTRUMENT_COLORS.get(r.instrument, REF_LINE),
                            ls=INSTRUMENT_LINESTYLES.get(r.instrument, "-"), lw=1.0,
                            label=r.instrument)
                if np.isfinite(r.k_refit_per_h):
                    w = s[(s[TIME_COL] >= r.pub_start_h) & (s[TIME_COL] <= r.pub_end_h)]
                    if len(w):
                        a = w["conc"].iloc[0]
                        xx = np.linspace(r.pub_start_h, r.pub_end_h, 50)
                        ax.semilogy(xx, a * np.exp(-r.k_refit_per_h * (xx - r.pub_start_h)),
                                    color=REF_LINE, lw=0.6)
            ax.set_ylabel(f"Burn {int(burn.replace('burn', '')):02d}\n" + r"PM (µg·m$^{-3}$)")
            ax.axvline(0.0, color=REF_LINE, lw=0.5)
        # Legend from the instrument list, so an instrument absent from the
        # first burn (the DustTrak in Burn 06) still appears.
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Line2D([0], [0], color=INSTRUMENT_COLORS.get(v[0], REF_LINE),
                          ls=INSTRUMENT_LINESTYLES.get(v[0], "-"), lw=1.0, label=v[0])
                   for v in INSTRUMENTS.values()]
        handles += [Line2D([0], [0], color=REF_LINE, lw=0.6, label="Published-window fit"),
                    Patch(color="#999999", alpha=0.25, lw=0, label="Common window")]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        axes[-1, 0].set_xlim(X_LO, X_HI)
        axes[-1, 0].set_xlabel("Time since garage closed (h)")
        return save_pdf_png(fig, stem, overwrite=overwrite)


def main() -> None:
    """Run the sensitivity analysis and write the dated tables and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace same-day ISO-dated outputs instead of raising")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    out_dir = get_common_file("burn_calcs") / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{date.today().isoformat()}_decay_window_sensitivity"

    df, series = analyze()
    if df.empty:
        print("No burns with two or more Morning Room instruments; nothing written.")
        return
    spread = spread_table(df)
    print("\nAcross-instrument spread (max k / min k):")
    print(spread.to_string(index=False))
    write_tables(df, spread, stem, args.overwrite)
    plot_curves(df, series, stem, args.overwrite)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
