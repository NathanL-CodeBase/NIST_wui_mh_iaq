"""
Burn 01 natural (no-filtration) size-dependent decay report from SMPS bands.

Purpose: Extract the four coarse SMPS band decay rates for Burn 01 (2024-04-26)
         from the already-fitted SMPS_decay_and_CADR.xlsx, convert each first-
         order rate k (h^-1) to a half-life (ln(2)/k, minutes), and write a
         compact CSV for the Section 3.4 rewrite. Burn 01 is the whole-house
         baseline with no portable air cleaner, so its band decays are natural
         deposition/infiltration losses, not filtration-driven removal. Every
         other burn's SMPS decay includes a CR Box and is not a natural-decay
         measurement.
Author:  Nathan Lima
Created: 2026-07-10
Updates:

Notes
-----
No decay fitting happens here. The fits, the 95 % CI (1.96 * standard error),
and the RSD > 10 % exclusion are produced upstream by
clean_air_delivery_rates_pmsizes.py (dataset = "SMPS"). If
SMPS_decay_and_CADR.xlsx is missing, run that script with dataset = "SMPS"
first.
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_paths import get_common_file

# Band label (as written by the upstream CADR script) mapped to its nm edges.
# The upstream labels carry the Greek capital sigma and the micro sign; match
# them exactly so the row lookup does not silently miss.
BAND_EDGES_NM = {
    "Ʃ9-100nm (µg/m³)": (9, 100),
    "Ʃ100-200nm (µg/m³)": (100, 200),
    "Ʃ200-300nm (µg/m³)": (200, 300),
    "Ʃ300-437nm (µg/m³)": (300, 437),
}

# RSD threshold matches the upstream fit-quality gate.
RSD_THRESHOLD = 0.10

BURN_ID = "burn1"


def _versioned_path(path: Path) -> Path:
    """Return ``path`` unchanged, or an ISO-date-suffixed variant if it exists.

    Never overwrites an existing output file silently (per project safety
    defaults); appends the current date before the suffix instead.

    Parameters
    ----------
    path : Path
        Intended output path.

    Returns
    -------
    Path
        The original path if free, otherwise ``<stem>_<YYYY-MM-DD><suffix>``.
    """
    if not path.exists():
        return path
    stamp = datetime.date.today().isoformat()
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def build_burn1_decay_report(xlsx_path: Path) -> pd.DataFrame:
    """Pull Burn 01 band rows and compute half-lives.

    Parameters
    ----------
    xlsx_path : Path
        Path to SMPS_decay_and_CADR.xlsx.

    Returns
    -------
    DataFrame
        One row per coarse band with columns: band, lower_nm, upper_nm,
        k_per_hour, k_ci95_per_hour, half_life_minutes, rsd, rsd_flag.
    """
    df = pd.read_excel(xlsx_path)

    burn1 = df[df["burn"] == BURN_ID]
    if burn1.empty:
        raise ValueError(
            f"No '{BURN_ID}' rows found in {xlsx_path.name}. "
            "Re-run clean_air_delivery_rates_pmsizes.py with dataset='SMPS'."
        )

    rows = []
    for band, (lo, hi) in BAND_EDGES_NM.items():
        band_row = burn1[burn1["pollutant"] == band]
        if band_row.empty:
            # An empty band means the upstream fit was excluded (RSD > 10 %)
            # or absent; record it as excluded rather than dropping silently.
            rows.append(
                {
                    "band": band,
                    "lower_nm": lo,
                    "upper_nm": hi,
                    "k_per_hour": np.nan,
                    "k_ci95_per_hour": np.nan,
                    "half_life_minutes": np.nan,
                    "rsd": np.nan,
                    "rsd_flag": "excluded_or_missing",
                }
            )
            continue

        k = float(band_row["decay"].iloc[0])
        k_ci95 = float(band_row["decay_uncertainty"].iloc[0])
        rsd = float(band_row["rsd"].iloc[0])
        # ln(2)/k gives the first-order half-life; convert hours to minutes.
        half_life_min = (np.log(2) / k * 60.0) if k > 0 else np.nan
        rsd_flag = "excluded" if rsd > RSD_THRESHOLD else "ok"

        rows.append(
            {
                "band": band,
                "lower_nm": lo,
                "upper_nm": hi,
                "k_per_hour": k,
                "k_ci95_per_hour": k_ci95,
                "half_life_minutes": half_life_min,
                "rsd": rsd,
                "rsd_flag": rsd_flag,
            }
        )

    return pd.DataFrame(rows)


def main():
    xlsx_path = get_common_file("burn_calcs") / "SMPS_decay_and_CADR.xlsx"
    if not xlsx_path.exists():
        sys.exit(
            f"Not found: {xlsx_path}\n"
            "Run clean_air_delivery_rates_pmsizes.py with dataset='SMPS' first."
        )

    report = build_burn1_decay_report(xlsx_path)

    # Output folder: smps_decay_report/ beside the burn_calcs data.
    out_dir = get_common_file("burn_calcs").parent / "smps_decay_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _versioned_path(out_dir / "smps_natural_decay_burn1.csv")
    report.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Console report for the Section 3.4 rewrite.
    print("Burn 01 natural (no-filtration) SMPS band decay")
    print("=" * 70)
    for _, r in report.iterrows():
        if r["rsd_flag"] in ("excluded", "excluded_or_missing"):
            print(
                f"  {r['lower_nm']:>3}-{r['upper_nm']:<3} nm : "
                f"EXCLUDED ({r['rsd_flag']}, RSD={r['rsd']})"
            )
            continue
        lo95 = r["k_per_hour"] - r["k_ci95_per_hour"]
        hi95 = r["k_per_hour"] + r["k_ci95_per_hour"]
        print(
            f"  {r['lower_nm']:>3}-{r['upper_nm']:<3} nm : "
            f"k = {r['k_per_hour']:.2f} h^-1 "
            f"(95% CI {lo95:.2f} to {hi95:.2f}); "
            f"half-life ~{r['half_life_minutes']:.1f} min"
        )

    # Ultrafine vs accumulation contrast (the specific numbers the text needs).
    uf = report[report["band"] == "Ʃ9-100nm (µg/m³)"]
    acc = report[report["band"].isin(["Ʃ200-300nm (µg/m³)", "Ʃ300-437nm (µg/m³)"])]
    if not uf.empty and not acc.empty and uf["rsd_flag"].iloc[0] == "ok":
        k_uf = uf["k_per_hour"].iloc[0]
        acc_ok = acc[acc["rsd_flag"] == "ok"]
        if not acc_ok.empty:
            k_acc = acc_ok["k_per_hour"].mean()
            print("-" * 70)
            print(
                f"  Ultrafine (9-100 nm) k = {k_uf:.2f} h^-1 exceeds "
                f"accumulation (200-437 nm) mean k = {k_acc:.2f} h^-1 by "
                f"{k_uf - k_acc:.2f} h^-1 ({(k_uf / k_acc - 1) * 100:.0f} %)."
            )

    print("=" * 70)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
